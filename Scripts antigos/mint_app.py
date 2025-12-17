import warnings
import pandas as pd
import numpy as np
import gradio as gr
import io
import tempfile
import os
import networkx as nx  # Biblioteca para construir a hierarquia (grafos)
from statsforecast.reconciliation import HierarchicalReconciliation  # Onde o MinT mora!

# --- Configurações Iniciais ---
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# --- CSS Personalizado (Mantido do seu exemplo!) ---
custom_css = """
.orange-button {
    background: linear-gradient(to right, #FF8C00, #FFA500) !important;
    color: white !important;
    border: none !important;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    transition: transform 0.2s, box-shadow 0.2s;
}
.orange-button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
}
"""

# --- FUNÇÕES CORE (LÓGICA DA RECONCILIAÇÃO) ---

def construir_matriz_s(df_hierarquia, all_unique_ids):
    """
    Constrói a Matriz de Agregação (S) a partir de uma lista de adjacência (Child, Parent).
    Usa networkx para montar o gráfico e encontrar os descendentes.
    """
    # Garante que os IDs dos pais também estejam na lista de nós, caso sejam raízes
    all_nodes = set(all_unique_ids)
    
    # Criar o grafo direcionado
    G = nx.DiGraph()
    for _, row in df_hierarquia.iterrows():
        G.add_edge(row['Parent_ID'], row['Child_ID']) # Pai -> Filho

    # Encontrar os nós "folha" (bottom-level) - são os que não têm filhos no grafo
    bottom_nodes = [node for node in all_nodes if not G.has_node(node) or G.out_degree(node) == 0]
    
    # Garante que a ordem das colunas (bottom_nodes) e do índice (all_nodes) seja consistente
    bottom_nodes = sorted(list(set(bottom_nodes)))
    all_nodes_sorted = sorted(list(all_nodes))
    
    # Inicializa a matriz S com zeros
    S_df = pd.DataFrame(0, index=all_nodes_sorted, columns=bottom_nodes)
    
    # Preenche a matriz S
    for node in all_nodes_sorted:
        if node in bottom_nodes:
            S_df.loc[node, node] = 1
        else:
            # Encontra todos os descendentes "folha" daquele nó
            try:
                descendants = nx.descendants(G, node)
                for desc in descendants:
                    if desc in bottom_nodes:
                        S_df.loc[node, desc] = 1
            except nx.NetworkXError:
                # Se o nó não estiver no grafo (ex: um nó folha que não é filho de ninguém)
                pass
                
    return S_df

def executar_reconciliacao(file_dados, file_hierarquia, progress=gr.Progress(track_tqdm=True)):
    """
    Executa o pipeline completo de reconciliação hierárquica.
    """
    if file_dados is None or file_hierarquia is None:
        raise gr.Error("Por favor, faça o upload dos DOIS arquivos: Dados e Hierarquia.")

    try:
        progress(0, desc="🔄 Carregando arquivos...")
        # Forçar os tipos de dados para evitar problemas com IDs numéricos
        df_dados = pd.read_csv(file_dados.name, dtype={'unique_id': str})
        df_hierarquia = pd.read_csv(file_hierarquia.name, dtype={'Child_ID': str, 'Parent_ID': str})

        # --- Validação dos Arquivos ---
        if 'unique_id' not in df_dados.columns or 'timestamp' not in df_dados.columns or \
           'forecast' not in df_dados.columns or 'actual' not in df_dados.columns:
            raise gr.Error("Arquivo de DADOS precisa ter as colunas: 'unique_id', 'timestamp', 'forecast', 'actual'")
        
        # [CORREÇÃO] Bloco duplicado removido daqui
        
        if 'Child_ID' not in df_hierarquia.columns or 'Parent_ID' not in df_hierarquia.columns:
            raise gr.Error("Arquivo de HIERARQUIA precisa ter as colunas: 'Child_ID', 'Parent_ID'")

        # --- [NOVO] Validação SHAP ---
        progress(0.1, desc="🔎 Procurando colunas SHAP...")
        shap_cols = sorted([col for col in df_dados.columns if col.lower().startswith('shap_')])
        
        if not shap_cols:
            raise gr.Error("NENHUMA coluna 'shap_...' encontrada! Para este método, o 'Arquivo de Dados' precisa conter as colunas de contribuição SHAP.")
        
        # Validar se a soma dos SHAPs bate com o forecast (MUITO IMPORTANTE!)
        try:
            shap_sum_test = df_dados[shap_cols].sum(axis=1)
            forecast_test = df_dados['forecast']
            # Usar 'isclose' para problemas de ponto flutuante
            if not np.allclose(shap_sum_test, forecast_test, atol=0.01):
                 gr.Warning(f"AVISO: A soma das colunas SHAP (~{shap_sum_test.iloc[0]:.2f}) não bate com a coluna 'forecast' ({forecast_test.iloc[0]:.2f}). "
                            "Certifique-se de que incluiu o 'shap_Intercept' ou 'base_value' no arquivo!")
        except Exception as e:
            gr.Warning(f"Não foi possível validar a soma dos SHAPs: {e}")

        # --- Construção da Hierarquia (Matriz S) ---
        progress(0.2, desc="🛠️ Construindo a Matriz de Hierarquia (S)...")
        all_unique_ids = df_dados['unique_id'].unique()
        
        # Validação extra: todos os nós da hierarquia estão nos dados?
        nodes_hierarquia = set(df_hierarquia['Child_ID']).union(set(df_hierarquia['Parent_ID']))
        nodes_dados = set(all_unique_ids)
        
        if not nodes_hierarquia.issubset(nodes_dados):
            missing_nodes = nodes_hierarquia - nodes_dados
            raise gr.Error(f"Erro! Os seguintes IDs da hierarquia não foram encontrados nos dados: {missing_nodes}")

        S_df = construir_matriz_s(df_hierarquia, all_unique_ids)

        # --- Preparação dos DataFrames (Y_hat, Y_hist) ---
        progress(0.5, desc="📊 Pivotando dados (Y_hist)...")
        df_dados['timestamp'] = pd.to_datetime(df_dados['timestamp'])

        # [MODIFICADO] Pivotar Y (histórico/atual) - Usado para o 'mint_cov'
        # Precisamos dos erros (actual - forecast) para a matriz de covariância
        Y_df_actual = df_dados.pivot(index='timestamp', columns='unique_id', values='actual').fillna(0)
        Y_df_forecast = df_dados.pivot(index='timestamp', columns='unique_id', values='forecast').fillna(0)
        
        # Y_df agora representa os RESIDUAIS (erros) do modelo original
        Y_df_residuals = Y_df_actual - Y_df_forecast
        
        # Garantir que as colunas (unique_id) estejam na MESMA ordem da Matriz S
        Y_df_residuals = Y_df_residuals[S_df.index]
        
        # --- [MODIFICADO] Reconciliação (A Mágica!) ---
        progress(0.7, desc="✨ Iniciando Reconciliação de Componentes...")
        reconciler = HierarchicalReconciliation()
        
        reconciled_components_dfs = [] # Para guardar os DFs de SHAP reconciliados
        Y_r_total_df = None # Para somar os componentes e ter o forecast final
        
        for i, shap_col in enumerate(shap_cols):
            progress(0.7 + (0.2 * (i / len(shap_cols))), desc=f"Componente {i+1}/{len(shap_cols)}: Reconciliando {shap_col}...")
            
            # Pivotar o componente SHAP (Y_hat para este componente)
            Y_hat_comp = df_dados.pivot(index='timestamp', columns='unique_id', values=shap_col).fillna(0)
            Y_hat_comp = Y_hat_comp[S_df.index] # Garantir a ordem
            
            # Reconciliar este componente!
            # Usamos os RESIDUAIS (Y_df_residuals) para o método 'mint_cov'
            Y_r_comp_df = reconciler.reconcile(
                Y_hat_df=Y_hat_comp,
                Y_df=Y_df_residuals, # Usando os residuais para a matriz de covariância
                S_df=S_df,
                method='mint_cov'
            )
            
            # Armazenar o componente reconciliado (para o Excel)
            comp_melted = Y_r_comp_df.reset_index().melt(
                id_vars='timestamp', 
                var_name='unique_id', 
                value_name=f"{shap_col}_reconciliado"
            )
            reconciled_components_dfs.append(comp_melted)
            
            # Somar para o forecast final
            if Y_r_total_df is None:
                Y_r_total_df = Y_r_comp_df
            else:
                Y_r_total_df = Y_r_total_df + Y_r_comp_df

        # --- Formatação da Saída ---
        progress(0.9, desc="💾 Formatando e Gerando Excel...")
        
        # Converter o total somado (forecast final) para o formato long
        df_reconciliado_long = Y_r_total_df.reset_index().melt(
            id_vars='timestamp', 
            var_name='unique_id', 
            value_name='forecast_reconciliado'
        )
        
        # Juntar com os dados originais para comparação
        df_final = pd.merge(
            df_dados,
            df_reconciliado_long,
            on=['timestamp', 'unique_id'],
            how='left'
        )
        
        # Juntar todos os SHAPs reconciliados individuais
        for comp_df in reconciled_components_dfs:
            df_final = pd.merge(df_final, comp_df, on=['timestamp', 'unique_id'], how='left')
        
        # Ordenar para ficar bonito
        df_final = df_final.sort_values(by=['unique_id', 'timestamp']).reset_index(drop=True)

        # Gerar arquivo Excel
        temp_dir = tempfile.mkdtemp()
        file_name = "Previsoes_Ajustadas_MinT.xlsx"
        file_path = os.path.join(temp_dir, file_name)

        df_final.to_excel(file_path, sheet_name='Dados_Ajustados_MinT', index=False)
        
        progress(1.0, desc="Pronto!")

        return (
            df_final, # Para a prévia no DataFrame
            gr.update(value=file_path, visible=True) # Para o botão de Download
        )

    except Exception as e:
        # Imprime o erro no console para debug
        print(f"Erro detalhado: {e}")
        import traceback
        traceback.print_exc()
        raise gr.Error(f"Ocorreu um erro: {str(e)}")


# --- CONSTRUÇÃO DA INTERFACE (GRADIO) ---

with gr.Blocks(theme=gr.themes.Soft(), title="Ajuste de Previsões (MinT)", css=custom_css) as demo:
    gr.Markdown("# 🚀 Ajustador de Previsões Hierárquicas (MinT)")
    gr.Markdown("Tamo junto, Ricardo! Faça o upload dos seus dados e da definição da hierarquia para aplicar a reconciliação MinT e deixar suas previsões consistentes.")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1. Arquivo de Dados (Long Format \"DNA\")")
            gr.Markdown("CSV (long format) com: `unique_id`, `timestamp`, `forecast`, `actual` e **TODAS as colunas `shap_...`** (incluindo `shap_Intercept`!)")
            file_dados_input = gr.File(label="Upload de Dados (.csv)")
        
        with gr.Column(scale=1):
            gr.Markdown("### 2. Arquivo de Hierarquia")
            gr.Markdown("CSV contendo: `Child_ID`, `Parent_ID` (Ex: 'SKU_123', 'Categoria_A')")
            file_hierarquia_input = gr.File(label="Upload da Hierarquia (.csv)")

    run_button = gr.Button("✨ Executar Ajuste MinT!", elem_classes=["orange-button"])

    with gr.Tabs() as results_tabs:
        with gr.TabItem("📊 Dados Reconciliados"):
            gr.Markdown("Prévia dos dados. `forecast_reconciliado` é a **soma dos SHAPs ajustados** (colunas `..._reconciliado`).")
            data_output = gr.DataFrame(label="Prévia dos Dados Ajustados", wrap=True)
        
        with gr.TabItem("💾 Download"):
            download_output = gr.File(label="Baixar Planilha Ajustada (.xlsx)", visible=False)

    # --- LÓGICA DOS EVENTOS ---
    
    run_button.click(
        executar_reconciliacao,
        inputs=[
            file_dados_input, 
            file_hierarquia_input
        ],
        outputs=[
            data_output,
            download_output
        ]
    )

if __name__ == "__main__":
    demo.launch(debug=True)


