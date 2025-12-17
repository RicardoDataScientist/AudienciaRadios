import warnings
import pandas as pd
import numpy as np
import gradio as gr
import io
import tempfile
import os
import networkx as nx  # Biblioteca para construir a hierarquia (grafos)
from hierarchicalforecast.core import HierarchicalReconciliation  # ✅ Correto
from hierarchicalforecast.methods import MinTrace  # Método de reconciliação

# --- Configurações Iniciais ---
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# --- CSS Personalizado ---
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

# --- FUNÇÕES CORE ---

def construir_matriz_s(df_hierarquia, all_unique_ids):
    """
    Constrói a Matriz de Agregação (S) a partir de uma lista de adjacência (Child, Parent).
    Usa networkx para montar o gráfico e encontrar os descendentes.
    """
    all_nodes = set(all_unique_ids)
    G = nx.DiGraph()
    for _, row in df_hierarquia.iterrows():
        G.add_edge(row['Parent_ID'], row['Child_ID'])  # Pai -> Filho

    bottom_nodes = [node for node in all_nodes if not G.has_node(node) or G.out_degree(node) == 0]
    bottom_nodes = sorted(list(set(bottom_nodes)))
    all_nodes_sorted = sorted(list(all_nodes))
    
    S_df = pd.DataFrame(0, index=all_nodes_sorted, columns=bottom_nodes)
    
    for node in all_nodes_sorted:
        if node in bottom_nodes:
            S_df.loc[node, node] = 1
        else:
            try:
                descendants = nx.descendants(G, node)
                for desc in descendants:
                    if desc in bottom_nodes:
                        S_df.loc[node, desc] = 1
            except nx.NetworkXError:
                pass
                
    return S_df


def executar_reconciliacao(file_dados, file_hierarquia, progress=gr.Progress(track_tqdm=True)):
    """
    Executa o pipeline completo de reconciliação hierárquica usando o novo pacote hierarchicalforecast.
    """
    if file_dados is None or file_hierarquia is None:
        raise gr.Error("Por favor, faça o upload dos DOIS arquivos: Dados e Hierarquia.")

    try:
        progress(0, desc="🔄 Carregando arquivos...")
        df_dados = pd.read_csv(file_dados.name, dtype={'unique_id': str})
        df_hierarquia = pd.read_csv(file_hierarquia.name, dtype={'Child_ID': str, 'Parent_ID': str})

        # --- Validação dos Arquivos ---
        if 'unique_id' not in df_dados.columns or 'timestamp' not in df_dados.columns or \
           'forecast' not in df_dados.columns or 'actual' not in df_dados.columns:
            raise gr.Error("Arquivo de DADOS precisa ter as colunas: 'unique_id', 'timestamp', 'forecast', 'actual'")
        
        if 'Child_ID' not in df_hierarquia.columns or 'Parent_ID' not in df_hierarquia.columns:
            raise gr.Error("Arquivo de HIERARQUIA precisa ter as colunas: 'Child_ID', 'Parent_ID'")

        # --- Validação SHAP ---
        progress(0.1, desc="🔎 Procurando colunas SHAP...")
        shap_cols = sorted([col for col in df_dados.columns if col.lower().startswith('shap_')])
        if not shap_cols:
            raise gr.Error("NENHUMA coluna 'shap_...' encontrada! Inclua as colunas SHAP (ex: shap_Intercept).")

        shap_sum_test = df_dados[shap_cols].sum(axis=1)
        forecast_test = df_dados['forecast']
        if not np.allclose(shap_sum_test, forecast_test, atol=0.01):
            gr.Warning("⚠️ A soma dos SHAPs não corresponde à previsão! Verifique se há 'shap_Intercept'.")

        # --- Construção da Hierarquia (Matriz S) ---
        progress(0.2, desc="🛠️ Construindo a Matriz S...")
        all_unique_ids = df_dados['unique_id'].unique()
        nodes_hierarquia = set(df_hierarquia['Child_ID']).union(set(df_hierarquia['Parent_ID']))
        nodes_dados = set(all_unique_ids)
        
        if not nodes_hierarquia.issubset(nodes_dados):
            missing_nodes = nodes_hierarquia - nodes_dados
            raise gr.Error(f"Erro! IDs da hierarquia ausentes nos dados: {missing_nodes}")

        S_df = construir_matriz_s(df_hierarquia, all_unique_ids)

        # --- Preparação dos DataFrames ---
        progress(0.5, desc="📊 Pivotando dados...")
        df_dados['timestamp'] = pd.to_datetime(df_dados['timestamp'])
        Y_df_actual = df_dados.pivot(index='timestamp', columns='unique_id', values='actual').fillna(0)
        Y_df_forecast = df_dados.pivot(index='timestamp', columns='unique_id', values='forecast').fillna(0)
        Y_df_residuals = Y_df_actual - Y_df_forecast
        Y_df_residuals = Y_df_residuals[S_df.index]

        # --- Reconciliação ---
        progress(0.7, desc="✨ Executando MinT...")
        reconciler = HierarchicalReconciliation(reconcilers=[MinTrace(method='mint_shrink')])

        reconciled_components_dfs = []
        Y_r_total_df = None

        for i, shap_col in enumerate(shap_cols):
            progress(0.7 + (0.2 * (i / len(shap_cols))), desc=f"Componente {i+1}/{len(shap_cols)}: {shap_col}")
            Y_hat_comp = df_dados.pivot(index='timestamp', columns='unique_id', values=shap_col).fillna(0)
            Y_hat_comp = Y_hat_comp[S_df.index]

            Y_r_comp_df = reconciler.fit_predict(
                Y_hat_df=Y_hat_comp,
                S=S_df,
                Y_df=Y_df_residuals
            )

            comp_melted = Y_r_comp_df.reset_index().melt(
                id_vars='timestamp',
                var_name='unique_id',
                value_name=f"{shap_col}_reconciliado"
            )
            reconciled_components_dfs.append(comp_melted)

            if Y_r_total_df is None:
                Y_r_total_df = Y_r_comp_df
            else:
                Y_r_total_df = Y_r_total_df + Y_r_comp_df

        # --- Formatação ---
        progress(0.9, desc="💾 Formatando resultados...")
        df_reconciliado_long = Y_r_total_df.reset_index().melt(
            id_vars='timestamp',
            var_name='unique_id',
            value_name='forecast_reconciliado'
        )

        df_final = pd.merge(df_dados, df_reconciliado_long, on=['timestamp', 'unique_id'], how='left')
        for comp_df in reconciled_components_dfs:
            df_final = pd.merge(df_final, comp_df, on=['timestamp', 'unique_id'], how='left')
        df_final = df_final.sort_values(by=['unique_id', 'timestamp']).reset_index(drop=True)

        # --- Exportar ---
        temp_dir = tempfile.mkdtemp()
        file_name = "Previsoes_Ajustadas_MinT.xlsx"
        file_path = os.path.join(temp_dir, file_name)
        df_final.to_excel(file_path, sheet_name='Dados_Ajustados_MinT', index=False)
        
        progress(1.0, desc="✅ Concluído!")

        return df_final, gr.update(value=file_path, visible=True)

    except Exception as e:
        print(f"Erro detalhado: {e}")
        import traceback
        traceback.print_exc()
        raise gr.Error(f"Ocorreu um erro: {str(e)}")


# --- INTERFACE GRADIO ---
with gr.Blocks(theme=gr.themes.Soft(), title="Ajuste de Previsões (MinT)", css=custom_css) as demo:
    gr.Markdown("# 🚀 Ajustador de Previsões Hierárquicas (MinT)")
    gr.Markdown("Tamo junto, Ricardo! Faça o upload dos seus dados e da definição da hierarquia para aplicar o MinT e deixar suas previsões consistentes.")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1. Arquivo de Dados (Long Format 'DNA')")
            file_dados_input = gr.File(label="Upload de Dados (.csv)")
        with gr.Column(scale=1):
            gr.Markdown("### 2. Arquivo de Hierarquia")
            file_hierarquia_input = gr.File(label="Upload da Hierarquia (.csv)")

    run_button = gr.Button("✨ Executar Ajuste MinT!", elem_classes=["orange-button"])

    with gr.Tabs() as results_tabs:
        with gr.TabItem("📊 Dados Reconciliados"):
            data_output = gr.DataFrame(label="Prévia dos Dados Ajustados", wrap=True)
        with gr.TabItem("💾 Download"):
            download_output = gr.File(label="Baixar Planilha Ajustada (.xlsx)", visible=False)

    run_button.click(
        executar_reconciliacao,
        inputs=[file_dados_input, file_hierarquia_input],
        outputs=[data_output, download_output]
    )

if __name__ == "__main__":
    demo.launch(debug=True)
