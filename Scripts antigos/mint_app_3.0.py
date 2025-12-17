import warnings
import pandas as pd
import numpy as np
import gradio as gr
import io
import tempfile
import os
import networkx as nx
from hierarchicalforecast.core import HierarchicalReconciliation
from hierarchicalforecast.methods import MinTrace
import json
import time

# ==========================
# CONFIGURAÇÕES INICIAIS
# ==========================
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

MINT_METHOD = "mint_shrink"  # Pode trocar para "ols" se quiser rodar sem matriz de covariância

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

# ==========================
# FUNÇÕES CORE
# ==========================

def construir_matriz_s(df_hierarquia, all_unique_ids):
    """Constrói a matriz de agregação (S_df)."""
    print("--- DEBUG: Iniciando construir_matriz_s ---")
    all_nodes = set(all_unique_ids)
    G = nx.DiGraph()

    nodes_hierarquia = set(df_hierarquia['Child_ID']).union(set(df_hierarquia['Parent_ID']))
    if not nodes_hierarquia.issubset(all_nodes):
        missing_nodes = nodes_hierarquia - all_nodes
        raise gr.Error(f"Erro! IDs ausentes nos dados: {missing_nodes}")

    for _, row in df_hierarquia.iterrows():
        G.add_edge(row['Parent_ID'], row['Child_ID'])

    parent_nodes = set(df_hierarquia['Parent_ID'])
    bottom_nodes = sorted(list(all_nodes - parent_nodes))
    all_nodes_sorted = sorted(list(all_nodes))

    S_df = pd.DataFrame(0, index=all_nodes_sorted, columns=bottom_nodes)
    for node in all_nodes_sorted:
        if node in bottom_nodes:
            S_df.loc[node, node] = 1
        elif G.has_node(node):
            descendants = nx.descendants(G, node)
            descendants.update(G.successors(node))
            for desc in descendants:
                if desc in bottom_nodes:
                    S_df.loc[node, desc] = 1

    S_df.insert(0, 'unique_id', S_df.index)
    S_df.reset_index(drop=True, inplace=True)
    print("--- DEBUG: construir_matriz_s concluído ---")
    return S_df


def passo1_carregar_dados(list_file_dados, progress=gr.Progress(track_tqdm=True)):
    print("\n--- DEBUG (PASSO 1) ---")
    if not list_file_dados:
        raise gr.Error("Por favor, envie pelo menos um arquivo Excel (.xlsx).")

    dfs = []
    for file_dados in list_file_dados:
        print(f"   - [DEBUG] Lendo: {file_dados.name}")
        df_temp = pd.read_excel(file_dados.name, engine='openpyxl', dtype={'unique_id': str})
        dfs.append(df_temp)
    df_dados = pd.concat(dfs, ignore_index=True)

    cols_necessarias = ['unique_id', 'timestamp', 'forecast', 'actual']
    if not all(col in df_dados.columns for col in cols_necessarias):
        raise gr.Error("Arquivos precisam conter: 'unique_id', 'timestamp', 'forecast', 'actual'.")

    base_col = 'shap_base' if 'shap_base' in df_dados.columns else (
        'shap_Intercept' if 'shap_Intercept' in df_dados.columns else None
    )
    if not base_col:
        raise gr.Error("Nenhuma coluna base SHAP ('shap_base' ou 'shap_Intercept') encontrada.")

    shap_cols = sorted([c for c in df_dados.columns if c.startswith('shap_') and c != base_col])
    if not shap_cols:
        raise gr.Error("Nenhuma coluna SHAP de componente encontrada.")

    all_unique_ids = sorted(df_dados['unique_id'].unique())

    app_state = {
        "df_dados_json": df_dados.to_json(orient='split', date_format='iso'),
        "base_col": base_col,
        "shap_cols": shap_cols
    }

    return (
        json.dumps(app_state),
        f"✅ {len(all_unique_ids)} IDs únicos encontrados. Agora, defina a hierarquia.",
        gr.update(visible=True),
        gr.update(visible=True),
        gr.update(choices=all_unique_ids),
        gr.update(choices=all_unique_ids)
    )


def passo2_executar_reconciliacao(app_state_string, df_hierarquia, progress=gr.Progress(track_tqdm=True)):
    inicio = time.time()
    print("\n--- DEBUG (PASSO 2 INÍCIO) ---")
    if not app_state_string:
        raise gr.Error("State vazio. Execute o Passo 1 novamente.")

    app_state = json.loads(app_state_string)
    df_dados = pd.read_json(io.StringIO(app_state["df_dados_json"]), orient='split')
    df_dados['timestamp'] = pd.to_datetime(df_dados['timestamp'])
    base_col = app_state["base_col"]
    shap_cols = app_state["shap_cols"]

    if df_hierarquia.empty:
        raise gr.Error("Defina pelo menos uma relação Pai → Filho.")

    all_unique_ids = df_dados['unique_id'].unique()
    tags_dict = {uid: [uid] for uid in all_unique_ids}

    S_df = construir_matriz_s(df_hierarquia, all_unique_ids)
    reconciler = HierarchicalReconciliation(reconcilers=[MinTrace(method=MINT_METHOD)])

    # Base SHAP pivotada
    Y_hat_base_df = df_dados.pivot(index='timestamp', columns='unique_id', values=base_col).fillna(0)

    reconciled_components = []
    Y_r_sum = None

    for i, shap_col in enumerate(shap_cols):
        progress(0.6 + 0.3 * (i / len(shap_cols)), desc=f"Reconciliação: {shap_col}")
        print(f"--- DEBUG: Reconciliação componente {i+1}/{len(shap_cols)}: {shap_col}")

        # Previsões (Y_hat)
        Y_hat_comp = df_dados.pivot(index='timestamp', columns='unique_id', values=shap_col).fillna(0)
        Y_hat_long = Y_hat_comp.reset_index().melt(
            id_vars='timestamp', var_name='unique_id', value_name='y_hat'
        ).rename(columns={'timestamp': 'ds'})

        # Dados reais + previsões
        Y_df_long = df_dados[['timestamp', 'unique_id', 'forecast', 'actual']].rename(
            columns={'timestamp': 'ds', 'forecast': 'y_hat', 'actual': 'y'}
        )

        try:
            Y_r_comp_df = reconciler.reconcile(
                Y_hat_df=Y_hat_long,
                S_df=S_df,
                tags=tags_dict,
                Y_df=Y_df_long
            )
        except Exception as e:
            raise gr.Error(f"Erro ao reconciliar {shap_col}: {str(e)}")

        # Reorganiza resultados
        Y_r_pivot = Y_r_comp_df.pivot(index='ds', columns='unique_id', values='y_hat')
        Y_r_sum = Y_r_pivot if Y_r_sum is None else Y_r_sum + Y_r_pivot

        melted = Y_r_comp_df.rename(columns={'y_hat': f'{shap_col}_reconciliado'})
        reconciled_components.append(melted)

    # ==========================
    # PATCH FINAL - timestamp fix
    # ==========================
    Y_r_total_df = Y_r_sum + Y_hat_base_df
    Y_r_total_df = Y_r_total_df.reset_index()

    if 'ds' in Y_r_total_df.columns and 'timestamp' not in Y_r_total_df.columns:
        Y_r_total_df = Y_r_total_df.rename(columns={'ds': 'timestamp'})

    if 'timestamp' not in Y_r_total_df.columns:
        raise gr.Error(f"Coluna de tempo não encontrada. Colunas disponíveis: {list(Y_r_total_df.columns)}")

    df_total_long = Y_r_total_df.melt(
        id_vars='timestamp',
        var_name='unique_id',
        value_name='forecast_reconciliado'
    )

    # =======================================
    # ✅ MERGE FINAL SEGURO
    # =======================================
    print("\n--- DEBUG: Iniciando merge final ---")

    df_final = pd.merge(df_dados, df_total_long, on=['timestamp', 'unique_id'], how='left')

    for comp_df in reconciled_components:
        # Mantém apenas colunas relevantes
        keep_cols = ['ds', 'unique_id'] + [c for c in comp_df.columns if c.endswith('_reconciliado')]
        comp_df_clean = comp_df[keep_cols].rename(columns={'ds': 'timestamp'})

        # Evita conflito de colunas duplicadas
        merge_cols = [c for c in comp_df_clean.columns if c not in df_final.columns or c in ['timestamp', 'unique_id']]
        df_final = pd.merge(df_final, comp_df_clean[merge_cols], on=['timestamp', 'unique_id'], how='left')

    # Exporta resultado final
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, "Previsoes_AjustADAS_MinT.xlsx")
    df_final.to_excel(file_path, sheet_name='Dados_Ajustados_MinT', index=False)

    tempo_total = time.time() - inicio
    print(f"--- DEBUG: Passo 2 finalizado com sucesso em {tempo_total:.2f}s ---")
    progress(1.0, desc="✅ Reconciliação concluída e planilha salva!")

    return df_final, gr.update(value=file_path, visible=True)

# ==========================
# FUNÇÕES AUXILIARES
# ==========================
def adicionar_relacao(pai, filhos_lista, df_atual):
    if not pai or not filhos_lista:
        raise gr.Error("Selecione um PAI e ao menos um FILHO.")
    novas = [{'Parent_ID': pai, 'Child_ID': filho} for filho in filhos_lista if pai != filho]
    df_novo = pd.DataFrame(novas)
    return pd.concat([df_atual, df_novo], ignore_index=True).drop_duplicates()

def limpar_hierarquia():
    return pd.DataFrame(columns=['Parent_ID', 'Child_ID'])

# ==========================
# INTERFACE GRADIO
# ==========================
with gr.Blocks(theme=gr.themes.Soft(), title="Ajuste de Previsões (MinT)", css=custom_css) as demo:
    app_state_textbox = gr.Textbox(visible=False)
    gr.Markdown("# 🚀 Ajustador de Previsões Hierárquicas (MinT)")
    gr.Markdown("Defina a hierarquia manualmente e execute o ajuste MinT (OLS ou Shrink).")

    with gr.Group():
        file_input = gr.File(label="Upload de Dados (.xlsx)", file_count="multiple")
        btn_carregar = gr.Button("1️⃣ Carregar Dados")

    with gr.Group(visible=False) as bloco_passo2:
        status = gr.Markdown("Aguardando dados...")
        dropdown_pai = gr.Dropdown(label="PAI", choices=[])
        checkbox_filhos = gr.CheckboxGroup(label="FILHOS", choices=[])
        btn_add = gr.Button("Adicionar Relação")
        btn_clear = gr.Button("Limpar Hierarquia")
        df_hierarquia_ui = gr.DataFrame(
            headers=['Parent_ID', 'Child_ID'],
            value=pd.DataFrame(columns=['Parent_ID', 'Child_ID']),
            label="Hierarquia"
        )

    with gr.Group(visible=False) as bloco_passo3:
        run_button = gr.Button("✨ Executar Ajuste MinT!", elem_classes=["orange-button"])

    with gr.Tabs():
        with gr.TabItem("📊 Dados Ajustados"):
            data_output = gr.DataFrame(label="Prévia dos Dados")
        with gr.TabItem("💾 Download"):
            download_output = gr.File(label="Baixar Planilha", visible=False)

    btn_carregar.click(
        fn=passo1_carregar_dados,
        inputs=[file_input],
        outputs=[app_state_textbox, status, bloco_passo2, bloco_passo3, dropdown_pai, checkbox_filhos]
    )
    btn_add.click(fn=adicionar_relacao, inputs=[dropdown_pai, checkbox_filhos, df_hierarquia_ui], outputs=[df_hierarquia_ui])
    btn_clear.click(fn=limpar_hierarquia, outputs=[df_hierarquia_ui])
    run_button.click(fn=passo2_executar_reconciliacao, inputs=[app_state_textbox, df_hierarquia_ui], outputs=[data_output, download_output])

if __name__ == "__main__":
    print("--- INICIANDO APP GRADIO ---")
    print(f"🧠 Método ativo: {MINT_METHOD}")
    demo.launch(debug=True)
