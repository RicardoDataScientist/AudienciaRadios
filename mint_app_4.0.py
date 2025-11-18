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

MINT_METHOD = "mint_shrink"  # ou "ols"

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
# FUNÇÕES AUXILIARES
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


def calcular_metricas(df_final):
    """Calcula MAPE, MAE e RMSE por unique_id e por tipo de previsão."""
    print("--- DEBUG: Calculando métricas ---")

    df_metricas = []
    metric_cols = [c for c in df_final.columns if c.startswith("forecast") or c.endswith("_reconciliado")]

    for col in metric_cols:
        if 'actual' not in df_final.columns:
            continue
        temp = df_final[['unique_id', 'actual', col]].dropna()
        if temp.empty:
            continue

        temp['erro_abs'] = np.abs(temp['actual'] - temp[col])
        temp['erro_pct'] = np.where(temp['actual'] != 0, temp['erro_abs'] / np.abs(temp['actual']), np.nan)
        temp['erro_quadrado'] = (temp['actual'] - temp[col]) ** 2

        resumo = temp.groupby('unique_id').agg(
            MAPE=('erro_pct', lambda x: np.nanmean(x) * 100),
            MAE=('erro_abs', 'mean'),
            RMSE=('erro_quadrado', lambda x: np.sqrt(np.nanmean(x)))
        ).reset_index()

        resumo['Tipo_Prev'] = col
        df_metricas.append(resumo)

    df_metricas = pd.concat(df_metricas, ignore_index=True)
    df_metricas = df_metricas[['unique_id', 'Tipo_Prev', 'MAPE', 'MAE', 'RMSE']].sort_values(['unique_id', 'Tipo_Prev'])
    print("--- DEBUG: Métricas calculadas com sucesso ---")

    # Exporta arquivo temporário
    temp_dir = tempfile.mkdtemp()
    metric_path = os.path.join(temp_dir, "Metricas_AjustADAS_MinT.xlsx")
    df_metricas.to_excel(metric_path, index=False)

    return df_metricas, metric_path


# ==========================
# ETAPA 1 - CARREGAR DADOS
# ==========================

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

# ==========================
# ETAPA 2 - RECONCILIAÇÃO
# ==========================

def passo2_executar_reconciliacao(app_state_string, df_hierarquia, progress=gr.Progress(track_tqdm=True)):
    inicio = time.time()
    print("\n--- DEBUG (PASSO 2 INÍCIO) ---")

    app_state = json.loads(app_state_string)
    df_dados = pd.read_json(io.StringIO(app_state["df_dados_json"]), orient='split')
    df_dados['timestamp'] = pd.to_datetime(df_dados['timestamp'])
    base_col = app_state["base_col"]
    shap_cols = app_state["shap_cols"]

    all_unique_ids = df_dados['unique_id'].unique()
    S_df = construir_matriz_s(df_hierarquia, all_unique_ids)
    reconciler = HierarchicalReconciliation(reconcilers=[MinTrace(method=MINT_METHOD)])
    tags_dict = {uid: [uid] for uid in all_unique_ids}

    Y_hat_base_df = df_dados.pivot(index='timestamp', columns='unique_id', values=base_col).fillna(0)
    reconciled_components = []
    Y_r_sum = None

    for i, shap_col in enumerate(shap_cols):
        progress(0.6 + 0.3 * (i / len(shap_cols)), desc=f"Reconciliação: {shap_col}")
        print(f"--- DEBUG: Reconciliação componente {i+1}/{len(shap_cols)}: {shap_col}")

        Y_hat_comp = df_dados.pivot(index='timestamp', columns='unique_id', values=shap_col).fillna(0)
        Y_hat_long = Y_hat_comp.reset_index().melt(id_vars='timestamp', var_name='unique_id', value_name='y_hat').rename(columns={'timestamp': 'ds'})
        Y_df_long = df_dados[['timestamp', 'unique_id', 'forecast', 'actual']].rename(columns={'timestamp': 'ds', 'forecast': 'y_hat', 'actual': 'y'})

        Y_r_comp_df = reconciler.reconcile(Y_hat_df=Y_hat_long, S_df=S_df, tags=tags_dict, Y_df=Y_df_long)
        Y_r_pivot = Y_r_comp_df.pivot(index='ds', columns='unique_id', values='y_hat')
        Y_r_sum = Y_r_pivot if Y_r_sum is None else Y_r_sum + Y_r_pivot
        melted = Y_r_comp_df.rename(columns={'y_hat': f'{shap_col}_reconciliado'})
        reconciled_components.append(melted)

    Y_r_total_df = Y_r_sum + Y_hat_base_df
    Y_r_total_df = Y_r_total_df.reset_index().rename(columns={'ds': 'timestamp'})
    df_total_long = Y_r_total_df.melt(id_vars='timestamp', var_name='unique_id', value_name='forecast_reconciliado')
    df_final = pd.merge(df_dados, df_total_long, on=['timestamp', 'unique_id'], how='left')

    for comp_df in reconciled_components:
        keep_cols = ['ds', 'unique_id'] + [c for c in comp_df.columns if c.endswith('_reconciliado')]
        comp_df_clean = comp_df[keep_cols].rename(columns={'ds': 'timestamp'})
        merge_cols = [c for c in comp_df_clean.columns if c not in df_final.columns or c in ['timestamp', 'unique_id']]
        df_final = pd.merge(df_final, comp_df_clean[merge_cols], on=['timestamp', 'unique_id'], how='left')

    # Exporta resultado reconciliado
    temp_dir = tempfile.mkdtemp()
    forecast_path = os.path.join(temp_dir, "Previsoes_AjustADAS_MinT.xlsx")
    df_final.to_excel(forecast_path, index=False)

    # Calcula métricas
    df_metricas, metric_path = calcular_metricas(df_final)

    tempo_total = time.time() - inicio
    print(f"--- DEBUG: Concluído em {tempo_total:.2f}s ---")

    progress(1.0, desc="✅ Reconciliação e métricas concluídas!")
    return df_final, df_metricas, gr.update(value=forecast_path, visible=True), gr.update(value=metric_path, visible=True)

# ==========================
# INTERFACE GRADIO
# ==========================
with gr.Blocks(theme=gr.themes.Soft(), title="Ajuste de Previsões (MinT)", css=custom_css) as demo:
    app_state_textbox = gr.Textbox(visible=False)

    gr.Markdown("# 🚀 Ajustador de Previsões Hierárquicas (MinT)")
    gr.Markdown("Executa reconciliação e avalia métricas (MAPE, MAE, RMSE) por série.")

    with gr.Group():
        file_input = gr.File(label="Upload de Dados (.xlsx)", file_count="multiple")
        btn_carregar = gr.Button("1️⃣ Carregar Dados")

    with gr.Group(visible=False) as bloco_passo2:
        status = gr.Markdown("Aguardando dados...")
        dropdown_pai = gr.Dropdown(label="PAI", choices=[])
        checkbox_filhos = gr.CheckboxGroup(label="FILHOS", choices=[])
        btn_add = gr.Button("Adicionar Relação")
        btn_clear = gr.Button("Limpar Hierarquia")
        df_hierarquia_ui = gr.DataFrame(headers=['Parent_ID', 'Child_ID'], value=pd.DataFrame(columns=['Parent_ID', 'Child_ID']), label="Hierarquia")

    with gr.Group(visible=False) as bloco_passo3:
        run_button = gr.Button("✨ Executar Ajuste MinT!", elem_classes=["orange-button"])

    with gr.Tabs():
        with gr.TabItem("📊 Dados Ajustados"):
            data_output = gr.DataFrame(label="Prévia dos Dados")
        with gr.TabItem("📈 Métricas"):
            metrics_output = gr.DataFrame(label="Métricas de Performance (MAPE, MAE, RMSE)")
        with gr.TabItem("💾 Downloads"):
            forecast_download = gr.File(label="Baixar Planilha Ajustada", visible=False)
            metrics_download = gr.File(label="Baixar Métricas", visible=False)

    btn_carregar.click(
        fn=passo1_carregar_dados,
        inputs=[file_input],
        outputs=[app_state_textbox, status, bloco_passo2, bloco_passo3, dropdown_pai, checkbox_filhos]
    )
    btn_add.click(lambda p, f, df: pd.concat([df, pd.DataFrame([{"Parent_ID": p, "Child_ID": c} for c in f if p != c])], ignore_index=True).drop_duplicates(), inputs=[dropdown_pai, checkbox_filhos, df_hierarquia_ui], outputs=[df_hierarquia_ui])
    btn_clear.click(lambda: pd.DataFrame(columns=['Parent_ID', 'Child_ID']), outputs=[df_hierarquia_ui])
    run_button.click(
        fn=passo2_executar_reconciliacao,
        inputs=[app_state_textbox, df_hierarquia_ui],
        outputs=[data_output, metrics_output, forecast_download, metrics_download]
    )

if __name__ == "__main__":
    print("--- INICIANDO APP GRADIO ---")
    print(f"🧠 Método ativo: {MINT_METHOD}")
    demo.launch(debug=True)
