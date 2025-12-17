# ==========================================
# AJUSTE TOP-DOWN POR PROPORÇÕES (SEM MinT)
# ==========================================

import warnings
import pandas as pd
import numpy as np
import gradio as gr
import io
import tempfile
import os
import json
import time
from collections import defaultdict

# ==========================
# CONFIGURAÇÕES INICIAIS
# ==========================
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

APP_METHOD = "topdown_proportions"

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

def _cols_ok(df, need=('unique_id','timestamp','forecast','actual')):
    return all(col in df.columns for col in need)

def _normalize_group(shares: pd.Series) -> pd.Series:
    """Normaliza proporções para somar 1 (tratando NaN/inf)."""
    s = shares.replace([np.inf, -np.inf], np.nan).astype(float)
    total = s.sum()
    if pd.isna(total) or total <= 0:
        return pd.Series(np.nan, index=s.index)
    return s / total

def _historical_fallback(df_hist_children: pd.DataFrame, k_months: int = 6) -> pd.Series:
    """
    Fallback de proporções com base no histórico recente (últimos k meses com 'actual').
    Espera df com colunas: ['timestamp','unique_id','actual'] contendo SOMENTE filhos.
    Retorna proporções médias por filho (Series index=unique_id).
    """
    if df_hist_children.empty:
        return pd.Series(dtype=float)
    # Ordena, pega últimos k meses distintos (onde há 'actual' válido)
    df_hist = df_hist_children.dropna(subset=['actual']).copy()
    if df_hist.empty:
        return pd.Series(dtype=float)

    # Usa últimos k meses por timestamp (considera todo o bloco)
    last_ts = sorted(df_hist['timestamp'].dt.to_period('M').unique())[-k_months:] if len(df_hist) else []
    if len(last_ts) == 0:
        # Média global das shares por filho ao longo de todo histórico
        # share_m = child_actual / sum_by_timestamp
        tmp = df_hist.copy()
        sum_by_ts = tmp.groupby('timestamp', as_index=False)['actual'].sum().rename(columns={'actual':'sum_ts'})
        tmp = tmp.merge(sum_by_ts, on='timestamp', how='left')
        tmp['share'] = np.where(tmp['sum_ts']>0, tmp['actual']/tmp['sum_ts'], np.nan)
        shares = tmp.groupby('unique_id')['share'].mean()
        return _normalize_group(shares).fillna(1.0 / tmp['unique_id'].nunique())
    else:
        # Filtra pelos últimos k meses
        periods = set(last_ts)
        tmp = df_hist[df_hist['timestamp'].dt.to_period('M').isin(periods)].copy()
        sum_by_ts = tmp.groupby('timestamp', as_index=False)['actual'].sum().rename(columns={'actual':'sum_ts'})
        tmp = tmp.merge(sum_by_ts, on='timestamp', how='left')
        tmp['share'] = np.where(tmp['sum_ts']>0, tmp['actual']/tmp['sum_ts'], np.nan)
        shares = tmp.groupby('unique_id')['share'].mean()
        shares = _normalize_group(shares)
        # Fallback final: se ainda ficou NaN, distribui igual
        if shares.isna().any() or shares.sum() <= 0:
            n = tmp['unique_id'].nunique()
            shares = pd.Series(1.0/n, index=shares.index)
        return shares

def calcular_metricas(df_final: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula métricas (MAPE, MAE, RMSE) comparando:
      - forecast original (coluna 'forecast')
      - forecast alocado por proporção (coluna 'forecast_ratio_alloc')
    Somente nas linhas com 'actual' e a respectiva coluna de previsão não nulas.
    """
    frames = []
    for col in ['forecast', 'forecast_ratio_alloc']:
        tmp = df_final[['unique_id','actual',col]].dropna()
        if tmp.empty:
            continue
        tmp['erro_abs'] = (tmp['actual'] - tmp[col]).abs()
        tmp['erro_pct'] = np.where(tmp['actual'] != 0,
                                   tmp['erro_abs'] / np.abs(tmp['actual']),
                                   np.nan)
        tmp['erro_qua'] = (tmp['actual'] - tmp[col])**2

        m = tmp.groupby('unique_id').agg(
            MAPE=('erro_pct', lambda x: np.nanmean(x) * 100),
            MAE=('erro_abs', 'mean'),
            RMSE=('erro_qua', lambda x: np.sqrt(np.nanmean(x)))
        ).reset_index()
        m['Tipo_Prev'] = col
        frames.append(m)

    if not frames:
        return pd.DataFrame(columns=['unique_id','Tipo_Prev','MAPE','MAE','RMSE'])

    out = pd.concat(frames, ignore_index=True)
    out = out[['unique_id','Tipo_Prev','MAPE','MAE','RMSE']].sort_values(['unique_id','Tipo_Prev'])
    return out

# ==========================
# ETAPA 1 - CARREGAR DADOS
# ==========================

def passo1_carregar_dados(list_file_dados, progress=gr.Progress(track_tqdm=True)):
    print("\n--- DEBUG (PASSO 1) ---")
    if not list_file_dados:
        raise gr.Error("Por favor, envie pelo menos um arquivo Excel (.xlsx).")

    dfs = []
    for f in list_file_dados:
        print(f"   - [DEBUG] Lendo: {f.name}")
        df_tmp = pd.read_excel(f.name, engine='openpyxl', dtype={'unique_id': str})
        dfs.append(df_tmp)

    df = pd.concat(dfs, ignore_index=True)
    if not _cols_ok(df):
        raise gr.Error("Arquivos precisam conter colunas: 'unique_id', 'timestamp', 'forecast', 'actual'.")

    # Tipagem e normalização mínima
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['unique_id'] = df['unique_id'].astype(str)

    all_ids = sorted(df['unique_id'].unique())
    app_state = {
        "df_json": df.to_json(orient='split', date_format='iso')
    }

    return (
        json.dumps(app_state),
        f"✅ {len(all_ids)} IDs únicos detectados. Selecione o PAI e adicione os FILHOS.",
        gr.update(visible=True),
        gr.update(visible=True),
        gr.update(choices=all_ids),
        gr.update(choices=all_ids)
    )

# ==========================
# ETAPA 2 - ALOCAÇÃO PROPORCIONAL
# ==========================

def passo2_executar_topdown(app_state_string, df_hierarquia, k_hist=6, progress=gr.Progress(track_tqdm=True)):
    """
    Executa alocação proporcional top-down:
      Para cada Parent_ID definido, em cada timestamp:
        - Obtém forecast do PAI (T).
        - Calcula shares dos FILHOS via forecasts dos filhos (normaliza).
        - Se soma filhos == 0, usa fallback:
            a) shares históricas (média dos últimos k meses com actual),
            b) se indisponível, divide igualmente.
        - Define forecast_ratio_alloc_filhos = shares * T
      Mantém forecast do PAI inalterado.
    """
    t0 = time.time()
    print("\n--- DEBUG (PASSO 2: TOP-DOWN PROPORTIONS) ---")

    # Leitura estado e dados
    app_state = json.loads(app_state_string)
    df = pd.read_json(io.StringIO(app_state['df_json']), orient='split')
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    if df_hierarquia is None or df_hierarquia.empty:
        raise gr.Error("Defina ao menos uma relação PAI → FILHOS na hierarquia.")

    # Sanitiza hierarquia e forma grupos por PAI
    df_h = df_hierarquia.dropna().copy()
    df_h = df_h[df_h['Parent_ID'] != df_h['Child_ID']]
    if df_h.empty:
        raise gr.Error("Hierarquia inválida. Verifique PAI ≠ FILHO.")

    groups = defaultdict(list)
    for _, row in df_h.iterrows():
        groups[str(row['Parent_ID'])].append(str(row['Child_ID']))

    # Vamos criar a coluna de saída
    df['forecast_ratio_alloc'] = np.nan
    df['proporcao_usada'] = np.nan  # opcional: para auditoria das shares finais aplicadas

    # Para cálculo de métricas depois, é útil saber quais linhas foram tratadas
    tratadas_mask_total = pd.Series(False, index=df.index)

    # Processa grupo a grupo
    parents = list(groups.keys())
    for gi, parent_id in enumerate(parents, start=1):
        children_ids = sorted(set(groups[parent_id]))
        print(f"[INFO] Grupo {gi}/{len(parents)}: PAI='{parent_id}' com {len(children_ids)} FILHOS.")

        # Filtra linhas relevantes
        df_parent = df[df['unique_id'] == parent_id].copy()
        if df_parent.empty:
            raise gr.Error(f"O PAI '{parent_id}' não foi encontrado nos dados enviados.")

        df_children = df[df['unique_id'].isin(children_ids)].copy()
        if df_children.empty:
            raise gr.Error(f"Nenhum FILHO de '{parent_id}' encontrado nos dados enviados.")

        # Trabalha por timestamp (usa união de datas onde exista pai OU filhos)
        all_ts = sorted(set(pd.concat([df_parent['timestamp'], df_children['timestamp']]).unique()))
        for j, ts in enumerate(all_ts, start=1):
            if j % 50 == 0:
                progress(0.1 + 0.8 * (j / max(len(all_ts), 1)), desc=f"Processando {parent_id}: {j}/{len(all_ts)} datas")

            # Forecast do pai nesta data
            p_row = df_parent.loc[df_parent['timestamp'] == ts]
            if p_row.empty:
                # Sem pai neste timestamp → não aloca
                continue
            T = p_row['forecast'].values[0]
            if pd.isna(T):
                # Se não há forecast do pai, não há o que distribuir neste ts
                continue

            # Filhos nesta data
            c_rows = df_children.loc[df_children['timestamp'] == ts, ['unique_id','forecast','actual']].copy()
            if c_rows.empty:
                continue

            # Shares a partir dos forecasts dos filhos
            shares = _normalize_group(c_rows.set_index('unique_id')['forecast'])

            # Fallback se soma==0 ou share inválida
            if shares.isna().any() or shares.sum() <= 0:
                # Usa histórico recente: pega últimos k meses com actual
                # Para isso, coletamos histórico dos filhos anterior ao ts
                hist_mask = (df_children['timestamp'] < ts)
                hist = df_children.loc[hist_mask, ['timestamp','unique_id','actual']]
                shares = _historical_fallback(hist, k_months=k_hist)

                # Se ainda não deu, distribui igual
                if shares.empty or shares.isna().any() or shares.sum() <= 0:
                    n = len(c_rows)
                    shares = pd.Series(1.0/n, index=c_rows['unique_id'].values)

            # Aplica alocação: yhat_alloc = share * T
            alloc = (shares * T).rename('alloc')

            # Commit na tabela principal
            # Índices das linhas dos filhos no timestamp
            idx_children_ts = df.index[(df['unique_id'].isin(alloc.index)) & (df['timestamp'] == ts)]
            # Map de proporções e valores alocados
            df.loc[idx_children_ts, 'forecast_ratio_alloc'] = df.loc[idx_children_ts, 'unique_id'].map(alloc)
            df.loc[idx_children_ts, 'proporcao_usada'] = df.loc[idx_children_ts, 'unique_id'].map(shares)

            tratadas_mask_total.loc[idx_children_ts] = True

        # Mantém PAI inalterado (se quiser, pode copiar o forecast do pai para ratio_alloc também)
        # Aqui: não mexemos no forecast do PAI; deixamos NaN em 'forecast_ratio_alloc' para o PAI.

    # MÉTRICAS (compara forecast original vs alocado quando existir 'actual')
    df_metricas = calcular_metricas(df)

    # EXPORTS
    temp_dir = tempfile.mkdtemp()
    out_forecast_path = os.path.join(temp_dir, "Previsoes_AjustADAS_TopDown.xlsx")
    out_metrics_path  = os.path.join(temp_dir, "Metricas_AjustADAS_TopDown.xlsx")

    # Ordena e salva
    order_cols = ['timestamp','unique_id','actual','forecast','forecast_ratio_alloc','proporcao_usada']
    extra_cols = [c for c in df.columns if c not in order_cols]
    df_out = df[order_cols + extra_cols]
    df_out.sort_values(['unique_id','timestamp'], inplace=True)
    df_out.to_excel(out_forecast_path, index=False)

    df_metricas.to_excel(out_metrics_path, index=False)

    elapsed = time.time() - t0
    print(f"--- DEBUG: Concluído em {elapsed:.2f}s ---")

    return (
        df_out.head(200),             # prévia
        df_metricas,
        gr.update(value=out_forecast_path, visible=True),
        gr.update(value=out_metrics_path,  visible=True)
    )

# ==========================
# INTERFACE GRADIO
# ==========================

with gr.Blocks(theme=gr.themes.Soft(), title="Ajuste de Previsões (Top-Down)", css=custom_css) as demo:
    app_state_textbox = gr.Textbox(visible=False)

    gr.Markdown("# 🚀 Ajustador de Previsões por Proporções (Top-Down)")
    gr.Markdown(
        "Distribui a **previsão do TOTAL (PAI)** entre os **SEGMENTOS (FILHOS)** "
        "segundo as **proporções previstas** dos filhos. "
        "Fallback: histórico recente (últimos _k_ meses) ou divisão igual."
    )

    with gr.Group():
        file_input = gr.File(label="Upload de Dados (.xlsx) — pode enviar múltiplos", file_count="multiple")
        btn_carregar = gr.Button("1️⃣ Carregar Dados")

    with gr.Group(visible=False) as bloco_passo2:
        status = gr.Markdown("Aguardando dados...")
        dropdown_pai = gr.Dropdown(label="PAI (Total)", choices=[])
        checkbox_filhos = gr.CheckboxGroup(label="FILHOS (Segmentos que somam o total)", choices=[])
        with gr.Row():
            btn_add = gr.Button("Adicionar Relação PAI → FILHOS")
            btn_clear = gr.Button("Limpar Hierarquia")
        df_hierarquia_ui = gr.DataFrame(
            headers=['Parent_ID', 'Child_ID'],
            value=pd.DataFrame(columns=['Parent_ID', 'Child_ID']),
            label="Hierarquia definida"
        )
        k_hist_slider = gr.Slider(1, 24, value=6, step=1, label="Meses de histórico para fallback de proporções (k)")

    with gr.Group(visible=False) as bloco_passo3:
        run_button = gr.Button("✨ Executar Alocação Top-Down", elem_classes=["orange-button"])

    with gr.Tabs():
        with gr.TabItem("📊 Dados Ajustados"):
            data_output = gr.DataFrame(label="Prévia dos Dados (200 linhas)")
        with gr.TabItem("📈 Métricas"):
            metrics_output = gr.DataFrame(label="Métricas de Performance (MAPE, MAE, RMSE)")
        with gr.TabItem("💾 Downloads"):
            forecast_download = gr.File(label="Baixar Planilha Ajustada", visible=False)
            metrics_download = gr.File(label="Baixar Métricas", visible=False)

    # Ações
    btn_carregar.click(
        fn=passo1_carregar_dados,
        inputs=[file_input],
        outputs=[app_state_textbox, status, bloco_passo2, bloco_passo3, dropdown_pai, checkbox_filhos]
    )

    # Adiciona linhas PAI → (cada) FILHO ao grid
    def _add_relations(parent, filhos, grid_df):
        if parent is None or not filhos:
            return grid_df
        add = pd.DataFrame([{"Parent_ID": parent, "Child_ID": c} for c in filhos if c != parent])
        if grid_df is None or grid_df.empty:
            return add.drop_duplicates()
        return pd.concat([grid_df, add], ignore_index=True).drop_duplicates()

    btn_add.click(
        fn=_add_relations,
        inputs=[dropdown_pai, checkbox_filhos, df_hierarquia_ui],
        outputs=[df_hierarquia_ui]
    )

    btn_clear.click(lambda: pd.DataFrame(columns=['Parent_ID', 'Child_ID']), outputs=[df_hierarquia_ui])

    run_button.click(
        fn=passo2_executar_topdown,
        inputs=[app_state_textbox, df_hierarquia_ui, k_hist_slider],
        outputs=[data_output, metrics_output, forecast_download, metrics_download]
    )

if __name__ == "__main__":
    print("--- INICIANDO APP GRADIO ---")
    print(f"🧠 Método ativo: {APP_METHOD}")
    demo.launch(debug=True)
