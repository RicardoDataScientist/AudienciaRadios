# ==========================================
# AJUSTE TOP-DOWN - ADAPTADO PARA "RESULTS"
# Adaptado por Janet para Ricardo
# Versão: Multi-Dimensão + Relatório Mágico (Baseline & 2025+)
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
import re
from collections import defaultdict

# ==========================
# CONFIGURAÇÕES INICIAIS
# ==========================
warnings.filterwarnings('ignore')

APP_METHOD = "topdown_results_magic_report"

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
# FUNÇÕES DE TRATAMENTO DE DADOS
# ==========================

def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Padroniza as colunas específicas do arquivo RESULTS.
    """
    df.columns = [str(c).strip().lower() for c in df.columns]
    
    # Mapa focado no seu arquivo RESULTS
    rename_map = {}
    for col in df.columns:
        if col in ['unique_id', 'id', 'segmento', 'zona']:
            rename_map[col] = 'unique_id'
        elif col in ['timestamp', 'data', 'ds']:
            rename_map[col] = 'timestamp'
        elif col in ['actual', 'y', 'real']:
            rename_map[col] = 'actual'
        elif col in ['forecast', 'yhat', 'previsao']:
            rename_map[col] = 'forecast'
            
    df = df.rename(columns=rename_map)
    
    # Garante colunas mínimas (se faltar actual, cria vazio)
    if 'actual' not in df.columns: df['actual'] = np.nan
    if 'forecast' not in df.columns: df['forecast'] = np.nan
        
    return df

def _detect_and_unpivot(df: pd.DataFrame) -> pd.DataFrame:
    """
    Verifica se precisa fazer unpivot. 
    Para o arquivo RESULTS, geralmente não precisa, mas mantemos por segurança.
    """
    # Se já tem as colunas sagradas, retorna direto!
    if 'timestamp' in df.columns and 'unique_id' in df.columns and ('forecast' in df.columns or 'actual' in df.columns):
        print("[AUTO-DETECT] Arquivo já está no formato padrão (Long). Pulei o unpivot.")
        return df

    # (Lógica de fallback para Wide, caso venha um arquivo diferente)
    cols = df.columns.tolist()
    date_cols = []
    id_cols = []
    date_pattern = re.compile(r'(\d{2,4}[-/]\d{1,2}|jan|fev|mar|apr|mai|jun|jul|aug|ago|set|sep|out|oct|nov|dec)', re.IGNORECASE)
    
    for c in cols:
        try:
            pd.to_datetime(c)
            date_cols.append(c)
        except:
            if date_pattern.search(str(c)):
                date_cols.append(c)
            else:
                id_cols.append(c)
    
    if len(date_cols) >= 3 and len(id_cols) > 0:
        print(f"[AUTO-DETECT] Formato Wide detectado. Convertendo...")
        df_melted = df.melt(id_vars=id_cols, value_vars=date_cols, var_name='timestamp', value_name='forecast')
        df_melted['actual'] = np.nan
        possible_ids = ['sku', 'produto', 'unique_id', 'id', 'cod', 'segmento', 'filial']
        found_id = next((c for c in id_cols if any(x in str(c).lower() for x in possible_ids)), id_cols[0])
        df_melted.rename(columns={found_id: 'unique_id'}, inplace=True)
        return df_melted
    
    return df

# ==========================
# FUNÇÕES DE LÓGICA DE NEGÓCIO
# ==========================

def _cols_ok(df, need=('unique_id','timestamp','forecast','actual')):
    return all(col in df.columns for col in need)

def _normalize_group(shares: pd.Series) -> pd.Series:
    """Normaliza proporções para somar 1."""
    s = shares.replace([np.inf, -np.inf], np.nan).astype(float)
    total = s.sum()
    if pd.isna(total) or total <= 0:
        return pd.Series(np.nan, index=s.index)
    return s / total

def _historical_fallback(df_hist_children: pd.DataFrame, k_months: int = 6) -> pd.Series:
    """Fallback usando histórico recente se a previsão for zero/nula."""
    if df_hist_children.empty:
        return pd.Series(dtype=float)
    
    df_hist = df_hist_children.dropna(subset=['actual']).copy()
    if df_hist.empty:
        return pd.Series(dtype=float)

    last_ts = sorted(df_hist['timestamp'].dt.to_period('M').unique())[-k_months:] if len(df_hist) else []
    
    if len(last_ts) == 0:
        # Média de todo o histórico se não tiver K meses
        tmp = df_hist.copy()
        sum_by_ts = tmp.groupby('timestamp', as_index=False)['actual'].sum().rename(columns={'actual':'sum_ts'})
        tmp = tmp.merge(sum_by_ts, on='timestamp', how='left')
        tmp['share'] = np.where(tmp['sum_ts']>0, tmp['actual']/tmp['sum_ts'], np.nan)
        shares = tmp.groupby('unique_id')['share'].mean()
        return _normalize_group(shares).fillna(1.0 / tmp['unique_id'].nunique())
    else:
        # Média dos últimos K meses
        periods = set(last_ts)
        tmp = df_hist[df_hist['timestamp'].dt.to_period('M').isin(periods)].copy()
        sum_by_ts = tmp.groupby('timestamp', as_index=False)['actual'].sum().rename(columns={'actual':'sum_ts'})
        tmp = tmp.merge(sum_by_ts, on='timestamp', how='left')
        tmp['share'] = np.where(tmp['sum_ts']>0, tmp['actual']/tmp['sum_ts'], np.nan)
        shares = tmp.groupby('unique_id')['share'].mean()
        shares = _normalize_group(shares)
        if shares.isna().any() or shares.sum() <= 0:
            n = tmp['unique_id'].nunique()
            shares = pd.Series(1.0/n, index=shares.index)
        return shares

def calcular_metricas(df_final: pd.DataFrame) -> pd.DataFrame:
    """Calcula MAPE, MAE e RMSE."""
    frames = []
    # Compara tanto o forecast original quanto o ajustado contra o actual
    for col in ['forecast', 'forecast_ratio_alloc']:
        if col not in df_final.columns: continue
        tmp = df_final[['unique_id','actual',col]].dropna()
        if tmp.empty: continue
        
        tmp['erro_abs'] = (tmp['actual'] - tmp[col]).abs()
        tmp['erro_pct'] = np.where(tmp['actual'] != 0, tmp['erro_abs'] / np.abs(tmp['actual']), np.nan)
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
    return out[['unique_id','Tipo_Prev','MAPE','MAE','RMSE']].sort_values(['unique_id','Tipo_Prev'])

# --- NOVA FUNÇÃO MÁGICA ---
def _gerar_relatorio_magico(df_full: pd.DataFrame, temp_dir: str) -> str:
    print("[INFO] Gerando Relatório Mágico (2025+ e Baselines)...")
    try:
        # Trabalha com cópia para não afetar o original
        df = df_full.copy()
        
        # Garante ordenação para o shift funcionar (Lag)
        df.sort_values(['unique_id', 'timestamp'], inplace=True)
        
        # 1. Baseline (Lag 12 meses) - Pega o REAL do ano anterior
        # Assumindo que os dados são mensais
        df['baseline_actual_ano_anterior'] = df.groupby('unique_id')['actual'].shift(12)
        
        # 2. Variação vs Baseline (Crescimento YoY)
        # Compara Forecast Ajustado com o Real do ano passado
        df['Var_Abs_AnoAnterior'] = df['forecast_ratio_alloc'] - df['baseline_actual_ano_anterior']
        df['Var_Pct_AnoAnterior'] = np.where(
            (df['baseline_actual_ano_anterior'].notna()) & (df['baseline_actual_ano_anterior'] != 0),
            (df['forecast_ratio_alloc'] / df['baseline_actual_ano_anterior']) - 1,
            np.nan
        )

        # 3. Variação Real vs Previsto (Acurácia)
        # Compara Real Atual vs Forecast Ajustado
        df['Var_Abs_Real_vs_Prev'] = df['actual'] - df['forecast_ratio_alloc']
        df['Var_Pct_Real_vs_Prev'] = np.where(
            (df['forecast_ratio_alloc'].notna()) & (df['forecast_ratio_alloc'] != 0),
            (df['actual'] / df['forecast_ratio_alloc']) - 1,
            np.nan
        )

        # 4. Filtro: Remover tudo antes de 2025
        df_filtered = df[df['timestamp'].dt.year >= 2025].copy()
        
        if df_filtered.empty:
            print("[AVISO] O filtro de data (>= 2025) removeu todas as linhas!")
            return None

        # 5. Seleção e Renomeação de Colunas
        # Tenta preservar 'tipo' se existir, senão ignora
        cols_order = ['timestamp', 'unique_id', 'tipo', 'actual', 'forecast', 'forecast_ratio_alloc',
                      'baseline_actual_ano_anterior', 
                      'Var_Abs_AnoAnterior', 'Var_Pct_AnoAnterior',
                      'Var_Abs_Real_vs_Prev', 'Var_Pct_Real_vs_Prev']
        
        # Filtra apenas colunas que existem no df
        final_cols = [c for c in cols_order if c in df_filtered.columns]
        
        df_final = df_filtered[final_cols].copy()
        
        # Renomeia para ficar bonito no Excel
        rename_dict = {
            'timestamp': 'Data',
            'unique_id': 'ID',
            'tipo': 'Tipo',
            'actual': 'Real',
            'forecast': 'Previsão_Original',
            'forecast_ratio_alloc': 'Previsão_Ajustada',
            'baseline_actual_ano_anterior': 'Real_Ano_Anterior',
            'Var_Abs_AnoAnterior': 'Var_Abs_YoY',
            'Var_Pct_AnoAnterior': 'Var_Pct_YoY',
            'Var_Abs_Real_vs_Prev': 'Var_Abs_Real_vs_Prev',
            'Var_Pct_Real_vs_Prev': 'Var_Pct_Real_vs_Prev'
        }
        df_final.rename(columns=rename_dict, inplace=True)
        
        # Remove Hora da Data
        if 'Data' in df_final.columns:
            df_final['Data'] = df_final['Data'].dt.date

        # Salva
        out_path = os.path.join(temp_dir, "Relatorio_Magico_2025.xlsx")
        df_final.to_excel(out_path, index=False)
        
        print(f"[SUCESSO] Relatório Mágico salvo em {out_path}")
        return out_path

    except Exception as e:
        print(f"[ERRO RELATORIO MAGICO] {e}")
        return None

def _salvar_formato_pbi(df_dados: pd.DataFrame, temp_dir: str) -> str:
    print("[INFO] Gerando Power BI Format...")
    try:
        # Mantém colunas extras como 'tipo' se existirem
        cols_fixas = ['timestamp', 'unique_id', 'actual', 'forecast', 'forecast_ratio_alloc', 'grupo_hierarquia']
        extra_cols = [c for c in df_dados.columns if c not in cols_fixas]
        
        cols_presentes = [c for c in cols_fixas if c in df_dados.columns] + extra_cols
        df_clean = df_dados[[c for c in cols_presentes if c in df_dados.columns]].copy()
        
        # Preenche o PAI (Total)
        parent_ids = df_clean[df_clean['forecast_ratio_alloc'].isna()]['unique_id'].unique()
        mask_total = df_clean['unique_id'].isin(parent_ids)
        if 'forecast' in df_clean.columns:
            df_clean.loc[mask_total, 'forecast_ratio_alloc'] = df_clean.loc[mask_total, 'forecast']
        
        # Cálculos de Variação
        df_clean['variacao_abs'] = df_clean['forecast_ratio_alloc'] - df_clean['actual']
        df_clean['variacao_pct'] = np.where(
            (df_clean['actual'].notna()) & (df_clean['actual'] != 0), 
            ((df_clean['forecast_ratio_alloc'] - df_clean['actual']) / df_clean['actual']),
            np.nan
        )
        
        df_clean[['forecast_ratio_alloc', 'variacao_abs']] = df_clean[['forecast_ratio_alloc', 'variacao_abs']].round(2)
        
        # Formata data para export
        df_clean.rename(columns={'timestamp': 'Data'}, inplace=True)
        if 'Data' in df_clean.columns:
            df_clean['Data'] = pd.to_datetime(df_clean['Data']).dt.date

        # Wide Base (para humanos)
        mapa_nomes_wide = {
            'actual': 'Real',
            'forecast': 'Previsão Original', 
            'forecast_ratio_alloc': 'Previsão Ajustada',
            'variacao_abs': 'Var Abs',
            'variacao_pct': 'Var %',
            'grupo_hierarquia': 'Dimensão/Grupo'
        }
        df_wide = df_clean.rename(columns=mapa_nomes_wide)
        
        # Long Base (para PBI)
        id_vars_long = ['Data', 'unique_id'] + [c for c in extra_cols if c in df_clean.columns]
        if 'grupo_hierarquia' in df_clean.columns:
            id_vars_long.append('grupo_hierarquia')
            
        value_vars_long = [c for c in ['actual', 'forecast', 'forecast_ratio_alloc'] if c in df_clean.columns]
        
        df_long = df_clean.melt(id_vars=id_vars_long, value_vars=value_vars_long, var_name='Metrica', value_name='Valor')

        out_path = os.path.join(temp_dir, "Dados_Para_PowerBI.xlsx")
        with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
            df_long.to_excel(writer, sheet_name='Base_Longa_PBI', index=False)
            df_wide.to_excel(writer, sheet_name='Base_Larga_Humana', index=False)
        
        return out_path
    except Exception as e:
        print(f"[ERRO PBI] {e}")
        return None

# ==========================
# ETAPA 1 - CARREGAR DADOS
# ==========================

def passo1_carregar_dados(list_file_dados, progress=gr.Progress(track_tqdm=True)):
    print("\n--- PASSO 1: LENDO ARQUIVO RESULTS ---")
    if not list_file_dados:
        raise gr.Error("Preciso do arquivo RESULTS.xlsx ou .csv!")

    dfs = []
    for f in list_file_dados:
        print(f"  - Lendo: {f.name}")
        try:
            # Suporte Híbrido: CSV ou Excel
            if f.name.lower().endswith('.csv'):
                try:
                    df_tmp = pd.read_csv(f.name)
                except:
                    df_tmp = pd.read_csv(f.name, sep=';')
            else:
                df_tmp = pd.read_excel(f.name, engine='openpyxl')
            
            # Padroniza
            df_tmp = _standardize_columns(df_tmp)
            df_tmp = _detect_and_unpivot(df_tmp)
            
            if 'unique_id' in df_tmp.columns:
                df_tmp['unique_id'] = df_tmp['unique_id'].astype(str)
                
            dfs.append(df_tmp)
        except Exception as e:
            raise gr.Error(f"Erro ao ler {f.name}: {str(e)}")

    df = pd.concat(dfs, ignore_index=True)
    df = _standardize_columns(df) 
    
    if not _cols_ok(df):
        msg = f"Colunas essenciais não encontradas.\nPreciso de: unique_id, timestamp, forecast, actual.\nAchei: {list(df.columns)}"
        raise gr.Error(msg)

    # Conversão de Data
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce').dt.normalize()
    df = df.dropna(subset=['timestamp'])
    
    # Conversão Numérica
    for col in ['actual', 'forecast']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    all_ids = sorted(df['unique_id'].unique())
    app_state = { "df_json": df.to_json(orient='split', date_format='iso') }

    preview_msg = f"✅ Arquivo RESULTS carregado com sucesso!\n{len(all_ids)} Séries encontradas (IDs)."
    
    return (
        json.dumps(app_state),
        preview_msg,
        gr.update(visible=True),
        gr.update(visible=True),
        gr.update(choices=all_ids),
        gr.update(choices=all_ids),
        df.head(10)
    )

# ==========================
# ETAPA 2 - ALOCAÇÃO
# ==========================

def passo2_executar_topdown(app_state_string, df_hierarquia, k_hist=6, progress=gr.Progress(track_tqdm=True)):
    print("\n--- PASSO 2: EXECUTANDO ALOCAÇÃO MULTI-DIMENSÃO ---")

    app_state = json.loads(app_state_string)
    df = pd.read_json(io.StringIO(app_state['df_json']), orient='split')
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    if df_hierarquia is None or df_hierarquia.empty:
        raise gr.Error("Defina a hierarquia e os GRUPOS (Ex: Classe, Região)!")

    # Prepara Hierarquia
    df_h = df_hierarquia.dropna().astype(str)
    df_h = df_h[df_h['Parent_ID'] != df_h['Child_ID']]
    
    # Se por acaso o usuário não usou grupos, preenche com 'Geral'
    if 'Group_ID' not in df_h.columns:
        df_h['Group_ID'] = 'Geral'
    
    groups = defaultdict(list)
    for _, row in df_h.iterrows():
        grupo = row['Group_ID'] if row['Group_ID'] and row['Group_ID'] != 'nan' else 'Geral'
        groups[(row['Parent_ID'], grupo)].append(row['Child_ID'])

    df['forecast_ratio_alloc'] = np.nan
    df['proporcao_usada'] = np.nan 
    df['grupo_hierarquia'] = np.nan 

    parent_ids_list = list(set([k[0] for k in groups.keys()]))
    valid_dates = df[df['unique_id'].isin(parent_ids_list) & df['forecast'].notna()]['timestamp'].unique()
    valid_dates = sorted(valid_dates)

    print(f"[INFO] Alocando dados para {len(valid_dates)} datas e {len(groups)} grupos de hierarquia...")

    for ts in progress.tqdm(valid_dates, desc="Processando Datas"):
        df_ts = df[df['timestamp'] == ts]
        
        for (parent_id, group_name), children_ids in groups.items():
            
            pai_row = df_ts[df_ts['unique_id'] == parent_id]
            if pai_row.empty: continue
            
            T = pai_row['forecast'].values[0]
            if pd.isna(T): continue

            filhos_rows = df_ts[df_ts['unique_id'].isin(children_ids)]
            if filhos_rows.empty: continue
            
            shares = _normalize_group(filhos_rows.set_index('unique_id')['forecast'])

            if shares.isna().any() or shares.sum() <= 0:
                hist_mask = (df['unique_id'].isin(children_ids)) & (df['timestamp'] < ts)
                hist = df.loc[hist_mask, ['timestamp','unique_id','actual']]
                shares = _historical_fallback(hist, k_months=k_hist)
                
                if shares.empty or shares.sum() <= 0:
                    shares = pd.Series(1.0/len(children_ids), index=children_ids)
            
            ids_filhos_presentes = filhos_rows['unique_id'].values
            shares = shares.reindex(ids_filhos_presentes).fillna(0)
            shares = _normalize_group(shares)
            
            alocacao = shares * T
            
            indices = filhos_rows.index
            df.loc[indices, 'forecast_ratio_alloc'] = df.loc[indices, 'unique_id'].map(alocacao)
            df.loc[indices, 'proporcao_usada'] = df.loc[indices, 'unique_id'].map(shares)
            df.loc[indices, 'grupo_hierarquia'] = group_name

    # Finalização
    print("[INFO] Calculando métricas e salvando...")
    df_metricas = calcular_metricas(df)

    temp_dir = tempfile.mkdtemp()
    out_forecast_path = os.path.join(temp_dir, "RESULTS_Ajustado_MultiDim.xlsx")
    out_metrics_path  = os.path.join(temp_dir, "RESULTS_Metricas.xlsx")

    # Formata Saída Principal
    df_out = df.copy()
    df_out.sort_values(['unique_id','timestamp'], inplace=True)
    df_out.rename(columns={'timestamp': 'Data'}, inplace=True)
    if 'Data' in df_out.columns: df_out['Data'] = df_out['Data'].dt.date
    
    cols_order = ['Data', 'unique_id', 'grupo_hierarquia', 'actual', 'forecast', 'forecast_ratio_alloc', 'proporcao_usada']
    cols_final = [c for c in cols_order if c in df_out.columns] + [c for c in df_out.columns if c not in cols_order]
    df_out = df_out[cols_final]
    
    df_out.to_excel(out_forecast_path, index=False)
    df_metricas.to_excel(out_metrics_path, index=False)
    out_pbi_path = _salvar_formato_pbi(df.copy(), temp_dir)
    
    # --- GERA O RELATÓRIO MÁGICO (NOVO) ---
    out_magic_path = _gerar_relatorio_magico(df, temp_dir)

    return (
        df_out.head(100),
        df_metricas,
        gr.update(value=out_forecast_path, visible=True),
        gr.update(value=out_metrics_path,  visible=True),
        gr.update(value=out_pbi_path, visible=True if out_pbi_path else False),
        gr.update(value=out_magic_path, visible=True if out_magic_path else False) # Output Mágico
    )

# ==========================
# UI GRADIO
# ==========================

with gr.Blocks(theme=gr.themes.Soft(), title="Ajuste Top-Down (RESULTS)", css=custom_css) as demo:
    app_state_textbox = gr.Textbox(visible=False)

    gr.Markdown("# 🚀 Ajustador Top-Down (Multi-Dimensão)")
    gr.Markdown(
        "**Novidade:** Agora com o **Relatório Mágico**! 🎩✨ \n"
        "Gera automaticamente um arquivo filtrado (2025+) com comparações de Baseline (Ano Anterior), "
        "Variação de Crescimento e Acurácia (Real vs Previsto)."
    )

    with gr.Group():
        file_input = gr.File(label="Arraste sua planilha RESULTS aqui (.xlsx ou .csv)", file_count="multiple")
        btn_carregar = gr.Button("1️⃣ Carregar Arquivo")
    
    status_msg = gr.Markdown("")

    with gr.Group(visible=True):
        preview_carga = gr.DataFrame(label="Preview (Dados Carregados)", interactive=False)

    with gr.Group(visible=False) as bloco_passo2:
        with gr.Row():
            with gr.Column(scale=1):
                dropdown_pai = gr.Dropdown(label="Quem é o PAI (Total)?", choices=[], interactive=True)
                checkbox_filhos = gr.Dropdown(label="Quem são os FILHOS?", choices=[], multiselect=True, interactive=True)
                txt_grupo = gr.Textbox(label="Nome do Grupo/Dimensão (Ex: Classe, Regiao)", value="Geral", placeholder="Digite o nome do agrupamento...")
                
                btn_add = gr.Button("➕ Adicionar Relação")
                btn_clear = gr.Button("🗑️ Limpar Tabela")
            
            with gr.Column(scale=2):
                df_hierarquia_ui = gr.DataFrame(
                    headers=['Parent_ID', 'Child_ID', 'Group_ID'],
                    value=pd.DataFrame(columns=['Parent_ID', 'Child_ID', 'Group_ID']),
                    label="Hierarquia Definida",
                    interactive=False
                )
        
        k_hist_slider = gr.Slider(1, 24, value=6, step=1, label="Fallback: Meses de Histórico")

    with gr.Group(visible=False) as bloco_passo3:
        run_button = gr.Button("✨ Calcular Ajuste Top-Down + Mágica", elem_classes=["orange-button"])

    with gr.Tabs():
        with gr.TabItem("📊 Planilha Ajustada"):
            data_output = gr.DataFrame(label="Dados Finais")
        with gr.TabItem("📈 Erros (Métricas)"):
            metrics_output = gr.DataFrame(label="MAPE / MAE")
        with gr.TabItem("💾 Baixar"):
            with gr.Row():
                forecast_download = gr.File(label="Download Excel Completo")
                metrics_download = gr.File(label="Download Métricas")
            with gr.Row():
                pbi_download = gr.File(label="Download Power BI Format")
                # --- NOVO BOTÃO DE DOWNLOAD ---
                magic_download = gr.File(label="🎩 Download Relatório Mágico (2025+)")

    # Callbacks
    btn_carregar.click(
        fn=passo1_carregar_dados,
        inputs=[file_input],
        outputs=[app_state_textbox, status_msg, bloco_passo2, bloco_passo3, dropdown_pai, checkbox_filhos, preview_carga]
    )

    def _add_relations(parent, filhos, grupo, grid_df):
        if not parent or not filhos: return grid_df
        if isinstance(filhos, str): filhos = [filhos]
        
        grupo_nome = grupo.strip() if grupo else "Geral"
        
        new_rows = pd.DataFrame([
            {"Parent_ID": parent, "Child_ID": c, "Group_ID": grupo_nome} 
            for c in filhos if c != parent
        ])
        
        if grid_df is None or grid_df.empty:
            return new_rows
            
        if 'Group_ID' not in grid_df.columns:
            grid_df['Group_ID'] = 'Geral'
            
        return pd.concat([grid_df, new_rows], ignore_index=True).drop_duplicates()

    btn_add.click(
        fn=_add_relations,
        inputs=[dropdown_pai, checkbox_filhos, txt_grupo, df_hierarquia_ui],
        outputs=[df_hierarquia_ui]
    )
    
    btn_clear.click(lambda: pd.DataFrame(columns=['Parent_ID', 'Child_ID', 'Group_ID']), outputs=[df_hierarquia_ui])

    run_button.click(
        fn=passo2_executar_topdown,
        inputs=[app_state_textbox, df_hierarquia_ui, k_hist_slider],
        outputs=[data_output, metrics_output, forecast_download, metrics_download, pbi_download, magic_download]
    )

if __name__ == "__main__":
    demo.launch()