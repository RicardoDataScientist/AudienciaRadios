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
# import locale # <-- REMOVIDO

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
        # Verifica se a coluna de forecast existe antes de tentar acessá-la
        if col not in df_final.columns:
            continue
            
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

def _salvar_formato_pbi(df_dados: pd.DataFrame, temp_dir: str) -> str:
    """
    Pega o DataFrame de saída, formata para 'long' (PBI), salva em temp_dir
    com 6 ABAS (3 Long/PBI, 3 Wide/Humanas) e retorna o path.
    *** VERSÃO MODIFICADA: Inclui cálculo de Variação (Abs e %) nas abas WIDE ***
    *** VERSÃO NAMI (V10): Remove * 100 e .round(2) da variacao_pct ***
    """
    print("[INFO] Formatando saída para Power BI (com 6 abas)...")
    
    # --- REMOVIDO: Bloco locale.setlocale ---
            
    try:
        # 1. Manter apenas colunas essenciais
        cols_essenciais = ['Data', 'unique_id', 'actual', 'forecast', 'forecast_ratio_alloc'] # Usa 'Data'
        cols_presentes = [col for col in cols_essenciais if col in df_dados.columns]
        df_clean = df_dados[cols_presentes].copy()
        df_clean.rename(columns={'Data': 'timestamp'}, inplace=True) # Renomeia pra lógica interna

        # --- NOVO FILTRO: Manter apenas dados com previsão ---
        print("[INFO] Filtrando dados: mantendo apenas linhas com 'forecast' não-nulo.")
        if 'forecast' in df_clean.columns:
             df_clean = df_clean.dropna(subset=['forecast'])
        
        if df_clean.empty:
             print("[AVISO] Todos os dados foram filtrados, PBI .xlsx ficará vazio.")
        # --- Fim do filtro ---

        # 2. Identificar o PAI (Total)
        parent_ids = df_clean[df_clean['forecast_ratio_alloc'].isna()]['unique_id'].unique()
        print(f"[INFO] IDs de PAI (Total) detectados para PBI: {list(parent_ids)}")

        # 3. O PULO DO GATO (Preencher PAI)
        mask_total = df_clean['unique_id'].isin(parent_ids)
        if 'forecast' in df_clean.columns:
            df_clean.loc[mask_total, 'forecast_ratio_alloc'] = df_clean.loc[mask_total, 'forecast']
        
        # --- MODIFICAÇÃO (RICARDO/NAMI): Adicionando cálculos de Variação (vs. REAL) ---
        print("[INFO] Calculando Variação (Ajustado vs Real) para abas WIDE.")
        df_clean['variacao_abs'] = df_clean['forecast_ratio_alloc'] - df_clean['actual']
        df_clean['variacao_pct'] = np.where(
            (df_clean['actual'].notna()) & (df_clean['actual'] != 0), 
            ((df_clean['forecast_ratio_alloc'] - df_clean['actual']) / df_clean['actual']), # <-- REMOVIDO * 100
            np.nan
        )
        
        # --- MODIFICAÇÃO NAMI: Arredondar valores calculados ---
        df_clean['forecast_ratio_alloc'] = df_clean['forecast_ratio_alloc'].round(2)
        df_clean['variacao_abs'] = df_clean['variacao_abs'].round(2)
        # df_clean['variacao_pct'] = df_clean['variacao_pct'].round(2) # <-- REMOVIDO
        # ---------------------------------------------------
        
        # --- NOVA ETAPA: Criar base WIDE (antes do MELT) ---
        mapa_nomes_wide = {
            'actual': 'Real (Actual)',
            'forecast': 'Previsão Original', 
            'forecast_ratio_alloc': 'Previsão Ajustada (Top-Down)',
            'variacao_abs': 'Variação Absoluta (Ajustado - Real)',
            'variacao_pct': 'Variação % (Ajustado - Real)'
        }
        df_wide_base = df_clean.rename(columns=mapa_nomes_wide)
        # ---------------------------------------------------

        # 4. Aplicar o MELT (Unpivot)
        id_vars = ['timestamp', 'unique_id']
        value_vars = [col for col in ['actual', 'forecast', 'forecast_ratio_alloc'] if col in df_clean.columns]
        
        df_long = df_clean.melt(
            id_vars=id_vars,
            value_vars=value_vars,
            var_name='Tipo_Metrica',
            value_name='Audiencia'
        )

        # 5. Renomear valores (para o LONG)
        mapa_nomes_long = {
            'actual': 'Real (Actual)',
            'forecast': 'Previsão Original',
            'forecast_ratio_alloc': 'Previsão Ajustada (Top-Down)'
        }
        df_long['Tipo_Metrica'] = df_long['Tipo_Metrica'].map(mapa_nomes_long).fillna(df_long['Tipo_Metrica'])

        # 6. Limpeza final e Formato de Data (para AMBOS os DFs)
        df_long = df_long.dropna(subset=['Audiencia'])
        
        # --- MODIFICAÇÃO NAMI (V6): ORDENAR PRIMEIRO, DEPOIS RENOMEAR (LONG) ---
        print("[INFO] Ordenando por data (real) e renomeando 'timestamp' (LONG)...")
        # 1. Ordena pela data real
        df_long = df_long.sort_values(by=['unique_id', 'timestamp', 'Tipo_Metrica'])
        # 2. Renomeia
        df_long.rename(columns={'timestamp': 'Data'}, inplace=True)
        # 3. --- MODIFICAÇÃO NAMI (V8): Força DATE (remove HORA) ---
        if 'Data' in df_long.columns:
            # df_long['Data'] já é datetime aqui, vindo do df_clean (que veio do df_dados)
            df_long['Data'] = df_long['Data'].dt.date
        # -----------------------------------------------------------------
        
        # --- MODIFICAÇÃO NAMI (V6): ORDENAR PRIMEIRO, DEPOIS RENOMEAR (WIDE) ---
        print("[INFO] Ordenando por data (real) e renomeando 'timestamp' (WIDE)...")
        # 1. Ordena pela data real
        df_wide_base = df_wide_base.sort_values(by=['unique_id', 'timestamp'])
        # 2. Renomeia
        df_wide_base.rename(columns={'timestamp': 'Data'}, inplace=True)
        # 3. --- MODIFICAÇÃO NAMI (V8): Força DATE (remove HORA) ---
        if 'Data' in df_wide_base.columns:
            # df_wide_base['Data'] já é datetime aqui
            df_wide_base['Data'] = df_wide_base['Data'].dt.date
        # -----------------------------------------------------------------
        
        # 7. --- MODIFICAÇÃO: Criar as 3 abas LONG (PBI) ---
        df_total_long = df_long[df_long['unique_id'].isin(parent_ids)].copy()
        df_segmentos_long = df_long[~df_long['unique_id'].isin(parent_ids)].copy()

        # 8. --- NOVA ETAPA: Criar as 3 abas WIDE (Humanos) ---
        df_total_wide = df_wide_base[df_wide_base['unique_id'].isin(parent_ids)].copy()
        df_segmentos_wide = df_wide_base[~df_wide_base['unique_id'].isin(parent_ids)].copy()

        # --- MODIFICAÇÃO NAMI (V4): Reordenar Colunas (LONG) ---
        col_order_long = ['Data', 'unique_id', 'Tipo_Metrica', 'Audiencia']
        df_long = df_long[col_order_long]
        df_total_long = df_total_long[col_order_long]
        df_segmentos_long = df_segmentos_long[col_order_long]
        
        # --- MODIFICAÇÃO NAMI (V4): Reordenar Colunas (WIDE) ---
        col_order_wide = [
            'Data', 'unique_id', 'Real (Actual)', 'Previsão Original', # <-- Mantido para o PBI (6 abas)
            'Previsão Ajustada (Top-Down)', 
            'Variação Absoluta (Ajustado - Real)', 'Variação % (Ajustado - Real)'
        ]
        # Garante que só usamos colunas que existem (ex: 'Real (Actual)' pode não existir)
        final_cols_wide = [c for c in col_order_wide if c in df_wide_base.columns]
        df_wide_base = df_wide_base[final_cols_wide]
        df_total_wide = df_total_wide[final_cols_wide]
        df_segmentos_wide = df_segmentos_wide[final_cols_wide]
        # ----------------------------------------------------

        # 9. Salvar com ExcelWriter (6 ABAS)
        out_pbi_path = os.path.join(temp_dir, "Dados_Para_PowerBI_v2.xlsx")
        
        with pd.ExcelWriter(out_pbi_path, engine='openpyxl') as writer:
            # Abas LONG (para PBI)
            df_long.to_excel(writer, sheet_name='Todos_Dados_Long', index=False)
            df_total_long.to_excel(writer, sheet_name='Total_Long', index=False)
            df_segmentos_long.to_excel(writer, sheet_name='Segmentos_Long', index=False)
            
            # Abas WIDE (para Humanos) - Agora com colunas de Variação
            df_wide_base.to_excel(writer, sheet_name='Todos_Dados_Wide', index=False)
            df_total_wide.to_excel(writer, sheet_name='Total_Wide', index=False)
            df_segmentos_wide.to_excel(writer, sheet_name='Segmentos_Wide', index=False)
        
        print(f"[INFO] Arquivo Power BI salvo em: {out_pbi_path} (com 6 abas)")
        return out_pbi_path
        
    except Exception as e:
        print(f"[ERRO] Falha ao gerar arquivo PBI: {e}")
        return None

# --- NOVA FUNÇÃO (RICARDO/NAMI): Salvar Resumo Apresentação (Apenas WIDE) ---
def _salvar_formato_resumo_apresentacao(df_dados: pd.DataFrame, temp_dir: str) -> str:
    """
    Salva um arquivo .xlsx de RESUMO contendo apenas as abas WIDE
    (com os cálculos de variação) para facilitar a apresentação.
    *** VERSÃO NAMI (V10): Remove * 100 e .round(2) da variacao_pct ***
    """
    print("[INFO] Formatando saída de RESUMO (Wide) para Apresentação...")
    
    # --- REMOVIDO: Bloco locale.setlocale ---

    try:
        # 1. Manter apenas colunas essenciais
        # 'forecast' ainda é necessário para o filtro e para preencher o PAI
        cols_essenciais = ['Data', 'unique_id', 'actual', 'forecast', 'forecast_ratio_alloc'] # Usa 'Data'
        cols_presentes = [col for col in cols_essenciais if col in df_dados.columns]
        df_clean = df_dados[cols_presentes].copy()
        df_clean.rename(columns={'Data': 'timestamp'}, inplace=True) # Renomeia pra lógica interna

        # 2. Filtro: Manter apenas dados com previsão
        if 'forecast' in df_clean.columns:
            df_clean = df_clean.dropna(subset=['forecast'])
        
        if df_clean.empty:
             print("[AVISO] Todos os dados foram filtrados, Resumo .xlsx ficará vazio.")

        # 3. Identificar o PAI (Total)
        parent_ids = df_clean[df_clean['forecast_ratio_alloc'].isna()]['unique_id'].unique()

        # 4. Preencher PAI (Ajustado = Original)
        mask_total = df_clean['unique_id'].isin(parent_ids)
        if 'forecast' in df_clean.columns:
            df_clean.loc[mask_total, 'forecast_ratio_alloc'] = df_clean.loc[mask_total, 'forecast']
        
        # 5. Cálculos de Variação (vs. REAL)
        df_clean['variacao_abs'] = df_clean['forecast_ratio_alloc'] - df_clean['actual']
        df_clean['variacao_pct'] = np.where(
            (df_clean['actual'].notna()) & (df_clean['actual'] != 0), 
            ((df_clean['forecast_ratio_alloc'] - df_clean['actual']) / df_clean['actual']), # <-- REMOVIDO * 100
            np.nan
        )
        
        # --- MODIFICAÇÃO NAMI: Arredondar valores calculados ---
        df_clean['forecast_ratio_alloc'] = df_clean['forecast_ratio_alloc'].round(2)
        df_clean['variacao_abs'] = df_clean['variacao_abs'].round(2)
        # df_clean['variacao_pct'] = df_clean['variacao_pct'].round(2) # <-- REMOVIDO
        # ---------------------------------------------------
        
        # 6. Criar base WIDE
        mapa_nomes_wide = {
            'actual': 'Real (Actual)',
            # 'forecast': 'Previsão Original', # <-- REMOVIDO (Task 2)
            'forecast_ratio_alloc': 'Previsão Ajustada (Top-Down)',
            'variacao_abs': 'Variação Absoluta (Ajustado - Real)',
            'variacao_pct': 'Variação % (Ajustado - Real)'
        }
        df_wide_base = df_clean.rename(columns=mapa_nomes_wide)

        # 7. Formato de Data e Rename
        # --- MODIFICAÇÃO NAMI (V6): ORDENAR PRIMEIRO, DEPOIS RENOMEAR (WIDE) ---
        print("[INFO] Ordenando por data (real) e renomeando 'timestamp' (Resumo WIDE)...")
        # 1. Ordena pela data real
        df_wide_base = df_wide_base.sort_values(by=['unique_id', 'timestamp'])
        # 2. Renomeia
        df_wide_base.rename(columns={'timestamp': 'Data'}, inplace=True)
        # 3. --- MODIFICAÇÃO NAMI (V8): Força DATE (remove HORA) ---
        if 'Data' in df_wide_base.columns:
            # df_wide_base['Data'] já é datetime aqui
            df_wide_base['Data'] = df_wide_base['Data'].dt.date
        # -----------------------------------------------------------------
        
        # 8. Criar as abas WIDE (Total e Segmentos)
        df_total_wide = df_wide_base[df_wide_base['unique_id'].isin(parent_ids)].copy()
        df_segmentos_wide = df_wide_base[~df_wide_base['unique_id'].isin(parent_ids)].copy()

        # --- MODIFICAÇÃO NAMI (V4): Reordenar Colunas (WIDE) ---
        col_order_wide = [
            'Data', 'unique_id', 'Real (Actual)', 
            # 'Previsão Original', # <-- REMOVIDO (Task 2)
            'Previsão Ajustada (Top-Down)', 
            'Variação Absoluta (Ajustado - Real)', 'Variação % (Ajustado - Real)'
        ]
        # Garante que só usamos colunas que existem (ex: 'Real (Actual)' pode não existir)
        final_cols_wide = [c for c in col_order_wide if c in df_wide_base.columns]
        df_wide_base = df_wide_base[final_cols_wide]
        df_total_wide = df_total_wide[final_cols_wide]
        df_segmentos_wide = df_segmentos_wide[final_cols_wide]
        # ----------------------------------------------------

        # 9. Salvar com ExcelWriter (APENAS 3 abas WIDE)
        out_resumo_path = os.path.join(temp_dir, "Resumo_Apresentacao_Wide.xlsx")
        
        with pd.ExcelWriter(out_resumo_path, engine='openpyxl') as writer:
            # Abas WIDE (para Humanos) - Agora com colunas de Variação
            df_wide_base.to_excel(writer, sheet_name='Todos_Dados_Wide', index=False)
            df_total_wide.to_excel(writer, sheet_name='Total_Wide', index=False)
            df_segmentos_wide.to_excel(writer, sheet_name='Segmentos_Wide', index=False)
        
        print(f"[INFO] Arquivo Resumo (Wide) salvo em: {out_resumo_path} (com 3 abas)")
        return out_resumo_path
        
    except Exception as e:
        print(f"[ERRO] Falha ao gerar arquivo Resumo (Wide): {e}")
        return None
# --- Fim da Nova Função ---


# ==========================
# ETAPA 1 - CARREGAR DADOS
# ==========================

def passo1_carregar_dados(list_file_dados, progress=gr.Progress(track_tqdm=True)):
    print("\n--- DEBUG (PASSO 1) ---")
    if not list_file_dados:
        raise gr.Error("Por favor, envie pelo menos um arquivo Excel (.xlsx).")

    dfs = []
    for f in list_file_dados:
        print(f"  - [DEBUG] Lendo: {f.name}")
        df_tmp = pd.read_excel(f.name, engine='openpyxl', dtype={'unique_id': str})
        dfs.append(df_tmp)

    df = pd.concat(dfs, ignore_index=True)
    if not _cols_ok(df):
        raise gr.Error("Arquivos precisam conter colunas: 'unique_id', 'timestamp', 'forecast', 'actual'.")

    # Tipagem e normalização mínima
    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.normalize() # <-- MODIFICADO (Remove HORA)
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

    # --- REFACTOR (Task 1): Loop por TEMPO (outer) e GRUPO (inner) ---
    print("[INFO] Iniciando processamento temporal (Outer Loop: Data, Inner Loop: Grupo)...")
    
    # 1. Pré-filtrar DFs de pais e filhos para lookup rápido
    df_parents_data = {
        parent_id: df[df['unique_id'] == parent_id].copy() 
        for parent_id in groups.keys()
    }
    df_children_data = {
        parent_id: df[df['unique_id'].isin(children_ids)].copy() 
        for parent_id, children_ids in groups.items()
    }
    
    # 2. Obter todos os timestamps onde há forecasts de PAIS
    # (O PAI é o driver da alocação)
    parent_ids_list = list(groups.keys())
    all_ts = sorted(df[df['unique_id'].isin(parent_ids_list) & df['forecast'].notna()]['timestamp'].unique())
    print(f"[INFO] Total de {len(all_ts)} timestamps com forecasts de PAIs para processar.")

    # 3. Outer loop: TEMPO
    # (Usa progress.tqdm para atualizar a barra do Gradio)
    for j, ts in enumerate(progress.tqdm(all_ts, desc="Processando datas"), start=1):
        
        # 4. Inner loop: GRUPOS (Pais)
        for parent_id in groups.keys():
            children_ids = groups[parent_id]
            
            # Lookup nos DFs pré-filtrados
            df_parent_group = df_parents_data[parent_id]
            df_children_group = df_children_data[parent_id]

            # Forecast do pai nesta data
            p_row = df_parent_group.loc[df_parent_group['timestamp'] == ts]
            if p_row.empty:
                continue # Pai não existe nesse TS (improvável se all_ts veio dele, mas seguro)
            T = p_row['forecast'].values[0]
            if pd.isna(T):
                continue # Garantia extra

            # Filhos nesta data
            c_rows = df_children_group.loc[df_children_group['timestamp'] == ts, ['unique_id','forecast','actual']].copy()
            if c_rows.empty:
                continue # Sem filhos nesse TS para alocar

            # Shares a partir dos forecasts dos filhos
            shares = _normalize_group(c_rows.set_index('unique_id')['forecast'])

            # Fallback se soma==0 ou share inválida
            if shares.isna().any() or shares.sum() <= 0:
                # Usa histórico recente: pega últimos k meses com actual
                # (Usa o df_children_group completo para olhar o histórico < ts)
                hist_mask = (df_children_group['timestamp'] < ts)
                hist = df_children_group.loc[hist_mask, ['timestamp','unique_id','actual']]
                shares = _historical_fallback(hist, k_months=k_hist)

                # Se ainda não deu, distribui igual
                if shares.empty or shares.isna().any() or shares.sum() <= 0:
                    n = len(c_rows)
                    # Garante que o index do shares use os unique_id dos filhos presentes *neste* TS
                    shares = pd.Series(1.0/n, index=c_rows['unique_id'].values)

            # Aplica alocação: yhat_alloc = share * T
            alloc = (shares * T).rename('alloc')

            # Commit na tabela principal
            # Índices das linhas dos filhos *neste* timestamp
            idx_children_ts = df.index[(df['unique_id'].isin(alloc.index)) & (df['timestamp'] == ts)]
            
            # Map de proporções e valores alocados
            # (Garante que só vai mapear valores para os filhos corretos)
            map_alloc = df.loc[idx_children_ts, 'unique_id'].map(alloc)
            map_shares = df.loc[idx_children_ts, 'unique_id'].map(shares)
            
            if not map_alloc.empty:
                df.loc[idx_children_ts, 'forecast_ratio_alloc'] = map_alloc
            if not map_shares.empty:
                df.loc[idx_children_ts, 'proporcao_usada'] = map_shares

            tratadas_mask_total.loc[idx_children_ts] = True

    # --- FIM DO REFACTOR ---


    # MÉTRICAS (compara forecast original vs alocado quando existir 'actual')
    print("[INFO] Calculando métricas de performance...")
    df_metricas = calcular_metricas(df)

    # EXPORTS
    temp_dir = tempfile.mkdtemp()
    out_forecast_path = os.path.join(temp_dir, "Previsoes_AjustADAS_TopDown.xlsx")
    out_metrics_path  = os.path.join(temp_dir, "Metricas_AjustADAS_TopDown.xlsx")

    # Ordena e salva o arquivo principal
    print("[INFO] Salvando arquivo completo (Previsoes_AjustADAS_TopDown.xlsx)...")
    order_cols = ['timestamp','unique_id','actual','forecast','forecast_ratio_alloc','proporcao_usada']
    extra_cols = [c for c in df.columns if c not in order_cols]
    # (Mantém 'forecast' no arquivo principal para auditoria)
    df_out = df[order_cols + extra_cols]
    # --- MODIFICAÇÃO NAMI (V7): Ordena por ID, depois Data ---
    df_out.sort_values(['unique_id','timestamp'], inplace=True)
    
    # --- MODIFICAÇÃO NAMI (V7): Renomeia e Reordena Colunas (Preview/Completo) ---
    df_out.rename(columns={'timestamp': 'Data'}, inplace=True)
    cols_reordenadas = ['Data', 'unique_id', 'actual', 'forecast', 'forecast_ratio_alloc', 'proporcao_usada']
    # Garante que só pegamos colunas que existem no df_out
    final_cols = [c for c in cols_reordenadas if c in df_out.columns] + \
                 [c for c in extra_cols if c in df_out.columns]
    df_out = df_out[final_cols]
    
    # --- MODIFICAÇÃO NAMI (V8): Força DATE (remove HORA) ---
    # if 'Data' in df_out.columns:
    #     df_out['Data'] = df_out['Data'].dt.date # <-- MUDANÇA AQUI (V9)
    # ---------------------------------------------------------------------
    
    # --- MODIFICAÇÃO NAMI (V9): Criar cópia para salvar (main) ---
    df_out_main = df_out.copy()
    if 'Data' in df_out_main.columns:
        df_out_main['Data'] = df_out_main['Data'].dt.date
    df_out_main.to_excel(out_forecast_path, index=False)
    # ---------------------------------------------------------

    # Salva métricas
    print("[INFO] Salvando métricas (Metricas_AJUSTADAS_TopDown.xlsx)...")
    df_metricas.to_excel(out_metrics_path, index=False)
    
    # --- NOVA ADIÇÃO: Gerar arquivo formatado para PBI ---
    out_pbi_path = _salvar_formato_pbi(df_out.copy(), temp_dir)
    # --- NOVA ADIÇÃO (RICARDO/NAMI): Gerar arquivo Resumo (Wide) ---
    out_resumo_path = _salvar_formato_resumo_apresentacao(df_out.copy(), temp_dir)
    # ---------------------------------------------------

    elapsed = time.time() - t0
    print(f"--- DEBUG: Concluído em {elapsed:.2f}s ---")

    # Prepara preview (removendo 'forecast' se solicitado, só para o preview)
    # df_preview = df_out.drop(columns=['forecast'], errors='ignore') # Decidi manter no preview
    
    return (
        df_out.head(200),            # prévia
        df_metricas,
        gr.update(value=out_forecast_path, visible=True),
        gr.update(value=out_metrics_path,  visible=True),
        # --- NOVA ADIÇÃO: Retorna o path do PBI ---
        gr.update(value=out_pbi_path, visible=True if out_pbi_path else False),
        # --- NOVA ADIÇÃO (RICARDO/NAMI): Retorna o path do Resumo ---
        gr.update(value=out_resumo_path, visible=True if out_resumo_path else False)
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
            forecast_download = gr.File(label="Baixar Planilha Ajustada (Completa)", visible=False)
            metrics_download = gr.File(label="Baixar Métricas", visible=False)
            # --- NOVA ADIÇÃO: Botão de download PBI ---
            pbi_download = gr.File(label="Baixar Dados para Power BI (Formatado)", visible=False)
            # --- NOVA ADIÇÃO (RICARDO/NAMI): Botão de download Resumo ---
            resumo_download = gr.File(label="Baixar Resumo para Apresentação (Wide XLSX)", visible=False)

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
        outputs=[
            data_output, 
            metrics_output, 
            forecast_download, 
            metrics_download,
            # --- NOVA ADIÇÃO: Output PBI ---
            pbi_download,
            # --- NOVA ADIÇÃO (RICARDO/NAMI): Output Resumo ---
            resumo_download
            ]
    )

if __name__ == "__main__":
    print("--- INICIANDO APP GRADIO ---")
    print(f"🧠 Método ativo: {APP_METHOD}")
    demo.launch(debug=True)

