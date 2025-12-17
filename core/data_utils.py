import re
import pandas as pd
import numpy as np

def sanitize_columns(df):
    """Limpeza ultra robusta para nomes de colunas."""
    novas_colunas = []
    for col in df.columns:
        clean_col = str(col).strip()
        clean_col = clean_col.lower()
        clean_col = re.sub(r'\s+', '_', clean_col)
        clean_col = re.sub(r'[^a-zA-Z0-9_]', '', clean_col)
        clean_col = clean_col.strip('_')
        novas_colunas.append(clean_col)
    df.columns = novas_colunas
    return df

def adicionar_features_temporais(df, features_selecionadas, target):
    """Adiciona features baseadas no índice de data/hora ANTES de qualquer outra etapa."""
    df_copy = df.copy()
    log_text = "⚙️ Adicionando features temporais selecionadas...\n"
    created_features = []

    if not features_selecionadas:
        log_text += " ⚠️ Nenhuma feature temporal selecionada para criação.\n"
        return df_copy, log_text, created_features

    feature_map = {
        "Mês": lambda d: d.index.month,
        "Ano": lambda d: d.index.year,
        "Dia da Semana": lambda d: d.index.dayofweek,
        "Dia do Ano": lambda d: d.index.dayofyear,
        "Semana do Ano": lambda d: d.index.isocalendar().week.astype(int)
    }

    for feature_nome_amigavel in features_selecionadas:
        if feature_nome_amigavel in feature_map:
            feature_nome_tecnico = sanitize_columns(pd.DataFrame(columns=[feature_nome_amigavel])).columns[0]
            df_copy[feature_nome_tecnico] = feature_map[feature_nome_amigavel](df_copy)
            log_text += f"    -> Feature '{feature_nome_tecnico}' criada.\n"
            created_features.append(feature_nome_tecnico)

    if "Features de Fourier (Anual)" in features_selecionadas:
        dayofyear = df_copy.index.dayofyear
        year_period = 365.25
        df_copy['fourier_sin_anual'] = np.sin(2 * np.pi * dayofyear / year_period)
        df_copy['fourier_cos_anual'] = np.cos(2 * np.pi * dayofyear / year_period)
        log_text += "    -> Features 'fourier_sin_anual' e 'fourier_cos_anual' criadas.\n"
        created_features.extend(['fourier_sin_anual', 'fourier_cos_anual'])

    if "Média Mensal Histórica (Target)" in features_selecionadas:
        if target in df_copy.columns:
            feature_name = f'media_mensal_historica_{target}'
            # Usamos expanding().mean() para evitar data leakage no cálculo da média histórica
            # .shift(1) garante que usamos apenas dados passados para cada mês
            df_copy[feature_name] = df_copy.groupby(df_copy.index.month)[target].transform(lambda x: x.expanding().mean().shift(1))
            log_text += f"    -> Feature '{feature_name}' criada (com expanding mean para evitar data leakage).\n"
            created_features.append(feature_name)
        else:
            log_text += f" ⚠️    Aviso: A variável target '{target}' não foi encontrada para criar a Média Mensal Histórica.\n"

    log_text += "--- ✅ Criação de features temporais concluída ---\n"
    return df_copy, log_text, created_features

def criar_features_de_lag(df, config_geracao):
    """Cria features de lag de forma configurável para evitar data leakage."""
    df_features = df.copy()
    log_text = "⚙️ Iniciando criação de features de lag...\n"
    for coluna_base, sufixo, lags in config_geracao:
        if coluna_base not in df_features.columns:
            msg = f" ⚠️    Aviso: Coluna '{coluna_base}' para lag não encontrada. Pulando.\n"
            log_text += msg
            continue
        log_text += f"    -> Processando lags para: '{coluna_base}'\n"
        for lag in lags:
            feature_name = f'lag{sufixo}_{lag}_meses'
            df_features[feature_name] = df_features[coluna_base].shift(lag)
    log_text += "--- ✅ Criação de features de lag concluída ---\n"
    return df_features, log_text

def criar_features_media_movel(df, config_geracao):
    """
    Cria features de média móvel de forma configurável para evitar data leakage.
    Aplica shift(1) antes de calcular a média móvel para garantir que usamos apenas dados passados.
    """
    df_features = df.copy()
    log_text = "⚙️ Iniciando criação de features de média móvel...\n"
    for coluna_base, sufixo, janelas in config_geracao:
        if coluna_base not in df_features.columns:
            msg = f" ⚠️    Aviso: Coluna '{coluna_base}' para média móvel não encontrada. Pulando.\n"
            log_text += msg
            continue
        log_text += f"    -> Processando médias móveis para: '{coluna_base}'\n"
        for janela in janelas:
            feature_name = f'ma{sufixo}_{janela}_meses'
            # Shift(1) garante que a média móvel no tempo T use dados de T-1, T-2...
            df_features[feature_name] = df_features[coluna_base].shift(1).rolling(window=janela).mean()
    log_text += "--- ✅ Criação de features de média móvel concluída ---\n"
    return df_features, log_text
