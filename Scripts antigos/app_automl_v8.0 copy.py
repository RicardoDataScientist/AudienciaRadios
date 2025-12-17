import warnings
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose
import traceback
import gradio as gr
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import io
import zipfile
import tempfile
import re
import os

# --- Configurações Iniciais ---
warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
warnings.simplefilter(action="ignore", category=FutureWarning)
sns.set_theme(style="whitegrid")

# --- CSS Personalizado ---
custom_css = """
/* Botões Laranja Estilizados */
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

/* Estilo "Dark Mode" para Checkboxes */
.dark-checkbox-group {
    background-color: #2c2c2c !important;
    border: 1px solid #444 !important;
    border-radius: 8px !importan;
    padding: 10px !important;
}
.dark-checkbox-group label span {
    color: #FFFFFF !important; /* Texto de cada opção */
}
.dark-checkbox-group input[type="checkbox"] {
    border: 1px solid #FFFFFF !important;
    accent-color: #FFA500 !important;
}
/* Títulos de Markdown (h3) dentro do grupo */
.dark-checkbox-group h3 {
    color: #FFA500 !important; /* Laranja para destacar o título */
    font-weight: bold;
    padding-bottom: 8px;
    border-bottom: 1px solid #444;
    margin-bottom: 12px;
}
"""

# --- FUNÇÕES CORE (Lógica DO MODELO) ---
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

def treinar_e_avaliar_cv(df_limpo, features_finais, target, teste_size, model_params, progress):
    """Treina o modelo final e gera resultados usando validação cruzada com janela deslizante."""
    log_text = "\n4. 🚀 Treinando modelo final com Validação Cruzada (Rolling Forecast)...\n"
    
    tscv = TimeSeriesSplit(n_splits=5, test_size=teste_size)
    all_preds, all_true = [], []
    log_text += f"📊 Configurando TimeSeriesSplit com {tscv.get_n_splits(df_limpo)} folds e tamanho de teste de {teste_size} meses.\n"
    
    X, y = df_limpo[features_finais], df_limpo[target]

    for fold, (train_index, test_index) in enumerate(tscv.split(X)):
        progress(0.7 + (0.2 * (fold / tscv.get_n_splits(df_limpo))), desc=f"Processando Fold {fold+1}/{tscv.get_n_splits(df_limpo)}...")
        log_text += f"    -> Fold {fold+1}: Treinando com {len(train_index)} amostras, prevendo {len(test_index)} amostras.\n"
        
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        modelo = xgb.XGBRegressor(**model_params)
        modelo.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        
        y_pred = modelo.predict(X_test)
        all_preds.append(pd.Series(y_pred, index=y_test.index))
        all_true.append(y_test)

    df_resultados = pd.concat([pd.concat(all_true), pd.concat(all_preds)], axis=1)
    df_resultados.columns = ['Real', 'Previsto']
    df_resultados = df_resultados.round(2)

    ## DESTAQUE - NAMI: Adicionando colunas de variação (Absoluta e %) ##
    df_resultados['Variacao_Absoluta'] = (df_resultados['Previsto'] - df_resultados['Real']).round(2)
    df_resultados['Variacao_Percentual_%'] = np.where(
        df_resultados['Real'] == 0, 
        np.nan, # Evita divisão por zero
        ((df_resultados['Previsto'] - df_resultados['Real']) / df_resultados['Real'])
    )
    # Arredonda para 4 casas decimais (ex: 0.1234 = 12.34%)
    df_resultados['Variacao_Percentual_%'] = df_resultados['Variacao_Percentual_%'].round(4)
    ## Fim do Destaque ##

    y_test_final, y_pred_final = df_resultados['Real'], df_resultados['Previsto']
    metricas = {
        'Métrica': ['MAPE', 'MAE', 'RMSE'],
        'Valor': [
            mean_absolute_percentage_error(y_test_final, y_pred_final),
            mean_absolute_error(y_test_final, y_pred_final),
            np.sqrt(mean_squared_error(y_test_final, y_pred_final))
        ]
    }
    df_metricas = pd.DataFrame(metricas).round(4)
    
    progress(0.95, desc="Gerando resultados e gráficos...")
    plt.close('all')

    # Gráfico de Previsão com Seaborn
    fig_pred, ax_pred = plt.subplots(figsize=(16, 3))
    sns.lineplot(data=df_resultados[['Real', 'Previsto']], ax=ax_pred, palette={'Real': '#0072B2', 'Previsto': '#ff0051'})
    ax_pred.set(title=f'Comparação Real vs. Previsto para {target.upper()} (Rolling Forecast)', xlabel='Data', ylabel="Média Mensal de Audiência (OPM)")
    ax_pred.legend(title='Série', bbox_to_anchor=(1.02, 1), loc='upper left')
    fig_pred.tight_layout()
    
    params_final = model_params.copy()
    params_final.pop('early_stopping_rounds', None)
    modelo_final = xgb.XGBRegressor(**params_final).fit(X, y)
    
    # Gráfico de Importância com Seaborn
    importances = pd.Series(modelo_final.feature_importances_, index=features_finais).sort_values(ascending=False).head(20)
    fig_imp, ax_imp = plt.subplots(figsize=(10, 8))
    sns.barplot(x=importances.values, y=importances.index, ax=ax_imp, palette="viridis_r")
    ax_imp.set_title(f'Importância das Features para {target.upper()} (Top 20)', fontsize=16)
    ax_imp.set_xlabel('Importância', fontsize=12)
    ax_imp.set_ylabel('Features', fontsize=12)
    ax_imp.tick_params(axis='y', labelsize=14)
    ax_imp.tick_params(axis='x', labelsize=14)
    fig_imp.tight_layout()
    
    # DESTAQUE-MINT: Declaramos shap_values e figs como None fora do try
    shap_values, fig_shap_summary, fig_shap_force = None, None, None
    try:
        explainer = shap.TreeExplainer(modelo_final)
        # DESTAQUE-MINT: Calculamos o SHAP em todo o dataset de treino (X)
        shap_values = explainer(X)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            plt.figure(); shap.summary_plot(shap_values, X, show=False, max_display=20); fig_shap_summary = plt.gcf(); fig_shap_summary.suptitle(f'Impacto Geral das Features em {target.upper()} (SHAP)', fontsize=16); fig_shap_summary.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.figure(); shap.force_plot(explainer.expected_value, shap_values.values[0,:], X.iloc[0,:], matplotlib=True, show=False); fig_shap_force = plt.gcf(); fig_shap_force.suptitle(f'Análise da Primeira Previsão para {target.upper()} (SHAP Force)', fontsize=16); fig_shap_force.tight_layout(rect=[0, 0.03, 1, 0.95])
    except Exception as e:
        log_text += f"\n⚠️ Aviso: Não foi possível gerar os gráficos SHAP. Erro: {e}\n"
        fig_shap_summary, fig_shap_force = None, None
        shap_values = None # Garante que é None em caso de falha
    
    # DESTAQUE-MINT: Adicionamos shap_values ao retorno
    return log_text, df_resultados, fig_pred, df_metricas, fig_imp, fig_shap_summary, fig_shap_force, modelo_final, shap_values


def prever_futuro(modelo, df_historico_base, features_finais, target, horizonte, config_geracao_lags, config_temporal, log_text=""):
    """Prevê o futuro de forma autorregressiva, recriando as features a cada passo."""
    log_text += "\n5. 🔮 Iniciando a previsão de valores futuros...\n"
    
    df_com_previsoes = df_historico_base.copy()
    ultima_data = df_com_previsoes.index.max()
    datas_futuras = pd.date_range(start=ultima_data + pd.DateOffset(months=1), periods=horizonte, freq='MS')
    log_text += f"    -> Gerando previsões para {horizonte} meses, de {datas_futuras[0].date()} até {datas_futuras[-1].date()}.\n"

    colunas_para_lag_base = {cfg[0] for cfg in config_geracao_lags}
    cols_exogenas = [c for c in colunas_para_lag_base if c != target]
    
    # DESTAQUE-MINT: Lista para armazenar os dataframes de features futuras
    all_features_para_prever = []

    for data_previsao in datas_futuras:
        df_com_previsoes.loc[data_previsao] = np.nan
        df_temp = df_com_previsoes.copy()
        
        # Recria features temporais e de lag
        df_temp, _, _ = adicionar_features_temporais(df_temp, config_temporal, target)
        df_temp, _ = criar_features_de_lag(df_temp, config_geracao_lags)
        
        # Pega a última linha (que contém as features para a data de previsão)
        features_para_prever = df_temp[features_finais].iloc[-1:]
        
        # DESTAQUE-MINT: Armazena o dataframe de features
        all_features_para_prever.append(features_para_prever)
        
        # Faz a previsão
        previsao = modelo.predict(features_para_prever)[0]
        
        # Atualiza o dataframe principal com a previsão para o próximo loop
        df_com_previsoes.loc[data_previsao, target] = previsao
        
        # Se houver colunas exógenas, preenche com o último valor (forward fill)
        if cols_exogenas:
            df_com_previsoes[cols_exogenas] = df_com_previsoes[cols_exogenas].ffill()

    df_resultado_futuro = df_com_previsoes.loc[datas_futuras]
    log_text += "--- ✅ Previsão futura concluída ---\n"
    
    # DESTAQUE-MINT: Concatena todos os dataframes de features futuras
    df_features_futuras = pd.concat(all_features_para_prever)
    
    # DESTAQUE-MINT: Retorna também o df_features_futuras
    return df_resultado_futuro, log_text, df_features_futuras

def gerar_grafico_decomposicao(df_historico, target):
    """Gera o gráfico de decomposição da série temporal usando statsmodels."""
    plt.close('all')
    try:
        decomposicao = seasonal_decompose(df_historico[target].dropna(), model='additive', period=12)
        
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
        
        decomposicao.observed.plot(ax=ax1, legend=False, color='gray')
        ax1.set_ylabel('Observado')
        ax1.set_title(f'Decomposição da Série Temporal para {target.upper()}', fontsize=16, fontweight='bold')
        
        decomposicao.trend.plot(ax=ax2, legend=False, color='#0072B2')
        ax2.set_ylabel('Tendência')
        
        decomposicao.seasonal.plot(ax=ax3, legend=False, color='green')
        ax3.set_ylabel('Sazonalidade')
        
        decomposicao.resid.plot(ax=ax4, legend=False, color='red', marker='o', linestyle='None', markersize=4)
        ax4.set_ylabel('Resíduos')
        
        fig.tight_layout()
        return fig
    except Exception as e:
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.text(0.5, 0.5, f'Não foi possível gerar a decomposição.\nErro: {e}', 
                horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=12, color='red')
        return fig

def gerar_grafico_sazonalidade_anual(df_historico, target):
    """Gera um gráfico focado no padrão de sazonalidade anual média."""
    plt.close('all')
    try:
        if len(df_historico[target].dropna()) < 24:
            raise ValueError("A série de dados é muito curta (< 24 meses) para uma análise de sazonalidade anual confiável.")

        decomposicao = seasonal_decompose(df_historico[target].dropna(), model='additive', period=12)
        
        df_sazonal = pd.DataFrame({
            'sazonalidade': decomposicao.seasonal,
            'mes': decomposicao.seasonal.index.month
        })
        
        media_sazonal_mensal = df_sazonal.groupby('mes')['sazonalidade'].mean()
        
        fig, ax = plt.subplots(figsize=(16, 3))
        media_sazonal_mensal.plot(kind='line', ax=ax, color='green', marker='o', linestyle='-')
        
        ax.set_title(f'Padrão de Sazonalidade Anual para {target.upper()}', fontsize=16, fontweight='bold')
        ax.set_ylabel('Impacto Sazonal Médio')
        ax.set_xlabel('Mês do Ano')
        ax.set_xticks(ticks=range(1, 13))
        ax.set_xticklabels(['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez'])
        ax.axhline(0, color='gray', linestyle='--', linewidth=1)
        ax.grid(True, which='major', linestyle='--', linewidth='0.5')
        
        fig.tight_layout()
        return fig

    except Exception as e:
        fig, ax = plt.subplots(figsize=(16, 3))
        ax.text(0.5, 0.5, f'Não foi possível gerar o gráfico de sazonalidade anual.\nErro: {e}', 
                horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=12, color='red')
        return fig

def gerar_graficos_previsao_futura(df_historico_limpo, df_previsoes_futuras, target, df_historico_completo):
    """Gera os 3 gráficos solicitados para a análise de previsão futura usando Seaborn."""
    plt.close('all')

    fig_futuro, ax1 = plt.subplots(figsize=(16, 3))
    df_historico_recente = df_historico_limpo.tail(12)
    sns.lineplot(data=df_historico_recente, x=df_historico_recente.index, y=target, ax=ax1, label='Real (Últimos 12 Meses)', color='#0072B2')
    sns.lineplot(data=df_previsoes_futuras, x=df_previsoes_futuras.index, y=target, ax=ax1, label='Previsto (Futuro)', color='#ff0051')
    ax1.set_title(f'Previsão Futura vs. Dados Históricos para {target.upper()}', fontsize=16, fontweight='bold')
    ax1.set(xlabel='Data', ylabel="Média Mensal de Audiência (OPM)")
    ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    fig_futuro.tight_layout()
    
    fig_futuro_media, ax2 = plt.subplots(figsize=(16, 3))
    media_mensal_historica = df_historico_completo.groupby(df_historico_completo.index.month)[target].mean()
    media_para_futuro = df_previsoes_futuras.index.map(lambda data: media_mensal_historica.get(data.month))
    
    df_plot_2 = pd.DataFrame({
        'Previsto (Futuro)': df_previsoes_futuras[target].values,
        'Média Mensal Histórica': media_para_futuro.values
    }, index=df_previsoes_futuras.index)
    
    sns.lineplot(data=df_plot_2, ax=ax2, palette={'Previsto (Futuro)': '#ff0051', 'Média Mensal Histórica': 'green'})
    ax2.lines[1].set_linestyle(':') 
    ax2.set_title(f'Previsão Futura vs. Média Mensal Histórica para {target.upper()}', fontsize=16, fontweight='bold')
    ax2.set(xlabel='Data', ylabel="Média Mensal de Audiência (OPM)")
    ax2.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    fig_futuro_media.tight_layout()
    
    fig_real_media, ax3 = plt.subplots(figsize=(16, 3))
    feature_media_hist = f'media_mensal_historica_{target}'
    if feature_media_hist in df_historico_limpo.columns:
        df_plot_3 = df_historico_limpo[[target, feature_media_hist]].rename(columns={target: 'Real', feature_media_hist: 'Média Mensal Histórica'})
        sns.lineplot(data=df_plot_3, ax=ax3, palette={'Real': '#0072B2', 'Média Mensal Histórica': 'green'})
        ax3.lines[1].set_linestyle(':')
        ax3.set_title('Comparativo: Real vs. Feature de Média Mensal Histórica', fontsize=16, fontweight='bold')
        ax3.set(xlabel='Data', ylabel="Média Mensal de Audiência (OPM)")
        ax3.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        fig_real_media.tight_layout()
    else:
        ax3.text(0.5, 0.5, 'Feature "Média Mensal Histórica" não foi criada.\nGráfico indisponível.', 
                 horizontalalignment='center', verticalalignment='center', transform=ax3.transAxes)

    return fig_futuro, fig_futuro_media, fig_real_media

def gerar_grafico_consolidado(df_historico_completo, df_resultados_cv, df_previsoes_futuras, target):
    """Gera um único gráfico consolidado mostrando a série histórica, as previsões do backtest (CV) e as previsões futuras."""
    plt.close('all')
    fig, ax = plt.subplots(figsize=(16, 3))

    # Pega o início do backtest para alinhar o início do plot dos dados reais
    start_date_backtest = df_resultados_cv.index.min()
    df_historico_plot = df_historico_completo.loc[df_historico_completo.index >= start_date_backtest]

    sns.lineplot(x=df_historico_plot.index, y=df_historico_plot[target], ax=ax, label='Real', color='#0072B2', linewidth=2)
    sns.lineplot(x=df_resultados_cv.index, y=df_resultados_cv['Previsto'], ax=ax, label='Previsto (Backtest CV)', color='#ff0051', linestyle='--', linewidth=2)
    sns.lineplot(x=df_previsoes_futuras.index, y=df_previsoes_futuras[target], ax=ax, label='Previsto (Futuro)', color='#D55E00', linewidth=2.5)

    ax.set_title(f'Visão Geral: Histórico, Backtest e Previsão Futura para {target.upper()}', fontsize=16, fontweight='bold')
    ax.set(xlabel='Data', ylabel="Média Mensal de Audiência (OPM)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.grid(True, which='major', linestyle='--', linewidth='0.5')

    fig.tight_layout()
    return fig


## DESTAQUE - NAMI: Adicionamos df_historico_completo para calcular o YoY no CSV futuro ##
def salvar_resultados_zip(target, df_metricas, features_finais, df_resultados_cv, df_previsoes_futuras, figs_dict, df_historico_completo=None):
    """Salva todos os artefatos em um arquivo zip."""
    file_prefix = f"{target}_"
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'a', zipfile.ZIP_DEFLATED, False) as zip_file:
        for name, fig in figs_dict.items():
            if fig:
                buf = io.BytesIO()
                fig.savefig(buf, format='png', bbox_inches='tight')
                zip_file.writestr(f"{file_prefix}{name}.png", buf.getvalue())
        
        if not df_metricas.empty:
            ## DESTAQUE - NAMI: Adicionado decimal=',' para o CSV de métricas ##
            zip_file.writestr(f"{file_prefix}metricas_cv.csv", df_metricas.to_csv(index=False, decimal=',').encode('utf-8'))
        
        if features_finais:
            features_text = ", ".join(features_finais)
            zip_file.writestr(f"{file_prefix}features_selecionadas.txt", features_text.encode('utf-8'))

        if not df_resultados_cv.empty:
            # df_resultados_cv já vem com as colunas de variação da função treinar_e_avaliar_cv
            ## DESTAQUE - NAMI: Adicionado decimal=',' para o CSV de CV ##
            zip_file.writestr(f"{file_prefix}previsoes_cv.csv", df_resultados_cv.to_csv(decimal=',').encode('utf-8'))
        
        if not df_previsoes_futuras.empty:
            ## DESTAQUE - NAMI: Bloco de cálculo YoY (Ano-contra-Ano) ##
            df_previsoes_para_salvar = df_previsoes_futuras.copy()
            
            # Renomeia a coluna target na previsão para clareza
            col_previsto = f'{target}_Previsto'
            df_previsoes_para_salvar = df_previsoes_para_salvar.rename(columns={target: col_previsto})

            if df_historico_completo is not None:
                try:
                    # Prepara o histórico real
                    col_anterior = f'{target}_Ano_Anterior'
                    df_historico_real = df_historico_completo[[target]].copy()
                    df_historico_real.columns = [col_anterior]
                    
                    # Prepara o df futuro para o merge
                    df_previsoes_para_salvar['data_ano_anterior'] = df_previsoes_para_salvar.index - pd.DateOffset(years=1)
                    
                    # Merge para buscar o valor do ano anterior
                    df_previsoes_para_salvar = pd.merge(
                        df_previsoes_para_salvar,
                        df_historico_real,
                        left_on='data_ano_anterior',
                        right_index=True,
                        how='left'
                    )
                    
                    # Calcula Variações YoY
                    df_previsoes_para_salvar['Variacao_Absoluta_YoY'] = (df_previsoes_para_salvar[col_previsto] - df_previsoes_para_salvar[col_anterior]).round(2)
                    df_previsoes_para_salvar['Variacao_Percentual_YoY_%'] = np.where(
                        df_previsoes_para_salvar[col_anterior] == 0,
                        np.nan,
                        ((df_previsoes_para_salvar[col_previsto] - df_previsoes_para_salvar[col_anterior]) / df_previsoes_para_salvar[col_anterior])
                    ).round(4)
                    
                    # Limpa colunas auxiliares
                    df_previsoes_para_salvar = df_previsoes_para_salvar.drop(columns=['data_ano_anterior'])
                
                except Exception as e:
                    print(f"Debug NAMI: Erro ao calcular YoY para CSV: {e}")
                    # Se falhar, pelo menos salva o que tem, mas sem o YoY
                    # Recria df_previsoes_para_salvar para garantir que não está num estado intermediário
                    df_previsoes_para_salvar = df_previsoes_futuras.copy()
                    df_previsoes_para_salvar = df_previsoes_para_salvar.rename(columns={target: col_previsto})
            
            ## Fim do Destaque ##
            ## DESTAQUE - NAMI: Adicionado decimal=',' para o CSV de previsões futuras ##
            zip_file.writestr(f"{file_prefix}previsoes_futuras.csv", df_previsoes_para_salvar.to_csv(decimal=',').encode('utf-8'))

    with tempfile.NamedTemporaryFile(delete=False, suffix='.zip', prefix=f"{target}_resultados_") as tmp_f:
        tmp_f.write(zip_buffer.getvalue())
        tmp_path = tmp_f.name

    return gr.update(value=tmp_path, visible=True, label=f"Download Resultados ({target}.zip)")

# --- DESTAQUE-MINT: Nova função para gerar o arquivo de dados para reconciliação ---
def gerar_arquivo_mint(target, df_limpo, df_resultados_cv, shap_values_historicos, features_finais, df_previsoes_futuras, df_features_futuras, modelo_final):
    """
    Gera um arquivo CSV completo contendo dados históricos, previsões de backtest,
    previsões futuras e os valores SHAP (contribuições) para cada um,
    já incluindo unique_id e timestamp.
    """
    try:
        # 1. Processar Dados Históricos
        # Começa com o dataframe limpo (features + target real)
        df_hist_base = df_limpo.copy()
        # Adiciona a previsão do backtest (terá NaNs onde não houve backtest)
        df_hist_base['forecast'] = df_resultados_cv['Previsto'] # Renomeado
        df_hist_base = df_hist_base.rename(columns={target: 'actual'}) # Renomeado
        df_hist_base['tipo'] = 'Historico'
        # DESTAQUE-MINT: Adiciona unique_id
        df_hist_base['unique_id'] = target 
        
        # Cria DataFrame de SHAP histórico
        df_shap_hist = pd.DataFrame(
            shap_values_historicos.values, 
            columns=[f"shap_{f}" for f in features_finais], 
            index=df_limpo.index
        )
        # DESTAQUE-MINT: Adiciona o base_value (shap_Intercept)
        df_hist_base['shap_base'] = shap_values_historicos.base_values
        
        # Junta tudo
        df_hist_combinado = pd.concat([df_hist_base, df_shap_hist], axis=1)

        # 2. Processar Dados Futuros
        # Calcula SHAP para as previsões futuras
        explainer_futuro = shap.TreeExplainer(modelo_final)
        shap_values_futuros = explainer_futuro(df_features_futuras[features_finais])
        
        # Cria DataFrame de SHAP futuro
        df_shap_futuro = pd.DataFrame(
            shap_values_futuros.values, 
            columns=[f"shap_{f}" for f in features_finais], 
            index=df_features_futuras.index
        )
        
        # Cria base para o dataframe futuro (começando com as features)
        df_futuro_base = df_features_futuras[features_finais].copy()
        df_futuro_base['actual'] = np.nan # Renomeado
        df_futuro_base['forecast'] = df_previsoes_futuras[target].values # Renomeado e .values para segurança
        df_futuro_base['tipo'] = 'Futuro_Forecast'
        # DESTAQUE-MINT: Adiciona unique_id
        df_futuro_base['unique_id'] = target
        # DESTAQUE-MINT: Adiciona o base_value (shap_Intercept)
        df_futuro_base['shap_base'] = shap_values_futuros.base_values
        
        # Junta tudo
        df_futuro_combinado = pd.concat([df_futuro_base, df_shap_futuro], axis=1)

        # 3. Combinar Histórico e Futuro
        df_final_mint = pd.concat([df_hist_combinado, df_futuro_combinado])
        
        # DESTAQUE-MINT (NOVO AJUSTE): Remove a HORA, mantendo apenas a DATA
        df_final_mint.index = df_final_mint.index.date
        
        # DESTAQUE-MINT: Renomeia o índice para timestamp ANTES de salvar
        df_final_mint.index.name = 'timestamp'
        
        # 4. Organizar Colunas (colocando unique_id e timestamp primeiro)
        cols_id = ['unique_id', 'tipo'] 
        cols_valores = ['actual', 'forecast', 'shap_base'] 
        cols_features = sorted([f for f in features_finais if f in df_final_mint.columns]) # Garante que só peguemos as existentes
        cols_shap = sorted([f"shap_{f}" for f in features_finais if f"shap_{f}" in df_final_mint.columns]) # Garante que só peguemos as existentes
        
        # Junta todas as colunas existentes na ordem desejada
        ordem_final = cols_id + cols_valores + cols_features + cols_shap
        # Pega colunas que podem ter sobrado (caso alguma feature não tenha SHAP, etc.)
        cols_restantes = [c for c in df_final_mint.columns if c not in ordem_final]
        
        df_final_mint = df_final_mint[ordem_final + cols_restantes] # Reordena

        # 5. Salvar em arquivo temporário (agora com índice timestamp sendo salvo como coluna)
        # DESTAQUE-MINT: Mudando de CSV para Excel (.xlsx) a pedido do Ricardo!
        with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx', prefix=f"{target}_mint_data_") as tmp_f:
            # Salva em Excel, engine='openpyxl' é o padrão para .xlsx
            df_final_mint.to_excel(tmp_f.name, index=True, engine='openpyxl') # index=True para salvar o 'timestamp'
            return gr.update(value=tmp_f.name, visible=True, label="Download Dados MinT (.xlsx)") # Atualiza label
            
    except Exception as e:
        print(f"Erro ao gerar arquivo MinT: {e}\n{traceback.format_exc()}")
        return gr.update(value=None, visible=False)

# --- FUNÇÕES DE PIPELINE ---

def executar_pipeline_auto(arquivo, coluna_data_orig, target_orig, colunas_features_orig, colunas_fixas_orig, temporal_features_orig, tamanho_previsao_meses, lags, n_estimators, learning_rate, max_depth, early_stopping_rounds, progress=gr.Progress(track_tqdm=True)):
    if not coluna_data_orig or not target_orig:
        raise gr.Error("Por favor, selecione a 'Coluna de data' e a 'Variável TARGET' antes de executar!")
    
    log_text = ""
    try:
        progress(0, desc="Carregando dados...")
        log_text += "1. Carregando e preparando dados...\n"
        # DESTAQUE-MINT: Adicionado 'None' para o novo output 'mint_output_file'
        yield log_text, None, None, None, None, None, None, gr.update(value=None, visible=False), "### Seleção de Features\n*Aguardando início do processo...*", gr.update(visible=False), "", [], None, None, None, None, None, None, gr.update(value=None, visible=False)
        
        df_original = pd.read_csv(arquivo.name) if arquivo.name.endswith('.csv') else pd.read_excel(arquivo.name)
        df = sanitize_columns(df_original.copy())
        
        coluna_data = sanitize_columns(pd.DataFrame(columns=[coluna_data_orig])).columns[0]
        target = sanitize_columns(pd.DataFrame(columns=[target_orig])).columns[0]
        colunas_features = sanitize_columns(pd.DataFrame(columns=colunas_features_orig)).columns.tolist()
        colunas_fixas = sanitize_columns(pd.DataFrame(columns=colunas_fixas_orig)).columns.tolist() if colunas_fixas_orig else []
        
        df[coluna_data] = pd.to_datetime(df[coluna_data]); df = df.set_index(coluna_data).sort_index()
        df_base_para_futuro = df.copy()

        progress(0.05, desc="Criando Features Temporais...")
        df, log_temporal, created_temporal_features = adicionar_features_temporais(df, temporal_features_orig, target)
        log_text += log_temporal

        colunas_fixas.extend(created_temporal_features)
        colunas_fixas = sorted(list(set(colunas_fixas)))
        
        data_final_treino_dt = df.index.max() - pd.DateOffset(months=int(tamanho_previsao_meses))
        datas_split = {'treino_fim': data_final_treino_dt}
        log_text += f"Divisão: Treino até {datas_split['treino_fim'].date()} (baseado em um horizonte de previsão de {tamanho_previsao_meses} meses)\n"
        
        progress(0.1, desc="Criando Features de Lag...")
        config_geracao = [(col, f'_{col}', lags) for col in colunas_features]
        df_features, log_criacao = criar_features_de_lag(df, config_geracao)
        log_text += log_criacao
        
        log_text += f"ℹ️ DataFrame com features geradas (antes do dropna) tem {len(df_features)} linhas.\n"
        
        progress(0.3, desc="Selecionando Features com CV...")
        treino_completo_com_nans = df_features[df_features.index <= datas_split['treino_fim']]
        log_text += "\n🛡️ Usando Validação Cruzada (TimeSeriesSplit) para seleção de features robusta...\n"
        
        # --- CONFIGURAÇÃO ROBUSTA DE VALIDAÇÃO CRUZADA ---
        n_splits_cv = 5
        tscv_selection = TimeSeriesSplit(n_splits=n_splits_cv, test_size=int(tamanho_previsao_meses))
        log_text += f"📊 CV para seleção configurado com {tscv_selection.get_n_splits(treino_completo_com_nans)} folds.\n"

        params = {'n_estimators': int(n_estimators), 'learning_rate': learning_rate, 'max_depth': int(max_depth),
                   'random_state': 42, 'early_stopping_rounds': int(early_stopping_rounds)}

        def calcular_mape_cv(features):
            if not features: return 1.0
            try:
                mape_scores = []
                df_teste_features = treino_completo_com_nans[features + [target]].dropna()
                
                if len(df_teste_features) < (tscv_selection.get_n_splits() + 1) * tscv_selection.test_size:
                    return float('inf')

                X_treino_selecao, y_treino_selecao = df_teste_features[features], df_teste_features[target]
                
                for train_idx, val_idx in tscv_selection.split(X_treino_selecao):
                    X_train, X_val = X_treino_selecao.iloc[train_idx], X_treino_selecao.iloc[val_idx]
                    y_train, y_val = y_treino_selecao.iloc[train_idx], y_treino_selecao.iloc[val_idx]

                    model = xgb.XGBRegressor(**params)
                    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
                    mape_scores.append(mean_absolute_percentage_error(y_val, model.predict(X_val)))
                
                return np.mean(mape_scores)
            except Exception: return float('inf')

        features_candidatas = [c for c in df_features.columns if c.startswith('lag_')]
        features_atuais = colunas_fixas.copy()
        mape_atual = calcular_mape_cv(features_atuais)
        
        log_text += f"\n--- 🤖 Iniciando a seleção com REFINAMENTO IMEDIATO e CV ---\n"
        log_text += f"MAPE de Referência (CV na base de treino com features fixas/temporais): {mape_atual:.5f}\n"
        
        iteration = 0
        while True:
            iteration += 1
            log_text += f"\n==================== ITERAÇÃO DE SELEÇÃO {iteration} ====================\n"
            log_text += f"MAPE CV atual: {mape_atual:.5f}\n"
            # DESTAQUE-MINT: Adicionado 'None' para o novo output 'mint_output_file'
            yield log_text, None, None, None, None, None, None, gr.update(value=None, visible=False), "### Seleção de Features\n*Aguardando início do processo...*", gr.update(visible=False), "", features_atuais, None, None, None, None, None, None, gr.update(value=None, visible=False)
            
            if not features_candidatas:
                log_text += "Não há mais features candidatas para testar. Finalizando seleção.\n"
                break

            log_text += "\n--- Fase 1: Buscando a melhor feature para ADICIONAR ---\n"
            # DESTAQUE-MINT: Adicionado 'None' para o novo output 'mint_output_file'
            yield log_text, None, None, None, None, None, None, gr.update(value=None, visible=False), "### Seleção de Features\n*Aguardando início do processo...*", gr.update(visible=False), "", features_atuais, None, None, None, None, None, None, gr.update(value=None, visible=False)

            resultados_adicao = {}
            for feature in features_candidatas:
                mape_teste = calcular_mape_cv(features_atuais + [feature])
                resultados_adicao[feature] = mape_teste
                log_text += f"    - Testando adicionar '{feature}': MAPE CV = {mape_teste:.5f}\n"
                # DESTAQUE-MINT: Adicionado 'None' para o novo output 'mint_output_file'
                yield log_text, None, None, None, None, None, None, gr.update(value=None, visible=False), "### Seleção de Features\n*Aguardando início do processo...*", gr.update(visible=False), "", features_atuais, None, None, None, None, None, None, gr.update(value=None, visible=False)
            
            melhor_nova_feature = min(resultados_adicao, key=resultados_adicao.get)
            melhor_mape_adicao = resultados_adicao[melhor_nova_feature]

            if melhor_mape_adicao >= mape_atual:
                log_text += f"\n❌ Nenhuma nova feature melhorou o MAPE CV. Finalizando.\n"
                break

            log_text += f"✅ Melhoria encontrada! Adicionando '{melhor_nova_feature}' (MAPE CV cai para {melhor_mape_adicao:.5f}).\n"
            # DESTAQUE-MINT: Adicionado 'None' para o novo output 'mint_output_file'
            yield log_text, None, None, None, None, None, None, gr.update(value=None, visible=False), "### Seleção de Features\n*Aguardando início do processo...*", gr.update(visible=False), "", features_atuais + [melhor_nova_feature], None, None, None, None, None, None, gr.update(value=None, visible=False)
            features_para_refinar = features_atuais + [melhor_nova_feature]
            mape_para_refinar = melhor_mape_adicao
            
            log_text += "\n--- Fase 2: Iniciando REFINAMENTO (tentando remover features antigas) ---\n"
            # DESTAQUE-MINT: Adicionado 'None' para o novo output 'mint_output_file'
            yield log_text, None, None, None, None, None, None, gr.update(value=None, visible=False), "### Seleção de Features\n*Aguardando início do processo...*", gr.update(visible=False), "", features_para_refinar, None, None, None, None, None, None, gr.update(value=None, visible=False)

            while True:
                features_removiveis = [f for f in features_para_refinar if f not in (colunas_fixas or []) and f != melhor_nova_feature]
                if not features_removiveis:
                    log_text += "    - Nenhuma feature antiga para remover. Fim do refinamento.\n"
                    break

                resultados_remocao = {}
                for f_remover in features_removiveis:
                    mape_teste = calcular_mape_cv([feat for feat in features_para_refinar if feat != f_remover])
                    resultados_remocao[f_remover] = mape_teste
                    log_text += f"    - Testando remover '{f_remover}': MAPE CV = {mape_teste:.5f}\n"
                    # DESTAQUE-MINT: Adicionado 'None' para o novo output 'mint_output_file'
                    yield log_text, None, None, None, None, None, None, gr.update(value=None, visible=False), "### Seleção de Features\n*Aguardando início do processo...*", gr.update(visible=False), "", features_para_refinar, None, None, None, None, None, None, gr.update(value=None, visible=False)
                
                melhor_remocao = min(resultados_remocao, key=resultados_remocao.get)
                mape_da_melhor_remocao = resultados_remocao[melhor_remocao]

                if mape_da_melhor_remocao < mape_para_refinar:
                    log_text += f"    💡 Refinamento! Removendo '{melhor_remocao}', MAPE CV cai para {mape_da_melhor_remocao:.5f}\n"
                    mape_para_refinar = mape_da_melhor_remocao
                    features_para_refinar.remove(melhor_remocao)
                else:
                    log_text += f"    - Nenhuma remoção melhorou mais o MAPE. Fim do refinamento.\n"
                    break
            
            features_atuais = features_para_refinar
            mape_atual = mape_para_refinar
            features_candidatas.remove(melhor_nova_feature)
            
        def format_features_md(features, title="### Features Finais Selecionadas (Auto)"):
            if not features: return f"{title}\n*Nenhuma feature selecionada...*"
            fixas_e_temporais = sorted([f for f in features if f in colunas_fixas])
            dinamicas = sorted([f for f in features if f not in colunas_fixas])
            md = f"{title}\n"
            if fixas_e_temporais: md += "**Fixas/Temporais:**\n" + "\n".join([f"- `{f}`" for f in fixas_e_temporais]) + "\n"
            if dinamicas: md += "**Selecionadas (Lags):**\n" + "\n".join([f"- `{f}`" for f in dinamicas])
            return md

        log_text += f"\n--- ✨ Seleção Concluída! ✨ ---\nMelhor MAPE na VALIDAÇÃO CRUZADA: {mape_atual:.5f}\n"
        # DESTAQUE-MINT: Adicionado 'None' para o novo output 'mint_output_file'
        yield log_text, None, None, None, None, None, None, gr.update(value=None, visible=False), "### Seleção de Features\n*Aguardando início do processo...*", gr.update(visible=False), "", features_atuais, None, None, None, None, None, None, gr.update(value=None, visible=False)
        
        features_finais = features_atuais
        features_md_final = format_features_md(features_finais)
        features_copy_str = ", ".join(features_finais)

        log_text += "\n⚙️ Preparando dataset final com as features selecionadas...\n"
        colunas_finais_treino = features_finais + [target]
        df_limpo = df_features[colunas_finais_treino].dropna()
        log_text += f"ℹ️ Dataset final (após dropna com features selecionadas) tem {len(df_limpo)} linhas.\n"
        log_text += "\n\n--- 🚀 Preparando para o treino final com as melhores features! Aguarde... ---\n"
        
        # DESTAQUE-MINT: Captura o novo 'shap_values_hist'
        log_treino_final, df_res, fig_pred, df_met, fig_imp, fig_shap_sum, fig_shap_force, modelo_final, shap_values_hist = treinar_e_avaliar_cv(df_limpo, features_finais, target, int(tamanho_previsao_meses), params, progress)
        log_text += log_treino_final

        config_geracao_futuro = [(col, f'_{col}', lags) for col in colunas_features]
        # DESTAQUE-MINT: Captura o novo 'df_features_futuras'
        df_previsoes_futuras, log_futuro, df_features_futuras = prever_futuro(modelo_final, df_base_para_futuro, features_finais, target, int(tamanho_previsao_meses), config_geracao_futuro, temporal_features_orig)
        log_text += log_futuro
        fig_fut, fig_fut_media, fig_real_media = gerar_graficos_previsao_futura(df_limpo, df_previsoes_futuras, target, df_base_para_futuro)
        
        fig_decomp = gerar_grafico_decomposicao(df_base_para_futuro, target)
        fig_sazonal = gerar_grafico_sazonalidade_anual(df_base_para_futuro, target)
        fig_consolidado = gerar_grafico_consolidado(df_base_para_futuro, df_res, df_previsoes_futuras, target)

        figs_para_salvar = {
            "previsao_cv": fig_pred, "importancia": fig_imp, "shap_summary": fig_shap_sum, 
            "shap_force": fig_shap_force, "previsao_futura": fig_fut, 
            "previsao_vs_media_hist": fig_fut_media, "real_vs_media_hist": fig_real_media,
            "decomposicao_serie": fig_decomp,
            "sazonalidade_anual": fig_sazonal,
            "visao_consolidada": fig_consolidado
        }
        
        ## DESTAQUE - NAMI: Passando df_base_para_futuro para salvar_resultados_zip ##
        zip_file = salvar_resultados_zip(target, df_met, features_finais, df_res, df_previsoes_futuras, figs_para_salvar, df_base_para_futuro)
        
        # DESTAQUE-MINT: Gera o arquivo MinT
        mint_file = gr.update(value=None, visible=False) # Default
        if shap_values_hist is not None:
             mint_file = gerar_arquivo_mint(target, df_limpo, df_res, shap_values_hist, features_finais, df_previsoes_futuras, df_features_futuras, modelo_final)
        else:
            log_text += "\n⚠️ Não foi possível gerar o arquivo MinT pois os valores SHAP não foram calculados."

        # DESTAQUE-MINT: Adiciona 'mint_file' ao yield final
        yield log_text, df_res, fig_pred, df_met, fig_imp, fig_shap_sum, fig_shap_force, zip_file, features_md_final, gr.update(visible=True), features_copy_str, features_finais, fig_fut, fig_fut_media, fig_real_media, fig_decomp, fig_sazonal, fig_consolidado, mint_file

    except Exception as e:
        log_text += f"\n❌ Ocorreu um erro: {e}\n\n--- Detalhes ---\n{traceback.format_exc()}"
        # DESTAQUE-MINT: Adicionado 'None' para o novo output 'mint_output_file'
        yield log_text, None, None, None, None, None, None, gr.update(value=None, visible=False), "### Erro\n*Ocorreu um erro durante a execução.*", gr.update(visible=False), "", [], None, None, None, None, None, None, gr.update(value=None, visible=False)

def executar_pipeline_manual(arquivo, coluna_data_orig, target_orig, tamanho_previsao_meses, temporal_features_orig, features_finais_selecionadas, colunas_configuradas_orig, n_estimators, learning_rate, max_depth, early_stopping_rounds, *lista_de_lags, progress=gr.Progress(track_tqdm=True)):
    if not coluna_data_orig or not target_orig:
        raise gr.Error("Por favor, selecione a 'Coluna de data' e a 'Variável TARGET' antes de executar!")
    if not features_finais_selecionadas:
        raise gr.Error("Nenhuma feature foi selecionada para o treino manual. Selecione ou cole as features na Etapa 4.")

    log_text = ""
    # DESTAQUE-MINT: Aumentado o número de 'None' de 14 para 15
    # DESTAQUE - NAMI: Aumentado para 16 (15 + 1 do mint_file)
    blank_outputs_for_yield = [None] * 16 
    try:
        progress(0, desc="Carregando dados...")
        log_text = "1. Carregando e preparando dados para treino manual...\n"
        
        initial_outputs = list(blank_outputs_for_yield)
        initial_outputs[8] = "### Features Selecionadas (Manual)\n*Aguardando execução...*" # Ajustado índice
        yield log_text, *initial_outputs
        
        df_original = pd.read_csv(arquivo.name) if arquivo.name.endswith('.csv') else pd.read_excel(arquivo.name)
        df = sanitize_columns(df_original.copy())

        coluna_data = sanitize_columns(pd.DataFrame(columns=[coluna_data_orig])).columns[0]
        target = sanitize_columns(pd.DataFrame(columns=[target_orig])).columns[0]
        colunas_configuradas = sanitize_columns(pd.DataFrame(columns=colunas_configuradas_orig)).columns.tolist()
        colunas_originais_sanitizadas = df.columns.tolist()

        df[coluna_data] = pd.to_datetime(df[coluna_data]); df = df.set_index(coluna_data).sort_index()
        df_base_para_futuro = df.copy()

        progress(0.1, desc="Criando Features Temporais...")
        df, log_temporal, _ = adicionar_features_temporais(df, temporal_features_orig, target)
        log_text += log_temporal

        progress(0.2, desc="Criando features de lag...")
        config_dict = {}
        for i, coluna in enumerate(colunas_configuradas):
            lags_selecionados = lista_de_lags[i]
            if lags_selecionados:
                if coluna not in config_dict: config_dict[coluna] = set()
                config_dict[coluna].update(lags_selecionados)

        log_text += "⚙️ Verificando features selecionadas para gerar lags necessários...\n"
        for feature in features_finais_selecionadas:
            if feature.startswith('lag_') and feature.endswith('_meses'):
                parts = feature.split('_')
                if len(parts) >= 4 and parts[-1] == 'meses' and parts[-2].isdigit():
                    try:
                        lag = int(parts[-2])
                        base_col = "_".join(parts[1:-2])
                        if base_col in colunas_originais_sanitizadas:
                            if base_col not in config_dict: config_dict[base_col] = set()
                            if lag not in config_dict[base_col]:
                                log_text += f"    -> Inferido lag={lag} para a coluna '{base_col}' a partir de '{feature}'.\n"
                                config_dict[base_col].add(lag)
                    except (ValueError, IndexError): continue

        config_geracao = []
        for coluna, lags_set in config_dict.items():
            if lags_set:
                config_geracao.append((coluna, f'_{coluna}', sorted(list(lags_set))))

        df_features, log_criacao = criar_features_de_lag(df, config_geracao)
        log_text += log_criacao

        log_text += "⚙️ Aplicando filtro de consistência para garantir reprodutibilidade...\n"
        features_finais_existentes = [f for f in features_finais_selecionadas if f in df_features.columns]
        features_descartadas = set(features_finais_selecionadas) - set(features_finais_existentes)
        if features_descartadas:
            log_text += f"⚠️ Aviso: As seguintes features foram selecionadas mas não puderam ser geradas ou encontradas e serão ignoradas: {', '.join(sorted(list(features_descartadas)))}\n"

        if not features_finais_existentes:
            raise ValueError("Nenhuma das features selecionadas pôde ser encontrada ou gerada. Verifique as configurações.")

        colunas_para_modelo = features_finais_existentes + [target]
        df_para_modelo = df_features[colunas_para_modelo]
        df_limpo = df_para_modelo.dropna()
        log_text += f"ℹ️ Para o processo, {len(df_features) - len(df_limpo)} linhas com dados ausentes foram removidas (baseado nas features selecionadas).\n"

        params = {'n_estimators': int(n_estimators), 'learning_rate': learning_rate, 'max_depth': int(max_depth), 'random_state': 42, 'early_stopping_rounds': int(early_stopping_rounds)}
        
        features_para_md = sorted(features_finais_existentes)
        features_selecionadas_md = "### Features Selecionadas (Manual):\n" + "\n".join([f"- `{f}`" for f in features_para_md])
        
        progress(0.5, desc="Treinando modelo com CV...")
        # DESTAQUE-MINT: Captura o novo 'shap_values_hist'
        log_treino_final, df_res, fig_pred, df_met, fig_imp, fig_shap_sum, fig_shap_force, modelo_final, shap_values_hist = treinar_e_avaliar_cv(df_limpo, features_finais_existentes, target, int(tamanho_previsao_meses), params, progress)
        log_text += log_treino_final

        # DESTAQUE-MINT: Captura o novo 'df_features_futuras'
        df_previsoes_futuras, log_futuro, df_features_futuras = prever_futuro(modelo_final, df_base_para_futuro, features_finais_existentes, target, int(tamanho_previsao_meses), config_geracao, temporal_features_orig)
        log_text += log_futuro
        fig_fut, fig_fut_media, fig_real_media = gerar_graficos_previsao_futura(df_limpo, df_previsoes_futuras, target, df_base_para_futuro)
        
        fig_decomp = gerar_grafico_decomposicao(df_base_para_futuro, target)
        fig_sazonal = gerar_grafico_sazonalidade_anual(df_base_para_futuro, target)
        fig_consolidado = gerar_grafico_consolidado(df_base_para_futuro, df_res, df_previsoes_futuras, target)

        figs_para_salvar = {
            "previsao_cv": fig_pred, "importancia": fig_imp, "shap_summary": fig_shap_sum, "shap_force": fig_shap_force,
            "previsao_futura": fig_fut, "previsao_vs_media_hist": fig_fut_media, "real_vs_media_hist": fig_real_media,
            "decomposicao_serie": fig_decomp,
            "sazonalidade_anual": fig_sazonal,
            "visao_consolidada": fig_consolidado
        }
        
        ## DESTAQUE - NAMI: Passando df_base_para_futuro para salvar_resultados_zip ##
        zip_file = salvar_resultados_zip(target, df_met, features_finais_existentes, df_res, df_previsoes_futuras, figs_para_salvar, df_base_para_futuro)
        
        # DESTAQUE-MINT: Gera o arquivo MinT
        mint_file = gr.update(value=None, visible=False) # Default
        if shap_values_hist is not None:
             mint_file = gerar_arquivo_mint(target, df_limpo, df_res, shap_values_hist, features_finais_existentes, df_previsoes_futuras, df_features_futuras, modelo_final)
        else:
            log_text += "\n⚠️ Não foi possível gerar o arquivo MinT pois os valores SHAP não foram calculados."

        # DESTAQUE-MINT: Adiciona 'mint_file' ao yield final
        yield log_text, df_res, fig_pred, df_met, fig_imp, fig_shap_sum, fig_shap_force, zip_file, features_selecionadas_md, fig_fut, fig_fut_media, fig_real_media, fig_decomp, fig_sazonal, fig_consolidado, mint_file
        
    except Exception as e:
        log_text += f"\n❌ Ocorreu um erro: {e}\n\n--- Detalhes ---\n{traceback.format_exc()}"
        error_outputs = list(blank_outputs_for_yield)
        error_outputs[8] = "### Erro\n*Ocorreu um erro durante a execução.*" # Ajustado índice
        yield log_text, *error_outputs

def processar_arquivo(arquivo):
    if arquivo is None: return [gr.update(visible=False)] * 10
    try:
        df = pd.read_csv(arquivo.name) if arquivo.name.endswith('.csv') else pd.read_excel(arquivo.name)
        df_original_cols = df.columns.tolist()
        df_sanitized = sanitize_columns(df.copy()) 
        colunas_sanitizadas = df_sanitized.columns.tolist()
        
        try:
            col_data_sanitizada = [c for c in colunas_sanitizadas if df_sanitized[c].astype(str).str.match(r'\d{4}-\d{2}-\d{2}').all()][0]
            col_data_original = df_original_cols[colunas_sanitizadas.index(col_data_sanitizada)]
        except (IndexError, Exception):
            col_data_original = df_original_cols[0]
        
        colunas_fixas_choices = df_original_cols
        feature_choices = [c for c in df_original_cols if c != col_data_original]

        return [
            gr.update(visible=True), gr.update(visible=True), gr.update(choices=df_original_cols, value=col_data_original),
            gr.update(choices=df_original_cols), gr.update(choices=feature_choices), gr.update(choices=colunas_fixas_choices),
            gr.update(choices=feature_choices), df_original_cols, colunas_fixas_choices, feature_choices
        ]
    except Exception as e:
        raise gr.Error(f"Erro ao ler o arquivo: {e}")

def update_manual_lag_ui(colunas_selecionadas):
    MAX_COLS = 15
    updates = []
    if not colunas_selecionadas: colunas_selecionadas = []
    for i in range(len(colunas_selecionadas)):
        if i < MAX_COLS:
            updates.append(gr.update(visible=True))
            updates.append(gr.update(value=f"**{colunas_selecionadas[i]}**"))

    for i in range(len(colunas_selecionadas), MAX_COLS):
        updates.append(gr.update(visible=False))
        updates.append(gr.update(value=""))
    return updates

def gerar_features_para_selecao_manual(arquivo, coluna_data_orig, target_orig, temporal_features_orig, colunas_configuradas_orig, *lista_de_lags):
    if not coluna_data_orig or not target_orig:
        raise gr.Error("Selecione a 'Coluna de data' e a 'Variável TARGET' primeiro!")

    df = pd.read_csv(arquivo.name) if arquivo.name.endswith('.csv') else pd.read_excel(arquivo.name)
    df = sanitize_columns(df)

    coluna_data = sanitize_columns(pd.DataFrame(columns=[coluna_data_orig])).columns[0]
    target = sanitize_columns(pd.DataFrame(columns=[target_orig])).columns[0]
    colunas_configuradas = sanitize_columns(pd.DataFrame(columns=colunas_configuradas_orig)).columns.tolist()
    
    df[coluna_data] = pd.to_datetime(df[coluna_data]); df = df.set_index(coluna_data).sort_index(); 
    
    df, _, created_temporal_features = adicionar_features_temporais(df, temporal_features_orig, target)
    
    config_geracao = []
    for i, coluna in enumerate(colunas_configuradas):
        lags_selecionados = lista_de_lags[i]
        if lags_selecionados:
            sufixo = f'_{coluna}'
            config_geracao.append((coluna, sufixo, lags_selecionados))

    df_features, _ = criar_features_de_lag(df, config_geracao)
    
    features_de_lag = [c for c in df_features.columns if c.startswith('lag_')]
    features_disponiveis = sorted(list(set(created_temporal_features + features_de_lag)))
    
    return gr.update(), gr.update(choices=features_disponiveis, value=features_disponiveis), features_disponiveis

def apply_pasted_features_to_checklist(pasted_text):
    if not pasted_text or not isinstance(pasted_text, str):
        return gr.update(), gr.update(value=[], choices=[])
    features = [f.strip() for f in pasted_text.split(',') if f.strip()]
    return gr.update(), gr.update(choices=features, value=features)

def reset_all():
    lag_ui_updates = update_manual_lag_ui([])
    return [
        # DESTAQUE-MINT: Adicionado gr.update() para o novo 'mint_output_file'
        gr.update(), gr.update(), gr.update(), gr.update(value=None), gr.update(value=None), gr.update(value=[]), gr.update(value=[]),
        gr.update(value=["Mês", "Features de Fourier (Anual)", "Média Mensal Histórica (Target)"]), gr.update(value=list(range(1, 13))),
        gr.update(value=["Mês", "Features de Fourier (Anual)", "Média Mensal Histórica (Target)"]), gr.update(value=[]),
        gr.update(visible=True), gr.update(value=""), gr.update(value=[], choices=[]),
        "", None, "", gr.update(visible=False), "", gr.update(value=None, visible=False), gr.update(value=None, visible=False, label="Download Dados MinT (.xlsx)"), # Label atualizada
        None, None, None, None, None, None, None, None, None, None, None,
    ] + lag_ui_updates

# --- CONSTRUÇÃO DA INTERFACE ---
with gr.Blocks(theme=gr.themes.Soft(), title="AutoML de Séries Temporais Pro", css=custom_css) as demo:
    gr.Markdown("# 🤖 AutoML para Séries Temporais com Feature Selection Pro v4.7 (com exportação MinT)")
    gr.Markdown("Faça o upload, configure o horizonte de previsão, o modo (automático ou manual) e rode um pipeline completo de modelagem com validação cruzada e previsão futura!")
    
    colunas_features_state = gr.State([])
    colunas_fixas_state = gr.State([])
    features_manuais_state = gr.State([])
    valid_features_state = gr.State([])
    auto_features_list_state = gr.State([])
    
    temporal_features_choices = ["Mês", "Ano", "Dia da Semana", "Dia do Ano", "Semana do Ano", "Features de Fourier (Anual)", "Média Mensal Histórica (Target)"]

    with gr.Row():
        arquivo_input = gr.File(label="Selecione seu arquivo (.csv ou .xlsx)", scale=3)
        reset_button = gr.Button("Limpar e Resetar Etapas", elem_classes=["orange-button"], visible=False, scale=1)

    with gr.Group(visible=False) as grupo_principal:
        with gr.Group():
            with gr.Row():
                coluna_data_input = gr.Dropdown(label="Coluna de data")
                tamanho_previsao_input = gr.Number(label="Horizonte de Previsão (meses)", value=6)
                coluna_target_input = gr.Dropdown(label="Variável TARGET (a prever)")

        with gr.Accordion("Parâmetros do Modelo (Ajuste Fino)", open=True):
            with gr.Group():
                n_estimators_input = gr.Slider(label="Número de Árvores (n_estimators)", minimum=50, maximum=1000, value=1000, step=50)
                learning_rate_input = gr.Slider(label="Taxa de Aprendizado (learning_rate)", minimum=0.01, maximum=0.3, value=0.1, step=0.01)
                max_depth_input = gr.Slider(label="Profundidade Máxima (max_depth)", minimum=3, maximum=10, value=6, step=1)
                early_stopping_input = gr.Number(label="Parada Antecipada (early_stopping_rounds)", value=20)

        with gr.Tabs():
            with gr.TabItem("AutoML com Seleção de Features"):
                with gr.Group(elem_classes=["dark-checkbox-group"]):
                    gr.Markdown("### 1. Features Temporais")
                    temporal_features_auto = gr.CheckboxGroup(choices=temporal_features_choices, value=["Mês", "Features de Fourier (Anual)", "Média Mensal Histórica (Target)"])
                    with gr.Row():
                        select_all_temp_auto = gr.Button("Selecionar Todos", scale=1, elem_classes=["orange-button"])
                        clear_temp_auto = gr.Button("Limpar", scale=1, elem_classes=["orange-button"])
                
                with gr.Group(elem_classes=["dark-checkbox-group"]):
                    gr.Markdown("### 2. Colunas para gerar features (lags)")
                    colunas_features_auto = gr.CheckboxGroup()
                    with gr.Row():
                        select_all_feat_auto = gr.Button("Selecionar Todos", scale=1, elem_classes=["orange-button"])
                        clear_feat_auto = gr.Button("Limpar", scale=1, elem_classes=["orange-button"])

                with gr.Group(elem_classes=["dark-checkbox-group"]):
                    gr.Markdown("### 3. Lags (meses)")
                    lags_auto = gr.CheckboxGroup(choices=list(range(1, 13)), value=list(range(1, 13)))
                    with gr.Row():
                        select_all_lags_auto = gr.Button("Selecionar Todos", scale=1, elem_classes=["orange-button"])
                        clear_lags_auto = gr.Button("Limpar", scale=1, elem_classes=["orange-button"])

                with gr.Group(elem_classes=["dark-checkbox-group"]):
                    gr.Markdown("### 4. Features FIXAS (opcional)")
                    colunas_fixas_auto = gr.CheckboxGroup()
                    with gr.Row():
                        select_all_fixas_auto = gr.Button("Selecionar Todos", scale=1, elem_classes=["orange-button"])
                        clear_fixas_auto = gr.Button("Limpar", scale=1, elem_classes=["orange-button"])
                
                run_button_auto = gr.Button("🚀 Executar Pipeline Automático!", variant="primary", scale=1)

            with gr.TabItem("Treino Manual Direto"):
                with gr.Group(elem_classes=["dark-checkbox-group"]):
                    gr.Markdown("### Passo 1: Selecione as features temporais")
                    temporal_features_manual = gr.CheckboxGroup(choices=temporal_features_choices, value=["Mês", "Features de Fourier (Anual)", "Média Mensal Histórica (Target)"])
                    with gr.Row():
                        select_all_temp_manual = gr.Button("Selecionar Todos", scale=1, elem_classes=["orange-button"])
                        clear_temp_manual = gr.Button("Limpar", scale=1, elem_classes=["orange-button"])
                
                with gr.Group(elem_classes=["dark-checkbox-group"]):
                    gr.Markdown("### Passo 2: Selecione as colunas para gerar features de LAG (opcional)")
                    colunas_features_manual = gr.CheckboxGroup()
                    with gr.Row():
                        select_all_feat_manual = gr.Button("Selecionar Todos", scale=1, elem_classes=["orange-button"])
                        clear_feat_manual = gr.Button("Limpar", scale=1, elem_classes=["orange-button"])
                
                with gr.Group():
                    gr.Markdown("### Passo 3: Configure os lags para cada coluna (opcional)")
                    lag_configs_ui = []
                    MAX_COLS_UI = 15
                    for i in range(MAX_COLS_UI):
                        with gr.Row(visible=False) as lag_row:
                            col_name = gr.Markdown()
                            gr.Markdown("Lags:")
                            col_lags = gr.CheckboxGroup(choices=list(range(1, 13)), value=[], interactive=True, elem_classes=["dark-checkbox-group"])
                            lag_configs_ui.append({'group': lag_row, 'name': col_name, 'lags': col_lags})
                    
                    generate_features_button = gr.Button("Gerar e Listar Features Disponíveis (para selecionar abaixo)")

                with gr.Group(visible=True, elem_classes=["dark-checkbox-group"]) as manual_select_group:
                    gr.Markdown("### Passo 4: Selecione/Cole as features finais e treine o modelo")
                    gr.Markdown("Use uma das opções abaixo para popular a lista de features e depois clique em 'Treinar'.")

                    with gr.Row():
                        paste_features_manual_textbox = gr.Textbox(label="Opção 1: Cole features (separadas por vírgula) e clique em Aplicar", scale=3, placeholder="lag_vendas_1_meses, mes, fourier_sin_anual, ...")
                        apply_pasted_features_button = gr.Button("Aplicar")
                    
                    paste_button_manual = gr.Button("Opção 2: Usar as features da última execução do AutoML")
                    
                    gr.Markdown("#### Revise as features que serão usadas no treino:")
                    manual_features_checklist = gr.CheckboxGroup(label="Features para o Modelo")
                    with gr.Row():
                        select_all_final_manual = gr.Button("Selecionar Todos", scale=1, elem_classes=["orange-button"])
                        clear_final_manual = gr.Button("Limpar", scale=1, elem_classes=["orange-button"])
                    
                    run_button_manual = gr.Button("🚀 Treinar com Features Selecionadas!", variant="primary")
        
        with gr.Group():
            gr.Markdown("## Resultados")
            log_output = gr.Textbox(label="Log de Execução", lines=15, interactive=False)
            with gr.Row():
                metricas_output = gr.DataFrame(label="Métricas de Performance (CV)", headers=['Métrica', 'Valor'])
                with gr.Column(scale=2):
                    features_selecionadas_output = gr.Markdown(label="Seleção de Features")
                    with gr.Row(visible=False) as copy_row:
                        features_copy_output = gr.Textbox(label="Lista de Features (para copiar)", show_label=False, interactive=False, show_copy_button=True, placeholder="A lista de features aparecerá aqui para cópia...")
                    download_output = gr.File(label="Download dos Gráficos e Resultados (.zip)", visible=False)
                    # DESTAQUE-MINT: Adicionado o novo componente de arquivo e label atualizada
                    mint_output_file = gr.File(label="Download Dados MinT (.xlsx)", visible=False)

            with gr.Tabs():
                with gr.TabItem("📊 Resultados da Previsão"):
                    plot_consolidado_output = gr.Plot(label="Visão Geral: Histórico, Backtest e Previsão Futura")
                    plot_pred_output = gr.Plot(label="Gráfico de Previsão (Foco no Backtest)")
                    dataframe_output = gr.DataFrame(label="Tabela com Previsões (Backtest)")
                with gr.TabItem("🧠 Análise do Modelo"):
                    plot_imp_output = gr.Plot(label="Importância das Features (XGBoost)")
                    plot_shap_summary_output = gr.Plot(label="Análise de Impacto Geral (SHAP Summary)")
                    plot_shap_force_output = gr.Plot(label="Análise de Previsão Individual (SHAP Force Plot)")
                with gr.TabItem("🔮 Previsão Futura"):
                    plot_futuro_output = gr.Plot(label="Previsão Futura vs. Histórico")
                    plot_futuro_vs_media_output = gr.Plot(label="Previsão Futura vs. Média Mensal Histórica")
                    plot_real_vs_media_output = gr.Plot(label="Real vs. Média Mensal Histórica")
                with gr.TabItem("🔎 Decomposição da Série"):
                    plot_decomp_output = gr.Plot(label="Decomposição de Tendência e Sazonalidade")
                    plot_sazonal_output = gr.Plot(label="Padrão de Sazonalidade Anual")

    # --- Lógica dos Eventos ---
    # DESTAQUE-MINT: Adicionado 'mint_output_file' às listas de outputs
    outputs_list_manual = [log_output, dataframe_output, plot_pred_output, metricas_output, plot_imp_output, plot_shap_summary_output, plot_shap_force_output, download_output, features_selecionadas_output, plot_futuro_output, plot_futuro_vs_media_output, plot_real_vs_media_output, plot_decomp_output, plot_sazonal_output, plot_consolidado_output, mint_output_file]
    outputs_list_auto = [log_output, dataframe_output, plot_pred_output, metricas_output, plot_imp_output, plot_shap_summary_output, plot_shap_force_output, download_output, features_selecionadas_output, copy_row, features_copy_output, auto_features_list_state, plot_futuro_output, plot_futuro_vs_media_output, plot_real_vs_media_output, plot_decomp_output, plot_sazonal_output, plot_consolidado_output, mint_output_file]
    
    arquivo_input.upload(processar_arquivo, [arquivo_input], [grupo_principal, reset_button, coluna_data_input, coluna_target_input, colunas_features_auto, colunas_fixas_auto, colunas_features_manual, colunas_features_state, colunas_fixas_state, valid_features_state])
    
    def atualizar_colunas_features(lista_colunas_total, data_selecionada):
        opcoes_validas = [col for col in lista_colunas_total if col != data_selecionada]
        return gr.update(choices=opcoes_validas), gr.update(choices=opcoes_validas), gr.update(choices=opcoes_validas), opcoes_validas

    coluna_data_input.change(atualizar_colunas_features, [colunas_features_state, coluna_data_input], [colunas_features_auto, colunas_features_manual, colunas_fixas_auto, valid_features_state])
    
    lag_ui_outputs_flat = []
    for config in lag_configs_ui:
        lag_ui_outputs_flat.extend([config['group'], config['name']])
    all_lag_checkboxes = [config['lags'] for config in lag_configs_ui]

    # DESTAQUE-MINT: Adicionado 'mint_output_file' à lista de componentes para resetar
    components_to_reset = [grupo_principal, reset_button, arquivo_input, coluna_data_input, coluna_target_input, colunas_features_auto, colunas_fixas_auto, temporal_features_auto, lags_auto, temporal_features_manual, colunas_features_manual, manual_select_group, paste_features_manual_textbox, manual_features_checklist, log_output, metricas_output, features_selecionadas_output, copy_row, features_copy_output, download_output, mint_output_file, plot_pred_output, dataframe_output, plot_imp_output, plot_shap_summary_output, plot_shap_force_output, plot_futuro_output, plot_futuro_vs_media_output, plot_real_vs_media_output, plot_decomp_output, plot_sazonal_output, plot_consolidado_output] + lag_ui_outputs_flat

    reset_button.click(reset_all, inputs=[], outputs=components_to_reset)
    colunas_features_manual.change(update_manual_lag_ui, [colunas_features_manual], lag_ui_outputs_flat)
    generate_features_button.click(gerar_features_para_selecao_manual, [arquivo_input, coluna_data_input, coluna_target_input, temporal_features_manual, colunas_features_manual] + all_lag_checkboxes, [manual_select_group, manual_features_checklist, features_manuais_state])
    apply_pasted_features_button.click(apply_pasted_features_to_checklist, inputs=[paste_features_manual_textbox], outputs=[manual_select_group, manual_features_checklist])
    paste_button_manual.click(lambda features_list: gr.update(value=features_list, choices=features_list), inputs=[auto_features_list_state], outputs=[manual_features_checklist])

    # --- Funções Auxiliares para Botões de Seleção ---
    def select_all_fn(choices): return gr.update(value=choices)
    def clear_fn(): return gr.update(value=[])

    # --- Conexões dos Botões de Seleção (Auto) ---
    select_all_temp_auto.click(lambda: select_all_fn(temporal_features_choices), None, temporal_features_auto)
    clear_temp_auto.click(clear_fn, None, temporal_features_auto)
    select_all_feat_auto.click(select_all_fn, [valid_features_state], colunas_features_auto)
    clear_feat_auto.click(clear_fn, None, colunas_features_auto)
    select_all_lags_auto.click(lambda: select_all_fn(list(range(1,13))), None, lags_auto)
    clear_lags_auto.click(clear_fn, None, lags_auto)
    select_all_fixas_auto.click(select_all_fn, [valid_features_state], colunas_fixas_auto)
    clear_fixas_auto.click(clear_fn, None, colunas_fixas_auto)

    # --- Conexões dos Botões de Seleção (Manual) ---
    select_all_temp_manual.click(lambda: select_all_fn(temporal_features_choices), None, temporal_features_manual)
    clear_temp_manual.click(clear_fn, None, temporal_features_manual)
    select_all_feat_manual.click(select_all_fn, [valid_features_state], colunas_features_manual)
    clear_feat_manual.click(clear_fn, None, colunas_features_manual)
    select_all_final_manual.click(select_all_fn, [features_manuais_state], manual_features_checklist)
    clear_final_manual.click(clear_fn, None, manual_features_checklist)
    
    run_button_manual.click(executar_pipeline_manual, inputs=[arquivo_input, coluna_data_input, coluna_target_input, tamanho_previsao_input, temporal_features_manual, manual_features_checklist, colunas_features_manual, n_estimators_input, learning_rate_input, max_depth_input, early_stopping_input] + all_lag_checkboxes, outputs=outputs_list_manual)
    run_button_auto.click(executar_pipeline_auto, inputs=[arquivo_input, coluna_data_input, coluna_target_input, colunas_features_auto, colunas_fixas_auto, temporal_features_auto, tamanho_previsao_input, lags_auto, n_estimators_input, learning_rate_input, max_depth_input, early_stopping_input], outputs=outputs_list_auto)

if __name__ == "__main__":
    demo.launch(debug=True)