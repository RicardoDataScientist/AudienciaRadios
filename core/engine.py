import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import warnings
from core.data_utils import adicionar_features_temporais, criar_features_de_lag

def treinar_modelo_quantilico(X_train, y_train, model_params, quantiles=[0.1, 0.5, 0.9]):
    """
    Treina 3 modelos XGBoost (Lower, Median, Upper) para Regressão Quantílica.
    """
    models = {}
    for q in quantiles:
        # Ajusta parâmetros para regressão quantílica
        params_q = model_params.copy()
        params_q['objective'] = 'reg:quantileerror'
        params_q['quantile_alpha'] = q
        # Remove early_stopping do kwargs do construtor se estiver lá, pois será passado no fit (ou ignorado aqui para simplificação)
        esr = params_q.pop('early_stopping_rounds', None) 
        
        model = xgb.XGBRegressor(**params_q)
        # Nota: reg:quantileerror é suportado em versões recentes do XGBoost.
        model.fit(X_train, y_train, verbose=False)
        models[q] = model
    return models

def treinar_e_avaliar_cv(df_limpo, features_finais, target, min_months_backtest, horizon, prediction_mode, model_params, progress):
    """
    Treina o modelo final usando Walk-Forward Validation dinâmico.
    n_splits é calculado para cobrir 'min_months_backtest' com passos de 'horizon'.
    """
    n_samples = len(df_limpo)
    
    # 1. Validação de Segurança: Garante pelo menos 12 meses de treino inicial
    min_train_samples = 12
    max_possible_backtest = n_samples - min_train_samples
    
    effective_backtest = min_months_backtest
    log_text = f"\n4. 🚀 Treinando modelo final com Walk-Forward Validation (Mode: {prediction_mode})...\n"

    if max_possible_backtest < min_months_backtest:
        effective_backtest = max(0, max_possible_backtest)
        log_text += f"⚠️ Aviso Adaptativo: Histórico curto ({n_samples} meses). Reduzindo janela de validação de {min_months_backtest} para {effective_backtest} meses para garantir 12 meses de treino inicial.\n"
    
    # 2. Cálculo de Splits (Walk-Forward)
    # Fórmula: Quantas vezes o horizonte cabe na janela de backtest?
    if horizon <= 0: horizon = 1 # Segurança
    
    n_splits = effective_backtest // horizon
    
    # Garante no mínimo 2 splits para ser um CV válido, a menos que não haja dados suficientes
    if n_splits < 2:
        log_text += f"⚠️ Aviso: Janela de teste ({effective_backtest} meses) curta para horizonte de {horizon} meses. Forçando 2 splits (pode reduzir treino inicial).\n"
        n_splits = 2

    # Configura TimeSeriesSplit
    # test_size = horizon (simula exatamente o cenário real de previsão)
    try:
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=horizon)
        log_text += f"📊 Configuração Walk-Forward: {n_splits} folds, Test Size = {horizon} meses (Horizonte), Total Backtest ≈ {n_splits * horizon} meses.\n"
    except ValueError as e:
        return f"❌ Erro Crítico no CV: {e}. Dados insuficientes para este horizonte.", pd.DataFrame(), None, pd.DataFrame(), None, None, None, None, None, 0.0

    all_preds_med, all_preds_low, all_preds_up, all_true = [], [], [], []
    
    X, y = df_limpo[features_finais], df_limpo[target]

    # Modelo final para exportação
    params_final = model_params.copy()
    params_final.pop('early_stopping_rounds', None)
    
    # TREINO DO MODELO FINAL (FULL DATA)
    if prediction_mode == "quantile":
        # Para quantile, o modelo principal é a mediana (0.5)
        params_median = params_final.copy()
        params_median['objective'] = 'reg:quantileerror'
        params_median['quantile_alpha'] = 0.5
        modelo_final = xgb.XGBRegressor(**params_median).fit(X, y)
    else:
        # Para normal, usamos erro quadrático
        params_normal = params_final.copy()
        params_normal['objective'] = 'reg:squarederror'
        modelo_final = xgb.XGBRegressor(**params_normal).fit(X, y)

    # LOOP DE VALIDAÇÃO CRUZADA
    for fold, (train_index, test_index) in enumerate(tscv.split(X)):
        progress(0.7 + (0.2 * (fold / tscv.get_n_splits(df_limpo))), desc=f"Processando Fold {fold+1}/{n_splits}...")
        
        # Log detalhado dos índices para conferência
        log_text += f"    -> Fold {fold+1}: Treino (0 a {train_index[-1]}) | Teste ({test_index[0]} a {test_index[-1]})\n"
        
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        if prediction_mode == "quantile":
            # Treina os 3 quantis
            models_fold = treinar_modelo_quantilico(X_train, y_train, params_final)
            y_pred_low = models_fold[0.1].predict(X_test)
            y_pred_med = models_fold[0.5].predict(X_test)
            y_pred_up = models_fold[0.9].predict(X_test)
        else:
            # Treina normal
            params_fold = params_final.copy()
            params_fold['objective'] = 'reg:squarederror'
            model_fold = xgb.XGBRegressor(**params_fold)
            model_fold.fit(X_train, y_train, verbose=False)
            
            y_pred_med = model_fold.predict(X_test)
            # No modo normal, calcularemos lower/upper DEPOIS com base no MAPE global do CV
            # Por enquanto, preenchemos com a própria previsão para manter o shape
            y_pred_low = y_pred_med 
            y_pred_up = y_pred_med

        all_preds_low.append(pd.Series(y_pred_low, index=y_test.index))
        all_preds_med.append(pd.Series(y_pred_med, index=y_test.index))
        all_preds_up.append(pd.Series(y_pred_up, index=y_test.index))
        all_true.append(y_test)

    # Consolida resultados
    y_true_concat = pd.concat(all_true)
    y_pred_med_concat = pd.concat(all_preds_med)
    y_pred_low_concat = pd.concat(all_preds_low)
    y_pred_up_concat = pd.concat(all_preds_up)

    df_resultados = pd.DataFrame({
        'Real': y_true_concat,
        'y_lower': y_pred_low_concat,
        'Previsto': y_pred_med_concat,
        'y_upper': y_pred_up_concat
    })

    # CÁLCULO DO MAPE GLOBAL E AJUSTE SE MODO NORMAL
    mape_global = mean_absolute_percentage_error(df_resultados['Real'], df_resultados['Previsto'])
    
    if prediction_mode == "normal":
        log_text += f"ℹ️ Modo Normal: Aplicando MAPE do CV ({mape_global:.2%}) como margem de erro.\n"
        # Recalcula bounds baseado no MAPE
        df_resultados['y_lower'] = df_resultados['Previsto'] * (1 - mape_global)
        df_resultados['y_upper'] = df_resultados['Previsto'] * (1 + mape_global)

    df_resultados = df_resultados.round(2)

    ## Colunas de Variação ##
    df_resultados['Variacao_Absoluta'] = (df_resultados['Previsto'] - df_resultados['Real']).round(2)
    df_resultados['Variacao_Percentual_%'] = np.where(
        df_resultados['Real'] == 0, 
        np.nan, 
        ((df_resultados['Previsto'] - df_resultados['Real']) / df_resultados['Real'])
    )
    df_resultados['Variacao_Percentual_%'] = df_resultados['Variacao_Percentual_%'].round(4)

    y_test_final, y_pred_final = df_resultados['Real'], df_resultados['Previsto']
    metricas = {
        'Métrica': ['MAPE', 'MAE', 'RMSE'],
        'Valor': [
            mape_global, # Já calculado
            mean_absolute_error(y_test_final, y_pred_final),
            np.sqrt(mean_squared_error(y_test_final, y_pred_final))
        ]
    }
    df_metricas = pd.DataFrame(metricas).round(4)
    
    progress(0.95, desc="Gerando resultados e gráficos...")
    plt.close('all')

    # Gráfico de Previsão
    fig_pred, ax_pred = plt.subplots(figsize=(16, 3))
    ax_pred.fill_between(df_resultados.index, df_resultados['y_lower'], df_resultados['y_upper'], color='#ff0051', alpha=0.15, label='Margem (Quantil ou MAPE)')
    sns.lineplot(data=df_resultados[['Real', 'Previsto']], ax=ax_pred, palette={'Real': '#0072B2', 'Previsto': '#ff0051'})
    ax_pred.set(title=f'Backtest: Real vs. Previsto ({prediction_mode.upper()}) para {target.upper()}', xlabel='Data', ylabel="Audiência")
    ax_pred.legend(title='Série', bbox_to_anchor=(1.02, 1), loc='upper left')
    fig_pred.tight_layout()
    
    # Gráfico de Importância
    importances = pd.Series(modelo_final.feature_importances_, index=features_finais).sort_values(ascending=False).head(20)
    fig_imp, ax_imp = plt.subplots(figsize=(10, 8))
    sns.barplot(x=importances.values, y=importances.index, ax=ax_imp, palette="viridis_r")
    ax_imp.set_title(f'Importância das Features ({target.upper()})', fontsize=16)
    fig_imp.tight_layout()
    
    # SHAP
    shap_values, fig_shap_summary, fig_shap_force = None, None, None
    try:
        explainer = shap.TreeExplainer(modelo_final)
        shap_values = explainer(X)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            plt.figure(); shap.summary_plot(shap_values, X, show=False, max_display=20); fig_shap_summary = plt.gcf(); fig_shap_summary.suptitle(f'Impacto SHAP', fontsize=16); fig_shap_summary.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.figure(); shap.force_plot(explainer.expected_value, shap_values.values[0,:], X.iloc[0,:], matplotlib=True, show=False); fig_shap_force = plt.gcf(); fig_shap_force.suptitle(f'SHAP Force (1ª Previsão)', fontsize=16); fig_shap_force.tight_layout(rect=[0, 0.03, 1, 0.95])
    except Exception as e:
        log_text += f"\n⚠️ Aviso: Não foi possível gerar os gráficos SHAP. Erro: {e}\n"
    
    # Retornamos também o MAPE GLOBAL para usar na previsão futura se for modo normal
    return log_text, df_resultados, fig_pred, df_metricas, fig_imp, fig_shap_summary, fig_shap_force, modelo_final, shap_values, mape_global


def prever_futuro(modelo_base_dummy, df_historico_base, features_finais, target, horizonte, config_geracao_lags, config_temporal, model_params, strategy='recursive', prediction_mode='quantile', mape_margin=0.0, log_text=""):
    """
    Prevê o futuro.
    Se prediction_mode='normal', usa mape_margin para gerar y_lower e y_upper.
    """
    log_text += f"\n5. 🔮 Iniciando previsão futura ({horizonte} meses). Mode: {prediction_mode}, Strategy: {strategy}...\n"
    
    df_com_previsoes = df_historico_base.copy()
    ultima_data = df_com_previsoes.index.max()
    datas_futuras = pd.date_range(start=ultima_data + pd.DateOffset(months=1), periods=horizonte, freq='MS')
    
    colunas_para_lag_base = {cfg[0] for cfg in config_geracao_lags}
    cols_exogenas = [c for c in colunas_para_lag_base if c != target]
    
    df_resultado_futuro = pd.DataFrame(index=datas_futuras, columns=[target, 'y_lower', 'y_upper'])
    all_features_para_prever = []
    
    params_clean = model_params.copy()
    params_clean.pop('early_stopping_rounds', None)

    # Função auxiliar para prever um passo (retorna low, med, up)
    def predict_step(model_or_models, X_step):
        if prediction_mode == 'quantile':
            low = model_or_models[0.1].predict(X_step)[0]
            med = model_or_models[0.5].predict(X_step)[0]
            up = model_or_models[0.9].predict(X_step)[0]
            return low, med, up
        else:
            # Modo Normal
            med = model_or_models.predict(X_step)[0]
            low = med * (1 - mape_margin)
            up = med * (1 + mape_margin)
            return low, med, up

    if strategy == 'recursive':
        # --- RECURSIVA ---
        log_text += "    -> Treinando modelo(s) base com todo histórico...\n"
        
        df_treino_full = df_historico_base.copy()
        df_treino_full, _, _ = adicionar_features_temporais(df_treino_full, config_temporal, target)
        df_treino_full, _ = criar_features_de_lag(df_treino_full, config_geracao_lags)
        df_treino_full = df_treino_full.dropna(subset=features_finais + [target])
        
        X_full = df_treino_full[features_finais]
        y_full = df_treino_full[target]
        
        if prediction_mode == 'quantile':
            models_recursive = treinar_modelo_quantilico(X_full, y_full, params_clean)
        else:
            params_n = params_clean.copy()
            params_n['objective'] = 'reg:squarederror'
            models_recursive = xgb.XGBRegressor(**params_n).fit(X_full, y_full)
        
        df_temp_loop = df_historico_base.copy()
        
        for data_previsao in datas_futuras:
            df_temp_loop.loc[data_previsao] = np.nan
            if cols_exogenas:
                 df_temp_loop[cols_exogenas] = df_temp_loop[cols_exogenas].ffill()

            df_temp_feats, _, _ = adicionar_features_temporais(df_temp_loop, config_temporal, target)
            df_temp_feats, _ = criar_features_de_lag(df_temp_feats, config_geracao_lags)
            
            features_step = df_temp_feats.loc[[data_previsao], features_finais]
            all_features_para_prever.append(features_step)
            
            pred_low, pred_med, pred_up = predict_step(models_recursive, features_step)
            
            df_resultado_futuro.loc[data_previsao, 'y_lower'] = pred_low
            df_resultado_futuro.loc[data_previsao, target] = pred_med
            df_resultado_futuro.loc[data_previsao, 'y_upper'] = pred_up
            
            # Atualiza target para recursão
            df_temp_loop.loc[data_previsao, target] = pred_med

    elif strategy == 'direct':
        # --- DIRECT ---
        log_text += "    -> Treinando modelos independentes por horizonte...\n"
        
        df_base_features = df_historico_base.copy()
        df_base_features, _, _ = adicionar_features_temporais(df_base_features, config_temporal, target)
        df_base_features, _ = criar_features_de_lag(df_base_features, config_geracao_lags)
        
        for i, data_previsao in enumerate(datas_futuras):
            horizonte_h = i + 1
            df_direct = df_base_features.copy()
            col_target_h = f"target_h{horizonte_h}"
            df_direct[col_target_h] = df_direct[target].shift(-horizonte_h)
            
            df_train_h = df_direct.dropna(subset=[col_target_h] + features_finais)
            X_train_h = df_train_h[features_finais]
            y_train_h = df_train_h[col_target_h]
            
            if prediction_mode == 'quantile':
                models_h = treinar_modelo_quantilico(X_train_h, y_train_h, params_clean)
            else:
                params_n = params_clean.copy()
                params_n['objective'] = 'reg:squarederror'
                models_h = xgb.XGBRegressor(**params_n).fit(X_train_h, y_train_h)
            
            features_presente = df_base_features.iloc[[-1]][features_finais]
            
            # Fake visualization entry
            features_presente_viz = features_presente.copy()
            features_presente_viz.index = [data_previsao]
            all_features_para_prever.append(features_presente_viz)
            
            pred_low, pred_med, pred_up = predict_step(models_h, features_presente)
            
            df_resultado_futuro.loc[data_previsao, 'y_lower'] = pred_low
            df_resultado_futuro.loc[data_previsao, target] = pred_med
            df_resultado_futuro.loc[data_previsao, 'y_upper'] = pred_up

    log_text += "--- ✅ Previsão futura concluída ---\n"
    
    if all_features_para_prever:
        df_features_futuras = pd.concat(all_features_para_prever)
    else:
        df_features_futuras = pd.DataFrame()
    
    return df_resultado_futuro, log_text, df_features_futuras
