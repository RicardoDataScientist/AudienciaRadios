import gradio as gr
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_percentage_error
import warnings
import traceback
import os

from core.data_utils import sanitize_columns, adicionar_features_temporais, criar_features_de_lag, criar_features_media_movel
from core.engine import treinar_e_avaliar_cv, prever_futuro
from utils.viz import gerar_grafico_decomposicao, gerar_grafico_sazonalidade_anual, gerar_graficos_previsao_futura, gerar_grafico_consolidado
from utils.export import salvar_resultados_zip, gerar_arquivo_mint

# --- Configurações Iniciais ---
warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
warnings.simplefilter(action="ignore", category=FutureWarning)

# --- Carregar CSS ---
with open("styles.css", "r") as f:
    custom_css = f.read()

# --- FUNÇÕES DE PIPELINE ---

def executar_pipeline_auto(arquivo, coluna_data_orig, target_orig, colunas_features_orig, colunas_fixas_orig, temporal_features_orig, tamanho_previsao_meses, backtest_size_meses, lags, ma_windows, n_estimators, learning_rate, max_depth, early_stopping_rounds, forecast_strategy, prediction_mode, progress=gr.Progress(track_tqdm=True)):
    if not coluna_data_orig or not target_orig:
        raise gr.Error("Por favor, selecione a 'Coluna de data' e a 'Variável TARGET' antes de executar!")
    
    log_text = ""
    try:
        progress(0, desc="Carregando dados...")
        log_text += "1. Carregando e preparando dados...\n"
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
        
        # Para seleção de features, usamos o horizonte de previsão como referência de split, 
        # mas o usuário pode querer backtest maior depois.
        # Vamos manter o tamanho_previsao para a seleção ser focada no curto prazo se for o caso.
        data_final_treino_dt = df.index.max() - pd.DateOffset(months=int(tamanho_previsao_meses))
        datas_split = {'treino_fim': data_final_treino_dt}
        log_text += f"Divisão (Seleção): Treino até {datas_split['treino_fim'].date()}\n"
        
        progress(0.1, desc="Criando Features de Lag e Médias Móveis...")
        config_geracao = [(col, f'_{col}', lags) for col in colunas_features]
        df_features, log_criacao = criar_features_de_lag(df, config_geracao)
        log_text += log_criacao

        config_ma = [(col, f'_{col}', ma_windows) for col in colunas_features]
        df_features, log_ma = criar_features_media_movel(df_features, config_ma)
        log_text += log_ma
        
        
        log_text += f"ℹ️ DataFrame com features geradas (antes do dropna) tem {len(df_features)} linhas.\n"
        
        progress(0.3, desc="Selecionando Features com CV...")
        treino_completo_com_nans = df_features[df_features.index <= datas_split['treino_fim']]
        log_text += "\n🛡️ Usando Validação Cruzada (TimeSeriesSplit) para seleção de features robusta...\n"
        
        tscv_selection = TimeSeriesSplit(n_splits=3, test_size=int(tamanho_previsao_meses))
        log_text += f"📊 CV para seleção configurado com {tscv_selection.get_n_splits(treino_completo_com_nans)} folds.\n"

        params = {'n_estimators': int(n_estimators), 'learning_rate': learning_rate, 'max_depth': int(max_depth),
                   'random_state': 42, 'early_stopping_rounds': int(early_stopping_rounds), 'objective': 'reg:squarederror'}

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

        features_candidatas = [c for c in df_features.columns if c.startswith('lag_') or c.startswith('ma_')]
        features_atuais = colunas_fixas.copy()
        mape_atual = calcular_mape_cv(features_atuais)
        
        log_text += f"\n--- 🤖 Iniciando a seleção com REFINAMENTO IMEDIATO e CV ---\n"
        log_text += f"MAPE de Referência (CV na base de treino): {mape_atual:.5f}\n"
        
        iteration = 0
        while True:
            iteration += 1
            log_text += f"\n==================== ITERAÇÃO DE SELEÇÃO {iteration} ====================\n"
            log_text += f"MAPE CV atual: {mape_atual:.5f}\n"
            yield log_text, None, None, None, None, None, None, gr.update(value=None, visible=False), "### Seleção de Features\n*Aguardando início do processo...*", gr.update(visible=False), "", features_atuais, None, None, None, None, None, None, gr.update(value=None, visible=False)
            
            if not features_candidatas:
                log_text += "Não há mais features candidatas para testar. Finalizando seleção.\n"
                break

            log_text += "\n--- Fase 1: Buscando a melhor feature para ADICIONAR ---\n"
            yield log_text, None, None, None, None, None, None, gr.update(value=None, visible=False), "### Seleção de Features\n*Aguardando início do processo...*", gr.update(visible=False), "", features_atuais, None, None, None, None, None, None, gr.update(value=None, visible=False)

            resultados_adicao = {}
            for feature in features_candidatas:
                mape_teste = calcular_mape_cv(features_atuais + [feature])
                resultados_adicao[feature] = mape_teste
                log_text += f"    - Testando adicionar '{feature}': MAPE CV = {mape_teste:.5f}\n"
                yield log_text, None, None, None, None, None, None, gr.update(value=None, visible=False), "### Seleção de Features\n*Aguardando início do processo...*", gr.update(visible=False), "", features_atuais, None, None, None, None, None, None, gr.update(value=None, visible=False)
            
            melhor_nova_feature = min(resultados_adicao, key=resultados_adicao.get)
            melhor_mape_adicao = resultados_adicao[melhor_nova_feature]

            if melhor_mape_adicao >= mape_atual:
                log_text += f"\n❌ Nenhuma nova feature melhorou o MAPE CV. Finalizando.\n"
                break

            log_text += f"✅ Melhoria encontrada! Adicionando '{melhor_nova_feature}' (MAPE CV cai para {melhor_mape_adicao:.5f}).\n"
            yield log_text, None, None, None, None, None, None, gr.update(value=None, visible=False), "### Seleção de Features\n*Aguardando início do processo...*", gr.update(visible=False), "", features_atuais + [melhor_nova_feature], None, None, None, None, None, None, gr.update(value=None, visible=False)
            features_para_refinar = features_atuais + [melhor_nova_feature]
            mape_para_refinar = melhor_mape_adicao
            
            log_text += "\n--- Fase 2: Iniciando REFINAMENTO (tentando remover features antigas) ---\n"
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
        yield log_text, None, None, None, None, None, None, gr.update(value=None, visible=False), "### Seleção de Features\n*Aguardando início do processo...*", gr.update(visible=False), "", features_atuais, None, None, None, None, None, None, gr.update(value=None, visible=False)
        
        features_finais = features_atuais
        features_md_final = format_features_md(features_finais)
        features_copy_str = ", ".join(features_finais)

        log_text += "\n⚙️ Preparando dataset final com as features selecionadas...\n"
        colunas_finais_treino = features_finais + [target]
        df_limpo = df_features[colunas_finais_treino].dropna()
        log_text += f"ℹ️ Dataset final (após dropna com features selecionadas) tem {len(df_limpo)} linhas.\n"
        log_text += "\n\n--- 🚀 Preparando para o treino final com as melhores features! Aguarde... ---\n"
        
        # TREINO FINAL COM OS NOVOS PARÂMETROS DE BACKTEST E MODE
        # Agora passamos tamanho_previsao_meses como 'horizon' para o CV
        log_treino_final, df_res, fig_pred, df_met, fig_imp, fig_shap_sum, fig_shap_force, modelo_final, shap_values_hist, mape_final = treinar_e_avaliar_cv(df_limpo, features_finais, target, int(backtest_size_meses), int(tamanho_previsao_meses), prediction_mode, params, progress)
        log_text += log_treino_final

        config_geracao_futuro = [(col, f'_{col}', lags) for col in colunas_features]
        
        # Previsão Futura
        config_ma_futuro = [(col, f'_{col}', ma_windows) for col in colunas_features]
        df_previsoes_futuras, log_futuro, df_features_futuras = prever_futuro(modelo_final, df_base_para_futuro, features_finais, target, int(tamanho_previsao_meses), config_geracao_futuro, config_ma_futuro, temporal_features_orig, params, strategy=forecast_strategy, prediction_mode=prediction_mode, mape_margin=mape_final)
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
        
        zip_file = salvar_resultados_zip(target, df_met, features_finais, df_res, df_previsoes_futuras, figs_para_salvar, df_base_para_futuro)
        
        mint_file = gr.update(value=None, visible=False)
        if shap_values_hist is not None:
             mint_file = gerar_arquivo_mint(target, df_limpo, df_res, shap_values_hist, features_finais, df_previsoes_futuras, df_features_futuras, modelo_final)
        else:
            log_text += "\n⚠️ Não foi possível gerar o arquivo MinT pois os valores SHAP não foram calculados."

        yield log_text, df_res, fig_pred, df_met, fig_imp, fig_shap_sum, fig_shap_force, zip_file, features_md_final, gr.update(visible=True), features_copy_str, features_finais, fig_fut, fig_fut_media, fig_real_media, fig_decomp, fig_sazonal, fig_consolidado, mint_file

    except Exception as e:
        log_text += f"\n❌ Ocorreu um erro: {e}\n\n--- Detalhes ---\n{traceback.format_exc()}"
        yield log_text, None, None, None, None, None, None, gr.update(value=None, visible=False), "### Erro\n*Ocorreu um erro durante a execução.*", gr.update(visible=False), "", [], None, None, None, None, None, None, gr.update(value=None, visible=False)

def executar_pipeline_manual(arquivo, coluna_data_orig, target_orig, tamanho_previsao_meses, backtest_size_meses, temporal_features_orig, features_finais_selecionadas, colunas_configuradas_orig, ma_windows, n_estimators, learning_rate, max_depth, early_stopping_rounds, forecast_strategy, prediction_mode, *lista_de_lags, progress=gr.Progress(track_tqdm=True)):
    if not coluna_data_orig or not target_orig:
        raise gr.Error("Por favor, selecione a 'Coluna de data' e a 'Variável TARGET' antes de executar!")
    if not features_finais_selecionadas:
        raise gr.Error("Nenhuma feature foi selecionada para o treino manual. Selecione ou cole as features na Etapa 4.")

    log_text = ""
    blank_outputs_for_yield = [None] * 16 
    try:
        progress(0, desc="Carregando dados...")
        log_text = "1. Carregando e preparando dados para treino manual...\n"
        
        initial_outputs = list(blank_outputs_for_yield)
        initial_outputs[8] = "### Features Selecionadas (Manual)\n*Aguardando execução...*"
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
        # Agora passamos tamanho_previsao_meses como 'horizon' para o CV
        log_treino_final, df_res, fig_pred, df_met, fig_imp, fig_shap_sum, fig_shap_force, modelo_final, shap_values_hist, mape_final = treinar_e_avaliar_cv(df_limpo, features_finais_existentes, target, int(backtest_size_meses), int(tamanho_previsao_meses), prediction_mode, params, progress)
        log_text += log_treino_final

        df_previsoes_futuras, log_futuro, df_features_futuras = prever_futuro(modelo_final, df_base_para_futuro, features_finais_existentes, target, int(tamanho_previsao_meses), config_geracao, config_ma, temporal_features_orig, params, strategy=forecast_strategy, prediction_mode=prediction_mode, mape_margin=mape_final)
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
        
        zip_file = salvar_resultados_zip(target, df_met, features_finais_existentes, df_res, df_previsoes_futuras, figs_para_salvar, df_base_para_futuro)
        
        mint_file = gr.update(value=None, visible=False)
        if shap_values_hist is not None:
             mint_file = gerar_arquivo_mint(target, df_limpo, df_res, shap_values_hist, features_finais_existentes, df_previsoes_futuras, df_features_futuras, modelo_final)
        else:
            log_text += "\n⚠️ Não foi possível gerar o arquivo MinT pois os valores SHAP não foram calculados."

        yield log_text, df_res, fig_pred, df_met, fig_imp, fig_shap_sum, fig_shap_force, zip_file, features_selecionadas_md, fig_fut, fig_fut_media, fig_real_media, fig_decomp, fig_sazonal, fig_consolidado, mint_file
        
    except Exception as e:
        log_text += f"\n❌ Ocorreu um erro: {e}\n\n--- Detalhes ---\n{traceback.format_exc()}"
        error_outputs = list(blank_outputs_for_yield)
        error_outputs[8] = "### Erro\n*Ocorreu um erro durante a execução.*"
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

def gerar_features_para_selecao_manual(arquivo, coluna_data_orig, target_orig, temporal_features_orig, colunas_configuradas_orig, ma_windows, *lista_de_lags):
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
    
    config_ma = [(col, f'_{col}', ma_windows) for col in colunas_configuradas]
    df_features, _ = criar_features_media_movel(df_features, config_ma)

    features_de_lag = [c for c in df_features.columns if c.startswith('lag_')]
    features_ma = [c for c in df_features.columns if c.startswith('ma_')]
    features_disponiveis = sorted(list(set(created_temporal_features + features_de_lag + features_ma)))
    
    return gr.update(), gr.update(choices=features_disponiveis, value=features_disponiveis), features_disponiveis

def apply_pasted_features_to_checklist(pasted_text):
    if not pasted_text or not isinstance(pasted_text, str):
        return gr.update(), gr.update(value=[], choices=[])
    features = [f.strip() for f in pasted_text.split(',') if f.strip()]
    return gr.update(), gr.update(choices=features, value=features)

def reset_all():
    lag_ui_updates = update_manual_lag_ui([])
    return [
        gr.update(), gr.update(), gr.update(), gr.update(value=None), gr.update(value=None), gr.update(value=[]), gr.update(value=[]),
        gr.update(value=["Mês", "Features de Fourier (Anual)", "Média Mensal Histórica (Target)"]), gr.update(value=list(range(1, 13))), gr.update(value=[3, 6, 9]),
        gr.update(value=["Mês", "Features de Fourier (Anual)", "Média Mensal Histórica (Target)"]), gr.update(value=[]), gr.update(value=[3, 6, 9]),
        gr.update(visible=True), gr.update(value=""), gr.update(value=[], choices=[]),
        "", None, "", gr.update(visible=False), "", gr.update(value=None, visible=False), gr.update(value=None, visible=False, label="Download Dados MinT (.xlsx)"),
        None, None, None, None, None, None, None, None, None, None, None,
    ] + lag_ui_updates

# --- CONSTRUÇÃO DA INTERFACE ---
with gr.Blocks(theme=gr.themes.Soft(), title="AutoML de Séries Temporais Pro", css=custom_css) as demo:
    gr.Markdown("# 🤖 AutoML para Séries Temporais com Feature Selection Pro v5.1 (Quantiles/Normal + Backtest Config)")
    gr.Markdown("Agora com opção de Regressão Normal (MAPE Margin) e controle independente do Backtest!")
    
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
                # NOVO CAMPO: Tamanho do Backtest
                backtest_size_input = gr.Number(label="Tamanho do Backtest (meses)", value=18)
                coluna_target_input = gr.Dropdown(label="Variável TARGET (a prever)")

        with gr.Accordion("Parâmetros do Modelo & Estratégia", open=True):
            with gr.Row():
                # NOVO CAMPO: Modo de Previsão
                prediction_mode_input = gr.Radio(choices=["quantile", "normal"], value="quantile", label="Modo de Previsão (Margem de Erro)", info="Quantile: Usa modelos otimista/pessimista (regressão quantílica). Normal: Usa modelo único e aplica o MAPE do backtest como margem.")
                forecast_strategy_input = gr.Radio(choices=["recursive", "direct"], value="recursive", label="Estratégia de Previsão", info="Recursive: Mais rápido, realimenta erros. Direct: Mais robusto para longo prazo.")
            with gr.Group():
                with gr.Row():
                    n_estimators_input = gr.Slider(label="Número de Árvores", minimum=50, maximum=1000, value=1000, step=50)
                    learning_rate_input = gr.Slider(label="Taxa de Aprendizado", minimum=0.01, maximum=0.3, value=0.1, step=0.01)
                    max_depth_input = gr.Slider(label="Profundidade Máxima", minimum=3, maximum=10, value=6, step=1)
                    early_stopping_input = gr.Number(label="Parada Antecipada", value=20)

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
                    gr.Markdown("### 4. Médias Móveis (meses)")
                    ma_auto = gr.CheckboxGroup(choices=[3, 6, 9, 12], value=[3, 6, 9])
                    with gr.Row():
                        select_all_ma_auto = gr.Button("Selecionar Todos", scale=1, elem_classes=["orange-button"])
                        clear_ma_auto = gr.Button("Limpar", scale=1, elem_classes=["orange-button"])

                with gr.Group(elem_classes=["dark-checkbox-group"]):
                    gr.Markdown("### 5. Features FIXAS (opcional)")
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
                    gr.Markdown("### Passo 3: Configure os lags para cada coluna (opcional) e Médias Móveis")
                    
                    with gr.Group(elem_classes=["dark-checkbox-group"]):
                        gr.Markdown("#### Médias Móveis (para todas as colunas selecionadas acima)")
                        ma_manual = gr.CheckboxGroup(choices=[3, 6, 9, 12], value=[3, 6, 9], label="Janelas de Média Móvel")

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
                metricas_output = gr.DataFrame(label="Métricas de Performance (CV - Mediana)", headers=['Métrica', 'Valor'])
                with gr.Column(scale=2):
                    features_selecionadas_output = gr.Markdown(label="Seleção de Features")
                    with gr.Row(visible=False) as copy_row:
                        features_copy_output = gr.Textbox(label="Lista de Features (para copiar)", show_label=False, interactive=False, show_copy_button=True, placeholder="A lista de features aparecerá aqui para cópia...")
                    download_output = gr.File(label="Download dos Gráficos e Resultados (.zip)", visible=False)
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

    components_to_reset = [grupo_principal, reset_button, arquivo_input, coluna_data_input, coluna_target_input, colunas_features_auto, colunas_fixas_auto, temporal_features_auto, lags_auto, ma_auto, temporal_features_manual, colunas_features_manual, ma_manual, manual_select_group, paste_features_manual_textbox, manual_features_checklist, log_output, metricas_output, features_selecionadas_output, copy_row, features_copy_output, download_output, mint_output_file, plot_pred_output, dataframe_output, plot_imp_output, plot_shap_summary_output, plot_shap_force_output, plot_futuro_output, plot_futuro_vs_media_output, plot_real_vs_media_output, plot_decomp_output, plot_sazonal_output, plot_consolidado_output] + lag_ui_outputs_flat

    reset_button.click(reset_all, inputs=[], outputs=components_to_reset)
    colunas_features_manual.change(update_manual_lag_ui, [colunas_features_manual], lag_ui_outputs_flat)
    generate_features_button.click(gerar_features_para_selecao_manual, [arquivo_input, coluna_data_input, coluna_target_input, temporal_features_manual, colunas_features_manual, ma_manual] + all_lag_checkboxes, [manual_select_group, manual_features_checklist, features_manuais_state])
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
    select_all_ma_auto.click(lambda: select_all_fn([3, 6, 9, 12]), None, ma_auto)
    clear_ma_auto.click(clear_fn, None, ma_auto)
    select_all_fixas_auto.click(select_all_fn, [valid_features_state], colunas_fixas_auto)
    clear_fixas_auto.click(clear_fn, None, colunas_fixas_auto)

    # --- Conexões dos Botões de Seleção (Manual) ---
    select_all_temp_manual.click(lambda: select_all_fn(temporal_features_choices), None, temporal_features_manual)
    clear_temp_manual.click(clear_fn, None, temporal_features_manual)
    select_all_feat_manual.click(select_all_fn, [valid_features_state], colunas_features_manual)
    clear_feat_manual.click(clear_fn, None, colunas_features_manual)
    select_all_final_manual.click(select_all_fn, [features_manuais_state], manual_features_checklist)
    clear_final_manual.click(clear_fn, None, manual_features_checklist)
    
    run_button_manual.click(executar_pipeline_manual, inputs=[arquivo_input, coluna_data_input, coluna_target_input, tamanho_previsao_input, backtest_size_input, temporal_features_manual, manual_features_checklist, colunas_features_manual, ma_manual, n_estimators_input, learning_rate_input, max_depth_input, early_stopping_input, forecast_strategy_input, prediction_mode_input] + all_lag_checkboxes, outputs=outputs_list_manual)
    run_button_auto.click(executar_pipeline_auto, inputs=[arquivo_input, coluna_data_input, coluna_target_input, colunas_features_auto, colunas_fixas_auto, temporal_features_auto, tamanho_previsao_input, backtest_size_input, lags_auto, ma_auto, n_estimators_input, learning_rate_input, max_depth_input, early_stopping_input, forecast_strategy_input, prediction_mode_input], outputs=outputs_list_auto)

if __name__ == "__main__":
    demo.launch(debug=True)
