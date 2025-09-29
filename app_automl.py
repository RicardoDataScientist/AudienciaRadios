import warnings
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, r2_score, mean_squared_error
import traceback
import gradio as gr
import matplotlib.pyplot as plt
import shap
import io
import zipfile
import tempfile

# --- Configurações Iniciais ---
warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
warnings.simplefilter(action="ignore", category=FutureWarning)
plt.style.use('seaborn-v0_8-whitegrid')


# --- FUNÇÕES CORE (LÓGICA DO MODELO) ---

def criar_features_temporais(df, config_geracao):
    """
    Cria features temporais (apenas lags) de forma configurável para evitar data leakage.
    """
    df_features = df.copy()
    log_text = "⚙️ Iniciando criação de features temporais (somente lags)...\n"
    for coluna_base, sufixo, lags in config_geracao:
        if coluna_base not in df_features.columns:
            msg = f" ⚠️  Aviso: Coluna '{coluna_base}' não encontrada. Pulando.\n"
            log_text += msg
            continue

        log_text += f"  -> Processando: '{coluna_base}'\n"
        for lag in lags:
            feature_name = f'lag{sufixo}_{lag}_meses'
            df_features[feature_name] = df_features[coluna_base].shift(lag)
    
    log_text += "--- ✅  Criação de features concluída ---\n"
    return df_features, log_text


def selecionar_features_com_refinamento(df, features_candidatas, features_fixas, target, datas_split, model_params, progress):
    """
    Realiza a seleção de features com uma lógica de refinamento imediato.
    """
    log_text = "\n🤖 Iniciando a seleção de features com REFINAMENTO IMEDIATO...\n"
    treino = df[(df.index >= datas_split['treino_inicio']) & (df.index <= datas_split['treino_fim'])]
    teste = df[(df.index >= datas_split['teste_inicio']) & (df.index <= datas_split['teste_fim'])]
    y_train = treino[target]
    y_test = teste[target]

    def calcular_mape(features):
        if not features: return float('inf')
        try:
            X_train, X_test = treino[features], teste[features]
            model = xgb.XGBRegressor(**model_params)
            model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
            return mean_absolute_percentage_error(y_test, model.predict(X_test))
        except Exception:
            return float('inf')

    features_atuais = features_fixas.copy() if features_fixas else []
    mape_atual = calcular_mape(features_atuais)
    candidatas_restantes = features_candidatas.copy()
    log_text += f"MAPE de Referência (features fixas, se houver): {mape_atual:.5f}\n"
    
    iteration = 0
    while True:
        iteration += 1
        log_text += f"\n{'='*10} ITERAÇÃO {iteration} {'='*10}\nMAPE atual: {mape_atual:.5f}\n"
        progress(iteration / (iteration + len(candidatas_restantes) + 1), desc=f"Iteração {iteration} | MAPE: {mape_atual:.5f}")

        if not candidatas_restantes:
            log_text += "Não há mais features candidatas para testar. Finalizando.\n"
            break

        resultados_adicao = {f: calcular_mape(features_atuais + [f]) for f in candidatas_restantes}
        melhor_nova_feature = min(resultados_adicao, key=resultados_adicao.get)
        melhor_mape_adicao = resultados_adicao[melhor_nova_feature]

        if melhor_mape_adicao >= mape_atual:
            log_text += f"❌ Nenhuma nova feature melhorou o MAPE. Finalizando.\n"
            break

        log_text += f"✅ Adicionando '{melhor_nova_feature}' (MAPE cai para {melhor_mape_adicao:.5f}).\n"
        features_para_refinar = features_atuais + [melhor_nova_feature]
        mape_para_refinar = melhor_mape_adicao
        
        while True:
            features_removiveis = [f for f in features_para_refinar if f not in (features_fixas or []) and f != melhor_nova_feature]
            if not features_removiveis: break

            resultados_remocao = {f: calcular_mape([feat for feat in features_para_refinar if feat != f]) for f in features_removiveis}
            melhor_remocao = min(resultados_remocao, key=resultados_remocao.get)
            mape_da_melhor_remocao = resultados_remocao[melhor_remocao]

            if mape_da_melhor_remocao < mape_para_refinar:
                log_text += f"   💡 Refinamento! Removendo '{melhor_remocao}', MAPE cai para {mape_da_melhor_remocao:.5f}\n"
                mape_para_refinar = mape_da_melhor_remocao
                features_para_refinar.remove(melhor_remocao)
            else:
                log_text += "   - Nenhuma remoção melhorou o MAPE. Fim do refinamento.\n"
                break
        
        features_atuais = features_para_refinar
        mape_atual = mape_para_refinar
        candidatas_restantes.remove(melhor_nova_feature)

    log_text += f"\n\n--- ✨ Seleção Concluída! ✨ ---\nMelhor MAPE: {mape_atual:.5f}\n"
    final_features = [f for f in features_atuais if f not in (features_fixas or [])] if features_fixas else features_atuais
    log_text += f"Features selecionadas: {len(final_features)}\n"
    
    return final_features, log_text

# --- FUNÇÕES DO GRADIO (CONTROLE DA INTERFACE) ---

def processar_arquivo(arquivo):
    if arquivo is None:
        return gr.update(choices=[], value=None), gr.update(choices=[], value=None), gr.update(choices=[], value=None), gr.update(choices=[], value=None), gr.update(value=None), gr.update(visible=False)
    try:
        df = pd.read_csv(arquivo.name) if arquivo.name.endswith('.csv') else pd.read_excel(arquivo.name)
        colunas = df.columns.tolist()
        
        col_data_candidata = [c for c in colunas if df[c].astype(str).str.match(r'\d{4}-\d{2}-\d{2}').all()]
        if col_data_candidata:
            primeira_col_data = col_data_candidata[0]
            df[primeira_col_data] = pd.to_datetime(df[primeira_col_data])
            data_min, data_max = df[primeira_col_data].min(), df[primeira_col_data].max()
            data_split_default = data_min + (data_max - data_min) * 0.8
            update_datepicker = gr.update(value=data_split_default.strftime('%Y-%m-%d'))
        else:
            primeira_col_data = colunas[0]
            update_datepicker = gr.update(value=None)

        colunas_fixas_choices = colunas + ['mes']
        return gr.update(choices=colunas, value=primeira_col_data), gr.update(choices=colunas), gr.update(choices=colunas), gr.update(choices=colunas_fixas_choices, value=['mes']), update_datepicker, gr.update(visible=True)
    except Exception as e:
        raise gr.Error(f"Erro ao ler o arquivo: {e}")

def executar_pipeline(arquivo, coluna_data, coluna_target, colunas_features, colunas_fixas, data_final_treino, lags, progress=gr.Progress(track_tqdm=True)):
    if not all([arquivo, coluna_data, coluna_target, colunas_features, data_final_treino]):
        raise gr.Error("Por favor, preencha todos os campos obrigatórios (Features Fixas é opcional)!")
        
    try:
        # 1. Carregar Dados
        progress(0, desc="Carregando dados...")
        log_text = "1. Carregando e preparando dados...\n"
        df = pd.read_csv(arquivo.name) if arquivo.name.endswith('.csv') else pd.read_excel(arquivo.name)
        df[coluna_data] = pd.to_datetime(df[coluna_data])
        df = df.set_index(coluna_data).sort_index()
        df['mes'] = df.index.month
        
        data_final_treino_dt = pd.to_datetime(data_final_treino)
        datas_split = {
            'treino_inicio': df.index.min(), 'treino_fim': data_final_treino_dt,
            'teste_inicio': data_final_treino_dt + pd.Timedelta(days=1), 'teste_fim': df.index.max()
        }
        log_text += f"Divisão: Treino até {datas_split['treino_fim'].date()}, Teste a partir de {datas_split['teste_inicio'].date()}\n"

        # 2. Criação de Features
        progress(0.1, desc="Criando Features...")
        config_geracao = [(col, f'_{col.lower().replace(" ", "_")[:10]}', lags) for col in colunas_features]
        df_features, log_criacao = criar_features_temporais(df, config_geracao)
        log_text += log_criacao

        # 3. Seleção de Features
        progress(0.3, desc="Selecionando Features...")
        features_candidatas = [c for c in df_features.columns if c.startswith('lag_')]
        params = {'n_estimators': 500, 'learning_rate': 0.05, 'random_state': 42, 'early_stopping_rounds': 20}
        
        melhores_features, log_selecao = selecionar_features_com_refinamento(df_features.dropna(), features_candidatas, colunas_fixas, coluna_target, datas_split, params, progress)
        log_text += log_selecao
        features_finais = (colunas_fixas or []) + melhores_features
        features_selecionadas_md = "### Features Selecionadas:\n" + "\n".join([f"- `{f}`" for f in melhores_features])

        # 4. Treinamento Final
        progress(0.8, desc="Treinando modelo final...")
        log_text += "\n4. Treinando modelo final...\n"
        df_final = df_features.dropna()
        treino = df_final[df_final.index <= datas_split['treino_fim']]
        teste = df_final[df_final.index >= datas_split['teste_inicio']]
        X_train, y_train = treino[features_finais], treino[coluna_target]
        X_test, y_test = teste[features_finais], teste[coluna_target]

        modelo_final = xgb.XGBRegressor(**params)
        modelo_final.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        
        # 5. Previsão, Métricas e Gráficos
        progress(0.9, desc="Gerando resultados e gráficos...")
        y_pred = modelo_final.predict(X_test)
        
        metricas = {
            'Métrica': ['MAPE', 'MAE', 'RMSE', 'R²'],
            'Valor': [
                mean_absolute_percentage_error(y_test, y_pred),
                mean_absolute_error(y_test, y_pred),
                np.sqrt(mean_squared_error(y_test, y_pred)),
                r2_score(y_test, y_pred)
            ]
        }
        df_metricas = pd.DataFrame(metricas).round(4)
        df_resultados = pd.DataFrame({'Real': y_test, 'Previsto': y_pred}, index=y_test.index).round(2)
        
        plt.close('all') # Limpa figuras anteriores
        
        # Gráficos
        fig_pred, ax_pred = plt.subplots(figsize=(12, 6)); df_resultados.plot(ax=ax_pred, color=['blue', 'red'], style=['-', '--']); ax_pred.set(title='Comparação Real vs. Previsto', xlabel='Data', ylabel=coluna_target); ax_pred.legend(); ax_pred.grid(True)
        
        importances = pd.Series(modelo_final.feature_importances_, index=features_finais).sort_values(ascending=False).head(20)
        fig_imp, ax_imp = plt.subplots(figsize=(10, 8)); importances.plot(kind='barh', ax=ax_imp, color='skyblue'); ax_imp.set(title='Importância das Features (Top 20)', xlabel='Importância'); ax_imp.invert_yaxis()
        
        explainer = shap.TreeExplainer(modelo_final); shap_values = explainer(X_test)
        
        plt.figure() # Garante uma nova figura para o SHAP Summary
        shap.summary_plot(shap_values, X_test, show=False, max_display=20); fig_shap_summary = plt.gcf(); fig_shap_summary.tight_layout()
        
        plt.figure() # Garante uma nova figura para o SHAP Force
        shap.force_plot(explainer.expected_value, shap_values.values[0,:], X_test.iloc[0,:], matplotlib=True, show=False); fig_shap_force = plt.gcf(); fig_shap_force.tight_layout()

        # Criar ZIP com gráficos
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'a', zipfile.ZIP_DEFLATED, False) as zip_file:
            for fig, name in [(fig_pred, "previsao.png"), (fig_imp, "importancia.png"), (fig_shap_summary, "shap_summary.png"), (fig_shap_force, "shap_force.png")]:
                buf = io.BytesIO()
                fig.savefig(buf, format='png', bbox_inches='tight')
                zip_file.writestr(name, buf.getvalue())
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp:
            tmp.write(zip_buffer.getvalue())
            zip_path = tmp.name

        plt.close('all') # Limpa figuras da memória
        return log_text, df_resultados, fig_pred, df_metricas, features_selecionadas_md, fig_imp, fig_shap_summary, fig_shap_force, gr.update(value=zip_path, visible=True)

    except Exception as e:
        log_text = f"❌ Ocorreu um erro: {e}\n\n--- Detalhes ---\n{traceback.format_exc()}"
        return log_text, None, None, None, None, None, None, None, gr.update(visible=False)


# --- CONSTRUÇÃO DA INTERFACE GRADIO ---
with gr.Blocks(theme=gr.themes.Soft(), title="AutoML de Séries Temporais") as demo:
    gr.Markdown("# 🤖 AutoML para Séries Temporais com Feature Selection Pro")
    gr.Markdown("Faça o upload do seu dataset, configure as opções e rode um pipeline completo de modelagem com análise de features!")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## 1. Configurações")
            arquivo_input = gr.File(label="Selecione seu arquivo (.csv ou .xlsx)")
            
            with gr.Group(visible=False) as grupo_config:
                coluna_data_input = gr.Dropdown(label="Coluna de data")
                data_final_treino_input = gr.Textbox(label="Data final do treino (AAAA-MM-DD)", placeholder="Ex: 2023-12-31")
                coluna_target_input = gr.Dropdown(label="Variável TARGET (a prever)")
                colunas_features_input = gr.CheckboxGroup(label="Colunas para gerar features (lags)")
                colunas_fixas_input = gr.CheckboxGroup(label="Features FIXAS (opcional)")
                lags_input = gr.CheckboxGroup(label="Lags (meses)", choices=list(range(1, 13)), value=[6, 12])
                run_button = gr.Button("🚀 Executar Pipeline!", variant="primary")

        with gr.Column(scale=2):
            gr.Markdown("## 2. Resultados")
            log_output = gr.Textbox(label="Log de Execução", lines=8, interactive=False)
            with gr.Row():
                metricas_output = gr.DataFrame(label="Métricas de Performance", headers=['Métrica', 'Valor'])
                features_selecionadas_output = gr.Markdown(label="Features Selecionadas")
            download_output = gr.File(label="Download dos Gráficos (.zip)", visible=False)

            with gr.Tabs():
                with gr.TabItem("📈 Previsão"):
                    plot_pred_output = gr.Plot(label="Gráfico de Previsão")
                    dataframe_output = gr.DataFrame(label="Tabela com Previsões")
                with gr.TabItem("🧠 Análise do Modelo"):
                    plot_imp_output = gr.Plot(label="Importância das Features (XGBoost)")
                    plot_shap_summary_output = gr.Plot(label="Análise de Impacto Geral (SHAP Summary)")
                    plot_shap_force_output = gr.Plot(label="Análise de Previsão Individual (SHAP Force Plot)")

    arquivo_input.upload(
        processar_arquivo,
        [arquivo_input],
        [coluna_data_input, coluna_target_input, colunas_features_input, colunas_fixas_input, data_final_treino_input, grupo_config]
    )
    
    run_button.click(
        executar_pipeline,
        [arquivo_input, coluna_data_input, coluna_target_input, colunas_features_input, colunas_fixas_input, data_final_treino_input, lags_input],
        [log_output, dataframe_output, plot_pred_output, metricas_output, features_selecionadas_output, plot_imp_output, plot_shap_summary_output, plot_shap_force_output, download_output]
    )

if __name__ == "__main__":
    demo.launch(debug=True)

