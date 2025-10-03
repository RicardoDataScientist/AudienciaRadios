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
import re

# --- Configurações Iniciais ---
warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
warnings.simplefilter(action="ignore", category=FutureWarning)
plt.style.use('seaborn-v0_8-whitegrid')

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
    border-radius: 8px !important;
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

# --- FUNÇÕES CORE (LÓGICA DO MODELO) ---
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

        log_text += f"   -> Processando: '{coluna_base}'\n"
        for lag in lags:
            feature_name = f'lag{sufixo}_{lag}_meses'
            df_features[feature_name] = df_features[coluna_base].shift(lag)
    
    log_text += "--- ✅  Criação de features concluída ---\n"
    return df_features, log_text

def treinar_e_avaliar(df_limpo, features_finais, target, datas_split, model_params, progress):
    """
    Função reutilizável para treinar o modelo final e gerar todos os resultados.
    Esta função usa o conjunto de TESTE real para a avaliação final.
    """
    log_text = "\n4. Treinando modelo final com as melhores features...\n"
    
    treino = df_limpo[df_limpo.index <= datas_split['treino_fim']]
    teste = df_limpo[(df_limpo.index >= datas_split['teste_inicio']) & (df_limpo.index <= datas_split['teste_fim'])]
    
    log_text += f"📊 Tamanho do dataset FINAL -> Treino: {len(treino)} linhas | Teste: {len(teste)} linhas.\n"
    
    if teste.empty:
        raise ValueError("O conjunto de teste está vazio. Verifique a divisão e o tamanho do seu dataset.")
    
    features_finais = sorted(list(set(features_finais)))
    
    X_train, y_train = treino[features_finais], treino[target]
    X_test, y_test = teste[features_finais], teste[target]

    modelo_final = xgb.XGBRegressor(**model_params)
    modelo_final.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    
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
    
    plt.close('all')
    
    fig_pred, ax_pred = plt.subplots(figsize=(12, 6)); df_resultados.plot(ax=ax_pred, color=['blue', 'red'], style=['-', '--']); ax_pred.set(title='Comparação Real vs. Previsto', xlabel='Data', ylabel=target); ax_pred.legend(); ax_pred.grid(True)
    
    importances = pd.Series(modelo_final.feature_importances_, index=features_finais).sort_values(ascending=False).head(20)
    fig_imp, ax_imp = plt.subplots(figsize=(10, 8)); importances.plot(kind='barh', ax=ax_imp, color='skyblue'); ax_imp.set(title='Importância das Features (Top 20)', xlabel='Importância'); ax_imp.invert_yaxis(); fig_imp.tight_layout()
    
    try:
        explainer = shap.TreeExplainer(modelo_final)
        shap_values = explainer(X_test)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            plt.figure(); shap.summary_plot(shap_values, X_test, show=False, max_display=20); fig_shap_summary = plt.gcf(); fig_shap_summary.tight_layout()
            plt.figure(); shap.force_plot(explainer.expected_value, shap_values.values[0,:], X_test.iloc[0,:], matplotlib=True, show=False); fig_shap_force = plt.gcf(); fig_shap_force.tight_layout()
    except Exception as e:
        log_text += f"\n⚠️ Aviso: Não foi possível gerar os gráficos SHAP. Erro: {e}\n"
        fig_shap_summary, fig_shap_force = None, None

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'a', zipfile.ZIP_DEFLATED, False) as zip_file:
        for fig, name in [(fig_pred, "previsao.png"), (fig_imp, "importancia.png"), (fig_shap_summary, "shap_summary.png"), (fig_shap_force, "shap_force.png")]:
            if fig:
                buf = io.BytesIO(); fig.savefig(buf, format='png', bbox_inches='tight'); zip_file.writestr(name, buf.getvalue())
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp:
        tmp.write(zip_buffer.getvalue()); zip_path = tmp.name

    plt.close('all')
    return log_text, df_resultados, fig_pred, df_metricas, fig_imp, fig_shap_summary, fig_shap_force, gr.update(value=zip_path, visible=True)

# --- FUNÇÕES DE PIPELINE ---

def executar_pipeline_auto(arquivo, coluna_data_orig, target_orig, colunas_features_orig, colunas_fixas_orig, tamanho_previsao_meses, lags, n_estimators, learning_rate, max_depth, early_stopping_rounds, progress=gr.Progress(track_tqdm=True)):
    log_text = ""
    no_update = gr.update()
    
    blank_outputs_para_nao_piscar = (
        no_update, no_update, no_update, no_update, 
        no_update, no_update, no_update, no_update
    )

    try:
        progress(0, desc="Carregando dados...")
        log_text += "1. Carregando e preparando dados...\n"
        yield log_text, None, None, None, None, None, None, gr.update(value=None, visible=False), "### Seleção de Features\n*Aguardando início do processo...*"
        
        df = pd.read_csv(arquivo.name) if arquivo.name.endswith('.csv') else pd.read_excel(arquivo.name)
        df = sanitize_columns(df)
        
        coluna_data = sanitize_columns(pd.DataFrame(columns=[coluna_data_orig])).columns[0]
        target = sanitize_columns(pd.DataFrame(columns=[target_orig])).columns[0]
        colunas_features = sanitize_columns(pd.DataFrame(columns=colunas_features_orig)).columns.tolist()
        colunas_fixas = sanitize_columns(pd.DataFrame(columns=colunas_fixas_orig)).columns.tolist() if colunas_fixas_orig else []
        
        df[coluna_data] = pd.to_datetime(df[coluna_data]); df = df.set_index(coluna_data).sort_index(); 
        if 'mes' not in colunas_fixas and 'mes' in df.columns is False:
            df['mes'] = df.index.month
        
        # --- NOVA LÓGICA DE DIVISÃO AUTOMÁTICA ---
        log_text += f"🧠 Definindo divisão: {tamanho_previsao_meses} meses para o conjunto de teste.\n"
        data_max = df.index.max()
        teste_fim = data_max
        teste_inicio = teste_fim - pd.DateOffset(months=tamanho_previsao_meses) + pd.Timedelta(days=1)
        treino_fim = teste_inicio - pd.Timedelta(days=1)
        treino_inicio = df.index.min()
        
        datas_split = {
            'treino_inicio': treino_inicio, 'treino_fim': treino_fim, 
            'teste_inicio': teste_inicio, 'teste_fim': teste_fim
        }
        log_text += f"Divisão -> Treino: {treino_inicio.date()} a {treino_fim.date()} | Teste: {teste_inicio.date()} a {teste_fim.date()}\n"
        # --- FIM DA NOVA LÓGICA ---
        
        progress(0.1, desc="Criando Features...")
        config_geracao = [(col, f'_{col}', lags) for col in colunas_features]
        df_features, log_criacao = criar_features_temporais(df, config_geracao)
        log_text += log_criacao
        
        maior_lag = max(lags) if lags else 0
        log_text += f"💡 Dica de consistência: Maior lag gerado foi {maior_lag}.\n"
        df_limpo = df_features.dropna()
        log_text += f"ℹ️ Para o processo, {len(df_features) - len(df_limpo)} linhas com dados ausentes foram removidas.\n"
        
        progress(0.3, desc="Selecionando Features...")
        
        treino_completo = df_limpo[df_limpo.index <= datas_split['treino_fim']]
        
        log_text += "\n🛡️ Criando conjunto de VALIDAÇÃO para uma seleção de features robusta...\n"
        
        # Usaremos os últimos 6 meses do treino para validação.
        try:
            data_inicio_validacao = datas_split['treino_fim'] - pd.DateOffset(months=5)
            if data_inicio_validacao < treino_completo.index.min():
                 data_inicio_validacao = treino_completo.index.min() + (datas_split['treino_fim'] - treino_completo.index.min()) * pd.Timedelta(days=1) * 0.5
        except Exception:
            data_inicio_validacao = treino_completo.index.min() + (datas_split['treino_fim'] - treino_completo.index.min()) * pd.Timedelta(days=1) * 0.5

        validacao = treino_completo[treino_completo.index >= data_inicio_validacao]
        treino_selecao = treino_completo[treino_completo.index < data_inicio_validacao]

        if treino_selecao.empty or validacao.empty:
            raise ValueError("Não foi possível criar os conjuntos de treino/validação. Verifique se seu período de treino é longo o suficiente (precisa ter mais de 6 meses) ou diminua o tamanho da previsão.")

        y_train_selecao = treino_selecao[target]
        y_valid = validacao[target]

        log_text += f"📊 Divisão para Seleção -> Treino: {len(treino_selecao)} linhas | Validação: {len(validacao)} linhas.\n"

        params = {'n_estimators': int(n_estimators), 'learning_rate': learning_rate, 'max_depth': int(max_depth),
                  'random_state': 42, 'early_stopping_rounds': int(early_stopping_rounds)}

        def calcular_mape(features):
            if not features: return 1.0
            try:
                X_train_s, X_valid_s = treino_selecao[features], validacao[features]
                model = xgb.XGBRegressor(**params)
                model.fit(X_train_s, y_train_selecao, eval_set=[(X_valid_s, y_valid)], verbose=False)
                return mean_absolute_percentage_error(y_valid, model.predict(X_valid_s))
            except Exception: return float('inf')

        features_candidatas = [c for c in df_limpo.columns if c.startswith('lag_')]
        features_atuais = colunas_fixas.copy() if colunas_fixas else []
        mape_atual = calcular_mape(features_atuais)
        candidatas_restantes = features_candidatas.copy()
        
        log_text += f"\n--- 🤖 Iniciando a seleção com REFINAMENTO IMEDIATO ---\n"
        log_text += f"MAPE de Referência (na validação): {mape_atual:.5f}\n"
        
        iteration = 0
        while True:
            iteration += 1
            log_text += f"\n==================== ITERAÇÃO DE SELEÇÃO {iteration} ====================\n"
            log_text += f"MAPE atual (validação): {mape_atual:.5f}\n"
            yield log_text, *blank_outputs_para_nao_piscar
            
            if not candidatas_restantes:
                log_text += "Não há mais features candidatas para testar. Finalizando seleção.\n"
                break

            log_text += "\n--- Fase 1: Buscando a melhor feature para ADICIONAR ---\n"
            yield log_text, *blank_outputs_para_nao_piscar

            resultados_adicao = {}
            for feature in candidatas_restantes:
                mape_teste = calcular_mape(features_atuais + [feature])
                resultados_adicao[feature] = mape_teste
                log_text += f"   - Testando adicionar '{feature}': MAPE = {mape_teste:.5f}\n"
                yield log_text, *blank_outputs_para_nao_piscar
            
            melhor_nova_feature = min(resultados_adicao, key=resultados_adicao.get)
            melhor_mape_adicao = resultados_adicao[melhor_nova_feature]

            if melhor_mape_adicao >= mape_atual:
                log_text += f"\n❌ Nenhuma nova feature melhorou o MAPE (melhor foi '{melhor_nova_feature}' com MAPE {melhor_mape_adicao:.5f}). Finalizando.\n"
                break

            log_text += f"✅ Melhoria encontrada! Adicionando '{melhor_nova_feature}' (MAPE cai para {melhor_mape_adicao:.5f}).\n"
            yield log_text, *blank_outputs_para_nao_piscar
            features_para_refinar = features_atuais + [melhor_nova_feature]
            mape_para_refinar = melhor_mape_adicao
            
            log_text += "\n--- Fase 2: Iniciando REFINAMENTO (tentando remover features antigas) ---\n"
            yield log_text, *blank_outputs_para_nao_piscar

            while True:
                features_removiveis = [f for f in features_para_refinar if f not in (colunas_fixas or []) and f != melhor_nova_feature]
                if not features_removiveis:
                    log_text += "   - Nenhuma feature antiga para remover. Fim do refinamento.\n"
                    break

                resultados_remocao = {}
                for f_remover in features_removiveis:
                    mape_teste = calcular_mape([feat for feat in features_para_refinar if feat != f_remover])
                    resultados_remocao[f_remover] = mape_teste
                    log_text += f"   - Testando remover '{f_remover}': MAPE = {mape_teste:.5f}\n"
                    yield log_text, *blank_outputs_para_nao_piscar
                
                melhor_remocao = min(resultados_remocao, key=resultados_remocao.get)
                mape_da_melhor_remocao = resultados_remocao[melhor_remocao]

                if mape_da_melhor_remocao < mape_para_refinar:
                    log_text += f"   💡 Refinamento! Removendo '{melhor_remocao}', MAPE cai para {mape_da_melhor_remocao:.5f}\n"
                    mape_para_refinar = mape_da_melhor_remocao
                    features_para_refinar.remove(melhor_remocao)
                else:
                    log_text += f"   - Nenhuma remoção melhorou mais o MAPE. Fim do refinamento.\n"
                    break
            
            features_atuais = features_para_refinar
            mape_atual = mape_para_refinar
            candidatas_restantes.remove(melhor_nova_feature)
            
        def format_features_md(features, title="### Features Finais Selecionadas (Auto)"):
            if not features: return f"{title}\n*Nenhuma feature selecionada...*"
            fixas = sorted([f for f in features if f in (colunas_fixas or [])])
            dinamicas = sorted([f for f in features if f not in (colunas_fixas or [])])
            md = f"{title}\n"
            if fixas: md += "**Fixas:**\n" + "\n".join([f"- `{f}`" for f in fixas]) + "\n"
            if dinamicas: md += "**Selecionadas:**\n" + "\n".join([f"- `{f}`" for f in dinamicas])
            return md

        log_text += f"\n--- ✨ Seleção Concluída! ✨ ---\nMelhor MAPE na VALIDAÇÃO: {mape_atual:.5f}\n"
        log_text += "\n\n--- 🚀 Preparando para o treino final com as melhores features! Aguarde... ---\n"
        
        yield log_text, *blank_outputs_para_nao_piscar
        
        features_finais = features_atuais
        features_md_final = format_features_md(features_finais)

        log_treino_final, df_res, fig_pred, df_met, fig_imp, fig_shap_sum, fig_shap_force, zip_file = treinar_e_avaliar(df_limpo, features_finais, target, datas_split, params, progress)
        log_text += log_treino_final
        
        yield log_text, df_res, fig_pred, df_met, fig_imp, fig_shap_sum, fig_shap_force, zip_file, features_md_final

    except Exception as e:
        log_text += f"\n❌ Ocorreu um erro: {e}\n\n--- Detalhes ---\n{traceback.format_exc()}"
        yield log_text, None, None, None, None, None, None, gr.update(value=None, visible=False), "### Erro\n*Ocorreu um erro durante a execução.*"


def executar_pipeline_manual(arquivo, coluna_data_orig, target_orig, tamanho_previsao_meses, features_finais_selecionadas, simular_drop, colunas_configuradas_orig, n_estimators, learning_rate, max_depth, early_stopping_rounds, *lista_de_lags, progress=gr.Progress(track_tqdm=True)):
    log_text = ""
    no_update = gr.update()
    
    def get_blank_outputs_manual():
        return no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update

    try:
        progress(0, desc="Carregando dados..."); 
        log_text = "1. Carregando e preparando dados para treino manual...\n"
        yield log_text, *get_blank_outputs_manual()
        
        df = pd.read_csv(arquivo.name) if arquivo.name.endswith('.csv') else pd.read_excel(arquivo.name)
        df = sanitize_columns(df)

        coluna_data = sanitize_columns(pd.DataFrame(columns=[coluna_data_orig])).columns[0]
        target = sanitize_columns(pd.DataFrame(columns=[target_orig])).columns[0]
        colunas_configuradas = sanitize_columns(pd.DataFrame(columns=colunas_configuradas_orig)).columns.tolist()

        df[coluna_data] = pd.to_datetime(df[coluna_data]); df = df.set_index(coluna_data).sort_index()

        if any('mes' in f for f in features_finais_selecionadas) and 'mes' in df.columns is False:
            df['mes'] = df.index.month
        
        progress(0.2, desc="Criando features...")
        config_geracao = []
        for i, coluna in enumerate(colunas_configuradas):
            lags_selecionados = lista_de_lags[i]
            if lags_selecionados:
                sufixo = f'_{coluna}'
                config_geracao.append((coluna, sufixo, lags_selecionados))
        df_features, log_criacao = criar_features_temporais(df, config_geracao)
        log_text += log_criacao

        if simular_drop and simular_drop > 0:
            df_features[f'__temp_consistency_lag__'] = df[target].shift(int(simular_drop))
            log_text += f"⚠️ Lag fantasma de {int(simular_drop)} meses criado para consistência.\n"

        df_limpo = df_features.dropna()
        log_text += f"ℹ️ Para o processo, {len(df_features) - len(df_limpo)} linhas com dados ausentes foram removidas.\n"

        # --- NOVA LÓGICA DE DIVISÃO AUTOMÁTICA (APLICADA AO MANUAL) ---
        log_text += f"🧠 Definindo divisão: {tamanho_previsao_meses} meses para o conjunto de teste.\n"
        data_max = df_limpo.index.max() # Usar df_limpo para garantir que a data máxima existe após o dropna
        teste_fim = data_max
        teste_inicio = teste_fim - pd.DateOffset(months=tamanho_previsao_meses) + pd.Timedelta(days=1)
        treino_fim = teste_inicio - pd.Timedelta(days=1)
        treino_inicio = df_limpo.index.min()

        datas_split = {
            'treino_inicio': treino_inicio, 'treino_fim': treino_fim, 
            'teste_inicio': teste_inicio, 'teste_fim': teste_fim
        }
        log_text += f"Divisão -> Treino: {treino_inicio.date()} a {treino_fim.date()} | Teste: {teste_inicio.date()} a {teste_fim.date()}\n"
        # --- FIM DA NOVA LÓGICA ---
        
        params = {
            'n_estimators': int(n_estimators), 'learning_rate': learning_rate, 'max_depth': int(max_depth),
            'random_state': 42, 'early_stopping_rounds': int(early_stopping_rounds)
        }
        
        features_para_md = sorted(features_finais_selecionadas)
        features_selecionadas_md = "### Features Selecionadas (Manual):\n" + "\n".join([f"- `{f}`" for f in features_para_md])
        
        progress(0.5, desc="Treinando modelo...")
        log_treino_final, df_res, fig_pred, df_met, fig_imp, fig_shap_sum, fig_shap_force, zip_file = treinar_e_avaliar(df_limpo, features_finais_selecionadas, target, datas_split, params, progress)
        log_text += log_treino_final

        yield log_text, df_res, fig_pred, df_met, fig_imp, fig_shap_sum, fig_shap_force, zip_file, features_selecionadas_md
        
    except Exception as e:
        log_text += f"❌ Ocorreu um erro: {e}\n\n--- Detalhes ---\n{traceback.format_exc()}"
        yield log_text, *get_blank_outputs_manual(), "### Erro\n*Ocorreu um erro durante a execução.*"


# --- FUNÇÕES DA INTERFACE ---

def processar_arquivo(arquivo):
    if arquivo is None: return [gr.update(visible=False)] * 9 # Ajustado para 9 saídas
    try:
        df = pd.read_csv(arquivo.name) if arquivo.name.endswith('.csv') else pd.read_excel(arquivo.name)
        df_original_cols = df.columns.tolist()
        df = sanitize_columns(df) 
        
        colunas_sanitizadas = df.columns.tolist()
        
        try:
            col_data_candidata_sanitizada = [c for c in colunas_sanitizadas if df[c].astype(str).str.match(r'\d{4}-\d{2}-\d{2}').all()][0]
            col_data_candidata_original = df_original_cols[colunas_sanitizadas.index(col_data_candidata_sanitizada)]
        except (IndexError, Exception):
            col_data_candidata_original = df_original_cols[0]
        
        colunas_fixas_choices = df_original_cols + ['mes']
        feature_choices = [c for c in df_original_cols if c != col_data_candidata_original]

        updates = [
            gr.update(visible=True), 
            gr.update(choices=df_original_cols, value=col_data_candidata_original),
            # O input de data foi removido, então não precisa de update para ele.
            gr.update(choices=df_original_cols), 
            gr.update(choices=feature_choices),
            gr.update(choices=colunas_fixas_choices), 
            gr.update(choices=feature_choices),
            df_original_cols, 
            colunas_fixas_choices,
            feature_choices
        ]
        return updates
    except Exception as e:
        raise gr.Error(f"Erro ao ler o arquivo: {e}")

def update_manual_lag_ui(colunas_selecionadas):
    MAX_COLS = 15
    updates = []
    if colunas_selecionadas:
        for i in range(len(colunas_selecionadas)):
            if i < MAX_COLS:
                updates.append(gr.update(visible=True))
                updates.append(gr.update(value=f"**{colunas_selecionadas[i]}**"))

        for i in range(len(colunas_selecionadas), MAX_COLS):
            updates.append(gr.update(visible=False))
            updates.append(gr.update(value=""))
    return updates


def gerar_features_para_selecao_manual(arquivo, coluna_data_orig, colunas_configuradas_orig, *lista_de_lags):
    df = pd.read_csv(arquivo.name) if arquivo.name.endswith('.csv') else pd.read_excel(arquivo.name)
    df = sanitize_columns(df)

    coluna_data = sanitize_columns(pd.DataFrame(columns=[coluna_data_orig])).columns[0]
    colunas_configuradas = sanitize_columns(pd.DataFrame(columns=colunas_configuradas_orig)).columns.tolist()
    
    df[coluna_data] = pd.to_datetime(df[coluna_data]); df = df.set_index(coluna_data).sort_index(); 
    
    if 'mes' not in df.columns:
        df['mes'] = df.index.month
    
    config_geracao = []
    for i, coluna in enumerate(colunas_configuradas):
        lags_selecionados = lista_de_lags[i]
        if lags_selecionados:
            sufixo = f'_{coluna}'
            config_geracao.append((coluna, sufixo, lags_selecionados))

    df_features, _ = criar_features_temporais(df, config_geracao)
    
    features_disponiveis = sorted([c for c in df_features.columns if c in df.columns or c.startswith('lag_') or c == 'mes'])
    
    return gr.update(visible=True), gr.update(choices=features_disponiveis, value=features_disponiveis), features_disponiveis


# --- CONSTRUÇÃO DA INTERFACE ---
with gr.Blocks(theme=gr.themes.Soft(), title="AutoML de Séries Temporais", css=custom_css) as demo:
    gr.Markdown("# 🤖 AutoML para Séries Temporais com Feature Selection Pro")
    gr.Markdown("Faça o upload, configure o modo (automático ou manual) e rode um pipeline completo de modelagem!")
    
    colunas_features_state = gr.State([])
    colunas_fixas_state = gr.State([])
    features_manuais_state = gr.State([])
    valid_features_state = gr.State([])

    with gr.Row():
        arquivo_input = gr.File(label="Selecione seu arquivo (.csv ou .xlsx)", scale=1)
    
    with gr.Group(visible=False) as grupo_principal:
        with gr.Group():
            with gr.Row():
                coluna_data_input = gr.Dropdown(label="Coluna de data")
                # --- ALTERAÇÃO DA INTERFACE ---
                tamanho_previsao_input = gr.Slider(label="Tamanho da Previsão (meses)", minimum=1, maximum=24, value=6, step=1, scale=1)
                coluna_target_input = gr.Dropdown(label="Variável TARGET (a prever)")

        with gr.Accordion("Parâmetros do Modelo (Ajuste Fino)", open=True):
            with gr.Group():
                n_estimators_input = gr.Slider(label="Número de Árvores (n_estimators)", minimum=50, maximum=1000, value=500, step=50)
                learning_rate_input = gr.Slider(label="Taxa de Aprendizado (learning_rate)", minimum=0.01, maximum=0.3, value=0.05, step=0.01)
                max_depth_input = gr.Slider(label="Profundidade Máxima (max_depth)", minimum=3, maximum=10, value=6, step=1)
                early_stopping_input = gr.Number(label="Parada Antecipada (early_stopping_rounds)", value=20)

        with gr.Tabs():
            with gr.TabItem("AutoML com Seleção de Features"):
                with gr.Group(elem_classes=["dark-checkbox-group"]):
                    gr.Markdown("### 1. Colunas para gerar features (lags)")
                    colunas_features_auto = gr.CheckboxGroup()
                    with gr.Row():
                        select_all_btn_feat_auto = gr.Button("Selecionar Todas", elem_classes=["orange-button"])
                        clear_btn_feat_auto = gr.Button("Limpar", elem_classes=["orange-button"])
                
                with gr.Group(elem_classes=["dark-checkbox-group"]):
                    gr.Markdown("### 2. Lags (meses)")
                    lags_auto = gr.CheckboxGroup(choices=list(range(1, 13)), value=[])
                    with gr.Row():
                        select_all_btn_lags_auto = gr.Button("Selecionar Todos", elem_classes=["orange-button"])
                        clear_btn_lags_auto = gr.Button("Limpar", elem_classes=["orange-button"])
                
                with gr.Group(elem_classes=["dark-checkbox-group"]):
                    gr.Markdown("### 3. Features FIXAS (opcional)")
                    colunas_fixas_auto = gr.CheckboxGroup()
                    with gr.Row():
                        select_all_btn_fixas_auto = gr.Button("Selecionar Todas", elem_classes=["orange-button"])
                        clear_btn_fixas_auto = gr.Button("Limpar", elem_classes=["orange-button"])
                
                gr.Markdown("---")
                run_button_auto = gr.Button("🚀 Executar Pipeline Automático!", variant="primary", scale=1)

            with gr.TabItem("Treino Manual Direto"):
                with gr.Group(elem_classes=["dark-checkbox-group"]):
                    gr.Markdown("### Passo 1: Selecione as colunas para gerar features")
                    colunas_features_manual = gr.CheckboxGroup()
                    with gr.Row():
                        select_all_btn_feat_manual = gr.Button("Selecionar Todas", elem_classes=["orange-button"])
                        clear_btn_feat_manual = gr.Button("Limpar", elem_classes=["orange-button"])
                
                with gr.Group():
                    gr.Markdown("### Passo 2: Configure os lags para cada coluna")
                    lag_configs_ui = []
                    MAX_COLS_UI = 15
                    for i in range(MAX_COLS_UI):
                        with gr.Row(visible=False) as lag_row:
                            col_name = gr.Markdown()
                            gr.Markdown("Lags:")
                            col_lags = gr.CheckboxGroup(choices=list(range(1, 13)), value=[], interactive=True, elem_classes=["dark-checkbox-group"])
                            lag_configs_ui.append({'group': lag_row, 'name': col_name, 'lags': col_lags})
                    
                    simular_drop_input = gr.Number(label="Para consistência, simular drop de X meses iniciais (opcional)", value=12)
                    generate_features_button = gr.Button("Gerar e Listar Features Disponíveis")

                with gr.Group(visible=False, elem_classes=["dark-checkbox-group"]) as manual_select_group:
                    gr.Markdown("### Passo 3: Selecione as features finais e treine o modelo")
                    manual_features_checklist = gr.CheckboxGroup()
                    with gr.Row():
                        select_all_btn_manual_final = gr.Button("Selecionar Todas", elem_classes=["orange-button"])
                        clear_btn_manual_final = gr.Button("Limpar", elem_classes=["orange-button"])
                    run_button_manual = gr.Button("🚀 Treinar com Features Selecionadas!", variant="primary")
        
        with gr.Group():
            gr.Markdown("## Resultados")
            log_output = gr.Textbox(label="Log de Execução", lines=15, interactive=False)
            with gr.Row():
                metricas_output = gr.DataFrame(label="Métricas de Performance", headers=['Métrica', 'Valor'])
                features_selecionadas_output = gr.Markdown(label="Seleção de Features")
            download_output = gr.File(label="Download dos Gráficos (.zip)", visible=False)

            with gr.Tabs():
                with gr.TabItem("📈 Previsão"):
                    plot_pred_output = gr.Plot(label="Gráfico de Previsão")
                    dataframe_output = gr.DataFrame(label="Tabela com Previsões")
                with gr.TabItem("🧠 Análise do Modelo"):
                    plot_imp_output = gr.Plot(label="Importância das Features (XGBoost)")
                    plot_shap_summary_output = gr.Plot(label="Análise de Impacto Geral (SHAP Summary)")
                    plot_shap_force_output = gr.Plot(label="Análise de Previsão Individual (SHAP Force Plot)")

    # --- Lógica dos Eventos ---
    outputs_list = [log_output, dataframe_output, plot_pred_output, metricas_output, plot_imp_output, plot_shap_summary_output, plot_shap_force_output, download_output, features_selecionadas_output]
    
    arquivo_input.upload(
        processar_arquivo,
        [arquivo_input],
        [grupo_principal, coluna_data_input, coluna_target_input, colunas_features_auto, colunas_fixas_auto, colunas_features_manual, colunas_features_state, colunas_fixas_state, valid_features_state]
    )
    
    def atualizar_colunas_features(lista_colunas_total, data_selecionada):
        opcoes_validas = [col for col in lista_colunas_total if col != data_selecionada]
        return gr.update(choices=opcoes_validas), gr.update(choices=opcoes_validas), opcoes_validas

    coluna_data_input.change(
        atualizar_colunas_features,
        [colunas_features_state, coluna_data_input],
        [colunas_features_auto, colunas_features_manual, valid_features_state]
    )
    
    select_all_btn_feat_auto.click(lambda x: gr.update(value=x), valid_features_state, colunas_features_auto)
    clear_btn_feat_auto.click(lambda: gr.update(value=[]), None, colunas_features_auto)
    select_all_btn_lags_auto.click(lambda: gr.update(value=list(range(1,13))), None, lags_auto)
    clear_btn_lags_auto.click(lambda: gr.update(value=[]), None, lags_auto)
    select_all_btn_fixas_auto.click(lambda x: gr.update(value=x), colunas_fixas_state, colunas_fixas_auto)
    clear_btn_fixas_auto.click(lambda: gr.update(value=[]), None, colunas_fixas_auto)
    select_all_btn_feat_manual.click(lambda x: gr.update(value=x), valid_features_state, colunas_features_manual)
    clear_btn_feat_manual.click(lambda: gr.update(value=[]), None, colunas_features_manual)
    select_all_btn_manual_final.click(lambda x: gr.update(value=x), features_manuais_state, manual_features_checklist)
    clear_btn_manual_final.click(lambda: gr.update(value=[]), None, manual_features_checklist)
    
    lag_ui_outputs_flat = []
    for config in lag_configs_ui:
        lag_ui_outputs_flat.extend([config['group'], config['name']])
    
    all_lag_checkboxes = [config['lags'] for config in lag_configs_ui]

    colunas_features_manual.change(
        update_manual_lag_ui,
        [colunas_features_manual],
        lag_ui_outputs_flat
    )

    generate_features_button.click(
        gerar_features_para_selecao_manual,
        [arquivo_input, coluna_data_input, colunas_features_manual] + all_lag_checkboxes,
        [manual_select_group, manual_features_checklist, features_manuais_state]
    )
    
    run_button_manual.click(
        executar_pipeline_manual,
        inputs=[arquivo_input, coluna_data_input, coluna_target_input, tamanho_previsao_input, manual_features_checklist, simular_drop_input, colunas_features_manual, n_estimators_input, learning_rate_input, max_depth_input, early_stopping_input] + all_lag_checkboxes,
        outputs=outputs_list
    )
    
    run_button_auto.click(
        executar_pipeline_auto,
        inputs=[arquivo_input, coluna_data_input, coluna_target_input, colunas_features_auto, colunas_fixas_auto, tamanho_previsao_input, lags_auto, n_estimators_input, learning_rate_input, max_depth_input, early_stopping_input],
        outputs=outputs_list
    )

if __name__ == "__main__":
    demo.launch(debug=True)
