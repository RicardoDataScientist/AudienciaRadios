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

# --- CSS Personalizado para o tema "Deus do Front-end" ---
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

/* Estilo "Dark Mode" para Checkboxes, aplicado ao Group */
.dark-checkbox-group .gradio-checkbox-group-label {
    color: #FFFFFF !important; /* Cor do label principal, se existir */
}
.dark-checkbox-group {
    background-color: #000000 !important; /* Fundo preto */
    border: 1px solid #444 !important;
    border-radius: 8px !important;
    padding: 10px !important;
}
.dark-checkbox-group label span {
    color: #FFFFFF !important; /* Texto de cada opção */
}
.dark-checkbox-group input[type="checkbox"] {
    border: 1px solid #FFFFFF !important; /* Quadrado branco */
    accent-color: #FFA500 !important; /* Cor do check quando marcado (laranja) */
}
"""

# --- FUNÇÕES CORE (LÓgica DO MODELO) ---
def sanitize_columns(df):
    """Limpeza ultra robusta para nomes de colunas."""
    novas_colunas = []
    for col in df.columns:
        clean_col = str(col).strip() # Garante que é string e remove espaços no início/fim
        clean_col = clean_col.lower() # Converte para minúsculas
        clean_col = re.sub(r'\s+', '_', clean_col) # Substitui espaços por _
        clean_col = re.sub(r'[^a-zA-Z0-9_]', '', clean_col) # Remove caracteres especiais
        clean_col = clean_col.strip('_') # Remove underscores no início/fim
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


def selecionar_features_com_refinamento(df_limpo, features_candidatas, features_fixas, target, datas_split, model_params, progress):
    """
    Realiza a seleção de features com uma lógica de refinamento imediato.
    AGORA RECEBE UM DF JÁ LIMPO.
    """
    log_text = "\n🤖 Iniciando a seleção de features com REFINAMENTO IMEDIATO...\n"
    treino = df_limpo[(df_limpo.index >= datas_split['treino_inicio']) & (df_limpo.index <= datas_split['treino_fim'])]
    teste = df_limpo[(df_limpo.index >= datas_split['teste_inicio']) & (df_limpo.index <= datas_split['teste_fim'])]
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

def treinar_e_avaliar(df_limpo, features_finais, target, datas_split, model_params, progress):
    """
    Função reutilizável para treinar o modelo final e gerar todos os resultados.
    AGORA RECEBE UM DF JÁ LIMPO.
    """
    log_text = "\n4. Treinando modelo final...\n"
    
    treino = df_limpo[df_limpo.index <= datas_split['treino_fim']]
    teste = df_limpo[df_limpo.index >= datas_split['teste_inicio']]
    
    log_text += f"📊 Tamanho do dataset final -> Treino: {len(treino)} linhas | Teste: {len(teste)} linhas.\n"
    
    if teste.empty:
        raise ValueError("O conjunto de teste está vazio. Verifique a data de divisão e o tamanho do seu dataset.")
    
    # Ordena as features para garantir consistência total
    features_finais = sorted(features_finais)
    
    X_train, y_train = treino[features_finais], treino[target]
    X_test, y_test = teste[features_finais], teste[target]

    log_text += f"\n🕵️  Inspecionando dados de entrada do modelo para garantir consistência...\n"
    log_text += f"Features Finais para o Modelo ({len(features_finais)}): {features_finais}\n\n"
    log_text += f"Primeiras 3 linhas de X_train:\n{X_train.head(3).to_string()}\n\n"
    log_text += f"Últimas 3 linhas de X_train:\n{X_train.tail(3).to_string()}\n\n"
    log_text += f"Primeiras 3 linhas de X_test:\n{X_test.head(3).to_string()}\n"

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
    
    explainer = shap.TreeExplainer(modelo_final); shap_values = explainer(X_test)
    
    plt.figure(); shap.summary_plot(shap_values, X_test, show=False, max_display=20); fig_shap_summary = plt.gcf(); fig_shap_summary.tight_layout()
    plt.figure(); shap.force_plot(explainer.expected_value, shap_values.values[0,:], X_test.iloc[0,:], matplotlib=True, show=False); fig_shap_force = plt.gcf(); fig_shap_force.tight_layout()

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'a', zipfile.ZIP_DEFLATED, False) as zip_file:
        for fig, name in [(fig_pred, "previsao.png"), (fig_imp, "importancia.png"), (fig_shap_summary, "shap_summary.png"), (fig_shap_force, "shap_force.png")]:
            buf = io.BytesIO(); fig.savefig(buf, format='png', bbox_inches='tight'); zip_file.writestr(name, buf.getvalue())
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp:
        tmp.write(zip_buffer.getvalue()); zip_path = tmp.name

    plt.close('all')
    return log_text, df_resultados, fig_pred, df_metricas, fig_imp, fig_shap_summary, fig_shap_force, gr.update(value=zip_path, visible=True)

# --- FUNÇÕES DE PIPELINE (CONTROLAM OS MODOS) ---

def executar_pipeline_auto(arquivo, coluna_data_orig, target_orig, colunas_features_orig, colunas_fixas_orig, data_final_treino, lags, progress=gr.Progress(track_tqdm=True)):
    try:
        progress(0, desc="Carregando dados..."); log_text = "1. Carregando e preparando dados...\n"
        df = pd.read_csv(arquivo.name) if arquivo.name.endswith('.csv') else pd.read_excel(arquivo.name)
        df = sanitize_columns(df) # Limpeza ROBUSTA dos nomes das colunas
        
        # Sanitiza os nomes das colunas recebidos da UI para corresponder ao df
        coluna_data = sanitize_columns(pd.DataFrame(columns=[coluna_data_orig])).columns[0]
        target = sanitize_columns(pd.DataFrame(columns=[target_orig])).columns[0]
        colunas_features = sanitize_columns(pd.DataFrame(columns=colunas_features_orig)).columns.tolist()
        colunas_fixas = sanitize_columns(pd.DataFrame(columns=colunas_fixas_orig)).columns.tolist() if colunas_fixas_orig else []
        
        df[coluna_data] = pd.to_datetime(df[coluna_data]); df = df.set_index(coluna_data).sort_index(); 
        if 'mes' not in colunas_fixas:
            df['mes'] = df.index.month
        
        data_final_treino_dt = pd.to_datetime(data_final_treino)
        datas_split = {'treino_inicio': df.index.min(), 'treino_fim': data_final_treino_dt, 'teste_inicio': data_final_treino_dt + pd.Timedelta(days=1), 'teste_fim': df.index.max()}
        log_text += f"Divisão: Treino até {datas_split['treino_fim'].date()}, Teste a partir de {datas_split['teste_inicio'].date()}\n"

        progress(0.1, desc="Criando Features...")
        # <<<<<<< MUDANÇA CRÍTICA: Usa o nome completo da coluna para o sufixo, evitando colisões.
        config_geracao = [(col, f'_{col}', lags) for col in colunas_features]
        df_features, log_criacao = criar_features_temporais(df, config_geracao); log_text += log_criacao
        
        maior_lag = max(lags) if lags else 0
        log_text += f"💡 Dica de consistência: Maior lag gerado foi {maior_lag}. Para replicar no modo manual, use este valor em 'simular drop'.\n"

        # --- PONTO ÚNICO DE LIMPEZA ---
        df_limpo = df_features.dropna()
        log_text += f"ℹ️ Para o processo, {len(df_features) - len(df_limpo)} linhas com dados ausentes foram removidas.\n"
        
        progress(0.3, desc="Selecionando Features...")
        features_candidatas = [c for c in df_features.columns if c.startswith('lag_')]
        params = {'n_estimators': 500, 'learning_rate': 0.05, 'random_state': 42, 'early_stopping_rounds': 20}
        
        melhores_features, log_selecao = selecionar_features_com_refinamento(df_limpo, features_candidatas, colunas_fixas, target, datas_split, params, progress); log_text += log_selecao
        
        features_finais = (colunas_fixas or []) + melhores_features
        
        # Ordena a lista para exibição na UI
        features_para_md = sorted(melhores_features)
        features_selecionadas_md = "### Features Selecionadas (Auto):\n" + "\n".join([f"- `{f}`" for f in features_para_md])

        log_treino, *results = treinar_e_avaliar(df_limpo, features_finais, target, datas_split, params, progress)
        log_text += log_treino

        return log_text, *results, features_selecionadas_md

    except Exception as e:
        log_text = f"❌ Ocorreu um erro: {e}\n\n--- Detalhes ---\n{traceback.format_exc()}"
        return log_text, None, None, None, None, None, None, None, None

def executar_pipeline_manual(arquivo, coluna_data_orig, target_orig, data_final_treino, features_finais_selecionadas, simular_drop, colunas_configuradas_orig, *lista_de_lags, progress=gr.Progress(track_tqdm=True)):
    try:
        progress(0, desc="Carregando dados..."); log_text = "1. Carregando e preparando dados para treino manual...\n"
        df = pd.read_csv(arquivo.name) if arquivo.name.endswith('.csv') else pd.read_excel(arquivo.name)
        df = sanitize_columns(df) # Limpeza ROBUSTA dos nomes das colunas

        coluna_data = sanitize_columns(pd.DataFrame(columns=[coluna_data_orig])).columns[0]
        target = sanitize_columns(pd.DataFrame(columns=[target_orig])).columns[0]
        colunas_configuradas = sanitize_columns(pd.DataFrame(columns=colunas_configuradas_orig)).columns.tolist()

        df[coluna_data] = pd.to_datetime(df[coluna_data]); df = df.set_index(coluna_data).sort_index()

        if any('mes' in f for f in features_finais_selecionadas):
            df['mes'] = df.index.month

        data_final_treino_dt = pd.to_datetime(data_final_treino)
        
        config_geracao = []
        for i, coluna in enumerate(colunas_configuradas):
            lags_selecionados = lista_de_lags[i]
            if lags_selecionados:
                # <<<<<<< MUDANÇA CRÍTICA: Usa o nome completo da coluna para o sufixo, evitando colisões.
                sufixo = f'_{coluna}'
                config_geracao.append((coluna, sufixo, lags_selecionados))
        df_features, _ = criar_features_temporais(df, config_geracao)

        if simular_drop and simular_drop > 0:
            df_features[f'__temp_consistency_lag__'] = df[target].shift(int(simular_drop))
            log_text += f"⚠️ Lag fantasma de {int(simular_drop)} meses criado para consistência.\n"

        # --- PONTO ÚNICO DE LIMPEZA ---
        df_limpo = df_features.dropna()
        log_text += f"ℹ️ Para o processo, {len(df_features) - len(df_limpo)} linhas com dados ausentes foram removidas.\n"

        datas_split = {'treino_inicio': df.index.min(), 'treino_fim': data_final_treino_dt, 'teste_inicio': data_final_treino_dt + pd.Timedelta(days=1), 'teste_fim': df.index.max()}
        log_text += f"Divisão: Treino até {datas_split['treino_fim'].date()}, Teste a partir de {datas_split['teste_inicio'].date()}\n"
        
        params = {'n_estimators': 500, 'learning_rate': 0.05, 'random_state': 42, 'early_stopping_rounds': 20}
        
        # Ordena a lista para exibição na UI
        features_para_md = sorted(features_finais_selecionadas)
        features_selecionadas_md = "### Features Selecionadas (Manual):\n" + "\n".join([f"- `{f}`" for f in features_para_md])

        log_treino, *results = treinar_e_avaliar(df_limpo, features_finais_selecionadas, target, datas_split, params, progress)
        log_text += log_treino

        return log_text, *results, features_selecionadas_md
        
    except Exception as e:
        log_text = f"❌ Ocorreu um erro: {e}\n\n--- Detalhes ---\n{traceback.format_exc()}"
        return log_text, None, None, None, None, None, None, None, None

# --- FUNÇÕES DA INTERFACE GRADIO ---

def processar_arquivo(arquivo):
    if arquivo is None: return [gr.update(visible=False)] * 9
    try:
        df = pd.read_csv(arquivo.name) if arquivo.name.endswith('.csv') else pd.read_excel(arquivo.name)
        df_original_cols = df.columns.tolist() # Guarda os nomes originais para a UI
        df = sanitize_columns(df) 
        
        colunas_sanitizadas = df.columns.tolist()
        
        # Tenta encontrar uma coluna de data após a sanitização
        try:
            col_data_candidata_sanitizada = [c for c in colunas_sanitizadas if df[c].astype(str).str.match(r'\d{4}-\d{2}-\d{2}').all()][0]
            # Mapeia de volta para o nome original
            col_data_candidata_original = df_original_cols[colunas_sanitizadas.index(col_data_candidata_sanitizada)]
        except (IndexError, Exception):
            col_data_candidata_original = df_original_cols[0]

        data_split_default_str = ""
        try:
            df[col_data_candidata_sanitizada] = pd.to_datetime(df[col_data_candidata_sanitizada])
            data_min, data_max = df[col_data_candidata_sanitizada].min(), df[col_data_candidata_sanitizada].max()
            data_split_default = data_min + (data_max - data_min) * 0.8
            data_split_default_str = data_split_default.strftime('%Y-%m-%d')
        except Exception:
            pass
        
        colunas_fixas_choices = df_original_cols + ['mes']
        updates = [
            gr.update(visible=True), # grupo_principal
            gr.update(choices=df_original_cols, value=col_data_candidata_original), # coluna_data_input
            gr.update(value=data_split_default_str), # data_final_treino_input
            gr.update(choices=df_original_cols), # coluna_target_input
            gr.update(choices=df_original_cols), # colunas_features_auto
            gr.update(choices=colunas_fixas_choices), # colunas_fixas_auto
            gr.update(choices=df_original_cols), # colunas_features_manual
            df_original_cols, colunas_fixas_choices # states
        ]
        return updates
    except Exception as e:
        raise gr.Error(f"Erro ao ler o arquivo: {e}")

def update_manual_lag_ui(colunas_selecionadas):
    """
    Atualiza a UI de configuração de lags no modo manual.
    Cria um grupo de configuração para cada coluna selecionada.
    """
    MAX_COLS = 15 # Define um limite para não sobrecarregar a UI
    updates = []
    
    # Gera updates para as colunas selecionadas
    for i in range(len(colunas_selecionadas)):
        if i < MAX_COLS:
            updates.append(gr.update(visible=True)) # Torna o grupo visível
            updates.append(gr.update(value=f"**{colunas_selecionadas[i]}**")) # Define o nome da coluna

    # Gera updates para esconder os grupos não utilizados
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
    
    df['mes'] = df.index.month # Gera 'mes' para que possa ser selecionada
    
    config_geracao = []
    for i, coluna in enumerate(colunas_configuradas):
        lags_selecionados = lista_de_lags[i]
        if lags_selecionados:
            # <<<<<<< MUDANÇA CRÍTICA: Usa o nome completo da coluna para o sufixo, evitando colisões.
            sufixo = f'_{coluna}'
            config_geracao.append((coluna, sufixo, lags_selecionados))

    df_features, _ = criar_features_temporais(df, config_geracao)
    
    features_disponiveis = sorted([c for c in df_features.columns if c in df.columns or c.startswith('lag_') or c == 'mes'])
    
    return gr.update(visible=True), gr.update(choices=features_disponiveis, value=features_disponiveis), features_disponiveis


# --- CONSTRUÇÃO DA INTERFACE ---
with gr.Blocks(theme=gr.themes.Soft(), title="AutoML de Séries Temporais", css=custom_css) as demo:
    gr.Markdown("# 🤖 AutoML para Séries Temporais com Feature Selection Pro")
    gr.Markdown("Faça o upload, configure o modo (automático ou manual) e rode um pipeline completo de modelagem!")
    
    # Estados para armazenar listas de choices para os botões "Selecionar Tudo"
    colunas_features_state = gr.State([])
    colunas_fixas_state = gr.State([])
    features_manuais_state = gr.State([])

    with gr.Row():
        arquivo_input = gr.File(label="Selecione seu arquivo (.csv ou .xlsx)", scale=1)
    
    with gr.Group(visible=False) as grupo_principal:
        with gr.Group():
            with gr.Row():
                coluna_data_input = gr.Dropdown(label="Coluna de data")
                data_final_treino_input = gr.Textbox(label="Data final do treino (AAAA-MM-DD)", placeholder="Ex: 2023-12-31")
                coluna_target_input = gr.Dropdown(label="Variável TARGET (a prever)")

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
            log_output = gr.Textbox(label="Log de Execução", lines=10, interactive=False)
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

    # --- Lógica dos Eventos ---
    arquivo_input.upload(
        processar_arquivo,
        [arquivo_input],
        [grupo_principal, coluna_data_input, data_final_treino_input, coluna_target_input, colunas_features_auto, colunas_fixas_auto, colunas_features_manual, colunas_features_state, colunas_fixas_state]
    )
    
    # Botões "Selecionar/Limpar"
    select_all_btn_feat_auto.click(lambda x: gr.update(value=x), colunas_features_state, colunas_features_auto)
    clear_btn_feat_auto.click(lambda: gr.update(value=[]), None, colunas_features_auto)
    select_all_btn_lags_auto.click(lambda: gr.update(value=list(range(1,13))), None, lags_auto)
    clear_btn_lags_auto.click(lambda: gr.update(value=[]), None, lags_auto)
    select_all_btn_fixas_auto.click(lambda x: gr.update(value=x), colunas_fixas_state, colunas_fixas_auto)
    clear_btn_fixas_auto.click(lambda: gr.update(value=[]), None, colunas_fixas_auto)
    select_all_btn_feat_manual.click(lambda x: gr.update(value=x), colunas_features_state, colunas_features_manual)
    clear_btn_feat_manual.click(lambda: gr.update(value=[]), None, colunas_features_manual)
    select_all_btn_manual_final.click(lambda x: gr.update(value=x), features_manuais_state, manual_features_checklist)
    clear_btn_manual_final.click(lambda: gr.update(value=[]), None, manual_features_checklist)
    
    # Lógica do Modo Manual
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
        [arquivo_input, coluna_data_input, coluna_target_input, data_final_treino_input, manual_features_checklist, simular_drop_input, colunas_features_manual] + all_lag_checkboxes,
        [log_output, dataframe_output, plot_pred_output, metricas_output, plot_imp_output, plot_shap_summary_output, plot_shap_force_output, download_output, features_selecionadas_output]
    )
    
    # Lógica do Modo Automático
    run_button_auto.click(
        executar_pipeline_auto,
        [arquivo_input, coluna_data_input, coluna_target_input, colunas_features_auto, colunas_fixas_auto, data_final_treino_input, lags_auto],
        [log_output, dataframe_output, plot_pred_output, metricas_output, plot_imp_output, plot_shap_summary_output, plot_shap_force_output, download_output, features_selecionadas_output]
    )

if __name__ == "__main__":
    demo.launch(debug=True)

