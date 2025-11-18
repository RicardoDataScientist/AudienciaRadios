import warnings
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import gradio as gr
import io
import zipfile
import tempfile
import os

# --- Configurações Iniciais ---
# Ignora avisos para uma interface mais limpa
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# --- CSS Personalizado para deixar o app mais bonito ---
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
.dark-orange-button {
    background: linear-gradient(to right, #D2691E, #CD853F) !important; /* Laranja mais escuro para contraste */
    color: white !important;
    border: none !important;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    transition: transform 0.2s, box-shadow 0.2s;
}
.dark-orange-button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
}
"""

# --- FUNÇÕES CORE (LÓGICA DA ANÁLISE) ---

def salvar_resultados_zip(target, df_metricas, figs_dict):
    """
    Salva as métricas e os gráficos gerados em um único arquivo .zip com o nome do target.
    """
    temp_dir = tempfile.mkdtemp()
    zip_path = os.path.join(temp_dir, f"Resultados_{target}.zip")

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'a', zipfile.ZIP_DEFLATED, False) as zip_file:
        if not df_metricas.empty:
            df_metricas_csv = df_metricas.copy()
            if 'Valor' in df_metricas_csv.columns:
                 df_metricas_csv['Valor'] = pd.to_numeric(df_metricas_csv['Valor'].astype(str).str.replace(',', ''))
            zip_file.writestr(f"metricas_{target}.csv", df_metricas_csv.to_csv(index=False).encode('utf-8'))

        for name, fig in figs_dict.items():
            if fig:
                buf = io.BytesIO()
                fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
                zip_file.writestr(f"grafico_{name}_{target}.png", buf.getvalue())
    
    with open(zip_path, 'wb') as f:
        f.write(zip_buffer.getvalue())

    return gr.update(value=zip_path, visible=True)


def executar_analise_regressao(arquivo, target_col, feature_cols, test_size_slider, n_estimators, max_depth, learning_rate, progress=gr.Progress(track_tqdm=True)):
    """
    Executa um pipeline completo de análise de regressão, com gráficos visualmente aprimorados.
    """
    if not target_col or not feature_cols:
        raise gr.Error("Por favor, selecione a variável TARGET e pelo menos uma FEATURE para continuar.")

    try:
        progress(0, desc="🎨 Preparando a paleta de cores e carregando dados...")
        
        # Estilo visual aprimorado para os gráficos
        sns.set_theme(style="whitegrid") # REMOVIDA A PALETA GLOBAL
        sns.set_context("talk", font_scale=0.8)
        
        df = pd.read_csv(arquivo.name) if arquivo.name.endswith('.csv') else pd.read_excel(arquivo.name)
        original_columns = df.columns.tolist()
        df.columns = [col.strip().lower().replace(' ', '_') for col in original_columns]

        column_map = dict(zip(original_columns, df.columns))
        target_col_clean = column_map[target_col]
        feature_cols_clean = [column_map[f] for f in feature_cols]

        if target_col_clean in feature_cols_clean:
            feature_cols_clean.remove(target_col_clean)

        if not feature_cols_clean:
            raise gr.Error("Nenhuma feature foi selecionada. Por favor, selecione pelo menos uma.")

        df_analise = df[[target_col_clean] + feature_cols_clean].dropna()
        if df_analise.empty:
             raise gr.Error("O DataFrame ficou vazio após remover valores ausentes. Verifique seus dados.")

        numeric_features = df_analise[feature_cols_clean].select_dtypes(include=np.number).columns.tolist()
        if not numeric_features:
            raise gr.Error("Nenhuma das features selecionadas é numérica.")
        
        feature_cols_clean = numeric_features
        df_analise = df_analise[[target_col_clean] + feature_cols_clean]

        progress(0.2, desc="📊 Gerando Heatmap de Correlação...")
        plt.close('all')
        corr_matrix = df_analise.corr()
        fig_corr, ax_corr = plt.subplots(figsize=(12, 10))
        # MELHORIA: Mapa de calor com cor original (divergente) e números maiores/negrito
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax_corr, linewidths=.5, linecolor='black', annot_kws={"size": 18, "weight": "bold"})
        ax_corr.set_title(f'Heatmap de Correlação (Alvo: {target_col})', fontsize=18, weight='bold')
        fig_corr.tight_layout()

        progress(0.4, desc="📈 Gerando gráficos de dispersão...")
        temp_abs_corr_col = 'abs_corr'
        corr_matrix[temp_abs_corr_col] = corr_matrix[target_col_clean].abs()
        top_corr_features = corr_matrix.sort_values(by=temp_abs_corr_col, ascending=False).index[1:6]
        corr_matrix.drop(columns=[temp_abs_corr_col], inplace=True)

        num_plots = len(top_corr_features)
        if num_plots > 0:
            fig_scatter, axes_scatter = plt.subplots(1, num_plots, figsize=(6 * num_plots, 5))
            if num_plots == 1: axes_scatter = [axes_scatter]
            fig_scatter.suptitle(f'Features Mais Correlacionadas vs. {target_col.upper()}', fontsize=20, weight='bold')
            for i, feature in enumerate(top_corr_features):
                original_feature_name = next((k for k, v in column_map.items() if v == feature), feature)
                # MELHORIA: Pontos azuis (padrão) e linha de regressão vermelha
                sns.regplot(
                    data=df_analise, x=feature, y=target_col_clean, ax=axes_scatter[i],
                    scatter_kws={'alpha': 0.7, 'edgecolor': 'w', 's': 60, 'linewidths': 0.5},
                    line_kws={'color': '#E74C3C', 'linewidth': 3, 'path_effects': [path_effects.SimpleLineShadow(shadow_color="black", alpha=0.3, offset=(1,-1)), path_effects.Normal()]}
                )
                axes_scatter[i].set_title(f'{original_feature_name}\n(Corr: {corr_matrix.loc[feature, target_col_clean]:.2f})', fontsize=14)
                axes_scatter[i].set_xlabel(original_feature_name, fontsize=12)
                axes_scatter[i].set_ylabel(target_col, fontsize=12)
            fig_scatter.tight_layout(rect=[0, 0.03, 1, 0.95])
        else:
            fig_scatter = None

        progress(0.6, desc="🤖 Treinando o modelo XGBoost...")
        X = df_analise[feature_cols_clean]
        y = df_analise[target_col_clean]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_slider, random_state=42)
        model = xgb.XGBRegressor(
            n_estimators=int(n_estimators), max_depth=int(max_depth), learning_rate=learning_rate,
            random_state=42, objective='reg:squarederror'
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        progress(0.8, desc="📝 Calculando métricas de performance...")
        metricas = {
            'Métrica': ['R² (R-squared)', 'MAE (Mean Absolute Error)', 'RMSE (Root Mean Squared Error)'],
            'Valor': [r2_score(y_test, y_pred), mean_absolute_error(y_test, y_pred), np.sqrt(mean_squared_error(y_test, y_pred))]
        }
        df_metricas = pd.DataFrame(metricas)
        
        progress(0.9, desc="✨ Gerando gráficos de diagnóstico...")
        # MELHORIA: Gráfico Real vs. Previsto com gradiente de cor para os resíduos
        fig_pred, ax_pred = plt.subplots(figsize=(10, 8))
        residuos = y_test - y_pred
        scatter_plot = ax_pred.scatter(y_test, y_pred, c=residuos, cmap='viridis_r', alpha=0.8, edgecolors='k', s=80)
        cbar = fig_pred.colorbar(scatter_plot)
        cbar.set_label('Resíduos (Real - Previsto)', rotation=270, labelpad=20, fontsize=12)
        ax_pred.plot([y.min(), y.max()], [y.min(), y.max()], '--', color='#D32F2F', linewidth=2.5, label="Linha Perfeita")
        ax_pred.set_xlabel(f"Valores Reais ({target_col})", fontsize=14)
        ax_pred.set_ylabel(f"Valores Previstos ({target_col})", fontsize=14)
        ax_pred.set_title("Real vs. Previsto", fontsize=18, weight='bold')
        ax_pred.legend(); ax_pred.grid(True, linestyle='--', alpha=0.6)

        original_feature_names = [next((k for k, v in column_map.items() if v == f), f) for f in feature_cols_clean]
        importances = pd.Series(model.feature_importances_, index=original_feature_names).sort_values(ascending=False)
        fig_imp, ax_imp = plt.subplots(figsize=(12, 8))
        # MELHORIA: Barras com bordas e um espaçamento para melhor leitura
        sns.barplot(x=importances.values, y=importances.index, ax=ax_imp, palette="plasma", edgecolor='black', linewidth=1.5)
        ax_imp.set_title('Importância das Features (XGBoost)', fontsize=18, weight='bold')
        ax_imp.set_xlabel('Importância Relativa', fontsize=14)
        ax_imp.tick_params(axis='y', labelsize=12)
        fig_imp.tight_layout()

        figs_para_salvar = {
            "correlacao": fig_corr, "dispersao": fig_scatter,
            "real_vs_previsto": fig_pred, "importancia_features": fig_imp
        }
        zip_file_path_update = salvar_resultados_zip(target_col.replace(" ", "_"), df_metricas, figs_para_salvar)
        
        df_metricas['Valor'] = df_metricas['Valor'].map('{:,.4f}'.format)
        return df_metricas, fig_corr, fig_scatter, fig_pred, fig_imp, zip_file_path_update

    except Exception as e:
        raise gr.Error(f"Ocorreu um erro: {str(e)}")

# --- FUNÇÕES AUXILIARES DA INTERFACE ---

def processar_arquivo(arquivo):
    if arquivo is None:
        return gr.update(visible=False), gr.update(choices=[], value=None), gr.update(choices=[], value=None), []
    try:
        df = pd.read_csv(arquivo.name) if arquivo.name.endswith('.csv') else pd.read_excel(arquivo.name)
        colunas = df.columns.tolist()
        return gr.update(visible=True), gr.update(choices=colunas), gr.update(choices=colunas, value=colunas), colunas
    except Exception as e:
        raise gr.Error(f"Erro ao ler o arquivo: {e}")

def atualizar_opcoes_feature(todas_as_colunas, target_selecionado):
    opcoes = [col for col in todas_as_colunas if col != target_selecionado]
    # Seleciona todas as features por padrão, exceto o target
    return gr.update(choices=opcoes, value=opcoes), opcoes

# --- CONSTRUÇÃO DA INTERFACE (GRADIO) ---

with gr.Blocks(theme=gr.themes.Soft(), title="Analisador de Regressão", css=custom_css) as demo:
    gr.Markdown("# 🔍 Analisador de Regressão com XGBoost")
    gr.Markdown("Faça o upload do seu dataset, configure a análise e treine um modelo para extrair insights valiosos!")

    todas_as_colunas_state = gr.State([])
    features_validas_state = gr.State([])

    with gr.Row():
        arquivo_input = gr.File(label="Selecione seu arquivo (.csv ou .xlsx)")

    with gr.Group(visible=False) as grupo_principal:
        with gr.Accordion("1. Configurações da Análise", open=True):
            with gr.Row():
                target_input = gr.Dropdown(label="Selecione a Variável TARGET (a prever)")
                features_input = gr.CheckboxGroup(label="Selecione as FEATURES (variáveis explicativas)")
            with gr.Row():
                gr.Markdown("") # Espaçador
                with gr.Column(scale=2):
                    select_all_btn = gr.Button("Selecionar Todas", elem_classes=["orange-button"])
                    clear_all_btn = gr.Button("Limpar Seleção", elem_classes=["dark-orange-button"])
        test_size_input = gr.Slider(label="Percentual para Teste", minimum=0.1, maximum=0.5, value=0.2, step=0.05)

        with gr.Accordion("2. Parâmetros do Modelo (Ajuste Fino)", open=True):
             with gr.Row():
                n_estimators_input = gr.Slider(label="Nº de Árvores (n_estimators)", minimum=50, maximum=2000, value=1000, step=50)
                max_depth_input = gr.Slider(label="Profundidade Máx. (max_depth)", minimum=2, maximum=10, value=5, step=1)
                learning_rate_input = gr.Slider(label="Taxa de Aprendizado (learning_rate)", minimum=0.01, maximum=0.3, value=0.1, step=0.01)

        run_button = gr.Button("🚀 Executar Análise!", elem_classes=["orange-button"])

    with gr.Tabs() as results_tabs:
        with gr.TabItem("📊 Métricas e Correlação"):
            with gr.Row():
                metricas_output = gr.DataFrame(label="Métricas de Performance do Modelo", scale=1)
                plot_corr_output = gr.Plot(label="Heatmap de Correlação", scale=2)

        with gr.TabItem("📈 Análise Visual das Features"):
             plot_scatter_output = gr.Plot(label="Dispersão das Features Mais Correlacionadas")

        with gr.TabItem("🧠 Diagnóstico do Modelo"):
            with gr.Row():
                plot_pred_output = gr.Plot(label="Diagnóstico: Real vs. Previsto")
                plot_imp_output = gr.Plot(label="Importância das Features")
        
        with gr.TabItem("💾 Download"):
            download_output = gr.File(label="Baixar Resultados (.zip)", visible=False)

    # --- LÓGICA DOS EVENTOS (CONECTANDO OS BOTÕES) ---
    arquivo_input.upload(
        processar_arquivo,
        [arquivo_input],
        [grupo_principal, target_input, features_input, todas_as_colunas_state]
    )

    target_input.change(
        atualizar_opcoes_feature,
        [todas_as_colunas_state, target_input],
        [features_input, features_validas_state]
    )
    
    select_all_btn.click(lambda choices: gr.update(value=choices), inputs=[features_validas_state], outputs=[features_input])
    clear_all_btn.click(lambda: gr.update(value=[]), None, features_input)

    run_button.click(
        executar_analise_regressao,
        inputs=[arquivo_input, target_input, features_input, test_size_input, n_estimators_input, max_depth_input, learning_rate_input],
        outputs=[metricas_output, plot_corr_output, plot_scatter_output, plot_pred_output, plot_imp_output, download_output]
    )

if __name__ == "__main__":
    demo.launch(debug=True)
