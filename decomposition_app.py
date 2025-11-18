import warnings
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import gradio as gr
import io
import tempfile
import os
from datetime import datetime
from statsmodels.tsa.seasonal import seasonal_decompose

# --- Configurações Iniciais ---
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
plt.rcParams['figure.constrained_layout.use'] = True
sns.set_theme(style="whitegrid")

# --- CSS Personalizado (o mesmo que você gosta) ---
custom_css = """
.orange-button {
    background: linear-gradient(to right, #007BFF, #0056b3) !important;
    color: white !important;
    border: none !important;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    transition: transform 0.2s, box-shadow 0.2s;
}
.orange-button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
    background: linear-gradient(to right, #0069d9, #004085) !important;
}
"""

# --- FUNÇÕES CORE (LÓGICA DA ANÁLISE) ---

def setup_analise(arquivo, eixo_x, data_inicio, data_fim, periodo_sazonal, modelo_decomp):
    """
    Função auxiliar que carrega, limpa e filtra os dados.
    É usada por ambas as funções de plotagem.
    """
    # 1. Validações Iniciais
    if arquivo is None:
        raise gr.Error("Por favor, faça o upload de um arquivo primeiro.")
    if not eixo_x:
        raise gr.Error("Selecione uma coluna para o Eixo X (data).")
    if not periodo_sazonal:
        raise gr.Error("Defina um Período Sazonal (ex: 7 para semanal, 12 para mensal).")
    
    try:
        periodo_int = int(periodo_sazonal)
        if periodo_int <= 1:
            raise ValueError()
    except ValueError:
        raise gr.Error("O Período Sazonal deve ser um número inteiro maior que 1.")

    # 2. Carregar e Limpar Dados
    df = pd.read_csv(arquivo.name) if arquivo.name.endswith('.csv') else pd.read_excel(arquivo.name)
    df_original_cols = df.copy()
    df.columns = [str(col).strip().lower().replace(' ', '_') for col in df.columns]
    column_map = {str(orig_col).strip().lower().replace(' ', '_'): orig_col for orig_col in df_original_cols.columns}
    eixo_x_clean = eixo_x.strip().lower().replace(' ', '_')

    # 3. Processar Eixo X (Data)
    df[eixo_x_clean] = pd.to_datetime(df[eixo_x_clean], errors='coerce')
    df.dropna(subset=[eixo_x_clean], inplace=True)
    df.set_index(eixo_x_clean, inplace=True)
    df.sort_index(inplace=True)

    # 4. Filtrar por Período
    df_filtrado = df.copy()
    if data_inicio:
        df_filtrado = df_filtrado.loc[df_filtrado.index >= pd.to_datetime(data_inicio)]
    if data_fim:
        df_filtrado = df_filtrado.loc[df_filtrado.index <= pd.to_datetime(data_fim)]

    if df_filtrado.empty:
        raise gr.Error("Nenhum dado encontrado para o período selecionado.")

    # 5. Definir Modelo
    model_str = 'additive' if modelo_decomp == "Aditivo" else 'multiplicative'
    
    return df_filtrado, periodo_int, model_str, column_map

def gerar_decomposicao_completa(arquivo, eixo_x, data_inicio, data_fim, periodo_sazonal, 
                                modelo_decomp, feature_unica, componentes_plotar, 
                                titulo_input, label_x_input, label_y_input, # <--- ADICIONADOS
                                progress=gr.Progress(track_tqdm=True)):
    """
    Gera a decomposição completa (Observado, Tendência, Sazonalidade, Resíduos)
    para UMA ÚNICA série temporal.
    """
    try:
        progress(0, desc="🚀 Preparando a análise...")
        df_filtrado, periodo_int, model_str, column_map = setup_analise(
            arquivo, eixo_x, data_inicio, data_fim, periodo_sazonal, modelo_decomp
        )

        if not feature_unica:
            raise gr.Error("Selecione uma 'Série para decompor' na Aba 1.")
        if not componentes_plotar:
            raise gr.Error("Selecione pelo menos um componente para plotar (ex: Tendência).")

        feature_clean = feature_unica.strip().lower().replace(' ', '_')
        feature_original = column_map.get(feature_clean, feature_clean)

        progress(0.3, desc=f"📈 Decompondo '{feature_original}'...")
        
        serie = df_filtrado[feature_clean].dropna()
        if len(serie) < 2 * periodo_int:
            raise gr.Error(f"Série muito curta para o período {periodo_int}. "
                           f"A série precisa ter pelo menos {2 * periodo_int} pontos. "
                           f"Série atual tem {len(serie)} pontos.")

        result = seasonal_decompose(serie, model=model_str, period=periodo_int)

        progress(0.6, desc="🎨 Desenhando os componentes...")
        
        num_plots = len(componentes_plotar)
        fig, axes = plt.subplots(num_plots, 1, figsize=(14, 3 * num_plots), sharex=True)
        
        # Garante que 'axes' seja sempre um array, mesmo com num_plots=1
        if num_plots == 1:
            axes = [axes]

        plot_map = {
            "Observado": (result.observed, "Observado", "blue"),
            "Tendência": (result.trend, "Tendência", "green"),
            "Sazonalidade": (result.seasonal, "Sazonalidade", "orange"),
            "Resíduos": (result.resid, "Resíduos", "red")
        }

        i = 0
        for componente in ["Observado", "Tendência", "Sazonalidade", "Resíduos"]:
            if componente in componentes_plotar:
                data, titulo, cor = plot_map[componente]
                data.plot(ax=axes[i], title=titulo, color=cor, legend=False)
                axes[i].set_ylabel(titulo)
                i += 1
        
        # --- LÓGICA DO TÍTULO E EIXOS CUSTOMIZADOS ---
        titulo_final = titulo_input if titulo_input else f'Decomposição {modelo_decomp} de: {feature_original} (Período={periodo_int})'
        fig.suptitle(titulo_final, fontsize=20, weight='bold', y=1.03)
        plt.xlabel(label_x_input if label_x_input else "Data")
        # label_y_input é ignorado aqui, pois os subplots têm seus próprios labels
        # --- FIM DA CUSTOMIZAÇÃO ---
        
        fig.autofmt_xdate()

        progress(0.9, desc="💾 Salvando o resultado...")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
            fig.savefig(tmpfile.name, format='png', dpi=120, bbox_inches='tight')
            caminho_download = tmpfile.name

        plt.close(fig)
        return fig, gr.update(value=caminho_download, visible=True)

    except Exception as e:
        plt.close('all')
        raise gr.Error(f"Oops! Aconteceu um erro: {str(e)}")


def gerar_comparativo_tendencias(arquivo, eixo_x, data_inicio, data_fim, periodo_sazonal, 
                                 modelo_decomp, features_multi, 
                                 titulo_input, label_x_input, label_y_input, # <--- ADICIONADOS
                                 progress=gr.Progress(track_tqdm=True)):
    """
    Gera um gráfico único comparando a TENDÊNCIA de VÁRIAS séries temporais.
    """
    try:
        progress(0, desc="🚀 Preparando a análise...")
        df_filtrado, periodo_int, model_str, column_map = setup_analise(
            arquivo, eixo_x, data_inicio, data_fim, periodo_sazonal, modelo_decomp
        )

        if not features_multi:
            raise gr.Error("Selecione pelo menos uma série para 'Comparar Tendências' na Aba 2.")

        features_clean = [f.strip().lower().replace(' ', '_') for f in features_multi]

        progress(0.3, desc="📈 Decompondo múltiplas séries...")
        
        fig, ax = plt.subplots(figsize=(14, 7))
        
        cores = sns.color_palette("husl", len(features_clean))

        for i, feature_clean in enumerate(features_clean):
            feature_original = column_map.get(feature_clean, feature_clean)
            serie = df_filtrado[feature_clean].dropna()
            
            if len(serie) < 2 * periodo_int:
                print(f"Aviso: Série '{feature_original}' ignorada (muito curta para o período {periodo_int}).")
                continue
                
            progress(0.3 + (i / len(features_clean)) * 0.5, 
                     desc=f"Decompondo '{feature_original}'...")
            
            result = seasonal_decompose(serie, model=model_str, period=periodo_int)
            
            # Label da legenda modificado (sem "Tendência - ")
            result.trend.plot(ax=ax, label=f'{feature_original}', 
                              color=cores[i], linewidth=2.5)

        # --- LÓGICA DO TÍTULO E EIXOS CUSTOMIZADOS ---
        titulo_final = titulo_input if titulo_input else f'Comparativo de Tendências ({modelo_decomp} | Período={periodo_int})'
        ax.set_title(titulo_final, fontsize=20, weight='bold')
        ax.set_xlabel(label_x_input if label_x_input else "Data")
        ax.set_ylabel(label_y_input if label_y_input else "Valor da Tendência")
        # --- FIM DA CUSTOMIZAÇÃO ---

        ax.legend(title="Séries", bbox_to_anchor=(1.04, 1), loc="upper left")
        fig.autofmt_xdate()

        progress(0.9, desc="💾 Salvando o resultado...")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
            fig.savefig(tmpfile.name, format='png', dpi=120, bbox_inches='tight')
            caminho_download = tmpfile.name

        plt.close(fig)
        return fig, gr.update(value=caminho_download, visible=True)

    except Exception as e:
        plt.close('all')
        raise gr.Error(f"Oops! Aconteceu um erro: {str(e)}")


# --- FUNÇÕES AUXILIARES DA INTERFACE ---

def processar_arquivo(arquivo):
    """
    Lê o arquivo, extrai colunas e o DataFrame para um estado.
    """
    if arquivo is None:
        return gr.update(visible=False), None, [], gr.update(choices=[], value=None)
    try:
        df = pd.read_csv(arquivo.name) if arquivo.name.endswith('.csv') else pd.read_excel(arquivo.name)
        colunas = df.columns.tolist()

        coluna_data_provavel = None
        for col in colunas:
            if 'data' in str(col).lower() or 'date' in str(col).lower() or 'time' in str(col).lower():
                try:
                    pd.to_datetime(df[col].dropna().iloc[:10], errors='raise')
                    coluna_data_provavel = col
                    break
                except (ValueError, TypeError, IndexError): continue

        return gr.update(visible=True), df, colunas, gr.update(choices=colunas, value=coluna_data_provavel)
    except Exception as e:
        raise gr.Error(f"Erro ao ler o arquivo: {e}")

def atualizar_opcoes(df, todas_colunas, eixo_x_selecionado):
    """
    Atualiza as opções de features e as listas de datas disponíveis.
    """
    if not eixo_x_selecionado:
        opcoes_features = todas_colunas
        updates_datas = (gr.update(choices=[], value=None), gr.update(choices=[], value=None))
    else:
        # Tenta pegar apenas colunas numéricas para features
        try:
            opcoes_features_numericas = df.select_dtypes(include=np.number).columns.tolist()
            opcoes_features = [col for col in opcoes_features_numericas if col != eixo_x_selecionado]
        except: # Fallback se o df não estiver pronto
             opcoes_features = [col for col in todas_colunas if col != eixo_x_selecionado]
        
        try:
            coluna_data_clean = str(eixo_x_selecionado).strip().lower().replace(' ', '_')
            temp_df = df.copy()
            temp_df.columns = [str(col).strip().lower().replace(' ', '_') for col in temp_df.columns]

            datas = pd.to_datetime(temp_df[coluna_data_clean], errors='coerce').dropna().dt.strftime('%Y-%m-%d').unique()
            datas_sorted = sorted(list(datas))

            updates_datas = (gr.update(choices=datas_sorted, value=datas_sorted[0] if datas_sorted else None),
                             gr.update(choices=datas_sorted, value=datas_sorted[-1] if datas_sorted else None))
        except Exception as e:
            print(f"Erro ao atualizar datas: {e}")
            updates_datas = (gr.update(choices=[], value=None), gr.update(choices=[], value=None))

    return (
        gr.update(choices=opcoes_features, value=None), # feature_unica_input
        gr.update(choices=opcoes_features, value=[]),   # features_multi_input
        updates_datas[0], # data_inicio
        updates_datas[1]  # data_fim
    )

# --- CONSTRUÇÃO DA INTERFACE (GRADIO) ---

with gr.Blocks(theme=gr.themes.Soft(), title="Decomposição de Séries Temporais", css=custom_css) as demo:
    
    gr.Markdown("# 📈 Decomposição de Séries Temporais")
    gr.Markdown("Faça o upload do seu dataset, escolha a série e os parâmetros para visualizar a Tendência, Sazonalidade e Resíduos.")

    # Estados para guardar os dados
    df_state = gr.State()
    todas_as_colunas_state = gr.State([])

    with gr.Row():
        arquivo_input = gr.File(label="Selecione seu arquivo (.csv ou .xlsx)")

    with gr.Group(visible=False) as grupo_principal:
        gr.Markdown("### 1. Configurações Principais da Análise")
        with gr.Row():
            eixo_x_input = gr.Dropdown(label="Eixo X (Coluna de Tempo/Período)")
            periodo_sazonal_input = gr.Number(label="Período Sazonal", 
                                              info="Ex: 7 (semanal), 12 (mensal), 52 (anual)", 
                                              step=1)
        with gr.Row():
            modelo_decomp_input = gr.Radio(choices=["Aditivo", "Multiplicativo"], 
                                           value="Aditivo", 
                                           label="Modelo de Decomposição")
        
        with gr.Accordion("🗓️ Filtrar por Período (Opcional)", open=False):
            with gr.Row():
                data_inicio_input = gr.Dropdown(label="Data de Início", interactive=True)
                data_fim_input = gr.Dropdown(label="Data Final", interactive=True)
        
        # --- NOVOS INPUTS DE CUSTOMIZAÇÃO ---
        with gr.Accordion("🎨 Customização de Títulos e Eixos (Opcional)", open=False):
            titulo_input = gr.Textbox(label="Título Personalizado", placeholder="Deixe em branco para o padrão")
            label_x_input = gr.Textbox(label="Legenda Eixo X", placeholder="Deixe em branco para 'Data'")
            label_y_input = gr.Textbox(label="Legenda Eixo Y", 
                                      placeholder="Deixe em branco para o padrão",
                                      info="Usado principalmente no gráfico de 'Comparar Tendências'.")
        # --- FIM DOS NOVOS INPUTS ---

        gr.Markdown("### 2. Escolha o Tipo de Gráfico")
        with gr.Tabs():
            with gr.TabItem("Decomposição Completa (Uma Série)"):
                gr.Markdown("Use esta aba para ver a decomposição completa (T-S-R) de *uma* série por vez.")
                feature_unica_input = gr.Dropdown(label="Escolha a série para decompor")
                componentes_plotar_input = gr.CheckboxGroup(
                    choices=["Observado", "Tendência", "Sazonalidade", "Resíduos"],
                    value=["Observado", "Tendência", "Sazonalidade", "Resíduos"],
                    label="Quais componentes plotar?"
                )
                run_button_unica = gr.Button("Gerar Decomposição Completa", elem_classes=["orange-button"])

            with gr.TabItem("Comparar Tendências (Várias Séries)"):
                gr.Markdown("Use esta aba para plotar *apenas a tendência* de *várias* séries no mesmo gráfico.")
                features_multi_input = gr.CheckboxGroup(label="Escolha as séries para comparar")
                run_button_multi = gr.Button("Gerar Comparativo de Tendências", elem_classes=["orange-button"])

    gr.Markdown("### 3. Resultados")
    with gr.Tabs() as results_tabs:
        with gr.TabItem("📈 Gráfico Gerado"):
            plot_output = gr.Plot(label="Seu Gráfico")
        with gr.TabItem("💾 Download"):
            download_output = gr.File(label="Baixar Gráfico (.png)", visible=False)

    # --- LÓGICA DOS EVENTOS (CONECTANDO OS BOTÕES) ---

    arquivo_input.upload(
        processar_arquivo,
        inputs=[arquivo_input],
        outputs=[grupo_principal, df_state, todas_as_colunas_state, eixo_x_input]
    )

    eixo_x_input.change(
        atualizar_opcoes,
        inputs=[df_state, todas_as_colunas_state, eixo_x_input],
        outputs=[feature_unica_input, features_multi_input, data_inicio_input, data_fim_input]
    )

    run_button_unica.click(
        gerar_decomposicao_completa,
        inputs=[arquivo_input, eixo_x_input, data_inicio_input, data_fim_input, 
                periodo_sazonal_input, modelo_decomp_input, feature_unica_input, 
                componentes_plotar_input,
                titulo_input, label_x_input, label_y_input], # <--- ADICIONADOS
        outputs=[plot_output, download_output]
    )

    run_button_multi.click(
        gerar_comparativo_tendencias,
        inputs=[arquivo_input, eixo_x_input, data_inicio_input, data_fim_input, 
                periodo_sazonal_input, modelo_decomp_input, features_multi_input,
                titulo_input, label_x_input, label_y_input], # <--- ADICIONADOS
        outputs=[plot_output, download_output]
    )

if __name__ == "__main__":
    demo.launch(debug=True)

