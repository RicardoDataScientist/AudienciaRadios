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

# --- Configurações Iniciais ---
# Ignora avisos para uma interface mais limpa
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
plt.rcParams['figure.constrained_layout.use'] = True


# --- CSS Personalizado para deixar o app mais bonito (reaproveitado do seu app anterior) ---
custom_css = """
.orange-button {
    background: linear-gradient(to right, #007BFF, #0056b3) !important; /* Azul moderno */
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
.dark-orange-button {
    background: linear-gradient(to right, #6c757d, #5a6268) !important; /* Cinza para ações secundárias */
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

# --- Função Auxiliar para Cores ---
def parse_matplotlib_color(color_string):
    """
    Converte uma string de cor no formato 'rgba(R,G,B,A)' (comum no Gradio)
    para uma tupla RGBA com floats entre 0 e 1, que o Matplotlib entende.
    """
    if isinstance(color_string, str) and color_string.startswith('rgba'):
        try:
            parts = color_string.strip().replace('rgba(', '').replace(')', '').split(',')
            r, g, b, a = [float(p.strip()) for p in parts]
            if r > 1.0 or g > 1.0 or b > 1.0: # Normaliza se estiver no formato 0-255
                r /= 255.0
                g /= 255.0
                b /= 255.0
            return (r, g, b, a)
        except (ValueError, IndexError):
            return None # Retorna None se não conseguir converter
    return color_string # Retorna a string original se não for 'rgba' (ex: #FFFFFF)


# --- FUNÇÕES CORE (LÓGICA DA GERAÇÃO DE GRÁFICOS) ---

def gerar_grafico(arquivo, eixo_x, features_y, data_inicio, data_fim, agrupamento,
                  titulo, tamanho, label_x, label_y, cor_fundo,
                  cor_fonte_titulo, cor_fonte_eixos, cor_fonte_legenda, efeito_sombra,
                  ajuste_eixo_y,
                  *cores, progress=gr.Progress(track_tqdm=True)):
    """
    Função principal que carrega os dados, filtra, agrupa pelo período e gera o gráfico de linhas.
    """
    # Validação inicial
    if arquivo is None:
        raise gr.Error("Por favor, faça o upload de um arquivo primeiro.")
    if not eixo_x:
        raise gr.Error("Selecione uma coluna para o Eixo X (geralmente uma data).")
    if not features_y:
        raise gr.Error("Selecione pelo menos uma feature para o Eixo Y.")

    try:
        progress(0, desc="🚀 Começando a análise...")

        df = pd.read_csv(arquivo.name) if arquivo.name.endswith('.csv') else pd.read_excel(arquivo.name)
        df_original_cols = df.copy()
        df.columns = [str(col).strip().lower().replace(' ', '_') for col in df.columns]

        column_map = {str(orig_col).strip().lower().replace(' ', '_'): orig_col for orig_col in df_original_cols.columns}

        eixo_x_clean = eixo_x.strip().lower().replace(' ', '_')
        features_y_clean = [f.strip().lower().replace(' ', '_') for f in features_y]

        progress(0.2, desc="📅 Processando a coluna de tempo...")
        df[eixo_x_clean] = pd.to_datetime(df[eixo_x_clean], errors='coerce')
        df.dropna(subset=[eixo_x_clean], inplace=True)

        df_filtrado = df.copy()
        if data_inicio:
            df_filtrado = df_filtrado[df_filtrado[eixo_x_clean] >= pd.to_datetime(data_inicio)]
        if data_fim:
            df_filtrado = df_filtrado[df_filtrado[eixo_x_clean] <= pd.to_datetime(data_fim)]

        if df_filtrado.empty:
            raise gr.Error("Nenhum dado encontrado para o período selecionado. Verifique as datas.")

        # --- NOVA SEÇÃO: AGRUPAMENTO DOS DADOS ---
        progress(0.4, desc=f"🔄 Agrupando dados ({agrupamento})...")
        df_plot = pd.DataFrame()
        if agrupamento != "Diário":
            df_filtrado.set_index(eixo_x_clean, inplace=True)
            
            # Garante que apenas features numéricas sejam usadas na agregação
            features_numericas = [f for f in features_y_clean if pd.api.types.is_numeric_dtype(df_filtrado[f])]
            if not features_numericas:
                 raise gr.Error("Nenhuma das features selecionadas é numérica. O agrupamento por média só funciona com números.")
            
            # Avisa sobre colunas ignoradas
            for f in features_y_clean:
                if f not in features_numericas:
                    print(f"Aviso: A coluna '{column_map.get(f, f)}' não é numérica e será ignorada no agrupamento.")

            if agrupamento == "Mensal":
                df_plot = df_filtrado[features_numericas].resample('M').mean()
            elif agrupamento == "Anual":
                df_plot = df_filtrado[features_numericas].resample('A').mean()
            
            df_plot.reset_index(inplace=True)
        else:
            df_plot = df_filtrado # Usa os dados diários como estão

        if df_plot.empty:
            raise gr.Error("Nenhum dado restou após o agrupamento. Verifique o período selecionado.")


        progress(0.5, desc="🎨 Desenhando o gráfico com Seaborn...")
        sns.set_theme(style="whitegrid")

        try:
            largura, altura = map(float, tamanho.replace(' ', '').split(','))
            tamanho_fig = (largura, altura)
        except:
            tamanho_fig = (16, 8) # Valor padrão em caso de erro

        fig, ax = plt.subplots(figsize=tamanho_fig)

        # --- Aplicando Customizações Avançadas ---
        parsed_cor_fundo = parse_matplotlib_color(cor_fundo)
        if parsed_cor_fundo:
            fig.patch.set_facecolor(parsed_cor_fundo)
            ax.set_facecolor(parsed_cor_fundo)

        cores_validas = [parse_matplotlib_color(c) for c in cores if c]
        mapa_de_cores = dict(zip(features_y_clean, cores_validas))

        for feature in features_y_clean:
            if feature in df_plot.columns and pd.api.types.is_numeric_dtype(df_plot[feature]):
                cor_linha = mapa_de_cores.get(feature)
                sns.lineplot(data=df_plot, x=eixo_x_clean, y=feature,
                             label=column_map.get(feature, feature), color=cor_linha,
                             linewidth=2.5, marker='o', markersize=5, ax=ax)
                if efeito_sombra:
                    ax.fill_between(df_plot[eixo_x_clean], df_plot[feature], color=cor_linha, alpha=0.2)
            else:
                print(f"Aviso: A coluna '{column_map.get(feature, feature)}' não é numérica ou não foi encontrada após o agrupamento e será ignorada.")

        # Customização de Títulos e Labels
        ax.set_title(titulo if titulo else f'Comparativo de Features vs. {eixo_x}',
                     fontsize=20, weight='bold', pad=20, color=parse_matplotlib_color(cor_fonte_titulo))
        ax.set_xlabel(label_x if label_x else eixo_x, fontsize=14, color=parse_matplotlib_color(cor_fonte_eixos))
        ax.set_ylabel(label_y if label_y else 'Valores (Média Agregada)', fontsize=14, color=parse_matplotlib_color(cor_fonte_eixos))

        # Ajuste do Eixo Y
        if ajuste_eixo_y == "Iniciar em Zero":
            ax.set_ylim(bottom=0)
        # Se for "Automático (Fit)", não faz nada, pois é o comportamento padrão.

        # Customização de Ticks e Legenda
        ax.tick_params(axis='both', colors=parse_matplotlib_color(cor_fonte_eixos))
        leg = ax.legend(title='Features', fontsize='large', title_fontsize='x-large',
                        bbox_to_anchor=(1.04, 1), loc="upper left")
        plt.setp(leg.get_texts(), color=parse_matplotlib_color(cor_fonte_legenda))
        plt.setp(leg.get_title(), color=parse_matplotlib_color(cor_fonte_legenda))

        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

        # --- Formatação de Datas (Adaptativa ao Agrupamento) ---
        fig.autofmt_xdate()
        if agrupamento == "Anual":
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        elif agrupamento == "Mensal":
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%Y'))
        else: # Diário
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=10))


        progress(0.9, desc="💾 Salvando o resultado...")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
            fig.savefig(tmpfile.name, format='png', dpi=150, bbox_inches='tight')
            caminho_download = tmpfile.name

        plt.close(fig) # Fecha a figura para liberar memória
        return fig, gr.update(value=caminho_download, visible=True)

    except Exception as e:
        plt.close('all') # Garante que figuras abertas sejam fechadas em caso de erro
        raise gr.Error(f"Oops! Aconteceu um erro: {str(e)}")


# --- FUNÇÕES AUXILIARES DA INTERFACE ---

def processar_arquivo(arquivo):
    """
    Lê o arquivo, extrai colunas e o DataFrame para um estado.
    """
    if arquivo is None:
        return gr.update(visible=False), None, [], gr.update(choices=[], value=None), gr.update(choices=[], value=None)
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

        return gr.update(visible=True), df, colunas, gr.update(choices=colunas, value=coluna_data_provavel), gr.update(choices=colunas, value=[])
    except Exception as e:
        raise gr.Error(f"Erro ao ler o arquivo: {e}")

def atualizar_features_e_datas(df, todas_colunas, eixo_x_selecionado):
    """
    Atualiza as opções de features e as listas de datas disponíveis com base no eixo X selecionado.
    """
    if not eixo_x_selecionado:
        opcoes_features = todas_colunas
        updates_datas = (gr.update(choices=[], value=None), gr.update(choices=[], value=None))
    else:
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

    return gr.update(choices=opcoes_features, value=[]), updates_datas[0], updates_datas[1]

def atualizar_seletores_de_cor(features_selecionadas, max_features):
    updates = []
    # Usando um palette mais robusto para mais cores
    cores_default = sns.color_palette("husl", len(features_selecionadas)).as_hex()
    for i in range(max_features):
        if i < len(features_selecionadas):
            feature = features_selecionadas[i]
            cor = cores_default[i % len(cores_default)]
            updates.append(gr.update(visible=True, label=f"Cor para '{feature}'", value=cor))
        else:
            updates.append(gr.update(visible=False, label=""))
    row_visibility_update = gr.update(visible=bool(features_selecionadas))
    return [row_visibility_update] + updates

# --- CONSTRUÇÃO DA INTERFACE (GRADIO) ---

with gr.Blocks(theme=gr.themes.Soft(), title="Visualizador de Séries Temporais", css=custom_css) as demo:
    MAX_FEATURES = 10

    gr.Markdown("# 📊 Visualizador Interativo de Séries Temporais")
    gr.Markdown("Faça o upload do seu dataset, escolha as colunas, o nível de agrupamento e o período para gerar gráficos comparativos incríveis!")

    df_state = gr.State()
    todas_as_colunas_state = gr.State([])

    with gr.Row():
        arquivo_input = gr.File(label="Selecione seu arquivo (.csv ou .xlsx)")

    with gr.Group(visible=False) as grupo_principal:
        with gr.Accordion("1. Configure sua Visualização", open=True):
            with gr.Row():
                eixo_x_input = gr.Dropdown(label="Eixo X (Coluna de Tempo/Período)")
                features_input = gr.CheckboxGroup(label="Eixo Y (Features a comparar)")
            
            # --- NOVO CONTROLE DE AGRUPAMENTO ---
            agrupamento_input = gr.Radio(
                choices=["Diário", "Mensal", "Anual"], 
                value="Diário", 
                label="Agrupar dados por (calcula a média para o período)",
                info="Escolha como agregar os dados no tempo. 'Diário' mostra os dados brutos."
            )

            with gr.Accordion("🗓️ Filtrar por Período (Opcional)", open=False):
                with gr.Row():
                    data_inicio_input = gr.Dropdown(label="Data de Início", interactive=True)
                    data_fim_input = gr.Dropdown(label="Data Final", interactive=True)

        with gr.Accordion("🎨 Escolha as Cores", open=True):
            with gr.Row(visible=False) as seletores_de_cor_area:
                color_pickers = [gr.ColorPicker(visible=False, interactive=True, label=f"Cor {i+1}") for i in range(MAX_FEATURES)]

        with gr.Accordion("⚙️ Customização Avançada do Gráfico", open=False):
            with gr.Row():
                titulo_input = gr.Textbox(label="Título do Gráfico", placeholder="Ex: Audiência Diária")
                tamanho_input = gr.Textbox(label="Tamanho (Largura,Altura)", value="16,8")
            with gr.Row():
                label_x_input = gr.Textbox(label="Legenda Eixo X", placeholder="Ex: Dia")
                label_y_input = gr.Textbox(label="Legenda Eixo Y", placeholder="Ex: Pontos de Audiência")
            with gr.Row():
                cor_fundo_input = gr.ColorPicker(label="Cor de Fundo", value="#FFFFFF")
                cor_fonte_titulo_input = gr.ColorPicker(label="Cor Fonte Título", value="#000000")
                cor_fonte_eixos_input = gr.ColorPicker(label="Cor Fonte Eixos", value="#000000")
                cor_fonte_legenda_input = gr.ColorPicker(label="Cor Fonte Legenda", value="#000000")
            with gr.Row():
                efeito_sombra_input = gr.Checkbox(label="Habilitar Sombreado (efeito gradiente)", value=True)
                ajuste_eixo_y_input = gr.Radio(choices=["Automático (Fit)", "Iniciar em Zero"], value="Automático (Fit)", label="Ajuste do Eixo Y")


        run_button = gr.Button("🚀 Gerar Gráfico!", elem_classes=["orange-button"])

    with gr.Tabs() as results_tabs:
        with gr.TabItem("📈 Gráfico Gerado"):
            plot_output = gr.Plot(label="Seu Gráfico Incrível")
        with gr.TabItem("💾 Download"):
            download_output = gr.File(label="Baixar Gráfico (.png)", visible=False)

    # --- LÓGICA DOS EVENTOS (CONECTANDO OS BOTÕES) ---

    arquivo_input.upload(
        processar_arquivo,
        inputs=[arquivo_input],
        outputs=[grupo_principal, df_state, todas_as_colunas_state, eixo_x_input, features_input]
    )

    eixo_x_input.change(
        atualizar_features_e_datas,
        inputs=[df_state, todas_as_colunas_state, eixo_x_input],
        outputs=[features_input, data_inicio_input, data_fim_input]
    )

    features_input.change(
        lambda sel: atualizar_seletores_de_cor(sel, MAX_FEATURES),
        inputs=[features_input],
        outputs=[seletores_de_cor_area] + color_pickers
    )

    run_button.click(
        gerar_grafico,
        inputs=[arquivo_input, eixo_x_input, features_input, data_inicio_input, data_fim_input,
                agrupamento_input,
                titulo_input, tamanho_input, label_x_input, label_y_input, cor_fundo_input,
                cor_fonte_titulo_input, cor_fonte_eixos_input, cor_fonte_legenda_input, efeito_sombra_input,
                ajuste_eixo_y_input] + color_pickers,
        outputs=[plot_output, download_output]
    )

if __name__ == "__main__":
    demo.launch(debug=True)
