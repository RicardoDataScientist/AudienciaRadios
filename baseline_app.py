import warnings
import pandas as pd
import numpy as np
import gradio as gr
import io
import json
import tempfile
import os
# openpyxl é necessário para o pd.ExcelWriter funcionar com .xlsx
try:
    import openpyxl
except ImportError:
    print("Aviso: openpyxl não instalado. O download do Excel pode falhar.")
    
from prophet import Prophet
import plotly.graph_objects as go
import plotly.io as pio

# Definindo o tema padrão do Plotly para ficar clean
pio.templates.default = "plotly_white"

# --- Configurações Iniciais ---
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# --- CSS Personalizado ---
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
"""

# --- FUNÇÕES CORE (LÓGICA DA ANÁLISE) ---

def limpar_nomes_colunas(df):
    """Padroniza os nomes das colunas para minúsculas e sem espaços."""
    original_columns = df.columns.tolist()
    # Converte tudo para string antes de limpar
    df.columns = [str(col).strip().lower().replace(' ', '_') for col in original_columns]
    column_map = dict(zip(original_columns, df.columns))
    return df, column_map

def carregar_e_preparar_arquivo(arquivo, progress=gr.Progress(track_tqdm=True)):
    """
    Lê o arquivo, identifica colunas de data e métricas,
    e salva o DF no state.
    """
    if arquivo is None:
        return gr.update(visible=False), None, gr.update(choices=[]), gr.update(choices=[])

    try:
        progress(0.2, desc="Lendo arquivo...")
        df = pd.read_csv(arquivo.name) if arquivo.name.endswith('.csv') else pd.read_excel(arquivo.name)
        
        progress(0.5, desc="Identificando colunas...")
        
        all_columns = df.columns.tolist()
        
        # Tenta adivinhar colunas numéricas (métricas)
        numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
        
        # Tenta adivinhar a coluna de data
        date_col_guess = all_columns[0]
        for col in all_columns:
            if str(col).lower() in ['data', 'date', 'periodo', 'período']:
                date_col_guess = col
                break
                
        progress(0.8, desc="Pronto para configurar!")
        
        # Salva o DF limpo no state como JSON
        df_json = df.to_json(orient='split', date_format='iso')
        
        # ATUALIZAÇÃO: Retorna uma LISTA no value de target_col_input
        return (
            gr.update(visible=True), 
            df_json, 
            gr.update(choices=all_columns, value=date_col_guess), 
            gr.update(choices=numeric_columns, value=[numeric_columns[0]] if numeric_columns else []) # Retorna lista!
        )
    
    except Exception as e:
        raise gr.Error(f"Erro ao ler o arquivo: {e}")


def gerar_baseline_e_previsao(df_json, date_col, target_cols, meses_para_prever, baseline_choices, data_filtro_str, progress=gr.Progress(track_tqdm=True)):
    """
    Executa a PREVISÃO com Prophet para MÚLTIPLAS COLUNAS e gera:
    1. Gráfico da primeira métrica.
    2. Tabela consolidada de todas as métricas.
    3. Arquivo Excel com uma aba por métrica.
    """
    
    # ATUALIZAÇÃO: Validação checa se target_cols (lista) não está vazia
    if not all([df_json, date_col, meses_para_prever]) or not target_cols:
        raise gr.Error("Por favor, preencha todos os campos e selecione pelo menos uma coluna para analisar! 🧐")
        
    if int(meses_para_prever) <= 0:
        raise gr.Error("O número de meses para prever deve ser maior que 0.")

    try:
        progress(0, desc="🔄 Carregando dados base...")
        df_base = pd.read_json(io.StringIO(df_json), orient='split')
        
        try:
            df_base[date_col] = pd.to_datetime(df_base[date_col])
        except Exception as e:
            raise gr.Error(f"Não consegui converter a coluna '{date_col}' para data. Verifique o formato. Erro: {e}")

        all_results_list = []
        figs_list = []
        
        # --- Configuração do Excel ---
        temp_dir = tempfile.mkdtemp()
        file_name = f"Baseline_Consolidado.xlsx" # NOVO NOME
        file_path = os.path.join(temp_dir, file_name)
        
        num_targets = len(target_cols)
        
        # --- LOOP PRINCIPAL: Roda para cada coluna selecionada ---
        for i, target_col in enumerate(target_cols):
            
            progress_pct = (i + 1) / num_targets
            progress(progress_pct, desc=f"Processando: {target_col} ({i+1}/{num_targets})")
                
            # --- CÓDIGO RESTAURADO: Todo o bloco de lógica do Prophet que havia sumido ---
            # Pega a série alvo e remove NaNs
            df_series = df_base[[date_col, target_col]].dropna()
            
            # Prophet exige colunas 'ds' (data) e 'y' (valor)
            df_prophet = df_series.rename(columns={date_col: 'ds', target_col: 'y'})
            
            if df_prophet.empty or len(df_prophet) < 2:
                print(f"Aviso: Pulando '{target_col}' por ter dados insuficientes (n={len(df_prophet)}).")
                continue # Pula para a próxima métrica
            
            # --- NOVO BASELINE: Ano Anterior (YoY) ---
            # Pega os dados históricos e "empurra" 1 ano pra frente
            df_prev_year = df_prophet[['ds', 'y']].copy()
            # Adiciona 1 ano à data
            df_prev_year['Data'] = df_prev_year['ds'] + pd.DateOffset(years=1)
            # Renomeia 'y' para o nome do baseline
            df_prev_year_baseline = df_prev_year[['Data', 'y']].rename(columns={'y': 'baseline_ano_anterior'})
            # --- Fim Novo Baseline ---
                
            # --- Lógica do Prophet (igual a antes) ---
            m = Prophet(
                yearly_seasonality=True, 
                weekly_seasonality=False, 
                daily_seasonality=False
            )
            m.fit(df_prophet)
            
            future = m.make_future_dataframe(periods=int(meses_para_prever), freq='MS')
            forecast = m.predict(future)
            
            # Baseline 1: Prophet (Tendência + Sazonalidade)
            forecast['baseline_prophet'] = forecast['trend'] + forecast['yearly']
            
            results_df = forecast.merge(df_prophet, on='ds', how='left')
            
            # --- Novos Baselines (igual a antes) ---
            results_df_final = results_df.rename(columns={'y': 'Original', 'ds': 'Data'})
            results_df_final['month'] = results_df_final['Data'].dt.month
            
            monthly_avg_hist = df_prophet.groupby(df_prophet['ds'].dt.month)['y'].mean().rename('baseline_avg_hist')
            results_df_final = results_df_final.merge(monthly_avg_hist, left_on='month', right_index=True, how='left')

            recent_data = df_prophet.tail(24) 
            monthly_avg_recent = recent_data.groupby(recent_data['ds'].dt.month)['y'].mean().rename('baseline_avg_recent_24m')
            monthly_avg_recent = monthly_avg_recent.combine_first(monthly_avg_hist) 
            results_df_final = results_df_final.merge(monthly_avg_recent, left_on='month', right_index=True, how='left')
            
            # --- NOVO: Merge do Baseline Ano Anterior ---
            results_df_final = results_df_final.merge(df_prev_year_baseline, on='Data', how='left')
            # --- FIM Merge ---
            
            # --- Fim Baselines ---

            colunas_tabela = [
                'Data', 
                'Original',
                'baseline_prophet',
                'baseline_avg_hist',
                'baseline_avg_recent_24m',
                'baseline_ano_anterior', # <-- ADICIONADO AQUI
                'yhat', 
                'yhat_lower',
                'yhat_upper',
                'trend',
                'yearly'
            ]
            cols_finais_base = [col for col in colunas_tabela if col in results_df_final.columns]
            results_df_final = results_df_final[cols_finais_base]
            
            # --- AGREGAÇÃO ---
            # Adiciona a coluna da métrica para identificação na tabela geral
            results_df_final['metrica'] = target_col
            # Reordena colunas
            cols_finais_agg = ['Data', 'metrica'] + [col for col in results_df_final.columns if col not in ['Data', 'metrica']]
            results_df_final_agg = results_df_final[cols_finais_agg]
            # --- FIM DO CÓDIGO RESTAURADO ---
            
            all_results_list.append(results_df_final_agg) # Adiciona na lista da tabela geral

            # --- GERAÇÃO DO GRÁFICO (só para o primeiro item) ---
            if i == 0:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=results_df_final['Data'], 
                    y=results_df_final['Original'], 
                        mode='markers',
                        name='Dado Original',
                        marker=dict(color='royalblue', size=5)
                    ))
                    
                if "Baseline Prophet (Tend+Saz)" in baseline_choices:
                    fig.add_trace(go.Scatter(
                        x=results_df_final['Data'], 
                        y=results_df_final['baseline_prophet'], 
                        mode='lines',
                        name='Baseline Prophet (Tend+Saz)',
                        line=dict(color='red', width=2, dash='dash')
                    ))
                    fig.add_trace(go.Scatter(
                        x=results_df_final['Data'],
                        y=results_df_final['yhat_upper'],
                        mode='lines',
                        line=dict(width=0, color='rgba(255, 0, 0, 0.2)'), 
                        name='Intervalo Confiança (Superior)',
                        showlegend=False
                    ))
                    fig.add_trace(go.Scatter(
                        x=results_df_final['Data'],
                        y=results_df_final['yhat_lower'],
                        mode='lines',
                        line=dict(width=0, color='rgba(255, 0, 0, 0.2)'), 
                        name='Intervalo Confiança (Inferior)',
                        fill='tonexty', 
                        fillcolor='rgba(255, 0, 0, 0.2)'
                    ))

                if "Baseline Média Histórica" in baseline_choices:
                    fig.add_trace(go.Scatter(
                        x=results_df_final['Data'], 
                        y=results_df_final['baseline_avg_hist'], 
                        mode='lines',
                        name='Baseline Média Histórica',
                        line=dict(color='green', width=2, dash='dot')
                    ))

                if 'Baseline Recente (Últ. 24m)' in baseline_choices:
                     fig.add_trace(go.Scatter(
                        x=results_df_final['Data'], 
                        y=results_df_final['baseline_avg_recent_24m'], 
                        mode='lines',
                        name='Baseline Recente (Últ. 24m)',
                        line=dict(color='purple', width=2, dash='dashdot')
                    ))
                
                # --- NOVO: Lógica de Plot para Ano Anterior ---
                if 'Baseline Ano Anterior (YoY)' in baseline_choices:
                     fig.add_trace(go.Scatter(
                        x=results_df_final['Data'], 
                        y=results_df_final['baseline_ano_anterior'], 
                        mode='lines',
                        name='Baseline Ano Anterior (YoY)',
                        line=dict(color='orange', width=2, dash='dash') # Nova cor
                    ))
                # --- FIM Novo Plot ---
                
                fig.update_layout(
                    title=f"Previsão de Baseline para '{target_col}' (Gráfico do 1º item)",
                    xaxis_title="Data",
                    yaxis_title="Audiência",
                    legend_title="Séries",
                    hovermode="x unified"
                )
                figs_list.append(fig) # Adiciona na lista de figuras
        
        # --- Fim do Loop ---

        
        # Consolida a tabela geral
        final_table_df = pd.concat(all_results_list, ignore_index=True) if all_results_list else pd.DataFrame()
        
        # --- NOVO: Aplicar Filtro de Data (se houver) ---
        tabela_filtrada_df = final_table_df.copy() # Começa com a tabela completa
        
        # data_filtro_str é o novo input do Gradio
        if data_filtro_str and not final_table_df.empty: 
            try:
                # Tenta converter o filtro para data
                data_filtro = pd.to_datetime(data_filtro_str)
                progress(0.92, desc=f"Filtrando dados a partir de {data_filtro_str}...")
                
                # Filtra a tabela final (que já tem a coluna 'Data' como datetime)
                tabela_filtrada_df = final_table_df[final_table_df['Data'] >= data_filtro].copy()
                
                if tabela_filtrada_df.empty:
                    print(f"Aviso: O filtro a partir de {data_filtro_str} não retornou nenhum dado.")
                    
            except Exception as e:
                print(f"Aviso: Não foi possível aplicar o filtro de data '{data_filtro_str}'. Retornando dados completos. Erro: {e}")
                # Se der erro no filtro, só ignora e usa a tabela completa
                tabela_filtrada_df = final_table_df # Reseta para a completa
        # --- FIM FILTRO DE DATA ---

        # Pega a primeira figura (ou uma figura vazia se nada foi processado)
        final_fig = figs_list[0] if figs_list else go.Figure().update_layout(title="Nenhum dado processado")

        # --- GERAÇÃO DO EXCEL (Aba Única) ---
        progress(0.95, desc="💾 Gerando planilha Excel consolidada...")
        
        # ATUALIZAÇÃO: Salva a 'tabela_filtrada_df'
        if not tabela_filtrada_df.empty:
            tabela_filtrada_df.to_excel(file_path, sheet_name='Baseline_Consolidado', index=False)
        elif not final_table_df.empty:
             # Se o filtro zerou os dados, salva um excel vazio (mas com cabeçalhos)
             tabela_filtrada_df.to_excel(file_path, sheet_name='Baseline_Consolidado', index=False)
        # --- FIM GERAÇÃO EXCEL ---

        progress(1.0, desc="Prontinho! ✨")

        return (
            final_fig, 
            tabela_filtrada_df, # ATUALIZAÇÃO: Retorna a tabela filtrada para a UI
            gr.update(value=file_path, visible=True)
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise gr.Error(f"Ocorreu um erro na análise: {str(e)}")


# --- CONSTRUÇÃO DA INTERFACE (GRADIO) ---

with gr.Blocks(theme=gr.themes.Soft(), title="Analisador de Baseline e Previsão", css=custom_css) as demo:
    gr.Markdown("# 📈 Analisador de Baseline e Previsão de Audiência")
    gr.Markdown("Carregue seus dados, descubra o **Baseline** e faça **previsões** futuras para *várias métricas*!")

    # State para armazenar o dataframe original
    original_df_state = gr.State() # Guarda o DF original (JSON)

    with gr.Row():
        arquivo_input = gr.File(label="Selecione seu arquivo (.csv ou .xlsx)", file_types=[".csv", ".xlsx"])

    with gr.Group(visible=False) as grupo_controles:
        with gr.Accordion("1. Configuração da Análise", open=True):
            with gr.Row():
                date_col_input = gr.Dropdown(label="Qual é a coluna de data? 🗓️")
                
                # ATUALIZAÇÃO: multiselect=True
                target_col_input = gr.Dropdown(
                    label="Quais colunas (segmentos) vamos analisar? 📊", 
                    multiselect=True
                )
            
            with gr.Row():
                meses_prever_input = gr.Number(label="Quantos meses prever no futuro? 🚀", value=3, precision=0)
            
            # NOVO CAMPO DE FILTRO
            with gr.Row():
                 data_filtro_input = gr.Textbox(
                    label="Filtrar outputs a partir de (AAAA-MM-DD):", 
                    placeholder="Deixe em branco para ver tudo"
                )

            with gr.Row():
                baseline_choices_input = gr.CheckboxGroup(
                    label="Quais baselines plotar? (Aparecerão no gráfico do 1º item e no Excel)", 
                    choices=[
                        "Baseline Prophet (Tend+Saz)", 
                        "Baseline Média Histórica", 
                        "Baseline Recente (Últ. 24m)",
                        "Baseline Ano Anterior (YoY)" # <-- ADICIONADO AQUI
                    ], 
                    value=["Baseline Prophet (Tend+Saz)"]
                )
        
        run_button = gr.Button("✨ Gerar Baseline e Previsão!", elem_classes=["orange-button"])

    with gr.Tabs() as results_tabs:
        with gr.TabItem("📈 Gráfico de Previsão (1º Item)"):
            plot_output = gr.Plot(label="Original vs. Previsão (Mostrando o primeiro item selecionado)")
        
        with gr.TabItem("📄 Tabela Consolidada"):
            table_output = gr.DataFrame(label="Dados Previstos e Componentes (Todas as Métricas)", wrap=True)

        # ATUALIZAÇÃO: Mudança nos labels de Download
        with gr.TabItem("💾 Download (Aba Única)"):
            download_output = gr.File(label="Baixar Planilha Consolidada (.xlsx)", visible=False, interactive=False)


    # --- LÓGICA DOS EVENTOS (CONECTANDO OS BOTÕES) ---
    
    # 1. Quando o arquivo é carregado
    arquivo_input.upload(
        carregar_e_preparar_arquivo,
        [arquivo_input],
        [grupo_controles, original_df_state, date_col_input, target_col_input]
    )

    # 2. Botão principal que roda a análise
    run_button.click(
        gerar_baseline_e_previsao,
        inputs=[
            original_df_state, 
            date_col_input, 
            target_col_input, # Agora envia uma lista
            meses_prever_input,
            baseline_choices_input,
            data_filtro_input # NOVO input
        ],
        outputs=[
            plot_output,
            table_output,
            download_output
        ]
    )

if __name__ == "__main__":
    demo.launch(debug=True)