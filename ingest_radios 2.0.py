import warnings
import pandas as pd
import numpy as np
import gradio as gr
import io
import tempfile
import os
import json

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
.dark-orange-button {
    background: linear-gradient(to right, #D2691E, #CD853F) !important;
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

def limpar_nomes_colunas(df):
    """Padroniza os nomes das colunas para minúsculas e sem espaços."""
    original_columns = df.columns.tolist()
    df.columns = [str(col).strip().lower().replace(' ', '_') for col in original_columns]
    column_map = dict(zip(original_columns, df.columns))
    return df, column_map

def executar_pivot_e_gerar_excel(df_json, agrupamento, metrica, progress=gr.Progress(track_tqdm=True)):
    """
    Executa o pipeline: Filtra -> Pivota -> Gera Excel.
    """
    if not all([df_json, agrupamento, metrica]):
        raise gr.Error("Por favor, selecione todas as opções: Arquivo, Agrupamento e Métrica.")

    try:
        progress(0, desc="🔄 Carregando dados...")
        df = pd.read_json(io.StringIO(df_json), orient='split')
        
        progress(0.3, desc=f" filtrando por '{agrupamento}'...")
        df_filtrado = df[df['agrupamento'] == agrupamento]
        
        if df_filtrado.empty:
            raise gr.Error(f"Nenhum dado encontrado para o agrupamento '{agrupamento}'.")
            
        progress(0.6, desc=f"Pivotando 'publico' usando '{metrica}' como valor...")
        
        # Faz o pivot
        df_pivot = df_filtrado.pivot_table(
            index='periodo', 
            columns='publico', 
            values=metrica
        ).reset_index()
        
        df_pivot.columns.name = None # Limpa o nome do índice das colunas
        df_pivot = df_pivot.fillna(0) # Boa prática preencher NaNs com 0
        
        # Ordenação temporal
        df_pivot['periodo'] = pd.to_datetime(df_pivot['periodo'])
        df_pivot = df_pivot.sort_values('periodo')
        
        # --- CORREÇÃO DA DATA (Formatando apenas a data) ---
        df_pivot['periodo'] = df_pivot['periodo'].dt.date
        
        progress(0.9, desc="💾 Gerando planilha Excel...")
        temp_dir = tempfile.mkdtemp()
        file_name = f"Pivot_{agrupamento.replace(' ', '_')}_{metrica}.xlsx"
        file_path = os.path.join(temp_dir, file_name)

        df_pivot.to_excel(file_path, sheet_name='Dados_Pivotados', index=False)
        
        progress(1.0, desc="Pronto!")

        return (
            df_pivot, # Para o DataFrame na UI
            gr.update(value=file_path, visible=True) # Para o gr.File
        )

    except Exception as e:
        print(f"Erro detalhado: {e}")
        import traceback
        traceback.print_exc()
        raise gr.Error(f"Ocorreu um erro: {str(e)}")

# --- FUNÇÕES AUXILIARES DA INTERFACE ---

def processar_arquivo(arquivo):
    """
    Lê o arquivo, limpa nomes, calcula MÉDIA MÓVEL (OPM 3M) e salva no state.
    """
    if arquivo is None:
        return gr.update(visible=False), None, gr.update(choices=[]), gr.update(choices=[])
    try:
        df = pd.read_csv(arquivo.name) if arquivo.name.endswith('.csv') else pd.read_excel(arquivo.name)
        df, column_map = limpar_nomes_colunas(df)
        
        # --- TRATAMENTO DE DATA ---
        try:
            df['periodo'] = pd.to_datetime(df['periodo'])
        except Exception as e:
            raise gr.Error(f"Erro ao converter a coluna 'PERIODO'. Verifique o formato. Erro: {e}")

        # --- CÁLCULO DA MÉDIA MÓVEL (OPM 3M) ---
        # 1. Verificamos se OPM existe
        if 'opm' in df.columns and 'agrupamento' in df.columns and 'publico' in df.columns:
            # 2. Ordenamos para garantir a sequência temporal correta
            df = df.sort_values(by=['agrupamento', 'publico', 'periodo'])
            
            # 3. Calculamos a média móvel agrupada
            # min_periods=1 garante que calcule mesmo se tiver menos de 3 meses no começo (opcional)
            # Se quiser estrito (só aparece no 3º mês), remova min_periods=1
            df['opm_3m'] = df.groupby(['agrupamento', 'publico'])['opm'].transform(
                lambda x: x.rolling(window=3, min_periods=1).mean()
            )
            # Arredondar para ficar bonito
            df['opm_3m'] = df['opm_3m'].round(2)
        
        # Converte periodo para date apenas para visualização/JSON final
        df['periodo'] = df['periodo'].dt.date
        
        # Listas para os dropdowns
        agrupamentos = df['agrupamento'].dropna().unique().tolist()
        agrupamentos.sort()
        
        # Métricas prováveis + a nova opm_3m
        metricas_padrao = ['opm', 'opm_3m', 'os', 'alc2', 'tmedind']
        metricas_validas = [m for m in metricas_padrao if m in df.columns]
        
        # Validações básicas
        required_cols = ['periodo', 'agrupamento', 'publico']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
             raise gr.Error(f"Colunas obrigatórias não encontradas: {missing}")

        if not metricas_validas:
            raise gr.Error("Nenhuma coluna de métrica conhecida encontrada.")

        # Salva o DF limpo no state como JSON
        df_json = df.to_json(orient='split', date_format='iso')
        
        return gr.update(visible=True), df_json, gr.update(choices=agrupamentos), gr.update(choices=metricas_validas)
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise gr.Error(f"Erro ao ler o arquivo: {e}")


# --- CONSTRUÇÃO DA INTERFACE (GRADIO) ---

with gr.Blocks(theme=gr.themes.Soft(), title="Processador e Pivot de Dados", css=custom_css) as demo:
    gr.Markdown("# 🛠️ Processador e Pivot de Dados")
    gr.Markdown("Filtre por **Agrupamento**, pivote por **Público** e gere sua planilha Excel pronta para análise!")

    # State para armazenar o dataframe original
    original_df_state = gr.State()

    with gr.Row():
        arquivo_input = gr.File(label="Selecione seu arquivo (.csv ou .xlsx)")

    with gr.Group(visible=False) as grupo_principal:
        with gr.Accordion("1. Configuração do Pivot", open=True):
            with gr.Row():
                agrupamento_input = gr.Dropdown(label="Selecione o AGRUPAMENTO para filtrar")
                valor_pivot_input = gr.Dropdown(label="Selecione a MÉTRICA para analisar (Valor do Pivot)")
        
        run_button = gr.Button("🚀 Gerar Planilha Pivotada!", elem_classes=["orange-button"])

    with gr.Tabs() as results_tabs:
        with gr.TabItem("📊 Dados Pivotados"):
            pivot_data_output = gr.DataFrame(label="Prévia dos Dados Pivotados", wrap=True)
        
        with gr.TabItem("💾 Download"):
            download_output = gr.File(label="Baixar Planilha (.xlsx)", visible=False)

    # --- LÓGICA DOS EVENTOS ---
    
    arquivo_input.upload(
        processar_arquivo,
        [arquivo_input],
        [grupo_principal, original_df_state, agrupamento_input, valor_pivot_input]
    )

    run_button.click(
        executar_pivot_e_gerar_excel,
        inputs=[original_df_state, agrupamento_input, valor_pivot_input],
        outputs=[pivot_data_output, download_output]
    )

if __name__ == "__main__":
    demo.launch()