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
"""

# --- FUNÇÕES CORE ---

def limpar_nomes_colunas(df):
    """Padroniza os nomes das colunas para minúsculas e sem espaços."""
    original_columns = df.columns.tolist()
    df.columns = [str(col).strip().lower().replace(' ', '_') for col in original_columns]
    return df

def executar_pivot_e_gerar_excel(df_json, agrupamento, metrica, progress=gr.Progress(track_tqdm=True)):
    """
    Filtra a rádio, extrai o TOTAL FM (Sexo Ambos) e une os dados para o XGBoost.
    """
    if not all([df_json, agrupamento, metrica]):
        raise gr.Error("Selecione Arquivo, Agrupamento e Métrica.")

    try:
        progress(0, desc="🔄 Carregando dados...")
        df = pd.read_json(io.StringIO(df_json), orient='split')
        df['periodo'] = pd.to_datetime(df['periodo']).dt.date

        # 1. Extrair a referência: TOTAL FM + SEXO AMBOS
        progress(0.2, desc="📈 Extraindo referência TOTAL FM...")
        df_total_fm = df[(df['agrupamento'] == 'TOTAL FM') & (df['publico'] == 'SEXO AMBOS')].copy()
        
        if df_total_fm.empty:
            # Caso não ache 'TOTAL FM' ou 'SEXO AMBOS', tentamos variações comuns de texto
            df_total_fm = df[
                (df['agrupamento'].str.contains('TOTAL FM', case=False, na=False)) & 
                (df['publico'].str.contains('AMBOS', case=False, na=False))
            ].copy()

        # Seleciona apenas as colunas necessárias para o merge posterior
        df_total_fm = df_total_fm[['periodo', metrica]].rename(columns={metrica: f'total_fm_ambos_{metrica}'})

        # 2. Filtrar a rádio selecionada
        progress(0.4, desc=f"🔍 Filtrando rádio '{agrupamento}'...")
        df_filtrado = df[df['agrupamento'] == agrupamento]
        
        if df_filtrado.empty:
            raise gr.Error(f"Nenhum dado para o agrupamento '{agrupamento}'.")

        # 3. Pivotar os dados da rádio (colunas serão os Públicos)
        progress(0.6, desc="🧩 Pivotando públicos...")
        df_pivot = df_filtrado.pivot_table(
            index='periodo', 
            columns='publico', 
            values=metrica
        ).reset_index()
        
        df_pivot.columns.name = None
        df_pivot = df_pivot.fillna(0)
        
        # 4. Unir (Merge) com a coluna do TOTAL FM
        progress(0.8, desc="🔗 Mesclando com TOTAL FM...")
        df_final = pd.merge(df_pivot, df_total_fm, on='periodo', how='left').fillna(0)
        
        # Ordenação temporal garantida
        df_final = df_final.sort_values('periodo')

        # 5. Gerar Excel
        progress(0.9, desc="💾 Gerando planilha...")
        temp_dir = tempfile.mkdtemp()
        file_name = f"Analise_{agrupamento.replace(' ', '_')}_{metrica}.xlsx"
        file_path = os.path.join(temp_dir, file_name)

        df_final.to_excel(file_path, sheet_name='Dados_para_ML', index=False)
        
        progress(1.0, desc="Pronto!")

        return df_final, gr.update(value=file_path, visible=True)

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise gr.Error(f"Erro no processamento: {str(e)}")

def processar_arquivo(arquivo):
    if arquivo is None:
        return gr.update(visible=False), None, gr.update(choices=[]), gr.update(choices=[])
    try:
        df = pd.read_csv(arquivo.name) if arquivo.name.endswith('.csv') else pd.read_excel(arquivo.name)
        df = limpar_nomes_colunas(df)
        
        df['periodo'] = pd.to_datetime(df['periodo'])
        
        # Cálculo da Média Móvel 3 Meses
        if 'opm' in df.columns and 'agrupamento' in df.columns and 'publico' in df.columns:
            df = df.sort_values(by=['agrupamento', 'publico', 'periodo'])
            df['opm_3m'] = df.groupby(['agrupamento', 'publico'])['opm'].transform(
                lambda x: x.rolling(window=3, min_periods=1).mean()
            ).round(2)
        
        # Lista de opções para a UI
        agrupamentos = sorted([x for x in df['agrupamento'].dropna().unique().tolist() if x != 'TOTAL FM'])
        metricas_validas = [m for m in ['opm', 'opm_3m', 'os', 'alc2', 'tmedind'] if m in df.columns]
        
        df_json = df.to_json(orient='split', date_format='iso')
        
        return gr.update(visible=True), df_json, gr.update(choices=agrupamentos), gr.update(choices=metricas_validas)
    
    except Exception as e:
        raise gr.Error(f"Erro ao ler arquivo: {e}")

# --- INTERFACE ---

with gr.Blocks(theme=gr.themes.Soft(), title="ML Data Prep - Rádio", css=custom_css) as demo:
    gr.Markdown("# 🛠️ Preparação de Dados para XGBoost")
    gr.Markdown("Esta ferramenta gera a tabela pivotada da rádio escolhida e injeta o **TOTAL FM (Sexo Ambos)** como coluna de feature.")

    original_df_state = gr.State()

    with gr.Row():
        arquivo_input = gr.File(label="Upload (.csv ou .xlsx)")

    with gr.Group(visible=False) as grupo_principal:
        with gr.Row():
            agrupamento_input = gr.Dropdown(label="Rádio para Previsão")
            valor_pivot_input = gr.Dropdown(label="Métrica (Target)")
        
        run_button = gr.Button("🚀 Gerar Tabela para Modelo", elem_classes=["orange-button"])

    with gr.Tabs():
        with gr.TabItem("📊 Preview (Rádio + Total FM)"):
            pivot_data_output = gr.DataFrame(label="Dados com Features", wrap=True)
        with gr.TabItem("💾 Download"):
            download_output = gr.File(label="Baixar Excel", visible=False)

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