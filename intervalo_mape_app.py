import warnings
import pandas as pd
import numpy as np
import gradio as gr
import io
import os
import tempfile
import plotly.graph_objects as go
import plotly.io as pio

# tema clean
pio.templates.default = "plotly_white"

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

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

# ------------------------------------------------------------------
# FUNÇÕES AUXILIARES
# ------------------------------------------------------------------

def carregar_planilha(arquivo, default_mape):
    """
    Lê o arquivo e devolve:
    1. O df original (em JSON)
    2. Um novo df (metrica, mape_percentual) para o usuário editar
    """
    if arquivo is None:
        return gr.update(visible=False), None, None, "Nenhum arquivo enviado."

    try:
        if arquivo.name.endswith(".csv"):
            df = pd.read_csv(arquivo.name)
        else:
            df = pd.read_excel(arquivo.name)

        # normaliza nomes
        df.columns = [str(c).strip().lower() for c in df.columns]

        if "metrica" not in df.columns or "previsto" not in df.columns:
            msg = "A planilha precisa ter as colunas 'metrica' e 'previsto'."
            return gr.update(visible=False), None, None, msg

        # --- NOVO: Limpa a coluna 'metrica' ---
        # Garante que não há espaços extras
        if "metrica" in df.columns:
            df["metrica"] = df["metrica"].astype(str).str.strip()
        # --- FIM NOVO ---

        if "data" in df.columns:
            try:
                df["data"] = pd.to_datetime(df["data"], errors='coerce')
                df["data"] = df["data"].dt.date
            except Exception as e:
                print(f"Aviso: Não foi possível formatar a coluna 'data'. Erro: {e}")

        metricas_unicas = sorted(df["metrica"].dropna().unique().tolist())
        df_json = df.to_json(orient="split", date_format="iso")

        if default_mape is None:
            default_mape = 5.0

        df_mapes = pd.DataFrame({
            "metrica": metricas_unicas,
            "mape_percentual": default_mape
        })

        return (
            gr.update(visible=True),
            df_json,
            df_mapes,
            f"Arquivo carregado. {len(df)} linhas, {len(metricas_unicas)} métricas. Edite os MAPEs abaixo:"
        )

    except Exception as e:
        return gr.update(visible=False), None, None, f"Erro ao ler arquivo: {e}"


def gerar_faixa(df_json, mape_input_df):
    """
    Cria upper/lower por linha usando a tabela de MAPEs editada pelo usuário.
    """
    if df_json is None:
        raise gr.Error("Carregue um arquivo primeiro.")
    
    if mape_input_df is None:
        raise gr.Error("A tabela de MAPEs está vazia. Carregue um arquivo.")

    df = pd.read_json(io.StringIO(df_json), orient="split")
    
    # --- NOVO: Limpa a coluna 'metrica' no DF recarregado ---
    if "metrica" in df.columns:
        df["metrica"] = df["metrica"].astype(str).str.strip()
    # --- FIM NOVO ---

    mape_dict = {}
    for _, row in mape_input_df.iterrows():
        # --- NOVO: Adiciona .strip() por segurança ---
        metrica = str(row["metrica"]).strip()
        
        try:
            mape_pct = float(row["mape_percentual"])
            
            # --- NOVO: Fallback para 0.0 se for NaN ---
            if pd.isna(mape_pct):
                mape_dict[metrica] = 0.0 # Era 0.05
            else:
                mape_dict[metrica] = mape_pct / 100.0
                
        except (ValueError, TypeError):
            # --- NOVO: Fallback para 0.0 se for nulo ou inválido ---
            # Se o usuário apagar o valor (None) ou digitar "abc",
            # vai cair aqui. O correto é ser 0%, e não 5%.
            mape_dict[metrica] = 0.0 # Era 0.05
    # --- FIM NOVO ---


    # cria colunas
    lowers = []
    uppers = []
    mape_usados = []

    for _, row in df.iterrows():
        # --- NOVO: Adiciona .strip() na busca ---
        met = str(row["metrica"]).strip()
        prev = row["previsto"]

        # --- NOVO: Fallback do .get() mudou para 0.0 ---
        # Se a métrica for nova ou tiver sido deletada da tabela de MAPEs,
        # usa 0% como padrão, em vez de 5%.
        mape_frac = mape_dict.get(met, 0.0) # Era 0.05

        lower = prev * (1 - mape_frac)
        upper = prev * (1 + mape_frac)

        lowers.append(lower)
        uppers.append(upper)
        mape_usados.append(mape_frac * 100.0)

    df["mape_usado_%"] = mape_usados
    df["lower"] = lowers
    df["upper"] = uppers

    if "data" in df.columns:
        try:
            df["data"] = pd.to_datetime(df["data"]).dt.date
        except Exception as e:
            print(f"Aviso: Não foi possível re-formatar 'data' antes de salvar: {e}")

    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, "faixa_confianca.xlsx")
    df.to_excel(file_path, index=False)

    return df, gr.update(value=file_path, visible=True)

# ------------------------------------------------------------------
# INTERFACE GRADIO
# ------------------------------------------------------------------

with gr.Blocks(theme=gr.themes.Soft(), title="Faixa de confiança por MAPE", css=custom_css) as demo:
    gr.Markdown("# 📏 Gerador de Faixa de Confiança por MAPE")
    gr.Markdown(
        "Envie uma planilha com as colunas **`metrica`** e **`previsto`**. "
        "Depois, edite os MAPEs (%) na tabela que aparecer."
    )

    df_state = gr.State()

    arquivo_input = gr.File(label="Selecione o arquivo (.csv ou .xlsx)", file_types=[".csv", ".xlsx"])

    with gr.Group(visible=False) as grupo_controles:
        gr.Markdown("### 1. Edite os MAPEs por Métrica")
        
        default_mape_input = gr.Number(
            label="Valor padrão (%) para pré-preencher a tabela",
            value=5.0,
            precision=2
        )
        
        gr.Markdown("Edite os MAPEs (%) para cada métrica na tabela abaixo:")
        mape_input_df = gr.Dataframe(
            headers=["metrica", "mape_percentual"],
            label="Tabela de MAPEs",
            # Define os tipos para a coluna 'metrica' ser string e 'mape_percentual' ser número
            datatype=["str", "number"],
            interactive=True,
            row_count=(10, "dynamic"),
            col_count=(2, "fixed")
        )

        run_button = gr.Button("✨ Gerar Faixa de Confiança", elem_classes=["orange-button"])

    with gr.Tabs():
        with gr.TabItem("📄 Tabela de Resultados"):
            table_output = gr.DataFrame(wrap=True, label="Resultados com Faixa")
        with gr.TabItem("💾 Download"):
            download_output = gr.File(label="Baixar Excel", visible=False, interactive=False)

    status_output = gr.Markdown()

    # --- Eventos (sem alteração aqui) ---
    
    arquivo_input.upload(
        carregar_planilha,
        [arquivo_input, default_mape_input],
        [grupo_controles, df_state, mape_input_df, status_output]
    )

    run_button.click(
        gerar_faixa,
        inputs=[df_state, mape_input_df],
        outputs=[table_output, download_output]
    )

if __name__ == "__main__":
    demo.launch(debug=True)