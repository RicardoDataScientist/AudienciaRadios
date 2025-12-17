import io
import zipfile
import tempfile
import gradio as gr
import pandas as pd
import numpy as np
import traceback
import shap

## DESTAQUE - NAMI: Adicionamos df_historico_completo para calcular o YoY no CSV futuro ##
def salvar_resultados_zip(target, df_metricas, features_finais, df_resultados_cv, df_previsoes_futuras, figs_dict, df_historico_completo=None):
    """Salva todos os artefatos em um arquivo zip."""
    file_prefix = f"{target}_"
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'a', zipfile.ZIP_DEFLATED, False) as zip_file:
        for name, fig in figs_dict.items():
            if fig:
                buf = io.BytesIO()
                fig.savefig(buf, format='png', bbox_inches='tight')
                zip_file.writestr(f"{file_prefix}{name}.png", buf.getvalue())
        
        if not df_metricas.empty:
            ## DESTAQUE - NAMI: Adicionado decimal=',' para o CSV de métricas ##
            zip_file.writestr(f"{file_prefix}metricas_cv.csv", df_metricas.to_csv(index=False, decimal=',').encode('utf-8'))
        
        if features_finais:
            features_text = ", ".join(features_finais)
            zip_file.writestr(f"{file_prefix}features_selecionadas.txt", features_text.encode('utf-8'))

        if not df_resultados_cv.empty:
            # df_resultados_cv já vem com as colunas de variação da função treinar_e_avaliar_cv
            ## DESTAQUE - NAMI: Adicionado decimal=',' para o CSV de CV ##
            zip_file.writestr(f"{file_prefix}previsoes_cv.csv", df_resultados_cv.to_csv(decimal=',').encode('utf-8'))
        
        if not df_previsoes_futuras.empty:
            ## DESTAQUE - NAMI: Bloco de cálculo YoY (Ano-contra-Ano) ##
            df_previsoes_para_salvar = df_previsoes_futuras.copy()
            
            # Renomeia a coluna target na previsão para clareza
            col_previsto = f'{target}_Previsto'
            df_previsoes_para_salvar = df_previsoes_para_salvar.rename(columns={target: col_previsto})

            if df_historico_completo is not None:
                try:
                    # Prepara o histórico real
                    col_anterior = f'{target}_Ano_Anterior'
                    df_historico_real = df_historico_completo[[target]].copy()
                    df_historico_real.columns = [col_anterior]
                    
                    # Prepara o df futuro para o merge
                    df_previsoes_para_salvar['data_ano_anterior'] = df_previsoes_para_salvar.index - pd.DateOffset(years=1)
                    
                    # Merge para buscar o valor do ano anterior
                    df_previsoes_para_salvar = pd.merge(
                        df_previsoes_para_salvar,
                        df_historico_real,
                        left_on='data_ano_anterior',
                        right_index=True,
                        how='left'
                    )
                    
                    # Calcula Variações YoY (Sobre a Mediana)
                    df_previsoes_para_salvar['Variacao_Absoluta_YoY'] = (df_previsoes_para_salvar[col_previsto] - df_previsoes_para_salvar[col_anterior]).round(2)
                    df_previsoes_para_salvar['Variacao_Percentual_YoY_%'] = np.where(
                        df_previsoes_para_salvar[col_anterior] == 0,
                        np.nan,
                        ((df_previsoes_para_salvar[col_previsto] - df_previsoes_para_salvar[col_anterior]) / df_previsoes_para_salvar[col_anterior])
                    ).round(4)
                    
                    # Limpa colunas auxiliares
                    df_previsoes_para_salvar = df_previsoes_para_salvar.drop(columns=['data_ano_anterior'])
                
                except Exception as e:
                    print(f"Debug NAMI: Erro ao calcular YoY para CSV: {e}")
                    # Se falhar, pelo menos salva o que tem, mas sem o YoY
                    # Recria df_previsoes_para_salvar para garantir que não está num estado intermediário
                    df_previsoes_para_salvar = df_previsoes_futuras.copy()
                    df_previsoes_para_salvar = df_previsoes_para_salvar.rename(columns={target: col_previsto})
            
            ## Fim do Destaque ##
            ## DESTAQUE - NAMI: Adicionado decimal=',' para o CSV de previsões futuras ##
            zip_file.writestr(f"{file_prefix}previsoes_futuras.csv", df_previsoes_para_salvar.to_csv(decimal=',').encode('utf-8'))

    with tempfile.NamedTemporaryFile(delete=False, suffix='.zip', prefix=f"{target}_resultados_") as tmp_f:
        tmp_f.write(zip_buffer.getvalue())
        tmp_path = tmp_f.name

    return gr.update(value=tmp_path, visible=True, label=f"Download Resultados ({target}.zip)")

# --- DESTAQUE-MINT: Nova função para gerar o arquivo de dados para reconciliação ---
def gerar_arquivo_mint(target, df_limpo, df_resultados_cv, shap_values_historicos, features_finais, df_previsoes_futuras, df_features_futuras, modelo_final):
    """
    Gera um arquivo CSV completo contendo dados históricos, previsões de backtest,
    previsões futuras e os valores SHAP (contribuições) para cada um,
    já incluindo unique_id e timestamp.
    """
    try:
        # 1. Processar Dados Históricos
        # Começa com o dataframe limpo (features + target real)
        df_hist_base = df_limpo.copy()
        # Adiciona a previsão do backtest (terá NaNs onde não houve backtest)
        df_hist_base['forecast'] = df_resultados_cv['Previsto'] # Renomeado
        df_hist_base = df_hist_base.rename(columns={target: 'actual'}) # Renomeado
        df_hist_base['tipo'] = 'Historico'
        # DESTAQUE-MINT: Adiciona unique_id
        df_hist_base['unique_id'] = target 
        
        # Cria DataFrame de SHAP histórico
        df_shap_hist = pd.DataFrame(
            shap_values_historicos.values, 
            columns=[f"shap_{f}" for f in features_finais], 
            index=df_limpo.index
        )
        # DESTAQUE-MINT: Adiciona o base_value (shap_Intercept)
        df_hist_base['shap_base'] = shap_values_historicos.base_values
        
        # Junta tudo
        df_hist_combinado = pd.concat([df_hist_base, df_shap_hist], axis=1)

        # 2. Processar Dados Futuros
        # Calcula SHAP para as previsões futuras (baseado no modelo mediana)
        explainer_futuro = shap.TreeExplainer(modelo_final)
        shap_values_futuros = explainer_futuro(df_features_futuras[features_finais])
        
        # Cria DataFrame de SHAP futuro
        df_shap_futuro = pd.DataFrame(
            shap_values_futuros.values, 
            columns=[f"shap_{f}" for f in features_finais], 
            index=df_features_futuras.index
        )
        
        # Cria base para o dataframe futuro (começando com as features)
        df_futuro_base = df_features_futuras[features_finais].copy()
        df_futuro_base['actual'] = np.nan # Renomeado
        df_futuro_base['forecast'] = df_previsoes_futuras[target].values # Renomeado e .values para segurança
        df_futuro_base['tipo'] = 'Futuro_Forecast'
        # DESTAQUE-MINT: Adiciona unique_id
        df_futuro_base['unique_id'] = target
        # DESTAQUE-MINT: Adiciona o base_value (shap_Intercept)
        df_futuro_base['shap_base'] = shap_values_futuros.base_values
        
        # Junta tudo
        df_futuro_combinado = pd.concat([df_futuro_base, df_shap_futuro], axis=1)

        # 3. Combinar Histórico e Futuro
        df_final_mint = pd.concat([df_hist_combinado, df_futuro_combinado])
        
        # DESTAQUE-MINT (NOVO AJUSTE): Remove a HORA, mantendo apenas a DATA
        df_final_mint.index = df_final_mint.index.date
        
        # DESTAQUE-MINT: Renomeia o índice para timestamp ANTES de salvar
        df_final_mint.index.name = 'timestamp'
        
        # 4. Organizar Colunas (colocando unique_id e timestamp primeiro)
        cols_id = ['unique_id', 'tipo'] 
        cols_valores = ['actual', 'forecast', 'shap_base'] 
        cols_features = sorted([f for f in features_finais if f in df_final_mint.columns]) # Garante que só peguemos as existentes
        cols_shap = sorted([f"shap_{f}" for f in features_finais if f"shap_{f}" in df_final_mint.columns]) # Garante que só peguemos as existentes
        
        # Junta todas as colunas existentes na ordem desejada
        ordem_final = cols_id + cols_valores + cols_features + cols_shap
        # Pega colunas que podem ter sobrado (caso alguma feature não tenha SHAP, etc.)
        cols_restantes = [c for c in df_final_mint.columns if c not in ordem_final]
        
        df_final_mint = df_final_mint[ordem_final + cols_restantes] # Reordena

        # 5. Salvar em arquivo temporário (agora com índice timestamp sendo salvo como coluna)
        # DESTAQUE-MINT: Mudando de CSV para Excel (.xlsx) a pedido do Ricardo!
        with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx', prefix=f"{target}_mint_data_") as tmp_f:
            # Salva em Excel, engine='openpyxl' é o padrão para .xlsx
            df_final_mint.to_excel(tmp_f.name, index=True, engine='openpyxl') # index=True para salvar o 'timestamp'
            return gr.update(value=tmp_f.name, visible=True, label="Download Dados MinT (.xlsx)") # Atualiza label
            
    except Exception as e:
        print(f"Erro ao gerar arquivo MinT: {e}\n{traceback.format_exc()}")
        return gr.update(value=None, visible=False)
