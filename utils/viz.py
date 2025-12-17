import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose

def gerar_grafico_decomposicao(df_historico, target):
    """Gera o gráfico de decomposição da série temporal usando statsmodels."""
    plt.close('all')
    try:
        decomposicao = seasonal_decompose(df_historico[target].dropna(), model='additive', period=12)
        
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
        
        decomposicao.observed.plot(ax=ax1, legend=False, color='gray')
        ax1.set_ylabel('Observado')
        ax1.set_title(f'Decomposição da Série Temporal para {target.upper()}', fontsize=16, fontweight='bold')
        
        decomposicao.trend.plot(ax=ax2, legend=False, color='#0072B2')
        ax2.set_ylabel('Tendência')
        
        decomposicao.seasonal.plot(ax=ax3, legend=False, color='green')
        ax3.set_ylabel('Sazonalidade')
        
        decomposicao.resid.plot(ax=ax4, legend=False, color='red', marker='o', linestyle='None', markersize=4)
        ax4.set_ylabel('Resíduos')
        
        fig.tight_layout()
        return fig
    except Exception as e:
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.text(0.5, 0.5, f'Não foi possível gerar a decomposição.\nErro: {e}', 
                horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=12, color='red')
        return fig

def gerar_grafico_sazonalidade_anual(df_historico, target):
    """Gera um gráfico focado no padrão de sazonalidade anual média."""
    plt.close('all')
    try:
        if len(df_historico[target].dropna()) < 24:
            raise ValueError("A série de dados é muito curta (< 24 meses) para uma análise de sazonalidade anual confiável.")

        decomposicao = seasonal_decompose(df_historico[target].dropna(), model='additive', period=12)
        
        df_sazonal = pd.DataFrame({
            'sazonalidade': decomposicao.seasonal,
            'mes': decomposicao.seasonal.index.month
        })
        
        media_sazonal_mensal = df_sazonal.groupby('mes')['sazonalidade'].mean()
        
        fig, ax = plt.subplots(figsize=(16, 3))
        media_sazonal_mensal.plot(kind='line', ax=ax, color='green', marker='o', linestyle='-')
        
        ax.set_title(f'Padrão de Sazonalidade Anual para {target.upper()}', fontsize=16, fontweight='bold')
        ax.set_ylabel('Impacto Sazonal Médio')
        ax.set_xlabel('Mês do Ano')
        ax.set_xticks(ticks=range(1, 13))
        ax.set_xticklabels(['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez'])
        ax.axhline(0, color='gray', linestyle='--', linewidth=1)
        ax.grid(True, which='major', linestyle='--', linewidth='0.5')
        
        fig.tight_layout()
        return fig

    except Exception as e:
        fig, ax = plt.subplots(figsize=(16, 3))
        ax.text(0.5, 0.5, f'Não foi possível gerar o gráfico de sazonalidade anual.\nErro: {e}', 
                horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=12, color='red')
        return fig

def gerar_graficos_previsao_futura(df_historico_limpo, df_previsoes_futuras, target, df_historico_completo):
    """Gera os 3 gráficos solicitados para a análise de previsão futura usando Seaborn e Quantis."""
    plt.close('all')

    # Garantir tipo float
    df_previsoes_futuras = df_previsoes_futuras.astype(float)

    fig_futuro, ax1 = plt.subplots(figsize=(16, 3))
    df_historico_recente = df_historico_limpo.tail(12)
    
    # Plot Histórico
    sns.lineplot(data=df_historico_recente, x=df_historico_recente.index, y=target, ax=ax1, label='Real (Últimos 12 Meses)', color='#0072B2')
    
    # Plot Faixa de Confiança Futura
    ax1.fill_between(df_previsoes_futuras.index, df_previsoes_futuras['y_lower'], df_previsoes_futuras['y_upper'], color='#ff0051', alpha=0.15, label='Intervalo Confiança')
    
    # Plot Previsão (Mediana/Main)
    sns.lineplot(data=df_previsoes_futuras, x=df_previsoes_futuras.index, y=target, ax=ax1, label='Previsto', color='#ff0051')
    
    ax1.set_title(f'Previsão Futura vs. Dados Históricos para {target.upper()}', fontsize=16, fontweight='bold')
    ax1.set(xlabel='Data', ylabel="Média Mensal de Audiência (OPM)")
    ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    fig_futuro.tight_layout()
    
    fig_futuro_media, ax2 = plt.subplots(figsize=(16, 3))
    media_mensal_historica = df_historico_completo.groupby(df_historico_completo.index.month)[target].mean()
    media_para_futuro = df_previsoes_futuras.index.map(lambda data: media_mensal_historica.get(data.month))
    
    df_plot_2 = pd.DataFrame({
        'Previsto (Futuro)': df_previsoes_futuras[target].values,
        'Média Mensal Histórica': media_para_futuro.values
    }, index=df_previsoes_futuras.index)
    
    # Adicionamos fill between também aqui para contexto
    ax2.fill_between(df_previsoes_futuras.index, df_previsoes_futuras['y_lower'], df_previsoes_futuras['y_upper'], color='#ff0051', alpha=0.1)

    sns.lineplot(data=df_plot_2, ax=ax2, palette={'Previsto (Futuro)': '#ff0051', 'Média Mensal Histórica': 'green'})
    ax2.lines[1].set_linestyle(':') 
    ax2.set_title(f'Previsão Futura vs. Média Mensal Histórica para {target.upper()}', fontsize=16, fontweight='bold')
    ax2.set(xlabel='Data', ylabel="Média Mensal de Audiência (OPM)")
    ax2.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    fig_futuro_media.tight_layout()
    
    fig_real_media, ax3 = plt.subplots(figsize=(16, 3))
    feature_media_hist = f'media_mensal_historica_{target}'
    if feature_media_hist in df_historico_limpo.columns:
        df_plot_3 = df_historico_limpo[[target, feature_media_hist]].rename(columns={target: 'Real', feature_media_hist: 'Média Mensal Histórica'})
        sns.lineplot(data=df_plot_3, ax=ax3, palette={'Real': '#0072B2', 'Média Mensal Histórica': 'green'})
        ax3.lines[1].set_linestyle(':')
        ax3.set_title('Comparativo: Real vs. Feature de Média Mensal Histórica', fontsize=16, fontweight='bold')
        ax3.set(xlabel='Data', ylabel="Média Mensal de Audiência (OPM)")
        ax3.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        fig_real_media.tight_layout()
    else:
        ax3.text(0.5, 0.5, 'Feature "Média Mensal Histórica" não foi criada.\nGráfico indisponível.', 
                 horizontalalignment='center', verticalalignment='center', transform=ax3.transAxes)

    return fig_futuro, fig_futuro_media, fig_real_media

def gerar_grafico_consolidado(df_historico_completo, df_resultados_cv, df_previsoes_futuras, target):
    """Gera um único gráfico consolidado mostrando a série histórica, as previsões do backtest (CV) e as previsões futuras."""
    plt.close('all')
    fig, ax = plt.subplots(figsize=(16, 3))
    
    df_previsoes_futuras = df_previsoes_futuras.astype(float)

    # Pega o início do backtest para alinhar o início do plot dos dados reais
    start_date_backtest = df_resultados_cv.index.min()
    df_historico_plot = df_historico_completo.loc[df_historico_completo.index >= start_date_backtest]

    # Faixas de confiança (Backtest e Futuro)
    ax.fill_between(df_resultados_cv.index, df_resultados_cv['y_lower'], df_resultados_cv['y_upper'], color='#ff0051', alpha=0.1)
    ax.fill_between(df_previsoes_futuras.index, df_previsoes_futuras['y_lower'], df_previsoes_futuras['y_upper'], color='#D55E00', alpha=0.15)

    sns.lineplot(x=df_historico_plot.index, y=df_historico_plot[target], ax=ax, label='Real', color='#0072B2', linewidth=2)
    sns.lineplot(x=df_resultados_cv.index, y=df_resultados_cv['Previsto'], ax=ax, label='Previsto (Backtest CV)', color='#ff0051', linestyle='--', linewidth=2)
    sns.lineplot(x=df_previsoes_futuras.index, y=df_previsoes_futuras[target], ax=ax, label='Previsto (Futuro)', color='#D55E00', linewidth=2.5)

    ax.set_title(f'Visão Geral: Histórico, Backtest e Previsão Futura para {target.upper()}', fontsize=16, fontweight='bold')
    ax.set(xlabel='Data', ylabel="Média Mensal de Audiência (OPM)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.grid(True, which='major', linestyle='--', linewidth='0.5')

    fig.tight_layout()
    return fig
