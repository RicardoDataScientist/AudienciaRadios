import warnings
import pandas as pd
import numpy as np
import geopandas
import geobr
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from shapely.geometry import MultiPolygon

# --- Configurações Iniciais ---
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# --- DADOS DE EXEMPLO (AQUI VOCÊ PODE PLUGAR SEUS DADOS REAIS) ---
# O ideal é carregar seu .csv ou .xlsx aqui com o pandas.
# Ex: df_real = pd.read_excel("audiencia03.xlsx")
# Por enquanto, vamos usar dados fictícios para o script funcionar.

print("📊 Criando dados de audiência de exemplo...")
cidades_selecionadas = ['Serra', 'Vitória', 'Viana', 'Cariacica', 'Guarapari']
dados_audiencia = pd.DataFrame({
    'MUNICIPIO': cidades_selecionadas,
    'AUDIENCIA_FM': np.random.randint(5, 50, size=len(cidades_selecionadas))
})
print("Dados de exemplo criados:")
print(dados_audiencia)

# --- LÓGICA PRINCIPAL PARA GERAR O MAPA ---

try:
    # --- 1. Preparar os dados do usuário ---
    # Renomeia as colunas do seu DataFrame para os nomes que o mapa espera
    # ATENÇÃO: Altere 'MUNICIPIO' e 'AUDIENCIA_FM' para os nomes exatos das suas colunas!
    df_usuario = dados_audiencia.rename(columns={'MUNICIPIO': 'name_muni', 'AUDIENCIA_FM': 'valor'})

    # --- 2. Baixar e preparar os dados do mapa ---
    print("\n🗺️ Baixando a malha geográfica do Espírito Santo...")
    mapa_es_completo = geobr.read_municipality(code_muni='ES', year=2020)

    # --- 3. Juntar seus dados com o mapa ---
    print("🔗 Unindo seus dados ao mapa...")
    mapa_com_dados = mapa_es_completo.merge(df_usuario[['name_muni', 'valor']], on='name_muni', how='left')
    
    # Filtra apenas para as cidades que o usuário selecionou
    mapa_filtrado = mapa_com_dados[mapa_com_dados['name_muni'].isin(cidades_selecionadas)].copy()
    mapa_filtrado['valor'].fillna(0, inplace=True) # Preenche com 0 cidades sem dados

    if mapa_filtrado.empty:
        raise ValueError("Nenhuma cidade correspondente encontrada. Verifique os nomes na sua planilha.")

    # --- 4. Limpar geometrias (remover ilhas distantes) ---
    print("🏝️ Removendo ilhas distantes para focar no continente...")
    geometrias_limpas = []
    for geom in mapa_filtrado['geometry']:
        if isinstance(geom, MultiPolygon):
            maior_poligono = max(geom.geoms, key=lambda p: p.area)
            geometrias_limpas.append(maior_poligono)
        else:
            geometrias_limpas.append(geom)
    mapa_filtrado['geometry'] = geopandas.GeoSeries(geometrias_limpas, index=mapa_filtrado.index, crs=mapa_filtrado.crs)

    # --- 5. Plotar o mapa ---
    print("🎨 Desenhando o mapa de calor geográfico...")
    plt.close('all')
    minx, miny, maxx, maxy = mapa_filtrado.total_bounds
    proporcao = (maxx - minx) / (maxy - miny) if (maxy - miny) != 0 else 1
    altura_polegadas, dpi = 8, 150
    largura_polegadas = altura_polegadas * proporcao
    
    fig, ax = plt.subplots(1, 1, figsize=(largura_polegadas, altura_polegadas), dpi=dpi)

    mapa_filtrado.plot(
        column='valor', ax=ax, edgecolor='black', linewidth=0.8,
        cmap='coolwarm', legend=True,
        legend_kwds={'label': "Pontos de Audiência", 'orientation': "vertical", "shrink": 0.6}
    )
    ax.set_xlim(minx, maxx); ax.set_ylim(miny, maxy)

    for idx, row in mapa_filtrado.iterrows():
        ponto_central = row.geometry.representative_point()
        texto_label = f"{row['name_muni']}\n({row['valor']:.1f})"
        ax.text(
            x=ponto_central.x, y=ponto_central.y, s=texto_label,
            ha='center', va='center', fontsize=9, fontweight='bold', color='#333',
            path_effects=[path_effects.withStroke(linewidth=3, foreground='white')]
        )

    ax.set_title(f'Heatmap de Audiência por Município', fontsize=18, weight='bold')
    ax.set_axis_off()
    fig.tight_layout()

    print("\n✅ Mapa gerado com sucesso! Exibindo...")
    plt.show()

except Exception as e:
    print(f"\n❌ Ocorreu um erro: {str(e)}")
