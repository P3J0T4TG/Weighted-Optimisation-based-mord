"""
Script para generar las tablas comparitivas de métodos a partir de los resultados de los experimentos.
"""

import pandas as pd
import matplotlib.pyplot as plt


# Definir la ruta del archivo y cargar los datos %
#archivo_xlsx_path = './prepared_results/20240619_193506_Todos.xlsx'
#archivo_xlsx_path = './prepared_results/20240626_141032_MAE_GridSearch+AMAE_bien.xlsx'
archivo_xlsx_path = './prepared_results/20240626_154803_CCR_GridSearch+AMAE_bien.xlsx'
df = pd.read_excel(archivo_xlsx_path)


#---------------------------------------------------------------------------------------------------------------------------------------#
#################
# PROCESAMIENTO #
#################
    #Primero vemos que datasets y metodos se han usado:
datasets_list = list(df['dataset'].unique())
modelos_list = list(df['estimator_name'].unique())
    #Creamos el dataset de salida cuyas columnas seran [Datasets, ...Modelos...]
columnas= ['dataset'] + modelos_list
df_final = pd.DataFrame(columns=columnas)

    #Para cada par (dataset, modelo) calcularemos su MAE medio y lo añadiremos a la tabla final
for dataset in datasets_list:
    fila= [dataset]
    for modelo in modelos_list:
        #Generamos la instancia de df_final que tiene la forma (dataset, MAE_modelo1, ...., MAE_modeloN)
        mae= df[(df['dataset']==dataset) & (df['estimator_name']==modelo)]['MAE'].mean()
        fila.append(mae)
    df_final.loc[len(df_final)]= fila
#-------------------------------------------------------------------------------------------------------------------------------------#

###############
# GENERACIÓN  #
###############

#Pasamos la tabla a formato latex
#print(df_final.to_latex(index=False))

# Guardar la tabla en un nuevo archivo
df_final.to_csv('./Tabla_comparativa.csv', index=False)

# Generar una imagen de la tabla
# fig, ax = plt.subplots(figsize=(12, 2))  # Ajusta el tamaño según sea necesario
# ax.axis('tight')
# ax.axis('off')
# ax.table(cellText=df_final.values, colLabels=df_final.columns, cellLoc = 'center', loc='center')

# plt.savefig('tabla_articulo.png', dpi=300)