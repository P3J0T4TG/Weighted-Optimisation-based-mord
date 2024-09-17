"""
Script para generar las tablas comparitivas de métodos a partir de run_results_collector.py
"""

import pandas as pd
import matplotlib.pyplot as plt


# Definir la ruta del archivo y cargar los datos %
archivo_xlsx_path = './prepared_results/20240828_143109_Lanzamiento_2D5.xlsx'
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
#df_final = pd.DataFrame(columns=columnas)

metrics=["QWK","MAE","1-off","CCR","MZE","MS","BalancedAccuracy","AMAE","MMAE"]
    #Para cada par (dataset, modelo) calcularemos su MAE medio y lo añadiremos a la tabla final
df_list=[]
for metrica in metrics:
    df_final = pd.DataFrame(columns=columnas)
    for dataset in datasets_list:
        fila= [dataset]
        for modelo in modelos_list:
            #Generamos la instancia de df_final que tiene la forma (dataset, Metrica_modelo_1, ...., Metrica_modelo_N)
            metric= df[(df['dataset']==dataset) & (df['estimator_name']==modelo)][metrica].mean() #Calculamos la media de la metrica
            #calculamos la desviacion estandar
            std= df[(df['dataset']==dataset) & (df['estimator_name']==modelo)][metrica].std()
            result= str(round(metric,3)) + " ± " + str(round(std,3))
            fila.append(result)
        df_final.loc[len(df_final)]= fila
    df_list.append(df_final)
#-------------------------------------------------------------------------------------------------------------------------------------#
#################
# GUARDAR TABLA #
#################

with pd.ExcelWriter('Metrics_2D5_std_.xlsx') as writer:
    for metrica,j in zip(metrics,range(len(metrics))):
        aux= df_list[j]
        aux.to_excel(writer, sheet_name=metrica, index=False)
#-------------------------------------------------------------------------------------------------------------------------------------#

###############
# GENERACIÓN  #
###############

#Pasamos la tabla a formato latex
#print(df_final.to_latex(index=False))

# Guardar la tabla en un nuevo archivo
#df_final.to_csv('./Lanzamiento_1D10_Resumen.csv', index=False)
#df_final.to_csv('./David_results_AMAE.csv', index=False)
# Generar una imagen de la tabla
# fig, ax = plt.subplots(figsize=(12, 2))  # Ajusta el tamaño según sea necesario
# ax.axis('tight')
# ax.axis('off')
# ax.table(cellText=df_final.values, colLabels=df_final.columns, cellLoc = 'center', loc='center')

# plt.savefig('tabla_articulo.png', dpi=300)
