import pandas as pd
import numpy as np
from scipy.stats import rankdata
import glob

###################
# CARGA DE DATOS  #
###################
#------------------> RESULTADOS TFM
#results_tfm=pd.read_excel('../20240626_154803_CCR_GridSearch+AMAE_bien.xlsx')
results_tfm=pd.read_excel('prepared_results/20240626_154803_CCR_GridSearch+AMAE_bien.xlsx')

#------------------> RESULTADOS REVIEW
#Ruta de los resultados del review
ruta= "Review/Ordinal/*.csv"
#leemos todos los archivos csv de la carpeta Ordinal y los guardamos en una lista de dataframes
df_list= [pd.read_csv(f,header=1) for f in glob.glob(ruta)]
#Creamos una lista con el nombre de los archivos csv y extraemos el nombre de los métodos
nombres= [f for f in glob.glob(ruta)]
metodos = [name.split("Results-")[1].split(".csv")[0] for name in nombres]
#Lista de Dataset
#dataset_list=resultados_tfm['dataset'].unique()
dataset_list=['contact-lenses', 'pasture', 'squash-stored', 'squash-unstored', 'tae', 'newthyroid', 'balance-scale', 'SWD', 'car', 'bondrate', 'toy', 'eucalyptus', 'LEV', 'automobile', 'winequality-red', 'ESL', 'ERA']


####################
# PREPROCESAMIENTO #
####################

# -----------------> PREPROCESAMIENTO TFM
# Extraemos del dataset las columnas necesarias
df_TFM = results_tfm[['dataset', 'estimator_name', 'random_state', 'MAE', 'CCR']]#, 'QWK', 'AMAE', 'MMAE']]
#ordenamos el dataset por  random_state, dataset, estimator_name
df_TFM = df_TFM.sort_values(by=['random_state', 'dataset', 'estimator_name'])
#reseteamos el indice
df_TFM = df_TFM.reset_index(drop=True)
#Calculamos el MZE a partir del CCR
df_TFM["MZE"]=1-df_TFM["CCR"]


# -----------------> PREPROCESAMIENTO REVIEW
for df, metodo in zip(df_list, metodos):
    print("Procesando: ", metodo, "...\n")
    """
    Se ha colado la primera columna de los df como el indice luego hay que arreglar esto:
    """
    df.reset_index(inplace=True)
    #corremos el encabezado una posicion a la izquierda
    encabezados=list(df.columns)
    #eliminamos el primer elemento
    encabezados.pop(0)
    #eliminamos ultima columna de df
    df.drop(df.columns[len(df.columns)-1], axis=1, inplace=True)
    df.columns=encabezados

    #Añadimos la columna "random_state" al df cuyo valor sera el indice de la fila
    df['random_state']=df.index

"""
Tras tener el df en el formato correcto, vamos a crear un df_REVIEW que contenga la misma estructura que el df_TFM
"""

df_REVIEW = pd.DataFrame(columns=['dataset', 'estimator_name', 'random_state', 'MAE', 'MZE'], index=range(1022))

#Iteramos sobre los dataframes de la lista df_list
for i in range(0, len(df_list)):
    print("Procesando: ", metodos[i], "...\n")
    #iteramos sobre los datasets
    for j in range(len(dataset_list)):
        print("Procesando dataset: ", dataset_list[j], "...\n")
        #Cada dataset ocupa 30 filas (las correspondientes a los 30 random_state)
        inicio = i*len(dataset_list)*30+j*30
        fin = inicio + 29
        #De la fila 0 a la 29 asignamos el dataset, el random_state y el metodo usado
        df_REVIEW.loc[inicio:fin, 'dataset'] = dataset_list[j]
        df_REVIEW.loc[inicio:fin, 'random_state'] = df_list[i]['random_state'].values[:30]
        df_REVIEW.loc[inicio:fin, 'estimator_name'] = metodos[i]
        
        #Asignamos las columnas MAE y MZE
        df_REVIEW.loc[inicio:fin, 'MZE'] = df_list[i].iloc[:, 3*j].values
        df_REVIEW.loc[inicio:fin, 'MAE'] = df_list[i].iloc[:, 3*j+1].values
        #Calculamos el CCR a partir del MZE
        df_REVIEW.loc[inicio:fin, 'CCR'] = 1-df_REVIEW.loc[inicio:fin, 'MZE']



#-----------------------------> UNION DE DATAFRAMES
#Unimos los dos dataframes
df_union = pd.concat([df_TFM, df_REVIEW],axis=0, ignore_index=True, sort=False)
#ordenamos el dataframe por dataset, estimator_name, random_state
df_union = df_union.sort_values(by=['dataset', 'random_state', 'estimator_name'])
#reseteamos el indice
df_union = df_union.reset_index(drop=True)




############
# RANKDATA #
############

#Asegúrate de que 'MAE' y 'MZE' son numéricos
df_union['MAE'] = pd.to_numeric(df_union['MAE'], errors='coerce')
df_union['MZE'] = pd.to_numeric(df_union['MZE'], errors='coerce')

# Realizamos el rankdata para cada semilla segun MAE y MZE
for i in range(0, len(df_union), df_union['estimator_name'].nunique()):
    df_union.loc[i:i+df_union['estimator_name'].nunique()-1, 'rank_MAE'] = rankdata(df_union.loc[i:i+df_union['estimator_name'].nunique()-1, 'MAE'].to_numpy())#, method='dense')
    df_union.loc[i:i+df_union['estimator_name'].nunique()-1, 'rank_MZE'] = rankdata(df_union.loc[i:i+df_union['estimator_name'].nunique()-1, 'MZE'].to_numpy())#, method='dense')


#Calculamos las medias de cada dataset
df_union['mean_rank_MAE'] = df_union.groupby(['dataset', 'estimator_name'])['rank_MAE'].transform('mean')
df_union['mean_rank_MZE'] = df_union.groupby(['dataset', 'estimator_name'])['rank_MZE'].transform('mean')

#Calculamos las medias de las medias obtenidas antes segun cada estimador
df_union['mean_mean_rank_MAE'] = df_union.groupby(['estimator_name'])['mean_rank_MAE'].transform('mean')
df_union['mean-mean_rank_MZE'] = df_union.groupby(['estimator_name'])['mean_rank_MZE'].transform('mean')

#extraemos de df_union una tabla que contenga dataset, estimator_name, mean_rank_MAE, mean_rank_MZE
tabla1 = df_union[['dataset', 'estimator_name', 'mean_rank_MAE', 'mean_rank_MZE']]
tabla1 = tabla1.drop_duplicates()
#extraemos una tabla que contenga estimator_name, mean_mean_rank_MAE, mean_mean_rank_MZE
tabla2 = df_union[['estimator_name', 'mean_mean_rank_MAE', 'mean-mean_rank_MZE']]
tabla2 = tabla2.drop_duplicates()


########
# SAVE #
########
#guardamos  df_union, tabla1 y tabla2 en un mismo archivo .xlsx
with pd.ExcelWriter('RANKING_def_v2.xlsx') as writer:
    df_union.to_excel(writer, sheet_name='Todo_rank', index=False)
    tabla1.to_excel(writer, sheet_name='Rank_por_dataset', index=False)
    tabla2.to_excel(writer, sheet_name='Rank_por_estimador', index=False)