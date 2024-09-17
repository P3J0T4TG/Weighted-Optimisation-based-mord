import pandas as pd
from pathlib import Path
import numpy as np
import os

# ruta de los datasets
data_dir_10 = Path('./0_Datasets/No_discretizados/Discretizados/')
bins=5

list_datasets = ['pyrim','auto','triazines','abalone','housing','wpbc','diabetes','stock','machine']

#data_df_resumen = pd.DataFrame(columns=['dataset','n_bins','#Patterns','#Attributes','#Classes','Class Distribution'])#,'#Patterns per class'])
data_df_resumen = pd.DataFrame(columns=['Dataset', '#Pat.', '#Attr.', '#Classes', 'Class distribution'])

list_df_datasets = []

for dataset in list_datasets:
    #######################
    # Cargamos los datos  #
    #######################
    #df_aux=pd.DataFrame(columns=['Resample','#Patterns','#Training Patterns','#Test Patterns','#Input Variables','#Patterns per class', 'Train_class_distribution', 'Test_class_distribution'])
    df_aux = pd.DataFrame(columns=['Resample', '#Patterns', '#Training Patterns', '#Test Patterns', '#Input Variables', 'Class distribution'])

    #Ruta dataset
    path_dataset = os.path.join(data_dir_10, str(bins) + '_bins', dataset + '/')
    for i in range(30):
        ##########################
        # CARGAMOS LOS RESAMPLES #
        ##########################
        train = pd.read_csv(path_dataset + 'train_'+ dataset +'.' + str(i), header=None, sep=' ' )
        test = pd.read_csv(path_dataset + 'test_'+ dataset +'.' + str(i), header=None, sep=' ')

#################
# PROCESAMIENTO #
#################
        # Concatena las columnas objetivo
        clases = pd.concat([train.iloc[:, -1], test.iloc[:, -1]])
        # Contamos el numero de valores para cada clase
        paternpp= clases.value_counts()
        paternpp_str = str(tuple(paternpp.values))  # Convertimos paternpp a lista y luego a cadena para asegurar la compatibilidad
        #--------->Sacamos los datos de cada resample
        #df_aux.loc[i] = [i,train.shape[0]+test.shape[0],train.shape[0],test.shape[0],train.shape[1]-1,paternpp_str, str(tuple(train.iloc[:, -1].value_counts().values)), str(tuple(test.iloc[:, -1].value_counts().values))]
        df_aux.loc[i] = [i, train.shape[0] + test.shape[0], train.shape[0], test.shape[0], train.shape[1] - 1, paternpp_str]

    #--------->Calculamos la media de los datos de los resamples
    num_classes = len(paternpp)
    class_distribution = df_aux['Class distribution'].iloc[0]  # Usamos la distribución de la primera iteración como ejemplo
    data_df_resumen.loc[len(data_df_resumen)] = [dataset, df_aux['#Patterns'].mean(), df_aux['#Input Variables'].mean(), num_classes, class_distribution]
    #data_df_resumen.loc[len(data_df_resumen)] = [dataset,bins,df_aux['#Patterns'].mean(),df_aux['#Training Patterns'].mean(),df_aux['#Test Patterns'].mean(),df_aux['#Input Variables'].mean()] #me gustaria añadir la media de Pattern per class, para cuando se haga la division por cuartiles
    #--------->Guardamos los datos de los resamples
    list_df_datasets.append(df_aux)

#################
# SAVE AS .xlsx #
#################

with pd.ExcelWriter('0.Resumen_Datasets_5_bins.xlsx') as writer:
    data_df_resumen.to_excel(writer, sheet_name='Resumen', index=False)
    for i in range(len(list_datasets)):
        list_df_datasets[i].to_excel(writer, sheet_name=list_datasets[i], index=False)





##################
# SAVE AS LATEX  #
##################

data_df_resumen.to_latex('0.Resumen_Datasets_5bins.tex', index=False)
