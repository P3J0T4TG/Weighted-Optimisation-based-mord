"""
Script para generar las tablas comparitivas de métodos a partir de los resultados del review.

Cada .csv del review son las métricas MZE, MAE y Time de un metodo en todos los datasets.
Para cada .csv se calcula la media de las métricas y se crea un dataframe con las medias.

"""
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

################################
# Rutas, Info y Carga de Datos #
################################

ruta= "Review/Ordinal/*.csv"
print("Leyendo .csv de la ruta: ", ruta, "...\n")
#leemos todos los archivos csv de la carpeta Ordinal y los guardamos en una lista de dataframes
df_list= [pd.read_csv(f,header=1) for f in glob.glob(ruta)]
#Creamos una lista con el nombre de los archivos csv y extraemos el nombre de los métodos
nombres= [f for f in glob.glob(ruta)]
metodos = [name.split("Results-")[1].split(".csv")[0] for name in nombres]
#Lista de Dataset
dataset_list=['contact-lenses', 'pasture', 'squash-stored', 'squash-unstored', 'tae', 'newthyroid', 'balance-scale', 'SWD', 'car', 'bondrate', 'toy', 'eucalyptus', 'LEV', 'automobile', 'winequality-red', 'ESL', 'ERA']

print("Dataframes cargados correctamente: ", metodos, "\n")


#Comprobaciond e donde se guardan los archivos
# Verificar si el directorio existe. Si no, crearlo.
directorio = "Review/Ordinal/Resumen"
if not os.path.exists(directorio):
    os.makedirs(directorio)

########################
# ESQUELETO DF FINALES #
########################
# df_MAE=pd.DataFrame(columns=["Dataset", metodos])
# df_MZE=pd.DataFrame(columns=["Dataset", metodos])
# df_Time=pd.DataFrame(columns=["Dataset", metodos])

df_MAE = pd.DataFrame({"Dataset": dataset_list})
df_MZE = pd.DataFrame({"Dataset": dataset_list})
df_Time = pd.DataFrame({"Dataset": dataset_list})

df_final_list=[]

###############################
# PROCESAMIENTO Y GENERACIÓN  #
###############################

for df, metodo in zip(df_list, metodos):
    print("Procesando: ", metodo, "...\n")
    #Calculamos la media de las metricas de un metodo
    df_mean=df.mean()
    df_final=pd.DataFrame(columns=["Dataset", "MZE", "MAE", "Time"])
    df_final["Dataset"]=dataset_list
    # rellenamos df_final con los elementos de df_mean, cada 3 filas de df_mean corresponden a un dataset
    df_final["MZE"] = df_mean[0::3].values
    df_final["MAE"] = df_mean[1::3].values
    df_final["Time"] = df_mean[2::3].values
    df_final_list.append(df_final)
    print("Guardando .csv de ", metodo, "...\n")
    #Guardamos df_final en un csv de nombre metodo
    df_final.to_csv("Review/Ordinal/Resumen/"+metodo+".csv", index=False)
    print("Guardado correctamente.\n")
    #Rellenamos los dataframes finales con los datos de df_final
    print("Rellenando dataframes finales...\n")

    df_MAE[metodo]=df_final["MAE"]
    df_MZE[metodo]=df_final["MZE"]
    df_Time[metodo]=df_final["Time"]

print("Dataframes finales rellenados correctamente.\n")
#Guardamos los dataframes finales en csv
df_MAE.to_csv("Review/Ordinal/Resumen/MAE_review.csv", index=False)
df_MZE.to_csv("Review/Ordinal/Resumen/MZE_review.csv", index=False)
df_Time.to_csv("Review/Ordinal/Resumen/Time_review.csv", index=False)





