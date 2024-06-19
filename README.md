# 19-06-2024

- Metodo logisticit_desb y logisticat_desb implementados (de dos formas distintas, pero se obtienen los mismos resultados, mas o menos tardan lo mismo en ejecutarse)
- Se han realizado los experimentos con todos los dataset y se han recopilado los resultados
	- ..._Desbalanceados.xlsx solo los resultados de los nuevos metodos
	- ..._Todos.xlsx con todos los resultados  de todos los métodos
- Los nuevos metodos heredan los errores de los originales (algunos fallos al calcular la metrica amaes, convergencia debil al usar lbfgs
	
# 15-04-2024 #

- He lanzado todos los experimentos con todos los resamples de todos los dataset menos bondrate
- En el archivo list_dataset_v2.txt está toda la información relativa a los errores y warnings que me han ido surgiendo durante los lanzamientos
- Quizas el mayor de los problemas se tiene con el dataset bondrate que da unos errores raros con la libreria de sklearn y joblib, además de predecir mal las etiquetas (estará todo enlazado)


