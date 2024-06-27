# 26-06-2024
- *Tablas.py* Añadido script para extraer las tablas comparativas de los metodos a partir de los resultados
- Añadidas las tablas extraidas de los resultados
- Métrica *AMAE* modificada para evitar el error de dividir por cero (se ha añadido al denominador un $\epsilon = 10^{⁻3}$)
- Añadido script que extrae los datos de los resultados del review
- Añadidos csv con MAE, MZE y Time del review (tablas de review: [Ordinal regression methods: survey and experimental study](http://dx.doi.org/10.1109/TKDE.2015.2457911) para compararlas con las generadas)

**Proximos arreglos**
- Falta unir las tablas para ponerlas bonicas
- Añadir Script para discretizar datasets de regresión (variable respuesta continua -> discreta)
- Aplicar la discretización a los datasets
- Realizar experimentos para estos
- Extraer resultados y tablas de estos y compararlos con los del review

# 19-06-2024

- Metodo logisticit_desb y logisticat_desb implementados (de dos formas distintas, pero se obtienen los mismos resultados, mas o menos tardan lo mismo en ejecutarse)
- Se han realizado los experimentos con todos los dataset y se han recopilado los resultados
	- ..._Desbalanceados.xlsx solo los resultados de los nuevos metodos
	- ..._Todos.xlsx con todos los resultados  de todos los métodos
- Los nuevos metodos heredan los errores de los originales (algunos fallos al calcular la metrica amaes, convergencia debil al usar lbfgs

# 15-04-2024 

- He lanzado todos los experimentos con todos los resamples de todos los dataset menos bondrate
- En el archivo list_dataset_v2.txt está toda la información relativa a los errores y warnings que me han ido surgiendo durante los lanzamientos
- Quizas el mayor de los problemas se tiene con el dataset bondrate que da unos errores raros con la libreria de sklearn y joblib, además de predecir mal las etiquetas (estará todo enlazado)


