# 1. Importar las bibliotecas necesarias
import numpy as np
from sklearn import preprocessing

# 2. Definir una función para calcular los pesos de las clases
def compute_class_weights(y):
    # 3. Contar el número de ocurrencias de cada clase en las etiquetas `y`
    n_classes = len(np.unique(y))
    n_samples = len(y)

    # 4. Calcular el peso de cada clase como el inverso de su frecuencia
    class_weights = n_samples / (n_classes * np.bincount(y))

    # 5. Normalizar los pesos de las clases para que sumen 1
    class_weights = class_weights / np.sum(class_weights)

    # 6. Devolver los pesos de las clases
    return class_weights