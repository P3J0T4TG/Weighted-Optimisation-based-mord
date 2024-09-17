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


def compute_class_weight_sklearn(class_weight, *, classes, y):
    """
    Estima los pesos de las clases para conjuntos de datos desequilibrados.
    
    Parámetros:
    - class_weight: Puede ser un diccionario de clases con sus respectivos pesos, 'balanced', o None.
    - classes: Array de las clases únicas en el conjunto de datos.
    - y: Array de las etiquetas del conjunto de datos.
    
    Retorna:
    - Array de pesos para las clases proporcionadas.
    """

    # Importa LabelEncoder de sklearn para codificar etiquetas con valor entre 0 y n_classes-1
    from sklearn.preprocessing import LabelEncoder

    # Verifica si todas las etiquetas en y están presentes en classes
    if set(y) - set(classes):
        raise ValueError("classes should include all valid labels that can be in y")
    
    # Si class_weight es None o está vacío, asigna pesos uniformes a todas las clases
    if class_weight is None or len(class_weight) == 0:
        weight = np.ones(classes.shape[0], dtype=np.float64, order="C")
    elif class_weight == "balanced":
        # Si class_weight es 'balanced', calcula los pesos basados en la frecuencia de las clases
        le = LabelEncoder()
        y_ind = le.fit_transform(y)  # Transforma las etiquetas a valores enteros
        if not all(np.isin(classes, le.classes_)):
            raise ValueError("classes should have valid labels that are in y")

        # Calcula los pesos inversos de la frecuencia de las clases
        recip_freq = len(y) / (len(le.classes_) * np.bincount(y_ind).astype(np.float64))
        weight = recip_freq[le.transform(classes)]  # Asigna los pesos calculados a las clases correspondientes
    else:
        # Si class_weight es un diccionario definido por el usuario, asigna los pesos especificados
        weight = np.ones(classes.shape[0], dtype=np.float64, order="C")
        unweighted_classes = []  # Lista para almacenar clases sin peso definido
        for i, c in enumerate(classes):
            if c in class_weight:
                weight[i] = class_weight[c]  # Asigna el peso definido por el usuario
            else:
                unweighted_classes.append(c)  # Añade a la lista si la clase no tiene peso definido

        # Verifica si todas las clases tienen un peso asignado
        n_weighted_classes = len(classes) - len(unweighted_classes)
        if unweighted_classes and n_weighted_classes != len(class_weight):
            unweighted_classes_user_friendly_str = np.array(unweighted_classes).tolist()
            raise ValueError(
                f"The classes, {unweighted_classes_user_friendly_str}, are not in class_weight"
            )

    return weight  # Retorna el array de pesos para las clases

import numpy as np
from sklearn.preprocessing import LabelEncoder

def compute_class_weights_improved(y, class_weight='balanced'):
    """
    Calcula los pesos de las clases, permitiendo diferentes estrategias de ponderación.
    """
    # Codificar etiquetas con valores entre 0 y n_classes-1
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    classes = np.unique(y_encoded)
    
    if class_weight == 'balanced':
        # Calcular el peso de cada clase como el inverso de su frecuencia, similar a la segunda función
        n_samples = len(y)
        recip_freq = n_samples / (len(classes) * np.bincount(y_encoded).astype(np.float64))
        class_weights = recip_freq[le.transform(classes)]
    else:
        # Asignar pesos uniformes si class_weight no es 'balanced'
        class_weights = np.ones(len(classes), dtype=np.float64)
    
    # Normalizar los pesos de las clases para que sumen 1
    class_weights = class_weights / np.sum(class_weights)
    
    return class_weights