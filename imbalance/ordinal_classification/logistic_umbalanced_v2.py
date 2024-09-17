"""
Modificar en este archivo la implementación de las clases desbalanceadas.
Una vez implementados los métodos, añadir dichas funciones a set_estimators.py
"""

import numpy as np
from scipy import optimize
from sklearn import base, metrics
from sklearn.utils.validation import check_X_y

def compute_class_weights(y):
    # Contar el número de ocurrencias de cada clase en las etiquetas `y`
    classes = np.unique(y)
    n_classes = len(classes)
    n_samples = len(y)

    #  Calcular el peso de cada clase como el inverso de su frecuencia
    class_weights = n_samples / (n_classes * np.bincount(y))

    #  Convertir los pesos de las clases a un diccionario
    class_weights_dict = {classes[i]: class_weights[i] for i in range(n_classes)}

    # Calculamos los pesos de las muestras a partir de los pesos de las clases
    sample_weight = np.ones(n_samples)                  #inicializamos los pesos de las muestras a 1
    for class_, weight in class_weights_dict.items():   #para cada clase y su peso
        sample_weight[y == class_] = weight             #asignamos el peso de la clase a las muestras de esa clase

    #  Devolver los pesos de las muestras 
    return sample_weight, class_weights_dict


def sigmoid(t): #NO NECESITA MODIFICACIONES
    # sigmoid function, 1 / (1 + exp(-t))
    # stable computation

    # Crear una máscara booleana que es Verdadera donde los elementos de `t` son mayores que 0.
    idx = t > 0
    # Crear un nuevo array del mismo tamaño que `t`, lleno de ceros.
    out = np.zeros_like(t)
    # Para los valores de `t` que son mayores que 0, calcula la función sigmoide de la manera normal.
    out[idx] = 1.0 / (1 + np.exp(-t[idx]))
    # Para los valores de `t` que no son mayores que 0, calcular el exponencial de `t` primero.
    exp_t = np.exp(t[~idx])
    # Luego, calcular la función sigmoide utilizando el exponencial de `t`.
    # Esto ayuda a evitar el desbordamiento numérico cuando `t` es un número muy grande.
    out[~idx] = exp_t / (1.0 + exp_t)
    return out


def log_loss(Z): #NO NECESITA MODIFICACIONES
    # stable computation of the logistic loss
    # Crear una máscara booleana que es Verdadera donde los elementos de `Z` son mayores que 0.
    idx = Z > 0
    # Crear un nuevo array del mismo tamaño que `Z`, lleno de ceros.
    out = np.zeros_like(Z)
    # Para los valores de `Z` que son mayores que 0, calcular la pérdida logarítmica de la manera normal.
    out[idx] = np.log(1 + np.exp(-Z[idx]))
    # Para los valores de `Z` que no son mayores que 0, calcular la pérdida logarítmica utilizando una fórmula alternativa.
    # Esto ayuda a evitar el desbordamiento numérico cuando `Z` es un número muy grande.
    out[~idx] = -Z[~idx] + np.log(1 + np.exp(Z[~idx]))
    return out


def obj_margin(x0, X, y, alpha, n_class, weights, L, sample_weight): #NO NECESITA MODIFICACIONES
    """
    Función objetivo para la formulación general basada en márgenes
    """

    # Extrae los parámetros del modelo de `x0`.
    w = x0[: X.shape[1]]
    # Extrae los parámetros de los márgenes de `x0`.
    c = x0[X.shape[1] :]
    # Calcula los márgenes.
    theta = L.dot(c)
    # Calcula los pesos de las clases para cada muestra.
    loss_fd = weights[y]
    # Calcula el producto de la matriz de características y los parámetros del modelo.
    Xw = X.dot(w)
    # Calcula la diferencia entre los márgenes y el producto de la matriz de características y los parámetros del modelo.
    Alpha = theta[:, None] - Xw  # (n_class - 1, n_samples)
    # Calcula una matriz que indica si cada muestra está en el lado correcto del margen.
    S = np.sign(np.arange(n_class - 1)[:, None] - y + 0.5)

## MODIFICAR PARA INCLUIR PESOS (?)
    # Calcula la pérdida logarítmica para cada muestra, ponderada por los pesos de las clases.
    err = loss_fd.T * log_loss(S * Alpha)
    # Si se han proporcionado pesos para las muestras, multiplica la pérdida por estos pesos.


    if sample_weight is not None:
        err *= sample_weight
    # Suma la pérdida ponderada para todas las muestras para obtener la pérdida total.
    obj = np.sum(err)
    # Añade la regularización a la pérdida total.
    obj += alpha * 0.5 * (np.dot(w, w))
    # Devuelve la pérdida total.
    return obj


def grad_margin(x0, X, y, alpha, n_class, weights, L, sample_weight): #NO NECESITA MODIFICACIONES
    """
    Gradiente para la formulación general basada en márgenes
    """

    # Extrae los parámetros del modelo de `x0`.
    w = x0[: X.shape[1]]
    # Extrae los parámetros de los márgenes de `x0`.
    c = x0[X.shape[1] :]
    # Calcula los márgenes.
    theta = L.dot(c)
    # Calcula los pesos de las clases para cada muestra.
    loss_fd = weights[y]
    # Calcula el producto de la matriz de características y los parámetros del modelo.
    Xw = X.dot(w)
    # Calcula la diferencia entre los márgenes y el producto de la matriz de características y los parámetros del modelo.
    Alpha = theta[:, None] - Xw  # (n_class - 1, n_samples)
    # Calcula una matriz que indica si cada muestra está en el lado correcto del margen.
    S = np.sign(np.arange(n_class - 1)[:, None] - y + 0.5)
    # Calcula el gradiente de la pérdida logarítmica para cada muestra, ponderada por los pesos de las clases.
    Sigma = S * loss_fd.T * sigmoid(-S * Alpha)
    # Si se han proporcionado pesos para las muestras, multiplica el gradiente de la pérdida por estos pesos.
    if sample_weight is not None:
        Sigma *= sample_weight
    # Calcula el gradiente de los parámetros del modelo.
    grad_w = X.T.dot(Sigma.sum(0)) + alpha * w
    # Calcula el gradiente de los márgenes.
    grad_theta = -Sigma.sum(1)
    # Calcula el gradiente de los parámetros de los márgenes.
    grad_c = L.T.dot(grad_theta)
    # Devuelve el gradiente de todos los parámetros.
    return np.concatenate((grad_w, grad_c), axis=0)


def threshold_fit(
    X,
    y,
    alpha,
    n_class,
    mode="AE",
    max_iter=1000,
    verbose=False,
    tol=1e-12,
    sample_weight=None,
):
    """
    Solve the general threshold-based ordinal regression model
    using the logistic loss as surrogate of the 0-1 loss

    Parameters
    ----------
    mode : string, one of {'AE', '0-1', 'SE'}


    -----> Añadido el argumento class_weight que se calcula en la funcion fit de los métodos

    """

    # Verifica que los datos de entrada sean válidos y estén en el formato correcto
    X, y = check_X_y(X, y, accept_sparse="csr")
    unique_y = np.sort(np.unique(y))
    if not np.all(unique_y == np.arange(unique_y.size)):
        raise ValueError(
            "Values in y must be %s, instead got %s"
            % (np.arange(unique_y.size), unique_y)
        )

    n_samples, n_features = X.shape

    # Crea una matriz que se utiliza para convertir entre los parámetros de los umbrales y los parámetros de los márgenes
    L = np.zeros((n_class - 1, n_class - 1))
    L[np.tril_indices(n_class - 1)] = 1.0

    # Dependiendo del valor del parámetro `mode`, se calcula una matriz que define cómo se calcula la pérdida para cada par de clases
    if mode == "AE":
        loss_fd = np.ones((n_class, n_class - 1))
    elif mode == "0-1":
        loss_fd = np.diag(np.ones(n_class - 1)) + np.diag(np.ones(n_class - 2), k=-1)
        loss_fd = np.vstack((loss_fd, np.zeros(n_class - 1)))
        loss_fd[-1, -1] = 1  # border case
    elif mode == "SE":
        a = np.arange(n_class - 1)
        b = np.arange(n_class)
        loss_fd = np.abs((a - b[:, None]) ** 2 - (a - b[:, None] + 1) ** 2)
    else:
        raise NotImplementedError

    # Inicializa un vector que contiene los parámetros iniciales del modelo
    x0 = np.zeros(n_features + n_class - 1)
    x0[X.shape[1] :] = np.arange(n_class - 1)
    options = {"maxiter": max_iter, "disp": verbose}
    if n_class > 2:
        # Define una lista de restricciones que limitan los valores que pueden tomar los parámetros del modelo
        bounds = [(None, None)] * (n_features + 1) + [(0, None)] * (n_class - 2)
    else:
        bounds = None

    # Utiliza la función `optimize.minimize` de SciPy para encontrar los parámetros del modelo que minimizan la función de pérdida
    sol = optimize.minimize(
        obj_margin,
        x0,
        method="L-BFGS-B",
        jac=grad_margin,
        bounds=bounds,
        options=options,
        args=(X, y, alpha, n_class, loss_fd, L, sample_weight),
        tol=tol,
    )
    if verbose and not sol.success:
        print(sol.message)

    # Extrae los coeficientes del modelo y los umbrales de la solución
    w, c = sol.x[: X.shape[1]], sol.x[X.shape[1] :]
    # Convierte los umbrales a los parámetros de los márgenes
    theta = L.dot(c)
    # Devuelve los coeficientes del modelo y los umbrales
    return w, theta


def threshold_predict(X, w, theta): #NO NECESITA MODIFICACIONES
    """
    Class numbers are assumed to be between 0 and k-1
    """
    # Calcula la diferencia entre los umbrales `theta` y el producto punto de los datos de entrada `X` y los coeficientes del modelo `w`
    tmp = theta[:, None] - np.asarray(X.dot(w))
    # Cuenta el número de umbrales que son mayores que la predicción del modelo para cada muestra de datos
    pred = np.sum(tmp < 0, axis=0).astype(int)
    # Devuelve las predicciones del modelo
    return pred


def threshold_proba(X, w, theta):
    """
    Class numbers are assumed to be between 0 and k-1. Assumes
    the `sigmoid` link function is used.
    """
    # Calcula la diferencia entre los umbrales `theta` y el producto escalar de los datos de entrada `X` y los coeficientes del modelo `w`
    eta = theta[:, None] - np.asarray(X.dot(w), dtype=np.float64)
    # Aplica la función sigmoide a `eta` y añade un cero al inicio y un uno al final de cada fila
    prob = np.pad(
        sigmoid(eta).T,
        pad_width=((0, 0), (1, 1)),
        mode="constant",
        constant_values=(0, 1),
    )
    # Calcula la diferencia entre elementos consecutivos en cada fila de `prob`
    # Esto da como resultado la probabilidad de cada clase para cada muestra de datos
    return np.diff(prob)


class LogisticAT_desb_v2(base.BaseEstimator):
    """
    Esta clase implementa el modelo de regresión logística ordinal (variante de Todos los Umbrales)

    Parámetros
    ----------
    alpha: float
        Parámetro de regularización. Cero significa sin regularización, valores más altos
        incrementan la regularización l2 al cuadrado.

    Referencias
    ----------
    J. D. M. Rennie y N. Srebro, "Loss Functions for Preference Levels :
    Regression with Discrete Ordered Labels," en Proceedings of the IJCAI
    Multidisciplinary Workshop on Advances in Preference Handling, 2005.
    """

    def __init__(self, alpha=1.0, verbose=0, max_iter=1000):
        # Inicializa los parámetros del modelo
        self.alpha = alpha
        self.verbose = verbose
        self.max_iter = max_iter

    def fit(self, X, y, sample_weight=None):
        # Ajusta el modelo a los datos de entrada `X` y las etiquetas `y`
        _y = np.array(y).astype(int)
        if np.abs(_y - y).sum() > 0.1:
            raise ValueError("y debe contener solo valores enteros")
        self.classes_ = np.unique(y)
        self.n_class_ = self.classes_.max() - self.classes_.min() + 1
        y_tmp = y - y.min()  # necesitamos clases que comiencen en cero
        self.sample_weight, self.weight=compute_class_weights(y_tmp) #calculamos los pesos de las clases
        self.coef_, self.theta_ = threshold_fit(
            X,
            y_tmp,
            self.alpha,
            self.n_class_,
            mode="AE",
            verbose=self.verbose,
            max_iter=self.max_iter,
            sample_weight=self.sample_weight,
        )
        return self

    def predict(self, X):
        # Realiza predicciones para los datos de entrada `X`
        return threshold_predict(X, self.coef_, self.theta_) + self.classes_.min()

    def predict_proba(self, X):
        # Calcula las probabilidades de cada clase para los datos de entrada `X`
        return threshold_proba(X, self.coef_, self.theta_)

    def score(self, X, y, sample_weight=None):
        # Calcula el error absoluto medio entre las predicciones del modelo y las etiquetas verdaderas `y`
        pred = self.predict(X)
        return -metrics.mean_absolute_error(pred, y, sample_weight=self.sample_weight)


class LogisticIT_desb_v2(base.BaseEstimator):
    """
    Esta clase implementa el modelo de regresión logística ordinal (variante de Umbral Inmediato).

    A diferencia del modelo OrdinalLogistic, esta variante minimiza un sustituto convexo de la pérdida 0-1,
    por lo tanto, la puntuación asociada con este objeto es la puntuación de precisión, es decir, la misma
    puntuación utilizada en los métodos de clasificación multiclase (sklearn.metrics.accuracy_score).

    Parámetros
    ----------
    alpha: float
        Parámetro de regularización. Cero significa sin regularización, valores más altos
        incrementan la regularización l2 al cuadrado.

    Referencias
    ----------
    J. D. M. Rennie y N. Srebro, "Loss Functions for Preference Levels :
    Regression with Discrete Ordered Labels," en Proceedings of the IJCAI
    Multidisciplinary Workshop on Advances in Preference Handling, 2005.
    """

    def __init__(self, alpha=1.0, verbose=0, max_iter=1000):
        # Inicializa los parámetros del modelo
        self.alpha = alpha
        self.verbose = verbose
        self.max_iter = max_iter

    def fit(self, X, y, sample_weight=None):
        # Ajusta el modelo a los datos de entrada `X` y las etiquetas `y`
        _y = np.array(y).astype(int)
        if np.abs(_y - y).sum() > 0.1:
            raise ValueError("y debe contener solo valores enteros")
        self.classes_ = np.unique(y)
        self.n_class_ = self.classes_.max() - self.classes_.min() + 1
        y_tmp = y - y.min()  # necesitamos clases que comiencen en cero
        self.sample_weight, self.weight=compute_class_weights(y_tmp) #calculamos los pesos de las clases
        self.coef_, self.theta_ = threshold_fit(
            X,
            y_tmp,
            self.alpha,
            self.n_class_,
            mode="0-1",
            #mode="SE",
            verbose=self.verbose,
            max_iter=self.max_iter,
            sample_weight=self.sample_weight,
        )
        return self

    def predict(self, X):
        # Realiza predicciones para los datos de entrada `X`
        return threshold_predict(X, self.coef_, self.theta_) + self.classes_.min()

    def predict_proba(self, X):
        # Calcula las probabilidades de cada clase para los datos de entrada `X`
        return threshold_proba(X, self.coef_, self.theta_)

    def score(self, X, y, sample_weight=None):
        # Calcula la precisión entre las predicciones del modelo y las etiquetas verdaderas `y`
        pred = self.predict(X)
        return metrics.accuracy_score(pred, y, sample_weight=self.sample_weight)

    
class LogisticSE_desb_v2(base.BaseEstimator):
    """
    Classifier that implements the ordinal logistic model
    (Squared Error variant).

    Contrary to the OrdinalLogistic model, this variant
    minimizes a convex surrogate of the 0-1 (?) loss ...

    TODO: double check this description (XXX)

    Parameters
    ----------
    alpha: float
        Regularization parameter. Zero is no regularization, higher values
        increase the squared l2 regularization.

    References
    ----------
    J. D. M. Rennie and N. Srebro, "Loss Functions for Preference Levels :
    Regression with Discrete Ordered Labels," in Proceedings of the IJCAI
    Multidisciplinary Workshop on Advances in Preference Handling, 2005.
    """

    def __init__(self, alpha=1.0, verbose=0, max_iter=100000):
        # Inicializa los parámetros del modelo
        self.alpha = alpha
        self.verbose = verbose
        self.max_iter = max_iter

    def fit(self, X, y, sample_weight=None):
        # Ajusta el modelo a los datos de entrada `X` y las etiquetas `y`
        _y = np.array(y).astype(int)
        if np.abs(_y - y).sum() > 1e-3:
            raise ValueError("y debe contener solo valores enteros")
        self.classes_ = np.unique(y)
        self.n_class_ = self.classes_.max() - self.classes_.min() + 1
        y_tmp = y - y.min()  # necesitamos clases que comiencen en cero
        self.sample_weight, self.weight=compute_class_weights(self.n_class_, y_tmp) #calculamos los pesos de las clases
        self.coef_, self.theta_ = threshold_fit(
            X,
            y_tmp,
            self.alpha,
            self.n_class_,
            mode="SE",
            verbose=self.verbose,
            max_iter=self.max_iter,
            sample_weight=self.sample_weight,
        )
        return self

    def predict(self, X):
        # Realiza predicciones para los datos de entrada `X`
        return threshold_predict(X, self.coef_, self.theta_) + self.classes_.min()

    def predict_proba(self, X):
        # Calcula las probabilidades de cada clase para los datos de entrada `X`
        return threshold_proba(X, self.coef_, self.theta_)

    def score(self, X, y, sample_weight=None):
        # Calcula el error cuadrado medio entre las predicciones del modelo y las etiquetas verdaderas `y`
        pred = self.predict(X)
        return -metrics.mean_squared_error(pred, y, sample_weight=self.sample_weight)