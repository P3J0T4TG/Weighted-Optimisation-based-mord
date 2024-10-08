"""
some ordinal regression algorithms

This implements the margin-based ordinal regression methods described
in http://arxiv.org/abs/1408.2327
"""

import numpy as np
from scipy import optimize
from sklearn import base, metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_X_y


def compute_class_weight(class_weight, *, classes, y):
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


def sigmoid(t):
    # sigmoid function, 1 / (1 + exp(-t))
    # stable computation
    idx = t > 0
    out = np.zeros_like(t)
    out[idx] = 1.0 / (1 + np.exp(-t[idx]))
    exp_t = np.exp(t[~idx])
    out[~idx] = exp_t / (1.0 + exp_t)
    return out


def log_loss(Z):
    # stable computation of the logistic loss
    idx = Z > 0
    out = np.zeros_like(Z)
    out[idx] = np.log(1 + np.exp(-Z[idx]))
    out[~idx] = -Z[~idx] + np.log(1 + np.exp(Z[~idx]))
    return out


def obj_margin(x0, X, y, alpha, n_class, weights, L, class_weight, sample_weight):
    """
    Objective function for the general margin-based formulation
    """

    w = x0[: X.shape[1]]
    c = x0[X.shape[1] :]  # noqa
    theta = L.dot(c)
    loss_fd = weights[y]

    Xw = X.dot(w)
    Alpha = theta[:, None] - Xw  # (n_class - 1, n_samples)
    S = np.sign(np.arange(n_class - 1)[:, None] - y + 0.5)

    err = loss_fd.T * log_loss(S * Alpha)
    if sample_weight is not None:
        err *= sample_weight
    # AÑADIDO
    if class_weight is not None:
        err *= class_weight
    # AÑADIDO
    obj = np.sum(err)
    obj += alpha * 0.5 * (np.dot(w, w))
    return obj


def grad_margin(x0, X, y, alpha, n_class, weights, L, class_weight, sample_weight):
    """
    Gradient for the general margin-based formulation
    """

    w = x0[: X.shape[1]]
    c = x0[X.shape[1] :]  # noqa
    theta = L.dot(c)
    loss_fd = weights[y]

    Xw = X.dot(w)
    Alpha = theta[:, None] - Xw  # (n_class - 1, n_samples)
    S = np.sign(np.arange(n_class - 1)[:, None] - y + 0.5)
    # Alpha[idx] *= -1
    # W[idx.T] *= -1

    Sigma = S * loss_fd.T * sigmoid(-S * Alpha)
    if sample_weight is not None:
        Sigma *= sample_weight

    # AÑADIDO
    if class_weight is not None:
        Sigma *= class_weight
    # AÑADIDO

    grad_w = X.T.dot(Sigma.sum(0)) + alpha * w

    grad_theta = -Sigma.sum(1)
    grad_c = L.T.dot(grad_theta)
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
    class_weight=None,
    sample_weight=None,
):
    """
    Solve the general threshold-based ordinal regression model
    using the logistic loss as surrogate of the 0-1 loss

    Parameters
    ----------
    mode : string, one of {'AE', '0-1', 'SE'}

    """

    X, y = check_X_y(X, y, accept_sparse="csr")
    unique_y = np.sort(np.unique(y))
    if not np.all(unique_y == np.arange(unique_y.size)):
        raise ValueError(
            "Values in y must be %s, instead got %s"
            % (np.arange(unique_y.size), unique_y)
        )

    n_samples, n_features = X.shape

    # convert from c to theta
    L = np.zeros((n_class - 1, n_class - 1))
    L[np.tril_indices(n_class - 1)] = 1.0

    if mode == "AE":
        # loss forward difference
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

    x0 = np.zeros(n_features + n_class - 1)
    x0[X.shape[1] :] = np.arange(n_class - 1)  # noqa

    options = {"maxiter": max_iter, "disp": verbose}
    if n_class > 2:
        bounds = [(None, None)] * (n_features + 1) + [(0, None)] * (n_class - 2)
    else:
        bounds = None

    # AÑADIDO
    enc = LabelEncoder()
    enc.fit(y)
    classes_ = enc.classes_
    class_weight_computed = compute_class_weight(class_weight, classes=classes_, y=y)
    class_weight_ = [class_weight_computed[y[i]] for i in range(len(y))]
    # AÑADIDO

    sol = optimize.minimize(
        obj_margin,
        x0,
        method="L-BFGS-B",
        jac=grad_margin,
        bounds=bounds,
        options=options,
        args=(X, y, alpha, n_class, loss_fd, L, class_weight_, sample_weight),
        tol=tol,
    )
    if verbose and not sol.success:
        print(sol.message)

    w, c = sol.x[: X.shape[1]], sol.x[X.shape[1] :]  # noqa
    theta = L.dot(c)
    return w, theta


def threshold_predict(X, w, theta):
    """
    Class numbers are assumed to be between 0 and k-1
    """
    tmp = theta[:, None] - np.asarray(X.dot(w))
    pred = np.sum(tmp < 0, axis=0).astype(int)
    return pred


def threshold_proba(X, w, theta):
    """
    Class numbers are assumed to be between 0 and k-1. Assumes
    the `sigmoid` link function is used.
    """
    eta = theta[:, None] - np.asarray(X.dot(w), dtype=np.float64)
    prob = np.pad(
        sigmoid(eta).T,
        pad_width=((0, 0), (1, 1)),
        mode="constant",
        constant_values=(0, 1),
    )
    return np.diff(prob)


class LogisticAT(base.BaseEstimator):
    """
    Classifier that implements the ordinal logistic model (All-Threshold variant)

    Parameters
    ----------
    alpha: float
        Regularization parameter. Zero is no regularization, higher values
        increate the squared l2 regularization.

    References
    ----------
    J. D. M. Rennie and N. Srebro, "Loss Functions for Preference Levels :
    Regression with Discrete Ordered Labels," in Proceedings of the IJCAI
    Multidisciplinary Workshop on Advances in Preference Handling, 2005.
    """

    def __init__(self, alpha=1.0, verbose=0, max_iter=1000, class_weight=None):
        self.alpha = alpha
        self.verbose = verbose
        self.max_iter = max_iter
        self.class_weight = class_weight

    def fit(self, X, y, sample_weight=None):
        _y = np.array(y).astype(int)
        if np.abs(_y - y).sum() > 0.1:
            raise ValueError("y must only contain integer values")
        self.classes_ = np.unique(y)
        self.n_class_ = self.classes_.max() - self.classes_.min() + 1
        y_tmp = y - y.min()  # we need classes that start at zero
        self.coef_, self.theta_ = threshold_fit(
            X,
            y_tmp,
            self.alpha,
            self.n_class_,
            mode="AE",
            verbose=self.verbose,
            max_iter=self.max_iter,
            class_weight=self.class_weight,
            sample_weight=sample_weight,
        )
        # print(self.coef_)
        # print("===========")
        # print(self.theta_)

        return self

    def predict(self, X):
        return threshold_predict(X, self.coef_, self.theta_) + self.classes_.min()

    def predict_proba(self, X):
        return threshold_proba(X, self.coef_, self.theta_)

    def score(self, X, y, sample_weight=None):
        pred = self.predict(X)
        return -metrics.mean_absolute_error(pred, y, sample_weight=sample_weight)


class LogisticIT(base.BaseEstimator):
    """
    Classifier that implements the ordinal logistic model
    (Immediate-Threshold variant).

    Contrary to the OrdinalLogistic model, this variant
    minimizes a convex surrogate of the 0-1 loss, hence
    the score associated with this object is the accuracy
    score, i.e. the same score used in multiclass
    classification methods (sklearn.metrics.accuracy_score).

    Parameters
    ----------
    alpha: float
        Regularization parameter. Zero is no regularization, higher values
        increate the squared l2 regularization.

    References
    ----------
    J. D. M. Rennie and N. Srebro, "Loss Functions for Preference Levels :
    Regression with Discrete Ordered Labels," in Proceedings of the IJCAI
    Multidisciplinary Workshop on Advances in Preference Handling, 2005.
    """

    def __init__(self, alpha=1.0, verbose=0, max_iter=1000, class_weight=None):
        self.alpha = alpha
        self.verbose = verbose
        self.max_iter = max_iter
        self.class_weight = class_weight

    def fit(self, X, y, sample_weight=None):
        _y = np.array(y).astype(int)
        if np.abs(_y - y).sum() > 0.1:
            raise ValueError("y must only contain integer values")
        self.classes_ = np.unique(y)
        self.n_class_ = self.classes_.max() - self.classes_.min() + 1
        y_tmp = y - y.min()  # we need classes that start at zero
        self.coef_, self.theta_ = threshold_fit(
            X,
            y_tmp,
            self.alpha,
            self.n_class_,
            mode="AE",
            verbose=self.verbose,
            max_iter=self.max_iter,
            class_weight=self.class_weight,
            sample_weight=sample_weight,
        )
        return self

    def predict(self, X):
        return threshold_predict(X, self.coef_, self.theta_) + self.classes_.min()

    def predict_proba(self, X):
        return threshold_proba(X, self.coef_, self.theta_)

    def score(self, X, y, sample_weight=None):
        pred = self.predict(X)
        return metrics.accuracy_score(pred, y, sample_weight=sample_weight)


class LogisticSE(base.BaseEstimator):
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
        self.alpha = alpha
        self.verbose = verbose
        self.max_iter = max_iter

    def fit(self, X, y, sample_weight=None):
        _y = np.array(y).astype(int)
        if np.abs(_y - y).sum() > 1e-3:
            raise ValueError("y must only contain integer values")
        self.classes_ = np.unique(y)
        self.n_class_ = self.classes_.max() - self.classes_.min() + 1
        y_tmp = y - y.min()  # we need classes that start at zero
        self.coef_, self.theta_ = threshold_fit(
            X,
            y_tmp,
            self.alpha,
            self.n_class_,
            mode="SE",
            verbose=self.verbose,
            max_iter=self.max_iter,
            class_weight=self.class_weight,
            sample_weight=sample_weight,
        )
        return self

    def predict(self, X):
        return threshold_predict(X, self.coef_, self.theta_) + self.classes_.min()

    def predict_proba(self, X):
        return threshold_proba(X, self.coef_, self.theta_)

    def score(self, X, y, sample_weight=None):
        pred = self.predict(X)
        return -metrics.mean_squared_error(pred, y, sample_weight=sample_weight)
