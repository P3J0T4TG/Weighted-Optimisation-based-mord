import numpy as np
from sklearn.metrics import confusion_matrix


def amae(y, ypred):
    cm = confusion_matrix(y, ypred)
    n_class = cm.shape[0]
    costes = np.reshape(np.tile(range(n_class), n_class), (n_class, n_class))
    costes = np.abs(costes - np.transpose(costes))
    errores = costes * cm
    #Se ha añadido un valor epsilon para evitar divisiones por cero; --> Faltaria ver el impacto de este
    amaes = np.sum(errores, axis=1) / (np.sum(cm, axis=1).astype("double") + 1e-10)
    #amaes = np.sum(errores, axis=1) / np.sum(cm, axis=1).astype("double")
    amaes = amaes[~np.isnan(amaes)]
    return np.mean(amaes)


def mmae(y, ypred):
    cm = confusion_matrix(y, ypred)
    n_class = cm.shape[0]
    costes = np.reshape(np.tile(range(n_class), n_class), (n_class, n_class))
    costes = np.abs(costes - np.transpose(costes))
    errores = costes * cm
    amaes = np.sum(errores, axis=1) / np.sum(cm, axis=1).astype("double")
    amaes = amaes[~np.isnan(amaes)]
    return amaes.max()
