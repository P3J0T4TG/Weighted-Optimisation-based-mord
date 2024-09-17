"""
Script que configura los estimadores y sus hiperparámetros para el lanzamiento de experimentos.

MODIFICAR ESTE ARCHIVO SÍ:
    * Se han añadido nuevos estimadores
    * Se han añadido nuevos hiperparámetros
    * Se quiere modificar la configuracion del GridSearchCV
"""
from sklearn.metrics import make_scorer, mean_absolute_error, accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from imbalance.metrics import amae

NOMINAL_CLASSIFIERS = [
    "logisticregressor",
]

ORDINAL_CLASSIFIERS = [
    "logisticat",
    "logisticit",
    "logisticat_desb_v2",  
    "logisticit_desb_v2",  
]

mae=mean_absolute_error

def MZE(y_true, y_pred):
    return 1 - accuracy_score(y_true, y_pred)

def set_estimator(estimator_name, random_state, n_jobs=-1, **kwargs):
    estimator_name = estimator_name.casefold()

    if estimator_name in NOMINAL_CLASSIFIERS:

        if estimator_name == "logisticregressor":
            from sklearn.linear_model import LogisticRegression

            param_grid = [
                {
                    "C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                    "max_iter": [1000, 2000, 3000, 5000],
                    "class_weight": ["balanced"],
                }
            ]
            estimator = LogisticRegression(random_state=random_state)

        else:
            raise NotImplementedError(
                f"Estimator {estimator_name} was included in NOMINAL_CLASSIFIERS "
                + "but not implemented in set_estimators function."
            )

        return GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            scoring=make_scorer(mae, greater_is_better=False),
            n_jobs=n_jobs,
            cv=StratifiedKFold(n_splits=2, shuffle=True, random_state=random_state),
            error_score="raise",
            **kwargs,
        )

    elif estimator_name in ORDINAL_CLASSIFIERS:

        if estimator_name == "logisticat":
            from mord import LogisticAT
            #from imbalance.ordinal_classification.threshold_based_David import LogisticAT

            param_grid = [
                {
                    "alpha": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                    "max_iter": [1000, 2000, 3000, 5000],
                    #"class_weight": ["balanced"],
                }
            ]
            estimator = LogisticAT()

        elif estimator_name == "logisticit":
            from mord import LogisticIT
            #from imbalance.ordinal_classification.threshold_based_David import LogisticIT

            param_grid = [
                {
                    "alpha": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                    "max_iter": [1000, 2000, 3000, 5000],
                    #"class_weight": ["balanced"],
                }
            ]
            estimator = LogisticIT()

        elif estimator_name == "logisticat_desb":
            from imbalance.ordinal_classification.logistic_umbalanced import LogisticAT_desb

            param_grid = [
                {
                    "alpha": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                    "max_iter": [1000, 2000, 3000, 5000],
                    #"class_weight": ["balanced"],
                }
            ]
            estimator = LogisticAT_desb()

        elif estimator_name == "logisticit_desb":
            from imbalance.ordinal_classification.logistic_umbalanced import LogisticIT_desb

            param_grid = [
                {
                    "alpha": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                    "max_iter": [1000, 2000, 3000, 5000],
                    #"class_weight": ["balanced"],
                }
            ]
            estimator = LogisticIT_desb()

#V2 DE LAS UMBALANCED
        elif estimator_name == "logisticat_desb_v2":
            from imbalance.ordinal_classification.logistic_umbalanced_v2 import LogisticAT_desb_v2
            param_grid = [
                {
                    "alpha": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                    "max_iter": [1000, 2000, 3000, 5000],
                    #"class_weight": ["balanced"],
                }
            ]
            estimator = LogisticAT_desb_v2()

        elif estimator_name == "logisticit_desb_v2":
            from imbalance.ordinal_classification.logistic_umbalanced_v2 import LogisticIT_desb_v2

            param_grid = [
                {
                    "alpha": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                    "max_iter": [1000, 2000, 3000, 5000],
                    #"class_weight": ["balanced"],
                }
            ]
            estimator = LogisticIT_desb_v2()

        else:
            raise NotImplementedError(
                f"Estimator {estimator_name} was included in ORDINAL_CLASSIFIERS "
                + "but not implemented in set_estimators function."
            )

        return GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            scoring=make_scorer(mae, greater_is_better=False),
            n_jobs=n_jobs,
            cv=StratifiedKFold(n_splits=2, shuffle=True, random_state=random_state),
            error_score="raise",
            **kwargs,
        )

    else:
        raise ValueError(f"Estimator {estimator_name} not recognised.")
