from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from imbalance.metrics import amae

NOMINAL_CLASSIFIERS = [
    "logisticregressor",
]

ORDINAL_CLASSIFIERS = [
    "logisticat",
    "logisticit",
    "logisticat_desb",  # Modificarlas y cuando estén listas meterlas en el pquete mord
    "logisticit_desb",  # ''        ''
]


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
            scoring=make_scorer(amae, greater_is_better=False),
            n_jobs=n_jobs,
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state),
            error_score="raise",
            **kwargs,
        )

    elif estimator_name in ORDINAL_CLASSIFIERS:

        if estimator_name == "logisticat":
            from mord import LogisticAT

            param_grid = [
                {
                    "alpha": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                    "max_iter": [1000, 2000, 3000, 5000],
                }
            ]
            estimator = LogisticAT()

        elif estimator_name == "logisticit":
            from mord import LogisticIT

            param_grid = [
                {
                    "alpha": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                    "max_iter": [1000, 2000, 3000, 5000],
                }
            ]
            estimator = LogisticIT()

        elif estimator_name == "logisticat_desb":
            from imbalance.ordinal_classification import LogisticAT_desb

            param_grid = [
                {
                    "alpha": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                    "max_iter": [1000, 2000, 3000, 5000],
                    # Aquí tendrías que poner "class_weight" : ["balanced"]
                }
            ]
            estimator = LogisticAT_desb()

        elif estimator_name == "logisticit_desb":
            from imbalance.ordinal_classification import LogisticIT_desb

            param_grid = [
                {
                    "alpha": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                    "max_iter": [1000, 2000, 3000, 5000],
                    # Aquí tendrías que poner "class_weight" : ["balanced"]
                }
            ]
            estimator = LogisticIT_desb()

        else:
            raise NotImplementedError(
                f"Estimator {estimator_name} was included in ORDINAL_CLASSIFIERS "
                + "but not implemented in set_estimators function."
            )

        return GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            scoring=make_scorer(amae, greater_is_better=False),
            n_jobs=n_jobs,
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state),
            error_score="raise",
            **kwargs,
        )

    else:
        raise ValueError(f"Estimator {estimator_name} not recognised.")
