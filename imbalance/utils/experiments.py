# -*- coding: utf-8 -*-
"""
Archivo con funciones para correr experimentos de clasificación.

NO MODIFICAR ESTE ARCHIVO
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
from dlmisc.results import load_results, write_results_file

from .set_estimators import set_estimator


def load_and_run_experiment(
    data_dir,
    results_dir,
    dataset,
    random_state=0,
    estimator_name="ridgeregressor",
    n_jobs=-1,
    interactive=False,
):
    from random import seed as random_seed

    # Fix seeds
    np.random.seed(random_state)
    random_seed(random_state)

    X_train, y_train, encoder = load_data(
        data_dir, dataset, partition="train", seed=random_state
    )
    X_test, y_test, _ = load_data(
        data_dir,
        dataset,
        partition="test",
        seed=random_state,
        encoder=encoder,
    )

    estimator = set_estimator(estimator_name, random_state=random_state, n_jobs=n_jobs)

    config = get_config(estimator, estimator_name, dataset, random_state)

    results = load_results(results_dir)
    if results is not None:
        if (
            results.find_experiment(
                config,
                deep=True,
            )
            is not None
        ):
            print("Experiment already run")
            return

    print("Running experiment...")

    if estimator_name is None:
        estimator_name = type(estimator).__name__

    start = int(round(time.time()))

    estimator.fit(X_train, y_train)

    train_probs = estimator.predict_proba(X_train)
    train_preds = estimator.classes_[np.argmax(train_probs, axis=1)]

    test_probs = estimator.predict_proba(X_test)
    test_preds = estimator.classes_[np.argmax(test_probs, axis=1)]

    total_time = int(round(time.time())) - start

    config = get_config(estimator, estimator_name, dataset, random_state)

    if not interactive:
        write_results_file(
            base_path=results_dir,
            name=estimator_name,
            config=config,
            predictions=test_preds,
            targets=y_test,
            rs=random_state,
            dataset=dataset,
            resample_id=random_state,
            train_predictions=train_preds,
            train_targets=y_train,
            time=total_time,
            best_params=estimator.best_params_,
        )
    else:
        train_metrics = compute_metrics(y_train, train_preds)
        print("train_metrics", train_metrics)

        test_metrics = compute_metrics(y_test, test_preds)
        print("test_metrics", test_metrics)


def load_data(data_dir, dataset, partition, seed, encoder=None):
    path = Path(data_dir) / dataset / f"{partition}_{dataset}.{seed}"
    df = pd.read_csv(path, sep=" ", header=None)

    X = df.values[:, :-1]
    y = df.values[:, -1]

    if partition == "train":
        from sklearn.preprocessing import LabelEncoder

        encoder = LabelEncoder()
        encoder = encoder.fit(y)
        y = encoder.transform(y)
    else:
        if encoder is None:
            raise ValueError(f"Encoder cannot be None for the {partition} partition.")
        y = encoder.transform(y)

    return X, y, encoder


def get_config(estimator, estimator_name, dataset, random_state):
    config = estimator.get_params().copy()
    config["estimator_name"] = estimator_name
    config["dataset"] = dataset
    config["random_state"] = random_state
    if "estimator" in config:
        del config["estimator"]
    if "scoring" in config:
        del config["scoring"]
    return config


def compute_metrics(targets, predictions):
    from dlmisc.metrics import accuracy_off1, minimum_sensitivity
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        cohen_kappa_score,
        mean_absolute_error,
        recall_score,
    )

    from imbalance.metrics import amae, mmae

    if len(predictions.shape) > 1:
        predictions = np.argmax(predictions, axis=1)

    metrics = {
        "QWK": cohen_kappa_score(targets, predictions, weights="quadratic"),
        "MAE": mean_absolute_error(targets, predictions),
        "1-off": accuracy_off1(targets, predictions),
        "CCR": accuracy_score(targets, predictions),
        "MZE": 1 - accuracy_score(targets, predictions),
        "MS": minimum_sensitivity(targets, predictions),
        "BalancedAccuracy": balanced_accuracy_score(targets, predictions),
        "AMAE": amae(targets, predictions),
        "MMAE": mmae(targets, predictions),
    }

    # Compute sensitivities for each class
    sensitivities = np.array(recall_score(targets, predictions, average=None))

    for i, sens in enumerate(sensitivities):
        metrics[f"Sens{i}"] = sens

    return metrics
