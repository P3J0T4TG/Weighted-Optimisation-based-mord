"""
Script para lanzar todos los experimentos de una
"""

import os

from run_experiment import run_experiment

estimator_names = ["logisticit", "logisticat", "logisticregressor"]
data_dir = "./0_Datasets/ordinal-regression/"
dataset = "toy"
path_dataset = os.path.join(data_dir, dataset)
N = 30
list_datasets = ["automobile", "balance-scale", "bondrate", "car", "contact-lenses", "ERA", "ESL", "eucalyptus", "LEV", "newthyroid", "pasture", "squash-stored", "squash-unstored", "SWD", "tae", "toy", "winequality-red"]

for dataset in list_datasets:
    for estimator_name in estimator_names:
        for resample in range(N):
            try:
                run_experiment(
                    [data_dir, "./results/", estimator_name, dataset, resample, -1]
                )
                print(f"Experiment {estimator_name} {resample} done")
            except SystemExit:
                print(
                    f"Experiment {estimator_name} {resample} already done, skipping to next"
                )
                continue
