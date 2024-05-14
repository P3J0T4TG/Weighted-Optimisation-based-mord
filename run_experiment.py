# -*- coding: utf-8 -*-
"""Classification Experiments: code for experiments as an alternative to orchestration.

This file is configured for runs of the main method with command line arguments, or for
single debugging runs. Results are written in a standard format.
"""

import sys

from imbalance.utils import load_and_run_experiment


def run_experiment(args):
    """Mechanism for testing estimator."""

    # cluster run (with args), this is fragile
    if len(args) > 1:  # cluster run, this is fragile
        print("Input args = ", args)
        data_dir = args[0]
        results_dir = args[1]
        estimator_name = args[2]
        dataset = args[3]
        random_state = int(args[4])
        n_jobs = int(args[5])

        load_and_run_experiment(
            data_dir,
            results_dir,
            dataset,
            random_state=random_state,
            estimator_name=estimator_name,
            n_jobs=n_jobs,
            interactive=False,
        )
    # local run (no args)
    else:
        # These are example parameters, change as required for local runs
        # Do not include paths to your local directories here in PRs
        # If threading is required, see the threaded version of this file
        data_dir = "./0_Datasets/ordinal-regression/"
        results_dir = "./results/"
        #estimator_name = "logisticregressor"
        estimator_name = "logisticat"
        # estimator_name = "logisticit"
        # estimator_name = "logisticat_desb"
        # estimator_name = "logisticit_desb"

        dataset = "tae"
        random_state = 0
        n_jobs = -1

        print(f"Local Run of {estimator_name}.")

        load_and_run_experiment(
            data_dir,
            results_dir,
            dataset,
            random_state=random_state,
            estimator_name=estimator_name,
            n_jobs=n_jobs,
            interactive=True,
        )


if __name__ == "__main__":
    """
    Example simple usage, with arguments input via script or hard coded for testing."""

    run_experiment(sys.argv)
