from datetime import datetime
from pathlib import Path

import numpy as np
from dlmisc.results import (
    create_excel_summary_from_dataframe,
    create_results_zip,
    load_results,
)
from scipy import stats

from imbalance.utils.experiments import compute_metrics

appendix = input("Appendix: ")
path = Path("./results")
results = load_results(path)
print("Results loaded")

# Fields from the experiment config that will be included in the dataframe as columns
config_columns_to_include = ["dataset", "estimator_name", "random_state", "estimators"]
best_params_columns_to_include = ["alpha", "C", "max_iter"]
filter_methods = [
    "logisticat",
    "logisticit",
    "logisticregressor",
    # "logisticat_desb",
    # "logisticit_desb",
]
filter_datasets = [
    "automobile",
    "balance-scale",
    "bondrate",
    "car",
    "contact-lenses",
    "ERA",
    "ESL",
    "eucalyptus",
    "LEV",
    "newthyroid",
    "pasture",
    "squash-stored",
    "squash-unstored",
    "SWD",
    "tae",
    "toy",
    "winequality-red"
]


def filter_fn(result):
    if (
        len(filter_methods) > 0
        and result.get_config()["estimator_name"] not in filter_methods
    ):
        return False

    if (
        len(filter_datasets) > 0
        and result.get_config()["dataset"] not in filter_datasets
    ):
        return False

    return True


df = results.get_dataframe(
    config_columns=config_columns_to_include,
    best_params_columns=best_params_columns_to_include,
    filter_fn=filter_fn,
    metrics_fn=compute_metrics,
    include_train=True,
    include_val=False,
)

df.sort_values(by=["dataset", "estimator_name", "random_state"], inplace=True)

# if len(filter_methods) > 0:
#     df = df[df["estimator_name"].isin(filter_methods)]

# if len(filter_datasets) > 0:
#     df = df[df["dataset"].isin(filter_datasets)]

# check that there is the same number of experiments per dataset and estimator
group_columns = ["estimator_name"]

# df_resamples = df[config_columns_to_include].groupby(group_columns)
# df_resamples_count = df_resamples.count()
# resamples = np.max(df_resamples_count.values)

# if resamples != stats.mode(df_resamples_count.values, axis=None)[0]:  # type: ignore
#     check = input(
#         f"The max number of resamples ({resamples}) is not the most repeated"
#         + f"({stats.mode(df_resamples_count.values, axis=None)[0][0]})."  # type: ignore
#         + f"Should it be {resamples}? Y/N"
#     )
#     if check == "N":
#         print("Check the results and run again.")
#         exit()

# # check which rs is missing for each dataset and estimator
# df_count_totals = df_resamples_count[df_resamples_count != resamples].dropna()

# if not df_count_totals.empty:
#     print(
#         "The following dataset and estimator combinations are missing some resamples:"
#     )
#     print(df_count_totals)
#     print("Check the results and run again.")

print(df)

output_path_wo_ext = (
    f'prepared_results/{datetime.now().strftime(r"%Y%m%d_%H%M%S")}_{appendix}'
)

# Columns to be used for grouping the summary dataframe

create_excel_summary_from_dataframe(df, f"{output_path_wo_ext}.xlsx", group_columns)
create_results_zip(path, f"{output_path_wo_ext}", "")


# create_multiple_metrics_latex_table_from_dataframe(
#     results_df=df,
#     destination_path=f"{output_path_wo_ext}.tex",
#     id_columns=["dataset_name", "loss_config.type", "loss_config.params.params_set"],
#     metric_columns={
#         "QWK": "max",
#         "MS": "max",
#         "MAE": "min",
#         "CCR": "max",
#         "1-off": "max",
#         "GMSEC": "max",
#     },
#     group_columns=[
#         "dataset_name",
#         "loss_config.type",
#         "loss_config.params.params_set",
#     ],
# )
