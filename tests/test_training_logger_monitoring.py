import pandas as pd

from utils.logger import GeneralTrainingLogger


def test_monitoring_panels_use_auroc_for_categories_and_mse_for_regression():
    logger = GeneralTrainingLogger(inspecting=True)
    logger.train_logs = [
        {
            "epoch": 1,
            "mace_0_auroc": 0.61,
            "patient_gender_auroc": 0.72,
            "instance_age_at_time_mean_squared_error": 0.84,
        }
    ]
    logger.val_logs = [
        {
            "epoch": 1,
            "mace_0_auroc": 0.59,
            "patient_gender_auroc": 0.70,
            "instance_age_at_time_mean_squared_error": 0.91,
        }
    ]
    endpoint_df = logger.endpoint_metrics_dataframe(["mace"])

    specs = logger.monitoring_panel_specs(
        ["mace", "patient_gender", "instance_age_at_time", "missing_target"],
        endpoint_df,
    )

    assert specs == [
        {
            "label": "mace",
            "kind": "endpoint_auroc",
            "metric": None,
            "ylabel": "AUROC",
        },
        {
            "label": "patient_gender",
            "kind": "direct_auroc",
            "metric": "patient_gender_auroc",
            "ylabel": "AUROC",
        },
        {
            "label": "instance_age_at_time",
            "kind": "direct_mse",
            "metric": "instance_age_at_time_mean_squared_error",
            "ylabel": "Mean squared error",
        },
    ]


def test_monitoring_panels_work_without_disease_endpoint_rows():
    logger = GeneralTrainingLogger(inspecting=True)
    logger.train_logs = [
        {"epoch": 1, "instance_systolic_bp_mean_squared_error": 0.75}
    ]
    logger.val_logs = [
        {"epoch": 1, "instance_systolic_bp_mean_squared_error": 0.80}
    ]

    specs = logger.monitoring_panel_specs(
        ["instance_systolic_bp"],
        pd.DataFrame(columns=["disease"]),
    )

    assert specs[0]["kind"] == "direct_mse"
    assert specs[0]["metric"] == "instance_systolic_bp_mean_squared_error"
