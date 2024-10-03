import qlib
import optuna
from qlib.constant import REG_CN
from qlib.utils import init_instance_by_config
from qlib.tests.config import CSI300_DATASET_CONFIG
from qlib.tests.data import GetData
from pathlib import Path


def objective(trial):
    task = {
        "model": {
            "class": "LGBModel",
            "module_path": "qlib.contrib.model.gbdt",
            "kwargs": {
                "loss": "mse",
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1),
                "learning_rate": trial.suggest_float("learning_rate", 0, 1),
                "subsample": trial.suggest_float("subsample", 0, 1),
                "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 1e4, log=True),
                "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 1e4, log=True),
                "max_depth": 10,
                "num_boost_round": 100,
                "num_leaves": trial.suggest_int("num_leaves", 1, 1024),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
                "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 50),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            },
        },
    }
    evals_result = dict()
    model = init_instance_by_config(task["model"])
    model.fit(dataset, evals_result=evals_result)
    return min(evals_result["valid"]['l2'])


if __name__ == "__main__":
    ## use qlib, not working well
    # provider_uri = "~/.qlib/qlib_data/cn_data"
    # GetData().qlib_data(target_dir=provider_uri, region=REG_CN, exists_skip=True)

    ## use local data
    provider_uri = "D:\qlib_data\qlib_cn"
    qlib.init(provider_uri=provider_uri, region="cn")

    dataset = init_instance_by_config(CSI300_DATASET_CONFIG)

    study = optuna.Study(study_name="LGBM_158", storage="sqlite:///db.sqlite3")
    study.optimize(objective, n_jobs=3, n_trials=10)
