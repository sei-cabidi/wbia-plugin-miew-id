import optuna
import yaml
from train import run
from helpers import get_config
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import pickle
import signal, sys


import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Load configuration file.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default_config.yaml",
        help="Path to the YAML configuration file. Default: configs/default_config.yaml",
    )

    parser.add_argument(
        "--n_trials",
        type=int,
        default=100,
        help="Number of trials. Default: 100",
    )


    return parser.parse_args()


def objective(trial, config):

    # Specify the parameters you want to optimize
    config.data.train.n_filter_min = trial.suggest_int("train.n_filter_min", 2, 5)
    # image_size = 256 #trial.suggest_categorical("image_size", [192, 256, 384, 440, 512])
    # config.data.image_size = [image_size, image_size]
    n_epochs = 20#trial.suggest_int("epochs", 20, 40)
    config.engine.epochs = n_epochs
    config.model_params.margin = trial.suggest_uniform("margin", 0.3, 0.7)
    config.model_params.s = trial.suggest_uniform("s", 20, 64)

    # The scheduler params are derived from one base paremeter to minimize the number of parameters to optimzie
    lr_base = trial.suggest_loguniform("lr_base", 1e-6, 1e-2)
    config.scheduler_params.lr_start = lr_base / 10
    config.scheduler_params.lr_max = lr_base * 10
    config.scheduler_params.lr_min = lr_base / 20

    print("trial number: ", trial.number)
    print("config: ", dict(config))

    if trial.number > 0:
        config.exp_name = config.exp_name.rsplit('_', 1)[0] + f"_t{trial.number}"
    else:
        config.exp_name = config.exp_name + f"_t{trial.number}"


    try:
        result = run(config)
    except Exception as e:
        print("Exception occured: ", e)
        print(trial)
        result = 0

    return result

## Probably only needed to reproduce the results of same seed
def signal_handler(signum, frame):
    global study_sampler
    print("\nSaving the current state of the study before exiting...")
    with open('study_sampler.pkl', 'wb') as f:
        pickle.dump(study, f)
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)


if __name__ == "__main__":
    args = parse_args()
    config_path = args.config
    n_trials = args.n_trials

    config = get_config(config_path)

    study_name = config.exp_name
    study_storage = f"sqlite:///{study_name}.db"
    study_sampler = TPESampler(seed=0)

    study = optuna.create_study(
        study_name=study_name, storage=study_storage,
        sampler=study_sampler, pruner=MedianPruner(), direction="maximize", load_if_exists=True
    )

    comb_objective = lambda trial: objective(trial, config)

    study.optimize(comb_objective, n_trials=n_trials)

    print("Best trial:")
    trial_ = study.best_trial

    print(f"Value: {trial_.value}")

    print("Best parameters:")
    for key, value in trial_.params.items():
        print(f"    {key}: {value}")

    # saves best parameters
    save_dict = trial_.params
    save_dict['best_score'] = trial_.value

    with open('sweep.pkl', 'wb') as f:
        pickle.dump(save_dict, f, pickle.HIGHEST_PROTOCOL)
