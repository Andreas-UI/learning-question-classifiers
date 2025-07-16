import mlflow


def get_or_create_experiment(experiment_name):
    if experiment := mlflow.get_experiment_by_name(experiment_name):
        print(f"MLFlow: '{experiment_name}' found !")
        return experiment.name
    else:
        print(f"MLFlow: '{experiment_name}' not found !")
        print(f"MLFlow: Creating '{experiment_name}' experiment...")
        experiment = mlflow.create_experiment(experiment_name)
        print(f"MLFlow: '{experiment_name}' created!")
        return experiment


def set_mlflow_experiment(experiment_name):
    get_or_create_experiment(experiment_name)
    print(f"MLFlow: Setting '{experiment_name}' experiment...")
    mlflow.set_experiment(experiment_name)
    print(f"MLFlow: '{experiment_name}' experiment is active !")
