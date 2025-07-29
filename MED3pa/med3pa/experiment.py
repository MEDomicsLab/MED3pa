"""
Orchestrates the execution of the med3pa method and integrates the functionality of other modules to run comprehensive experiments. 
It includes classes to manage and store results ``Med3paResults``, execute experiments like ``Med3paExperiment`` and ``Med3paDetectronExperiment``, and integrate results from the Detectron method ``Med3paDetectronResults``
"""
# import json
import os
import matplotlib.pyplot as plt
from typing import Any, Dict, List, Tuple, Type, Union

# import numpy as np
from checkpointer import checkpoint
from sklearn.model_selection import train_test_split

from MED3pa.datasets import DatasetsManager
from MED3pa.detectron.experiment import DetectronExperiment, DetectronResult, DetectronStrategy, \
    EnhancedDisagreementStrategy
from MED3pa.med3pa.mdr import MDRCalculator
from MED3pa.med3pa.models import APCModel, IPCModel, MPCModel
from MED3pa.med3pa.profiles import Profile, ProfilesManager
from MED3pa.med3pa.results import Med3paResults, Med3paRecord
from MED3pa.med3pa.uncertainty import *
from MED3pa.models.base import BaseModelManager
from MED3pa.models.classification_metrics import *
from MED3pa.models.concrete_regressors import *


class Med3paExperiment:
    """
    Class to run the MED3PA method experiment.
    """

    @staticmethod
    @checkpoint(root_path="checkpoints", verbosity=False)
    def run(datasets_manager: DatasetsManager,
            base_model_manager: BaseModelManager = None,
            uncertainty_metric: str = 'absolute_error',
            ipc_type: str = 'RandomForestRegressor',
            ipc_params: Dict = None,
            ipc_grid_params: Dict = None,
            ipc_cv: int = 4,
            pretrained_ipc: str = None,
            apc_params: Dict = None,
            apc_grid_params: Dict = None,
            apc_cv: int = 4,
            pretrained_apc: str = None,
            samples_ratio_min: int = 0,
            samples_ratio_max: int = 50,
            samples_ratio_step: int = 5,
            med3pa_metrics: List[str] = None,
            evaluate_models: bool = False,
            use_ref_models: bool = False,
            mode: str = 'mpc',
            models_metrics: List[str] = None) -> Med3paResults:

        """
        Runs the MED3PA experiment on both reference and testing sets.

        Args:
            datasets_manager (DatasetsManager): the datasets manager containing the dataset to use in the experiment.
            base_model_manager (BaseModelManager, optional): Instance of BaseModelManager to get the base model,
                by default None.
            uncertainty_metric (str, optional): the uncertainty metric ysed to calculate uncertainty,
                by default absolute_error.
            ipc_type (str, optional): The regressor model to use for IPC, by default RandomForestRegressor.
            ipc_params (dict, optional): Parameters for initializing the IPC regressor model, by default None.
            ipc_grid_params (dict, optional): Grid search parameters for optimizing the IPC model, by default None.
            ipc_cv (int, optional): Number of cross-validation folds for optimizing the IPC model, by default None.
            pretrained_ipc (str, optional): path to a pretrained ipc, by default None.
            apc_params (dict, optional): Parameters for initializing the APC regressor model, by default None.
            apc_grid_params (dict, optional): Grid search parameters for optimizing the APC model, by default None.
            apc_cv (int, optional): Number of cross-validation folds for optimizing the APC model, by default None.
            pretrained_apc (str, optional): path to a pretrained apc, by default None.
            use_ref_models (bool, optional): whether or not to use the trained IPC and APC models from the reference set
                on the test set.
            samples_ratio_min (int, optional): Minimum sample ratio, by default 0.
            samples_ratio_max (int, optional): Maximum sample ratio, by default 50.
            samples_ratio_step (int, optional): Step size for sample ratio, by default 5.
            med3pa_metrics (list of str, optional): List of metrics to calculate, by default, multiple metrics included.
            evaluate_models (bool, optional): Whether to evaluate the models, by default False.
            mode (str): The modality of dataset, either 'ipc', 'apc', or 'mpc'.
            models_metrics (list of str, optional): List of metrics for model evaluation,
                by default ['MSE', 'RMSE', 'MAE'].

        Returns:
            Med3paResults: the results of the MED3PA experiment on the reference set and testing set.
        """
        if med3pa_metrics is None:
            med3pa_metrics = ['Accuracy', 'BalancedAccuracy', 'Precision', 'Recall', 'F1Score',
                              'Specificity', 'Sensitivity', 'Auc', 'LogLoss', 'Auprc', 'NPV', 'PPV', 'MCC']

        if models_metrics is None:
            models_metrics = ['MSE', 'RMSE', 'MAE']

        results_ref = None
        if datasets_manager.reference_set is not None:
            print("Running MED3pa Experiment on the reference set:")
            results_ref, ipc_config, apc_config = Med3paExperiment._run_by_set(datasets_manager=datasets_manager,
                                                                               set='reference',
                                                                               base_model_manager=base_model_manager,
                                                                               uncertainty_metric=uncertainty_metric,
                                                                               ipc_type=ipc_type,
                                                                               ipc_params=ipc_params,
                                                                               ipc_grid_params=ipc_grid_params,
                                                                               ipc_cv=ipc_cv,
                                                                               pretrained_ipc=pretrained_ipc,
                                                                               apc_params=apc_params,
                                                                               apc_grid_params=apc_grid_params,
                                                                               apc_cv=apc_cv,
                                                                               pretrained_apc=pretrained_apc,
                                                                               samples_ratio_min=samples_ratio_min,
                                                                               samples_ratio_max=samples_ratio_max,
                                                                               samples_ratio_step=samples_ratio_step,
                                                                               med3pa_metrics=med3pa_metrics,
                                                                               evaluate_models=evaluate_models,
                                                                               models_metrics=models_metrics,
                                                                               mode=mode)
        print("Running MED3pa Experiment on the test set:")
        if use_ref_models:
            if results_ref is None:
                raise ValueError("use_ref_models cannot be true if reference set is None. ")
            results_testing, ipc_config, apc_config = Med3paExperiment._run_by_set(datasets_manager=datasets_manager,
                                                                                   set='testing',
                                                                                   base_model_manager=base_model_manager,
                                                                                   uncertainty_metric=uncertainty_metric,
                                                                                   ipc_type=ipc_type,
                                                                                   ipc_params=ipc_params,
                                                                                   ipc_grid_params=ipc_grid_params,
                                                                                   ipc_cv=ipc_cv,
                                                                                   pretrained_ipc=pretrained_ipc,
                                                                                   ipc_instance=ipc_config,
                                                                                   apc_params=apc_params,
                                                                                   apc_grid_params=apc_grid_params,
                                                                                   apc_cv=apc_cv,
                                                                                   pretrained_apc=pretrained_apc,
                                                                                   apc_instance=apc_config,
                                                                                   samples_ratio_min=samples_ratio_min,
                                                                                   samples_ratio_max=samples_ratio_max,
                                                                                   samples_ratio_step=samples_ratio_step,
                                                                                   med3pa_metrics=med3pa_metrics,
                                                                                   evaluate_models=evaluate_models,
                                                                                   models_metrics=models_metrics,
                                                                                   mode=mode)
        else:
            results_testing, ipc_config, apc_config = Med3paExperiment._run_by_set(datasets_manager=datasets_manager,
                                                                                   set='testing',
                                                                                   base_model_manager=base_model_manager,
                                                                                   uncertainty_metric=uncertainty_metric,
                                                                                   ipc_type=ipc_type,
                                                                                   ipc_params=ipc_params,
                                                                                   ipc_grid_params=ipc_grid_params,
                                                                                   ipc_cv=ipc_cv,
                                                                                   pretrained_ipc=pretrained_ipc,
                                                                                   ipc_instance=None,
                                                                                   apc_params=apc_params,
                                                                                   apc_grid_params=apc_grid_params,
                                                                                   apc_cv=apc_cv,
                                                                                   pretrained_apc=pretrained_apc,
                                                                                   apc_instance=None,
                                                                                   samples_ratio_min=samples_ratio_min,
                                                                                   samples_ratio_max=samples_ratio_max,
                                                                                   samples_ratio_step=samples_ratio_step,
                                                                                   med3pa_metrics=med3pa_metrics,
                                                                                   evaluate_models=evaluate_models,
                                                                                   models_metrics=models_metrics,
                                                                                   mode=mode)

        results = Med3paResults(results_ref, results_testing)
        med3pa_params = {
            'uncertainty_metric': uncertainty_metric,
            'samples_ratio_min': samples_ratio_min,
            'samples_ratio_max': samples_ratio_max,
            'samples_ratio_step': samples_ratio_step,
            'med3pa_metrics': med3pa_metrics,
            'evaluate_models': evaluate_models,
            'models_evaluation_metrics': models_metrics,
            'mode': mode,
            'ipc_model': ipc_config.get_info(),
            'apc_model': apc_config.get_info() if apc_config is not None else None,
        }
        experiment_config = {
            'experiment_name': "Med3paExperiment",
            'datasets': datasets_manager.get_info(),
            'base_model': base_model_manager.get_info() if base_model_manager is not None else None,
            'med3pa_params': med3pa_params
        }
        results.set_experiment_config(experiment_config)
        results.set_models(ipc_config, apc_config)
        return results

    @staticmethod
    def _run_by_set(datasets_manager: DatasetsManager,
                    set: str = 'reference',
                    base_model_manager: BaseModelManager = None,
                    uncertainty_metric: str = 'absolute_error',
                    ipc_type: str = 'RandomForestRegressor',
                    ipc_params: Dict = None,
                    ipc_grid_params: Dict = None,
                    ipc_cv: int = 4,
                    pretrained_ipc: str = None,
                    ipc_instance: IPCModel = None,
                    apc_params: Dict = None,
                    apc_grid_params: Dict = None,
                    apc_cv: int = 4,
                    apc_instance: APCModel = None,
                    pretrained_apc: str = None,
                    samples_ratio_min: int = 0,
                    samples_ratio_max: int = 50,
                    samples_ratio_step: int = 5,
                    med3pa_metrics: List[str] = None,
                    evaluate_models: bool = False,
                    mode: str = 'mpc',
                    models_metrics: List[str] = None) -> Tuple[Med3paRecord, dict, dict]:

        """
        Orchestrates the MED3PA experiment on one specific set of the dataset.

        Args:
            datasets_manager (DatasetsManager): the datasets manager containing the dataset to use in the experiment.
            base_model_manager (BaseModelManager, optional): Instance of BaseModelManager to get the base model,
                by default None.
            uncertainty_metric (str, optional): the uncertainty metric ysed to calculate uncertainty,
                by default absolute_error.
            ipc_type (str, optional): The regressor model to use for IPC, by default RandomForestRegressor.
            ipc_params (dict, optional): Parameters for initializing the IPC regressor model, by default None.
            ipc_grid_params (dict, optional): Grid search parameters for optimizing the IPC model, by default None.
            ipc_cv (int, optional): Number of cross-validation folds for optimizing the IPC model, by default None.
            apc_params (dict, optional): Parameters for initializing the APC regressor model, by default None.
            apc_grid_params (dict, optional): Grid search parameters for optimizing the APC model, by default None.
            apc_cv (int, optional): Number of cross-validation folds for optimizing the APC model, by default None.
            samples_ratio_min (int, optional): Minimum sample ratio, by default 0.
            samples_ratio_max (int, optional): Maximum sample ratio, by default 50.
            samples_ratio_step (int, optional): Step size for sample ratio, by default 5.
            med3pa_metrics (list of str, optional): List of metrics to calculate.
            evaluate_models (bool, optional): Whether to evaluate the models, by default False.
            mode (str): The modality of dataset, either 'ipc', 'apc', or 'mpc'.
            models_metrics (list of str, optional): List of metrics for model evaluation.

        Returns:
            Med3paRecord: the results of the MED3PA experiment.
        """

        # Step 1 : datasets and base model setting
        # Retrieve the dataset based on the set type
        if set == 'reference':
            dataset = datasets_manager.get_dataset_by_type(dataset_type="reference", return_instance=True)
        elif set == 'testing':
            dataset = datasets_manager.get_dataset_by_type(dataset_type="testing", return_instance=True)
        else:
            raise ValueError("The set must be either the reference set or the testing set")

        # retrieve different dataset components needed for the experiment
        x = dataset.get_observations()
        y_true = dataset.get_true_labels()
        predicted_probabilities = dataset.get_pseudo_probabilities()
        features = datasets_manager.get_column_labels()
        threshold = None

        # Initialize base model and predict probabilities if not provided
        if base_model_manager is None and predicted_probabilities is None:
            raise ValueError("Either the base model or the predicted probabilities should be provided!")

        if predicted_probabilities is None:
            # base_model = base_model_manager.get_instance()
            predicted_probabilities = base_model_manager.predict_proba(x)[:, 1]  # base_model.predict(x, True)
            threshold = base_model_manager.threshold

        dataset.set_pseudo_probs_labels(predicted_probabilities, threshold)

        # Step 2 : Mode and metrics setup
        valid_modes = ['mpc', 'apc', 'ipc']
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode '{mode}'. The mode must be one of {valid_modes}.")

        if med3pa_metrics == []:
            med3pa_metrics = ClassificationEvaluationMetrics.supported_metrics()

        # Step 3 : Calculate uncertainty values
        uncertainty_calc = UncertaintyCalculator(uncertainty_metric)
        uncertainty_values = uncertainty_calc.calculate_uncertainty(x, predicted_probabilities, y_true)

        # Step 4: Set up splits to evaluate the models
        if evaluate_models:
            x_train, x_test, uncertainty_train, uncertainty_test, y_train, y_test = train_test_split(x,
                                                                                                     uncertainty_values,
                                                                                                     y_true,
                                                                                                     test_size=0.1,
                                                                                                     random_state=42)
        else:
            x_train = x
            uncertainty_train = uncertainty_values

            # Split the data if pretrained models are not available
        if pretrained_ipc is None and pretrained_apc is None:
            # Split the data in half: one half for IPC, the other half for APC
            x_ipc, x_apc, uncertainty_ipc, uncertainty_apc, y_ipc, _ = train_test_split(
                x_train, uncertainty_train, y_train, test_size=0.5, random_state=42)
        else:
            x_ipc, uncertainty_ipc = x_train, uncertainty_train
            x_apc, uncertainty_apc = x_train, uncertainty_train

        results = Med3paRecord()

        # Step 5: Create and train IPCModel
        if pretrained_ipc is None and ipc_instance is None:
            IPC_model = IPCModel(model_name=ipc_type, params=ipc_params, pretrained_model=None)
            if ipc_type == 'EnsembleRandomForestRegressor':
                class_1_prop = np.sum(y_ipc) / len(y_ipc)
                sample_weight = np.where(y_ipc == 0, 1 / (1 - class_1_prop), 1 / class_1_prop)
                IPC_model.train(x_ipc, uncertainty_ipc, sample_weight=sample_weight)
            else:
                IPC_model.train(x_ipc, uncertainty_ipc)
            print("IPC Model training complete.")
            # optimize IPC model if grid params were provided
            if ipc_grid_params is not None:
                if len(uncertainty_values) > 4:
                    # No optimization if 4 or less samples
                    if ipc_type == 'EnsembleRandomForestRegressor':
                        IPC_model.optimize(ipc_grid_params, ipc_cv, x_train, uncertainty_train, sample_weight)
                    else:
                        IPC_model.optimize(ipc_grid_params, ipc_cv, x_train, uncertainty_train)
                    print("IPC Model optimization complete.")
        elif pretrained_ipc is not None:
            IPC_model = IPCModel(model_name=ipc_type, params=ipc_params, pretrained_model=pretrained_ipc)
            print("Loaded a pretrained IPC model.")
        else:
            IPC_model = ipc_instance
            print("Used a trained IPC instance.")

        # Predict IPC values
        IPC_values = IPC_model.predict(x)
        print("Individualized confidence scores calculated.")
        # Save the calculated confidence scores by the APCmodel
        ipc_dataset = dataset.clone()
        ipc_dataset.set_confidence_scores(IPC_values)
        results.set_dataset(mode="ipc", dataset=ipc_dataset)
        results.set_confidence_scores(IPC_values, "ipc")
        metrics_by_dr = MDRCalculator.calc_metrics_by_dr(dataset=ipc_dataset, confidence_scores=IPC_values,
                                                         metrics_list=med3pa_metrics)
        results.set_metrics_by_dr(metrics_by_dr)
        if mode in ['mpc', 'apc']:

            # Step 6: Create and train APCModel
            IPC_values = IPC_model.predict(x_apc)
            if pretrained_apc is None and apc_instance is None:
                APC_model = APCModel(features=features, params=apc_params)
                APC_model.train(x_apc, IPC_values)
                print("APC Model training complete.")
                # optimize APC model if grid params were provided
                if apc_grid_params is not None:
                    APC_model.optimize(apc_grid_params, apc_cv, x_apc, uncertainty_apc)
                    print("APC Model optimization complete.")
            elif pretrained_apc is not None:
                APC_model = APCModel(features=features, params=apc_params, pretrained_model=pretrained_apc)
                APC_model.train(x_apc, IPC_values)
                print("Loaded a pretrained APC model.")
            else:
                APC_model = apc_instance
                print("Used a trainde IPC instance.")

            # Predict APC values
            APC_values = APC_model.predict(x_apc)
            print("Aggregated confidence scores calculated.")
            # Save the tree structure created by the APCModel
            tree = APC_model.treeRepresentation
            results.set_tree(tree=tree)
            # Save the calculated confidence scores by the APCmodel
            apc_dataset = dataset.clone()
            apc_dataset.set_confidence_scores(APC_model.predict(x))
            results.set_dataset(mode="apc", dataset=apc_dataset)
            results.set_confidence_scores(APC_model.predict(x), "apc")

            # Step 7: Create and train MPCModel
            if mode == 'mpc':
                # Create and predict MPC values
                MPC_model = MPCModel(IPC_values=IPC_model.predict(x), APC_values=APC_model.predict(x))
                MPC_values = MPC_model.predict()
                # Save the calculated confidence scores by the MPCmodel
                mpc_dataset = dataset.clone()
                mpc_dataset.set_confidence_scores(MPC_values)
                results.set_dataset(mode="mpc", dataset=mpc_dataset)
                results.set_confidence_scores(MPC_values, "mpc")
            else:
                MPC_model = MPCModel(APC_values=APC_values)
                MPC_values = MPC_model.predict()
                mpc_dataset = dataset.clone()
                mpc_dataset.set_confidence_scores(MPC_values)

            print("Mixed confidence scores calculated.")

            # Step 8: Calculate the profiles for the different samples_ratio and drs
            profiles_manager = ProfilesManager(features)
            for samples_ratio in range(samples_ratio_min, samples_ratio_max + 1, samples_ratio_step):
                # Calculate profiles and their metrics by declaration rate
                MDRCalculator.calc_profiles(profiles_manager, tree, mpc_dataset, features, MPC_values, samples_ratio)
                MDRCalculator.calc_metrics_by_profiles(profiles_manager, mpc_dataset, features, MPC_values,
                                                       samples_ratio, med3pa_metrics)
                results.set_profiles_manager(profiles_manager)
                print("Results extracted for minimum_samples_ratio = ", samples_ratio)

        if mode in ['mpc', 'apc']:
            ipc_config = IPC_model
            apc_config = APC_model
            if evaluate_models:
                IPC_evaluation = IPC_model.evaluate(x_test, uncertainty_test, models_metrics)
                APC_evaluation = APC_model.evaluate(x_test, uncertainty_test, models_metrics)
                results.set_models_evaluation(IPC_evaluation, APC_evaluation)
        else:
            ipc_config = IPC_model
            apc_config = None
            if evaluate_models:
                IPC_evaluation = IPC_model.evaluate(x_test, uncertainty_test, models_metrics)
                results.set_models_evaluation(IPC_evaluation, None)

        return results, ipc_config, apc_config


class Med3paDetectronExperiment:
    @staticmethod
    @checkpoint(root_path="checkpoints", verbosity=False)
    def run(datasets: DatasetsManager,
            base_model_manager: BaseModelManager,
            uncertainty_metric: str = 'absolute_error',
            training_params: Dict = None,
            samples_size: int = 20,
            samples_size_profiles: int = 10,
            ensemble_size: int = 10,
            num_calibration_runs: int = 100,
            patience: int = 3,
            test_strategies: Union[str, List[str]] = "enhanced_disagreement_strategy",
            allow_margin: bool = False,
            margin: float = 0.05,
            ipc_type: str = 'RandomForestRegressor',
            ipc_params: Dict = None,
            ipc_grid_params: Dict = None,
            ipc_cv: int = None,
            pretrained_ipc: str = None,
            apc_params: Dict = None,
            apc_grid_params: Dict = None,
            apc_cv: int = None,
            pretrained_apc: str = None,
            samples_ratio_min: int = 0,
            samples_ratio_max: int = 50,
            samples_ratio_step: int = 5,
            med3pa_metrics: List[str] = ['Accuracy', 'BalancedAccuracy', 'Precision', 'Recall', 'F1Score',
                                         'Specificity', 'Sensitivity', 'Auc', 'LogLoss', 'Auprc', 'NPV', 'PPV', 'MCC'],
            evaluate_models: bool = False,
            use_ref_models: bool = False,
            models_metrics: List[str] = ['MSE', 'RMSE', 'MAE'],
            mode: str = 'mpc',
            all_dr: bool = False,
            prev_med3pa_results=None) -> Med3paResults:
        """Runs the MED3PA and Detectron experiment.
        Args:
            datasets (DatasetsManager): The datasets manager instance.
            training_params (dict): Parameters for training the models.
            base_model_manager (BaseModelManager): The base model manager instance.
            uncertainty_metric (str, optional): the uncertainty metric ysed to calculate uncertainty, by default absolute_error.
            samples_size (int, optional): Sample size for the Detectron experiment, by default 20.
            samples_size_profiles (int, optional): Sample size for Profiles Detectron experiment, by default 10.
            ensemble_size (int, optional): Number of models in the ensemble, by default 10.
            num_calibration_runs (int, optional): Number of calibration runs, by default 100.
            patience (int, optional): Patience for early stopping, by default 3.
            test_strategies (Union[str, List[str]): strategies for testing disagreement, by default enhanced_disagreement_strategies.
            allow_margin (bool, optional): Whether to allow a margin in the test, by default False.
            margin (float, optional): Margin value for the test, by default 0.05.
            ipc_type (str, optional): The regressor model to use for IPC, by default RandomForestRegressor.
            ipc_params (dict, optional): Parameters for initializing the IPC regressor model, by default None.
            ipc_grid_params (dict, optional): Grid search parameters for optimizing the IPC model, by default None.
            ipc_cv (int, optional): Number of cross-validation folds for optimizing the IPC model, by default None.
            pretrained_ipc (str, optional): path to a pretrained ipc, by default None.
            apc_params (dict, optional): Parameters for initializing the APC regressor model, by default None.
            apc_grid_params (dict, optional): Grid search parameters for optimizing the APC model, by default None.
            pretrained_apc (str, optional): path to a pretrained apc, by default None.
            apc_cv (int, optional): Number of cross-validation folds for optimizing the APC model, by default None.
            samples_ratio_min (int, optional): Minimum sample ratio, by default 0.
            samples_ratio_max (int, optional): Maximum sample ratio, by default 50.
            samples_ratio_step (int, optional): Step size for sample ratio, by default 5.
            med3pa_metrics (list of str, optional): List of metrics to calculate, by default ['Auc', 'Accuracy', 'BalancedAccuracy'].
            evaluate_models (bool, optional): Whether to evaluate the models, by default False.
            models_metrics (list of str, optional): List of metrics for model evaluation, by default ['MSE', 'RMSE'].
            all_dr (bool, optional): Whether to run for all declaration rates, by default False.
        Returns:
            Med3paResults: Results of MED3pa on reference and testing sets, plus Detectron Results.
        """

        valid_modes = ['mpc', 'apc']

        if mode not in valid_modes:
            raise ValueError(f"Invalid mode '{mode}'. The mode must be one of {valid_modes}.")

        if prev_med3pa_results is not None:
            print("Using previous med3pa_results to execute Med3paDetectron experiment")
            med3pa_results = prev_med3pa_results

        else:
            med3pa_results = Med3paExperiment.run(datasets_manager=datasets,
                                                  base_model_manager=base_model_manager,
                                                  uncertainty_metric=uncertainty_metric,
                                                  ipc_params=ipc_params, ipc_grid_params=ipc_grid_params, ipc_cv=ipc_cv,
                                                  ipc_type=ipc_type, pretrained_ipc=pretrained_ipc,
                                                  apc_params=apc_params, apc_grid_params=apc_grid_params, apc_cv=apc_cv,
                                                  pretrained_apc=pretrained_apc,
                                                  evaluate_models=evaluate_models, models_metrics=models_metrics,
                                                  samples_ratio_min=samples_ratio_min,
                                                  samples_ratio_max=samples_ratio_max,
                                                  samples_ratio_step=samples_ratio_step,
                                                  med3pa_metrics=med3pa_metrics, mode=mode,
                                                  use_ref_models=use_ref_models)

        # print("Running Global Detectron Experiment:")
        # detectron_results = DetectronExperiment.run(datasets=datasets, training_params=training_params,
        #                                             base_model_manager=base_model_manager,
        #                                             samples_size=samples_size,
        #                                             num_calibration_runs=num_calibration_runs,
        #                                             ensemble_size=ensemble_size,
        #                                             patience=patience, allow_margin=allow_margin, margin=margin)
        # detectron_results.analyze_results(test_strategies)

        # if med3pa_results.test_record.get_confidence_scores("mpc") is not None:
        #     confidence_scores = med3pa_results.test_record.get_confidence_scores("mpc")
        # elif med3pa_results.test_record.get_confidence_scores("apc") is not None:
        #     confidence_scores = med3pa_results.test_record.get_confidence_scores("apc")
        # else:
        #     raise ValueError("the confidence scores were not calculated!")
        if mode == "mpc":
            confidence_scores = med3pa_results.test_record.get_confidence_scores("mpc")
        elif mode == "apc":
            confidence_scores = med3pa_results.test_record.get_confidence_scores("apc")
        else:
            raise ValueError("the confidence scores were not calculated!")

        print("Running Profiled Detectron Experiment:")
        detectron_results, detectron_profiles_res = MDRCalculator.detectron_by_profiles(datasets=datasets,
                                                                                        profiles_manager=med3pa_results.test_record.get_profiles_manager(),
                                                                                        training_params=training_params,
                                                                                        base_model_manager=base_model_manager,
                                                                                        confidence_scores=confidence_scores,
                                                                                        samples_size=samples_size_profiles,
                                                                                        num_calibration_runs=num_calibration_runs,
                                                                                        ensemble_size=ensemble_size,
                                                                                        patience=patience,
                                                                                        strategies=test_strategies,
                                                                                        allow_margin=allow_margin,
                                                                                        margin=margin,
                                                                                        all_dr=all_dr)
        if detectron_results is None:
            detectron_results = DetectronExperiment.run(datasets=datasets, training_params=training_params,
                                                        base_model_manager=base_model_manager,
                                                        samples_size=samples_size,
                                                        num_calibration_runs=num_calibration_runs,
                                                        ensemble_size=ensemble_size,
                                                        patience=patience, allow_margin=allow_margin, margin=margin)
        detectron_results.analyze_results(test_strategies)

        med3pa_params = {
            'uncertainty_metric': uncertainty_metric,
            'samples_ratio_min': samples_ratio_min,
            'samples_ratio_max': samples_ratio_max,
            'samples_ratio_step': samples_ratio_step,
            'med3pa_metrics': med3pa_metrics,
            'evaluate_models': evaluate_models,
            'models_evaluation_metrics': models_metrics,
            'mode': mode,

        }

        detectron_params = {
            'samples_size': samples_size,
            'profiles_samples_size': samples_size_profiles,
            'cdcs_ensemble_size': ensemble_size,
            'num_runs': num_calibration_runs,
            'patience': patience,
            'allow_margin': allow_margin,
            'margin': margin,
            'additional_training_params': training_params,
        }
        experiment_config = {
            'experiment_name': "Med3paDetectronExperiment",
            'med3pa_detectron_params': {},
        }

        experiment_config['med3pa_detectron_params']['detectron_params'] = detectron_params
        experiment_config['med3pa_detectron_params']['med3pa_params'] = med3pa_params

        med3pa_results.set_detectron_results(detectron_results)
        med3pa_results.set_experiment_config(experiment_config)

        return med3pa_results
