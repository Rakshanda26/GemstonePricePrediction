import os
import sys
import mlflow
import mlflow.sklearn
import numpy as np
from urllib.parse import urlparse
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.utils.utils import load_object
from src.logger.logging import logging
from src.exception.exception import customexception


class ModelEvaluation:
    def __init__(self):
        """
        Initialize the ModelEvaluation class and log the start of evaluation.
        """
        logging.info("Evaluation process started.")

    def eval_metrics(self, actual, pred):
        """
        Calculate evaluation metrics: RMSE, MAE, and R2 Score.
        Args:
            actual: Actual target values.
            pred: Predicted target values.
        Returns:
            rmse: Root Mean Squared Error.
            mae: Mean Absolute Error.
            r2: R-squared score.
        """
        try:
            rmse = np.sqrt(mean_squared_error(actual, pred))
            mae = mean_absolute_error(actual, pred)
            r2 = r2_score(actual, pred)
            logging.info("Evaluation metrics computed successfully.")
            return rmse, mae, r2
        except Exception as e:
            logging.error("Error occurred while calculating evaluation metrics.")
            raise customexception(e, sys)

    def initiate_model_evaluation(self, train_array, test_array):
        """
        Evaluate the model on the test data and log metrics using MLflow.
        Args:
            train_array: Array containing the training data.
            test_array: Array containing the test data.
        """
        try:
            # Extract features and target labels from test data
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            # Path to the saved model
            model_path = os.path.join("artifacts", "model.pkl")
            
            # Load the model object
            model = load_object(model_path)
            logging.info("Model loaded successfully.")

            # Determine the MLflow tracking URI scheme
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
            logging.info(f"Tracking URL type: {tracking_url_type_store}")

            # Start an MLflow run
            with mlflow.start_run():
                # Make predictions and calculate evaluation metrics
                predictions = model.predict(X_test)
                rmse, mae, r2 = self.eval_metrics(y_test, predictions)

                # Log metrics to MLflow
                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("mae", mae)
                mlflow.log_metric("r2", r2)

                # Log the model
                if tracking_url_type_store != "file":
                    # Log and register the model with MLflow Model Registry
                    mlflow.sklearn.log_model(model, "model", registered_model_name="ml_model")
                else:
                    # Log the model locally
                    mlflow.sklearn.log_model(model, "model")
                
                logging.info("Model evaluation and logging completed successfully.")

        except Exception as e:
            logging.error("Error occurred during model evaluation.")
            raise customexception(e, sys)
