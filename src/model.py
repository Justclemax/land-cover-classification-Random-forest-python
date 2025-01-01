from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from tqdm import tqdm
import pandas as pd
from loguru import logger
import argparse
from common import transform_color, extraction_data_in_optic_image
import os
from dotenv import load_dotenv
import time
from sklearn.base import clone

load_dotenv()


class ProgressRandomForest(RandomForestClassifier):
    """
    Wrapper for RandomForestClassifier to add progress tracking using tqdm.
    """

    def fit(self, X, y, sample_weight=None):
        # Override the `fit` method to include progress tracking
        self.n_jobs = 1  # Ensure jobs are handled sequentially for tracking
        self._tqdm = tqdm(total=self.n_estimators, desc="Training Progress", unit="trees")

        original_fit = super().fit  # Store the original fit method

        def monitored_fit(estimator, X, y, sample_weight=None, **fit_params):
            """Monitored fit for individual trees in the ensemble."""
            self._tqdm.update(1)  # Update the progress bar
            return original_fit(estimator, X, y, sample_weight=sample_weight, **fit_params)

        self._fit = monitored_fit  # Override the fit function
        result = super().fit(X, y, sample_weight=sample_weight)
        self._tqdm.close()  # Close the tqdm bar after training
        return result


def build_model(data, seeds: int = 2, report_save_path: str = "../data/classification_report.csv"):
    """
    Builds a RandomForestClassifier model, trains it, and evaluates it with a confusion matrix and classification report.

    Parameters:
    - data (pd.DataFrame): The input data to train and test the model.
    - seeds (int): Random seed for reproducibility.
    - report_save_path (str): Path to save the classification report.

    Returns:
    - model: The trained model.
    """
    try:
        # Ensure data contains the necessary 'value' column
        if data is None or "value" not in data.columns:
            raise ValueError("The required column 'value' is missing in the input data.")

        # Split data into training and testing sets
        logger.info("Starting data split...")
        train, test = train_test_split(data, train_size=0.8, random_state=seeds)
        logger.info(f"Data split completed: {len(train)} training samples, {len(test)} testing samples.")

        # Define the features (predictors)
        predictors = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"]

        # Initialize the ProgressRandomForest with 100 trees
        model = ProgressRandomForest(n_estimators=100, random_state=seeds)
        logger.info("RandomForest model with progress tracking initialized.")

        # Start training the model with tqdm
        logger.info("Training the model...")
        start_time = time.time()  # Start timing the training process
        model.fit(train[predictors], train["value"])
        training_time = time.time() - start_time  # Calculate training time
        logger.info(f"Model training completed in {training_time:.2f} seconds.")

        # Test the model and make predictions
        logger.info("Testing the model on the test set...")
        test_predictions = model.predict(test[predictors])

        # Confusion matrix visualization
        logger.info("Generating confusion matrix...")
        cm = confusion_matrix(test["value"], test_predictions)
        cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
        cm_display.plot(cmap='Blues')
        logger.info("Confusion matrix displayed.")

        # Generate the classification report
        logger.info("Generating classification report...")
        report = classification_report(test["value"], test_predictions, output_dict=True)
        report_df = pd.DataFrame(report).transpose()

        # Save the report to a CSV file
        report_df.to_csv(report_save_path, index=True)
        logger.success(f"Classification report saved successfully to {report_save_path}")

    except Exception as e:
        logger.critical(f"Unexpected error occurred during model building and evaluation: {e}")
        raise

    return model


def main():
    # Initialisation
    parser = argparse.ArgumentParser(description="Process and visualize land cover data.")
    parser.add_argument("--color_data", required=False, help="Path to the land cover color JSON file.",
                        default=os.getenv('COLOR_DATA'))
    parser.add_argument("--sample_data", required=False, help="Path to the GeoJSON or Shapefile sample data.",
                        default=os.getenv("SAMPLE_DATA_PATH"))
    parser.add_argument("--data_path_raster_image", required=False, help="Path to raster image",
                        default=os.getenv("PATH_DATA_IMAGE"))
    parser.add_argument("--report_save_path", required=False, help="Path to save the classification report.",
                        default="../data/classification_report.csv")

    # Parse les arguments
    args = parser.parse_args()

    try:
        # Log information and data transformation
        logger.info("Loading and transforming color data...")
        transform_result = transform_color(args.color_data)

        # Log and extract data from the sample and optical image
        logger.info("Extracting data from optical image and sample...")
        data = extraction_data_in_optic_image(args.sample_data, args.data_path_raster_image, transform_result)

        # Build and evaluate the model
        logger.info("Building and evaluating the model...")
        build_model(data=data, report_save_path=args.report_save_path)

    except Exception as e:
        logger.critical(f"Critical error in the main script: {e}")
        return None


if __name__ == "__main__":
    main()