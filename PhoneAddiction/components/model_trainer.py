import os, sys
from urllib.parse import urlparse

import mlflow
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

from PhoneAddiction.exception.exception import NetworkSecurityException 
from PhoneAddiction.logging.logger import logging
from PhoneAddiction.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from PhoneAddiction.entity.config_entity import ModelTrainerConfig
from PhoneAddiction.utils.ml_utils.model.estimator import NetworkModel
from PhoneAddiction.utils.main_utils.utils import save_object, load_object, load_numpy_array_data, evaluate_models

# MLFlow setup
mlflow.set_tracking_uri("file:./mlruns")


# ✅ Custom function to build MLP classifier
def build_mlp(input_dim, num_classes=3, lr=0.001):
    model = Sequential([
        Dense(128, activation='relu', input_dim=input_dim),
        BatchNormalization(),
        Dropout(0.4),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')   # classification
    ])

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        metrics=['accuracy']
    )
    return model


class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig,
                 data_transformation_artifact: DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def track_mlflow(self, best_model, y_true, y_pred, framework="sklearn"):
        """Log metrics in MLFlow"""
        acc = accuracy_score(y_true, y_pred)
        mlflow.log_metric("accuracy", acc)

        if framework == "sklearn":
            mlflow.sklearn.log_model(best_model, "model")
        elif framework == "keras":
            mlflow.keras.log_model(best_model, "model")

    def train_model(self, X_train, y_train, X_test, y_test):
        # ------------------
        # Define sklearn models
        models = {
            "Logistic Regression": LogisticRegression(max_iter=500, solver='liblinear'),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(),
            "KNN": KNeighborsClassifier()
        }

        params = {
            "Logistic Regression": {'C': [0.01, 0.1, 1, 10], 'penalty': ['l1', 'l2']},
            "Decision Tree": {'criterion': ['gini', 'entropy'], 'max_depth': [3, 5, 10, None]},
            "Random Forest": {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, None]},
            "KNN": {'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance']}
        }

        # ------------------
        # Evaluate sklearn models
        model_report: dict = evaluate_models(
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
            models=models, param=params
        )

        # Pick best sklearn model
        best_model_score = max(sorted(model_report.values()))
        best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
        best_model = models[best_model_name]

        # Predict & metrics for sklearn model
        y_test_pred = best_model.predict(X_test)

        with mlflow.start_run(run_name=best_model_name):
            self.track_mlflow(best_model, y_test, y_test_pred, framework="sklearn")

        # ------------------
        # Train custom Keras MLP
        keras_model = build_mlp(input_dim=X_train.shape[1], num_classes=len(np.unique(y_train)))
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        keras_model.fit(
            X_train, y_train,
            validation_split=0.2,
            epochs=50,
            batch_size=16,
            callbacks=[early_stop],
            verbose=0
        )

        y_test_pred_keras = np.argmax(keras_model.predict(X_test), axis=1)

        with mlflow.start_run(run_name="Keras-MLP"):
            self.track_mlflow(keras_model, y_test, y_test_pred_keras, framework="keras")

        # ------------------
        # Save best sklearn model (main pipeline)
        preprocessor = load_object(self.data_transformation_artifact.transformed_object_file_path)
        model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
        os.makedirs(model_dir_path, exist_ok=True)

        Network_Model = NetworkModel(preprocessor=preprocessor, model=best_model)
        save_object(self.model_trainer_config.trained_model_file_path, obj=Network_Model)
        save_object("final_model/model.pkl", best_model)

        # ------------------
        # Artifact
        model_trainer_artifact = ModelTrainerArtifact(
            trained_model_file_path=self.model_trainer_config.trained_model_file_path,
            train_metric_artifact={"accuracy": accuracy_score(y_train, best_model.predict(X_train))},
            test_metric_artifact={"accuracy": accuracy_score(y_test, y_test_pred)}
        )

        logging.info(f"Model trainer artifact: {model_trainer_artifact}")
        return model_trainer_artifact
    
    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            # Load arrays
            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            X_train, y_train, X_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1],
            )

            return self.train_model(X_train, y_train, X_test, y_test)
        except Exception as e:
            raise NetworkSecurityException(e, sys)
