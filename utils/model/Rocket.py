# Import required libraries
import os
import logging
import pandas as pd
import numpy as np
from sklearn import metrics
from sktime.transformations.panel.rocket import Rocket
from sklearn.linear_model import RidgeClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

exercise_map = {
    "Side-Lateral-Raise": 10000,
    "Lunge" : 20000,
}

class RocketTransformerClassifier:
    
    def __init__(self, exercise):
        self.classifiers_mapping = {}
        self.exercise = exercise

    def fit_rocket(self, x_train, y_train):
        # Initialize and fit Rocket transformer
        rocket = Rocket(num_kernels=exercise_map[self.exercise], normalise=False)
        rocket.fit(x_train)
        x_training_transform = rocket.transform(x_train)

        # Normalize the transformed data
        scaler = StandardScaler()
        x_training_transform = scaler.fit_transform(x_training_transform)

        # Train RidgeClassifier with normalized transformed data
        classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
        classifier.fit(x_training_transform, y_train)

        # Store the transformer, scaler, and classifier
        self.classifiers_mapping["transformer"] = rocket
        self.classifiers_mapping["scaler"] = scaler
        self.classifiers_mapping["classifier"] = classifier

    def evaluate(self, x_val, y_val):
        rocket = self.classifiers_mapping["transformer"]
        scaler = self.classifiers_mapping["scaler"]
        classifier = self.classifiers_mapping["classifier"]
    
        # Transform and normalize test data
        x_val_transform = rocket.transform(x_val)
        x_val_transform = scaler.transform(x_val_transform)
    
        # Predict and evaluate
        predictions = classifier.predict(x_val_transform)
        accuracy = metrics.accuracy_score(y_val, predictions)

        # logger.info("-----------------------------------------------")
        # logger.info(f"Accuracy: {accuracy}")

        return accuracy


    def predict_rocket(self, x_test, y_test):
        # Retrieve transformer, scaler, and classifier
        rocket = self.classifiers_mapping["transformer"]
        scaler = self.classifiers_mapping["scaler"]
        classifier = self.classifiers_mapping["classifier"]
    
        # Transform and normalize test data
        x_test_transform = rocket.transform(x_test)
        x_test_transform = scaler.transform(x_test_transform)
    
        # Predict and evaluate
        predictions = classifier.predict(x_test_transform)
        accuracy = metrics.accuracy_score(y_test, predictions)
        confusion_matrix = metrics.confusion_matrix(y_test, predictions)
        classification_report = metrics.classification_report(y_test, predictions)
    
        # logger.info("-----------------------------------------------")
        # logger.info(f"Accuracy: {accuracy}")
        # logger.info("\nConfusion Matrix:\n" + str(confusion_matrix))
        # logger.info("\nClassification Report:\n" + classification_report)
    
        return accuracy, confusion_matrix, classification_report
    
    