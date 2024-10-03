import pandas as pd

from constants import FEATURE_NAMES, EDIBILITY_CLASS
from typing import List
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import random
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import shap
import xgboost

FILE_PATH = 'data/mushroom/agaricus-lepiota.data'

def get_mushroom_features_table(file_path, features: List[str]=None) -> pd.DataFrame:
    """
    Reads a file containing mushroom data and returns a pandas DataFrame.

    Args:
        file_path (str): Path to the file.
        features (List[str], optional): List of feature names to include in the DataFrame.
            If None, default names for columns will be applied.

    Returns:
        pd.DataFrame: DataFrame containing the mushroom data.

    Raises:
        FileNotFoundError: If the specified file path does not exist.
        PermissionError: If the file cannot be opened due to permission issues.
        IOError: If there's an unexpected error while opening or reading the file.
    """
    try:
        with open(file_path, 'r') as file:
            data = [line.strip().split(',') for line in file]
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found: {file_path}")
    except PermissionError as e:
        raise PermissionError(f"Permission denied: {file_path}")
    except IOError as e:
        raise IOError(f"Error opening file: {file_path}")
    return pd.DataFrame(data, columns=features)


def encode_edibility(labels: pd.Series, positive_label: str) -> pd.Series:
    """
    Encodes edibility labels as binary values.

    Args:
        labels (pd.Series): Series containing edibility labels.
        positive_label (str): Label representing poisonous mushrooms.

    Returns:
        pd.Series: Series containing binary encoded edibility labels (0 for edible, 1 for poisonous).
    """
    return (labels == positive_label).astype(int)


def get_encoded_df(features_table: pd.DataFrame) -> pd.DataFrame:
    """
    Encodes categorical features in a DataFrame using label encoding.

    Args:
        features_table (pd.DataFrame): DataFrame containing categorical features.

    Returns:
        pd.DataFrame: DataFrame with categorical features encoded as numerical values.
    """
    encoded_df = pd.DataFrame()
    label_encoder = LabelEncoder()
    for col in features_table.columns:
        encoded_df[col] = label_encoder.fit_transform(features_table[col])
    return encoded_df

# MushroomEdibilityModel blabla
class MushroomEdibilityModel:
    """
    Machine learning model for predicting mushroom edibility and get feature importance.

    This class facilitates training and evaluation of a model for classifying mushrooms as edible or poisonous based on their features. It performs data preprocessing steps such as label encoding and train-test split.

    Attributes:
        model (object): The machine learning model object (type depends on the specific model implementation).
        data_train (pd.DataFrame): DataFrame containing training data features.
        data_test (pd.DataFrame): DataFrame containing testing data features.
        label_train (pd.Series): Series containing encoded edibility labels for training data.
        label_test (pd.Series): Series containing encoded edibility labels for testing data.
    """
    def __init__(self, model: object, features_table: pd.DataFrame, label_column: str, positive_label: str='p'):
        """
        Initializes the MushroomEdibilityModel object.

        Args:
            model (object): The machine learning model object to be used for training.
            features_table (pd.DataFrame): DataFrame containing mushroom features.
            label_column (str): Name of the column in the DataFrame containing edibility labels.
            positive_label (str, optional): Label representing edible mushrooms (default is 'p').
        """
        labels = features_table[label_column]
        features_table_without_label = features_table.drop(label_column, axis=1)
        encoded_label = encode_edibility(labels, positive_label)
        encoded_data = get_encoded_df(features_table_without_label)
        self.model = model
        self.data_train, self.data_test, self.label_train, self.label_test = train_test_split(encoded_data, encoded_label, test_size = 0.25, random_state = 4)

    def train(self):
        """
        Trains the machine learning model on the prepared training data.
        """
        model.fit(self.data_train, self.label_train)

    def evaluate(self):
        """
        Evaluates the model's performance on the testing data.

        Returns:
            confusion_matrix: A confusion matrix representing the model's performance.
        """
        label_predicted = model.predict(self.data_test)
        conf_matrix = confusion_matrix(self.label_test, label_predicted)
        return conf_matrix


    def plot_feature_importance_for_model(self):
        """
        Visualizes the feature importance for the trained model.
    
        This method assumes the model being used has a `feature_importances_` attribute or similar functionality to extract feature importance scores.
    
        Raises:
            AttributeError: If the model object lacks a `feature_importances_` attribute.
        """
        try:
            # Sort feature importances and column names together by importance
            sorted_features = sorted(zip(self.data_train.columns, self.model.feature_importances_), key=lambda x: x[1], reverse=True)
            feature_names, importances = zip(*sorted_features)
    
            plt.figure(figsize=(10, 6))
            sns.barplot(x=importances, y=feature_names)
            plt.xlabel('Feature Importance')
            plt.ylabel('Feature')
            plt.title('Feature Importance Plot')
            plt.show()
        except AttributeError:
            print("The model object does not have a 'feature_importances_' attribute. Feature importance cannot be plotted.")
        

if __name__ == '__main__':
    df = get_mushroom_features_table(FILE_PATH, FEATURE_NAMES)

    model = RandomForestClassifier(random_state=4)
    # model = xgboost.XGBClassifier(random_state=4) 
    
    mushroom_edibility = MushroomEdibilityModel(model, df, 'poisonous', 'p')
    
    mushroom_edibility.train()
    print(mushroom_edibility.evaluate())
    mushroom_edibility.plot_feature_importance_for_model()