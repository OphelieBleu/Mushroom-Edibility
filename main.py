import pandas as pd

from features import FEATURE_NAMES, EDIBILITY_CLASS
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

class ReadDatasetFile:
    def __init__(self, file_path: str, columns: List[str] = None):
        self.file_path = file_path
        self.columns = columns

    def check_file_extension_csv(self):
        file_extension = self.file_path.split('.')[-1].lower()
        return file_extension == '.csv'

    def convert_text_file_to_dataframe(self):
        data = []
        with open(self.file_path, 'r') as file:
            for line in file:
                data.append(line.strip().split(','))
        return pd.DataFrame(data, columns=self.columns)

    def file_converter_to_dataframe(self):
        if self.check_file_extension_csv():
            return pd.read_csv(self.file_path)
        else:
            return self.convert_text_file_to_dataframe()

class MushroomEdibilityModel():
    def __init__(self, data: pd.DataFrame, label_column: str, pos_label: str='p'):
        self.labels = data[label_column]
        self.data = data.drop(label_column, axis=1)
        self.encoded_label = self.encode_edibility(self.labels, pos_label)
        self.encoded_data = self.get_categorical_df(self.data)
        self.data_train, self.data_test, self.label_train, self.label_test = train_test_split(self.encoded_data, self.encoded_label, test_size = 0.25, random_state = 4)

    def get_categorical_df(self, data):
        encoded_df = pd.DataFrame()
        label_encoder = LabelEncoder()
        for col in data.columns:
            encoded_df[col] = label_encoder.fit_transform(data[col])
        return encoded_df

    def encode_edibility(self, labels: pd.Series, pos_label: str):
        return (labels == pos_label).astype(int)

    def get_model_from_type(self, model_type: str=None):
        if model_type == 'lr':
            model = LogisticRegression(random_state=4)
        else:
            model = RandomForestClassifier(random_state=4)
        return model

    def generate_report(self, model_type: str=None):
        model = self.get_model_from_type()
        model.fit(self.data_train, self.label_train)
        confusion_matrix = self.evaluate_model(model)
        print(confusion_matrix)
        self.get_feature_importance_for_model(model)

    def evaluate_model(self, model):
        label_predicted = model.predict(self.data_test)
        conf_matrix = confusion_matrix(self.label_test, label_predicted)
        return conf_matrix

    def get_feature_importance_for_model(self, model):
        plt.figure(figsize=(10, 6))
        sns.barplot(x=model.feature_importances_, y=self.data_train.columns)
        plt.xlabel('Feature Importance')
        plt.ylabel('Feature')
        plt.title('Feature Importance Plot')
        plt.show()

if __name__ == '__main__':
    FILE_PATH = 'data/mushroom/agaricus-lepiota.data'
    dataset_reader = ReadDatasetFile(FILE_PATH, FEATURE_NAMES)
    df = dataset_reader.file_converter_to_dataframe()
    mushroom_edibility = MushroomEdibilityModel(df, 'poisonous')
    mushroom_edibility.generate_report()