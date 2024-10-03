# Mushroom-Edibility

This code provides a framework for building and evaluating machine learning models to predict the edibility of mushrooms based on different features. 

The framework has different capabilities:
- Data Loading
- Data Preprocessing and encoding of numerical values into numlerical ones
- Model Creation and Training
- Model Evaluation
- Feature Importance Visualisation

## How to Use

Install Dependencies:
```console
pip install -r requirements.txt
```

### Modify Configuration (Optional)
- Update FILE_PATH with the mushroom data file.
- Customize FEATURE_NAMES in constants.py if your data has different feature names.
- Adjust positive_label in encode_edibility to specify the label indicating poisonous mushrooms (default is 'p').
  
### How to run

```console
python main.py  # Assuming the script is named this
```

The script will print the confusion matrix and plot feature importance.


```console
df = get_mushroom_features_table(FILE_PATH, FEATURE_NAMES)
model = RandomForestClassifier(random_state=4)  # Or other scikit-learn model
mushroom_edibility = MushroomEdibilityModel(model, df, 'poisonous', 'p')
mushroom_edibility.train()
print(mushroom_edibility.evaluate())
mushroom_edibility.plot_feature_importance_for_model()
```

### Extensibility

- Other machine learning models an be used (e.g., DecisionTreeClassifier, LogisticRegression, XGBoostClassifier) by replacing model in the example usage.
- Other file can also be used for the analysis
- Additional categorical features can also be added