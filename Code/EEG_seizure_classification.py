

import sklearn
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import pickle
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import LocalOutlierFactor, KNeighborsRegressor, KNeighborsClassifier

from sklearn.model_selection import (train_test_split, cross_val_score,
GridSearchCV, StratifiedKFold, KFold)

from sklearn.preprocessing import (StandardScaler, Normalizer, RobustScaler,
QuantileTransformer, PowerTransformer, LabelEncoder, OneHotEncoder, OrdinalEncoder)

from sklearn.metrics import (classification_report, mean_squared_error, 
confusion_matrix, plot_confusion_matrix, ConfusionMatrixDisplay, accuracy_score,
balanced_accuracy_score)

# Loading the data from .csv
data = pd.read_csv('Epileptic Seizure Recognition.csv', index_col = 0)

# Re-labelling all four non-epileptic pacient classes to 0
data.loc[data["y"] > 1 , "y"] = 0

# Printing dimensions of the data
print(data.shape)
# Separate Data into a Training and Validation Datasets
test_size = 0.20 # Allocating 80/20 % split of the data for train/test
seed = 10

x_train, x_test, y_train, y_test = train_test_split(data.drop(axis=1,labels=["y"]), 
                                                    data["y"],
                                                    test_size=test_size, 
                                                    random_state=seed,
                                                    stratify=data["y"])

# Printing the dimensions of the training and validation datasets
print(f"Train x: {x_train.shape}\nTrain y: {y_train.shape}")
print(f"Test x: {x_test.shape}\nTest y   : {y_test.shape}")
# Train a baseline

# instantiate a knn object
knn = KNeighborsClassifier(n_neighbors=5)

# train the model
knn.fit(x_train, y_train)

# predict
y_pred_knn = knn.predict(x_test)

# evaluate
print("Accuracy:", balanced_accuracy_score(y_test, y_pred_knn))
# Create a PIPELINE to investigate different Normalization techniques

# Standardize the dataset
pipelines_list = []
pipelines_list.append(('NonScaledKnn', 
                  Pipeline([('KNN',
                             KNeighborsClassifier(n_neighbors=5,n_jobs=-1))])))
pipelines_list.append(('ScaledKnn', 
                  Pipeline([('Scaler', 
                             StandardScaler()),
                            ('KNN',
                             KNeighborsClassifier(n_neighbors=5, n_jobs=-1))])))

pipelines_list.append(('NormalizedKnn', 
                  Pipeline([('Normalizer', 
                             Normalizer()),
                            ('KNN',
                             KNeighborsClassifier(n_neighbors=5, n_jobs=-1))])))

pipelines_list.append(('RobustedKnn', 
                  Pipeline([('Robust', 
                             RobustScaler()),
                            ('KNN',
                             KNeighborsClassifier(n_neighbors=5, n_jobs=-1))])))

pipelines_list.append(('QuantiledKnn', 
                  Pipeline([('Quantile', 
                             QuantileTransformer()),
                            ('KNN',
                             KNeighborsClassifier(n_neighbors=5, n_jobs=-1))])))

pipelines_list.append(('PoweredKnn', 
                  Pipeline([('Power', 
                             PowerTransformer()),
                            ('KNN',
                             KNeighborsClassifier(n_neighbors=5, n_jobs=-1))])))
# Cross-validation allows us to compare different 
# machine learning methods and get a sense of how
# well they will work in practice

# Test options and evaluation metric
num_folds = 10
scoring = 'balanced_accuracy'

results = []
names = []

for name, model in pipelines_list:
  # k-fold size
  kfold = KFold(n_splits=num_folds)
  # cross validation
  cv_results = cross_val_score(model, x_train, y_train,cv=kfold,scoring=scoring)
  # store results
  results.append(-cv_results)
  names.append(name)
  print("%s Mean: %f Std: %f" % (name, 
                                      cv_results.mean(), 
                                      cv_results.std()))
  # KNN Algorithm tuning (beat the baseline)

# hyperparameter
k_values = np.array([3,5,11,15,20,25])
weights = ['uniform','distance']
algorithm = ['auto', 'ball_tree', 'kd_tree', 'brute']
metric = [1,2]

# number of combinations
models = (len(k_values) * len(weights) * len(algorithm) * len(metric))
rounds = models * num_folds

print(f"Number of models: {models}")
print(f"Complexity of evaluation: {rounds} rounds")

param_grid = dict(n_neighbors=k_values, 
                  weights=weights,
                  algorithm=algorithm,
                  p=metric)

# Instantiate a normalization algorithm
pt = PowerTransformer()
# Learning the stats from feature
pt.fit(x_train)
# Transform x train
scaler = pt.transform(x_train)

# instantiate a model
model = KNeighborsClassifier()

# Test options and evaluation metric
num_folds = 10
scoring = 'balanced_accuracy'

# Grid Searching with cross-validation
kfold = KFold(n_splits=num_folds)
grid = GridSearchCV(estimator=model, 
                    param_grid=param_grid, 
                    scoring=scoring,
                    cv=kfold)
# train the model
grid_result = grid.fit(scaler, y_train)

# Print results
print("Best: %f using %s" % (grid_result.best_score_, 
                             grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
  print("%f (%f) with: %r" % (mean, stdev, param))

# predict using the best estimator
# pay attention in normalizer instance that was the same used in the train

predict = grid_result.best_estimator_.predict(pt.transform(x_test))

print("Accuracy:", balanced_accuracy_score(y_test, predict))
# Save the model using pickle
with open('pipe2.pkl', 'wb') as file:
  pickle.dump(grid_result, file)
  # Under the production environment [pickle]
with open('pipe2.pkl', 'rb') as file:
  model = pickle.load(file)

predict = model.best_estimator_.predict(pt.transform(x_test))
print("Accuracy:", balanced_accuracy_score(y_test, predict))

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
# Random Forest using RandomForestClassifier
rnd_clf = RandomForestClassifier(n_estimators=500,
                                 max_leaf_nodes=16, 
                                 random_state=42)
rnd_clf.fit(x_train, y_train)

# Predict
predict = rnd_clf.predict(x_test)

# Evaluate
print("Accuracy:", balanced_accuracy_score(y_test, predict))
# global varibles
seed = 15
num_folds = 10
#scoring = {'AUC': 'roc_auc', 'Accuracy': make_scorer(accuracy_score)}
scoring = 'balanced_accuracy'
# See documentation for more info
# https://scikit-learn.org/stable/auto_examples/compose/plot_compare_reduction.html#sphx-glr-auto-examples-compose-plot-compare-reduction-py

# create a dictionary with the hyperparameters
search_space = [{"n_estimators": [100,200,300,400],
                 "criterion": ["gini","entropy"],
                 "max_leaf_nodes": [4,16,32,64,128],
                 "random_state": [seed],}]

# create grid search
kfold = StratifiedKFold(n_splits=num_folds,random_state=seed,shuffle=True)
model = RandomForestClassifier()
# see other scoring
# https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter

grid = RandomizedSearchCV(estimator=model, 
                    param_distributions=search_space,
                    n_iter=20,
                    cv=kfold,
                    verbose = 2,
                    scoring=scoring,
                    return_train_score=True,
                    n_jobs=-1,
                    refit="Accuracy")

# fit grid search
grid_result = grid.fit(x_train,y_train)
# Print results
print("Best: %f using %s" % (grid_result.best_score_, 
                             grid_result.best_params_))
# predict using the best estimator

predict = grid_result.best_estimator_.predict(x_test)

print("Accuracy:", balanced_accuracy_score(y_test, predict))
# Save the model using pickle
with open('pipeRF2.pkl', 'wb') as file:
  pickle.dump(grid_result, file)
  # Under the production environment [pickle]
with open('pipeRF2.pkl', 'rb') as file:
  model = pickle.load(file)

predict = model.best_estimator_.predict(pt.transform(x_test))
print("Accuracy:", balanced_accuracy_score(y_test, predict))
