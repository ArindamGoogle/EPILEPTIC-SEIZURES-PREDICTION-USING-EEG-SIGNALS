# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 15:16:46 2021

@author: EARTH
"""


import pandas as pd
import numpy as np
data = pd.read_csv('data.csv')
data.head()
data.describe()
data.info()
data.isnull().values.any()
data['y'].value_counts()
data = data.iloc[:,1:]
data['y'] = data['y'].replace([5,4,3,2],0)
#data.iloc[:,:-1]
#data.iloc[:,-1]
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
X = data.iloc[:,:-1]
y = data.iloc[:,-1]

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=0)
# normalizing:
standard_scaler = preprocessing.StandardScaler()
X_train = pd.DataFrame(standard_scaler.fit_transform(X_train))
X_test = pd.DataFrame(standard_scaler.fit_transform(X_test))
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,roc_auc_score,confusion_matrix,average_precision_score
clf = DecisionTreeClassifier()
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test,y_pred)
accuracy
roc_auc = roc_auc_score(y_test,y_pred)
roc_auc
avg_precision = average_precision_score(y_test,y_pred)
avg_precision
cm = confusion_matrix(y_test,y_pred)
cm
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

predictions = [round(value) for value in y_pred]

# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
roc_auc = roc_auc_score(y_test,y_pred)
print("ROC-AUC: %.2f%%" % (accuracy * 100.0))
avg_precision = average_precision_score(y_test,y_pred)
print("Average Precision : %.2f%%" % (avg_precision*100.0))
cm = confusion_matrix(y_test,predictions)
print("Confusion Matrix : ")
print(cm)
from xgboost import XGBClassifier
model = XGBClassifier()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]

# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
roc_auc = roc_auc_score(y_test,predictions)
print("ROC-AUC: %.2f%%" % (roc_auc * 100.0))
avg_precision = average_precision_score(y_test,predictions)
print("Average Precision : %.2f%%" % (avg_precision*100.0))
cm = confusion_matrix(y_test,predictions)
print("Confusion Matrix : ")
print(cm)
len(X_train.columns)
import tensorflow
from keras.models import Sequential
from keras.layers import Dense
import numpy
model = Sequential()
model.add(Dense(200,input_dim=178,activation='relu'))
model.add(Dense(180,activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=10)
scores = model.evaluate(X_test, y_test)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
pred = model.predict(X_test)
pred = [round(x[0]) for x in pred]
roc_auc = roc_auc_score(y_test,pred)
print("ROC-AUC: %.2f%%" % (roc_auc * 100.0))
avg_precision = average_precision_score(y_test,pred)
print("Average Precision : %.2f%%" % (avg_precision*100.0))
cm = confusion_matrix(y_test,pred)
print("Confusion Matrix : ")
print(cm)
X_train.shape
x_train = X_train.values.reshape(X_train.shape[0],X_train.shape[1],1)
from keras.layers import Conv1D,LSTM,MaxPool1D,Dropout
CNN1D = Sequential()
CNN1D.add(Conv1D(10,10,activation='elu',input_shape=(178,1)))
CNN1D.add(MaxPool1D(4,1))
CNN1D.add(Dropout(0.2))
CNN1D.add(LSTM(128, return_sequences=True))
CNN1D.add(LSTM(64)) 
CNN1D.add(Dense(1,activation='sigmoid'))
CNN1D.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
CNN1D.fit(x_train, y_train, epochs=5, batch_size=100)

x_test = X_test.values.reshape(X_test.shape[0],X_test.shape[1],1)
scores = CNN1D.evaluate(x_test, y_test)
print("\n%s: %.2f%%" % (CNN1D.metrics_names[1], scores[1]*100))
pred = CNN1D.predict(x_test)
pred = [round(x[0]) for x in pred]
roc_auc = roc_auc_score(y_test,pred)
print("ROC-AUC: %.2f%%" % (roc_auc * 100.0))
avg_precision = average_precision_score(y_test,pred)
print("Average Precision : %.2f%%" % (avg_precision*100.0))
cm = confusion_matrix(y_test,pred)
print("Confusion Matrix : ")
print(cm)
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
#SVM WITH RBF KERNEL
svm_clf = SVC(C=4,gamma=1.3)
svm_clf = svm_clf.fit(X_train, y_train)
predict = svm_clf.predict(X_test)
#pred = [round(x[0]) for x in predict]
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
roc_auc = roc_auc_score(y_test,predict)
print("ROC-AUC: %.2f%%" % (roc_auc * 100.0))
avg_precision = average_precision_score(y_test,predict)
print("Average Precision : %.2f%%" % (avg_precision*100.0))
#print(accuracy_score(y_test, predict))
#print(classification_report(y_test, predict))
#accuracies.append(accuracy_score(y_test, predict))
#f_scores.append(f1_score(y_test, predict,average = "macro"))
cm = confusion_matrix(y_test,predict)
print("Confusion Matrix : ")
print(cm)
linear_svm_clf = LinearSVC(C=0.1)
linear_svm_clf = linear_svm_clf.fit(X_train, y_train)
predict = linear_svm_clf.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
roc_auc = roc_auc_score(y_test,predict)
print("ROC-AUC: %.2f%%" % (roc_auc * 100.0))
avg_precision = average_precision_score(y_test,predict)
print("Average Precision : %.2f%%" % (avg_precision*100.0))
cm = confusion_matrix(y_test,predict)
print("Confusion Matrix : ")
print(cm)
print()
print()
print(classification_report(y_test, predict))