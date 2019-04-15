import pandas as pd
import numpy as np
import csv
import matplotlib as mpl
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn import tree



df = pd.read_csv("data.csv", header=None)
# You might not need this next line if you do not care about losing information
# about flow_id etc. All you actually need to feed your machine learning model
# are features and output label.
columns_list = ['srceIP',
                'srcePort',
                'destIP',
                'destPort',
                'protocol',
                'sentPkts',
                'sentBytes',
                'recvPkts',
                'recvBytes',
                'totalPkts',
                'totalBytes',
                'timeStart',
                'duration',
                'label']
df.columns = columns_list
features = ['protocol', 'sentPkts', 'sentBytes', 'recvPkts', 'recvBytes', 'totalPkts', 'totalBytes', 'duration']

print(df)

X = df[features]
y = df['label']

acc_scores = 0
for i in range(0, 10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

    # Decision Trees algorithm
    clf = tree.DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    f1Score = f1_score(y_pred, y_test, average=None)
    print("Decision Tree F1 score: " + str(f1Score))

    # Neural network (MultiPerceptron Classifier)
    # clf = MLPClassifier()
    # clf.fit(X_train, y_train)

    #SVM's
    #clf = SVC(gamma='auto')     #SVC USE THIS
    #clf = LinearSVC()  #Linear SVC
    #clf.fit(X_train, y_train)


    #here you are supposed to calculate the evaluation measures indicated in the project proposal (accuracy, F-score etc)
    result = clf.score(X_test, y_test)  #accuracy score
    print("Accuracy score for : " + str(result))