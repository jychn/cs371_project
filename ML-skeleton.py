import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
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
import scikitplot as skplt

def average(numList):
    return sum(numList) / len(numList)

# Read in CSV file into pandas Dataframe data structure
df = pd.read_csv("data.csv", header=None)

# Label columns in dataframe
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

# Create list of features to test in model
features = [
            'protocol',
            'sentPkts',
            'sentBytes',
            'recvPkts',
            'recvBytes',
            'totalPkts',
            'totalBytes',
            'duration'
            ]

# ******************************************************************************
# //////////
# EVALUATION PART 1 - Evaluating ML algortihms by feature
# //////////
# Bar graph with average accuracy for each feature
'''
x_axis = features
y_axis = []
for feature in features:
    X = df[features]
    y = df['label']
    accuracyScores = []
    for i in range(0, 10):
        # Split dataset into training set and testing set
        # train_test_split tutorial: https://www.youtube.com/watch?v=fwY9Qv96DJY
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

        # DECISION TREE CLASSIFIER MACHINE LEARNING MODEL
        dtc = tree.DecisionTreeClassifier()
        dtc.fit(X_train, y_train)
        y_pred = dtc.predict(X_test)
        accuracyScores.append(accuracy_score(y_test, y_pred))

    y_axis.append(average(accuracyScores))

y_pos = np.arange(len(x_axis))

plt.bar(y_pos, y_axis, width=0.5, align='center', alpha=0.3)
plt.xticks(y_pos, x_axis)
plt.ylabel('Accuracy')
plt.title('Decision Tree Classifier: Accuracy by Feature')

plt.show()
print("Graph for model accuracy given each feature."
'''
# Line chart with accuracy of each feature over 10 iterations
x_axis = list(range(1,11))
y_axis = []
for feature in features:
    X = df[features]
    y = df['label']
    accuracyScores = []
    for i in range(0, 10):
        # Split dataset into training set and testing set
        # train_test_split tutorial: https://www.youtube.com/watch?v=fwY9Qv96DJY
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

        # DECISION TREE CLASSIFIER MACHINE LEARNING MODEL
        dtc = tree.DecisionTreeClassifier()
        dtc.fit(X_train, y_train)
        y_pred = dtc.predict(X_test)
        accuracyScores.append(accuracy_score(y_test, y_pred))

    y_axis.append(accuracyScores)

colors = ["#72a871","#f49169","#a1572a","#a2f622","#f23931","#8172f6","#1ac1e4","#5d4856"]
i = 0
for y_set in y_axis:
    line_chart = plt.plot(x_axis, y_set, colors[i])
    i += 1
plt.title("Accuracy of Each Feature Over 10 Iterations")
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.legend(features, loc=4)
plt.show()

input()

# ******************************************************************************


# ******************************************************************************
# //////////
# EVALUATION PART 2 - Comparing the different ML algorithms
# //////////
print("Features used for training models: ")
for feature in features:
    print("\t" + feature)

# List of labels follow 1 = Web Browsing, 2 = Video Streaming,
# 3 = Video Conferencing, 4 = File Downloading
labels = ['1', '2', '3', '4']

# Create X and Y values from datafram
X = df[features]
y = df['label']
#print(X)
#print(y)

# Since machine learning model will be ran 10 times for cross validation, create
# lists to store results of each metric for each ML alogirthm.

# For DecisionTreeClassifier
dtcAccuracyScores = []
dtcPrecisionScores =[]
dtcRecallScores = []
dtcF1Scores = []

# For Neural Network (Multi-layer Perceptron Classifier)
mlpAccuracyScores = []
mlpPrecisionScores =[]
mlpRecallScores = []
mlpF1Scores = []

# For support vector machines
svcAccuracyScores = []
svcPrecisionScores =[]
svcRecallScores = []
svcF1Scores = []

for i in range(0, 10):
    # Split dataset into training set and testing set
    # train_test_split tutorial: https://www.youtube.com/watch?v=fwY9Qv96DJY
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

    # DECISION TREE CLASSIFIER MACHINE LEARNING MODEL
    dtc = tree.DecisionTreeClassifier()
    dtc.fit(X_train, y_train)
    y_pred = dtc.predict(X_test)
    dtcAccuracyScores.append(accuracy_score(y_test, y_pred))
    dtcPrecisionScores.append(precision_score(y_test, y_pred, average='weighted', labels=np.unique(y_pred)))
    dtcRecallScores.append(recall_score(y_test, y_pred, average='weighted', labels=np.unique(y_pred)))
    dtcF1Scores.append(f1_score(y_test, y_pred, average='weighted', labels=np.unique(y_pred)))
    skplt.metrics.plot_confusion_matrix(y_test, y_pred, normalize=True)
    # plt.show()
    # print(classification_report(y_test, y_pred, target_names=labels))

    # NEURAL NETWORK (MULTI-LAYER PERCEPTRON CLASSIFIER) MACHINE LEARNING MODEL
    mlp = MLPClassifier()
    mlp.fit(X_train, y_train)
    y_pred = mlp.predict(X_test)
    mlpAccuracyScores.append(accuracy_score(y_test, y_pred))
    mlpPrecisionScores.append(precision_score(y_test, y_pred, average='weighted', labels=np.unique(y_pred)))
    mlpRecallScores.append(recall_score(y_test, y_pred, average='weighted', labels=np.unique(y_pred)))
    mlpF1Scores.append(f1_score(y_test, y_pred, average='weighted', labels=np.unique(y_pred)))


    # SUPPORT VECTOR MACHINE MACHINE LEARNING MODEL
    svc = SVC(gamma='auto')     #SVC USE THIS
    # svc = LinearSVC()  #Linear SVC
    svc.fit(X_train, y_train)
    y_pred = svc.predict(X_test)
    svcAccuracyScores.append(accuracy_score(y_test, y_pred))
    svcPrecisionScores.append(precision_score(y_test, y_pred, average='weighted', labels=np.unique(y_pred)))
    svcRecallScores.append(recall_score(y_test, y_pred, average='weighted', labels=np.unique(y_pred)))
    svcF1Scores.append(f1_score(y_test, y_pred, average='weighted', labels=np.unique(y_pred)))


# EVALUATING METRICS INFORMATION
# Print out average metrics for each algorithm
# Average accuracy for each algorithm
print("Average accuracy of dtc: " + str(average(dtcAccuracyScores)))
print("Average accuracy of mlp: " + str(average(mlpAccuracyScores)))
print("Average accuracy of svc: " + str(average(svcAccuracyScores)))


# Average precision for each algorithm
print("Average precision of dtc: " + str(average(dtcPrecisionScores)))
print("Average precision of mlp: " + str(average(mlpPrecisionScores)))
print("Average precision of svc: " + str(average(svcPrecisionScores)))

# Average recall for each algorithm
print("Average recall of dtc: " + str(average(dtcRecallScores)))
print("Average recall of mlp: " + str(average(mlpRecallScores)))
print("Average recall of svc: " + str(average(svcRecallScores)))

# Average F1 scores for each algorithm
print("Average F1 score of dtc: " + str(average(dtcF1Scores)))
print("Average F1 score of mlp: " + str(average(mlpF1Scores)))
print("Average F1 score of svc: " + str(average(svcF1Scores)))


'''
# Print out arrays to see resulting metric values
print("Accuracy scores for dtc: " + str(dtcAccuracyScores))
print("Precision scores for dtc: " + str(dtcPrecisionScores))
print("Recall scores for dtc: " + str(dtcRecallScores))
print("F1 scores for dtc: " + str(dtcF1Scores))
print()
print("Accuracy scores for mlp: " + str(mlpAccuracyScores))
print("Precision scores for mlp: " + str(PrecisionScores))
print("Recall scores for mlp: " + str(mlpRecallScores))
print("F1 scores for mlp: " + str(mlpF1Scores))
print()
print("Accuracy scores for svc: " + str(svcAccuracyScores))
print("Precision scores for svc: " + str(svcPrecisionScores))
print("Recall scores for svc: " + str(svcRecallScores))
print("F1 scores for svc: " + str(svcF1Scores))

'''
