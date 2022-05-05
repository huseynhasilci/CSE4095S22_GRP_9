import json
import os
import csv
import string

import xlsxwriter
import numpy as np
import pandas as pd
import nltk
import math
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from zemberek import TurkishMorphology

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, precision_score

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
df = pd.read_csv(r'../../latest_hukums_with_classes_csv_file1.csv')

#happy_lines = []
#lines = df['ictihat'].values.tolist()
#for line in lines:
#    tokens = word_tokenize(line)
#    tokens = [w.lower() for w in tokens]

#    table = str.maketrans('', '', string.punctuation)
#    stripped = [w.translate(table) for w in tokens]

#    words = [word for word in stripped if word.isalpha()]
#    happy_lines.append(words)
#happy_lines = np.asarray(happy_lines, dtype=object)
#happy_lines.reshape((19280, 1))
# print(df)
le = LabelEncoder()
for column in df.columns:
    temp_new = le.fit_transform(df[column].astype('category'))
    df.drop(labels=[column], axis="columns", inplace=True)
    df[column] = temp_new
#print(happy_lines[0][0])
X = df.iloc[:, 1:2]
y = df.iloc[:, 2:]
print(X)
print(y)
# X = X.drop("Unnamed: 0", axis=1, inplace=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

classifiers = [
    # KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
    LogisticRegression(solver='lbfgs', max_iter=3000),
    MultinomialNB()
]
for i in classifiers:
    classifier = i

    classifier.fit(X_train, np.ravel(y_train))

    y_pred = classifier.predict(X_test)
    print(classifier.score(X_test, y_test))
    print(confusion_matrix(y_test, y_pred))
    #print(precision_score(y_test, y_pred, average='macro', zero_division=1))
    #print(precision_score(y_test, y_pred, average='micro', zero_division=1))
    #print(precision_score(y_test, y_pred, average='weighted', zero_division=1))
    #print(precision_score(y_test, y_pred, average=None, zero_division=1))
    print(classification_report(y_test, y_pred, zero_division=1))