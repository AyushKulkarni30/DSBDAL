import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("Social_Network_Ads.csv")
print(dataset.head(5))
print(dataset.shape)

x = dataset.iloc[:, [2,3]].values
y = dataset.iloc[:, 4].values
print(x)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.30, random_state=4)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)
print(y_pred)
print(y_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

Accuracy = (74 + 31) / 120
print(Accuracy)

Error_rate = (5+10)/120
print(Error_rate)

from sklearn.metrics import precision_score, recall_score
print(precision_score(y_test, y_pred))
print(recall_score(y_test, y_pred))

from sklearn.metrics import f1_score
print(f1_score(y_test, y_pred))