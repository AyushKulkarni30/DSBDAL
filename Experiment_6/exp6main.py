import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

iris = pd.read_csv("Iris.csv")
print(iris.head(5))

print(iris['Species'].unique())
print(iris.drop(columns="Id", inplace=True))

g = sns.relplot(x="SepalLengthCm", y='SepalWidthCm', data=iris, hue='Species', style='Species')
g.fig.set_size_inches(10, 5)
plt.show()

plt.figure(figsize=(15,10))
plt.subplot(2, 2, 1)
sns.violinplot(x='Species', y='PetalLengthCm', data=iris)
plt.subplot(2, 2, 2)
sns.violinplot(x='Species', y='PetalWidthCm', data=iris)
plt.subplot(2, 2, 3)
sns.violinplot(x='Species', y='SepalLengthCm', data=iris)
plt.subplot(2, 2, 4)
sns.violinplot(x='Species', y='SepalWidthCm', data=iris)
plt.show()

plt.subplots(figsize=(10, 7))
sns.violinplot(data=iris)
sns.swarmplot(data=iris)
plt.show()

print(iris.plot.area(y=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'], alpha=0.5))

print(iris.corr())

plt.subplots(figsize = (8,8))
sns.heatmap(iris.corr(), annot=True, fmt="f").set_title("Corelation of attributes (petal and sepal")
plt.show()
