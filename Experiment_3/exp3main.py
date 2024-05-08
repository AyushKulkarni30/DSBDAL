import numpy
from scipy import stats
from collections import Counter

speed = [99, 86, 87,88, 111, 86, 103, 87, 94, 78, 77, 85, 86]

x = numpy.mean(speed)
print(x)

y = numpy.median(speed)
print(y)

z = stats.mode(speed)
print(z)

n_num = [1, 2, 3, 4, 5]
n = len(n_num)
get_sum = sum(n_num)
mean = get_sum / n
print(f"Mean / Average is : {mean}")


n_num.sort()

if n % 2 == 0:
    median1 = n_num[n//2]
    median2 = n_num[n//2 - 1]
    median = (median1 + median2) / 2
else:
    median = n_num[n//2]
print("Median is : " + str(median))

n_num = [1, 2, 3, 4, 5]
n = len(n_num)

data = Counter(n_num)
get_mode = dict(data)
mode = [k for k, v in get_mode.items() if v == max(list(data.values()))]

if len(mode) == n:
    get_mode = "No mode found"
else:
    get_mode = "Mode is/are : " + ','.join(map(str, mode))

print(get_mode)

import pandas as pd
df = pd.DataFrame({'A' : ['a', 'b', 'c', 'c', 'a', 'b'],
                   'B' : [0, 1, 1, 0, 1, 0]}, dtype = "category")
print(df.dtypes)


df = pd.DataFrame({'A' : ['a', 'b', 'c', 'c', 'a', 'b'],
                   'B' : [0, 1, 1, 0, 1, 0],
                   'C' : [7, 8, 9, 5, 3, 6]})

df['A'] = df['A'].astype('category')

print(df)
print(df.groupby(['A', 'B']).mean().reset_index())

import pandas as pd

data = pd.read_csv("Iris.csv")

print('Iris-setosa')
setosa = data['Species'] == 'Iris-setosa'
print(data[setosa].describe())

print("\nIris-versicolor")
versicolor = data['Species'] == 'Iris-versicolor'
print(data[versicolor].describe())

print("\nIris-virginica")
virginica = data['Species'] == 'Iris-virginica'
print(data[virginica].describe())