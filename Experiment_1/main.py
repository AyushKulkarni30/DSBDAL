import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("StudentsPerformance.csv")
df.head(15)
df.isnull().sum()
df.describe()
# df.dtypes
df.dropna(axis=1)

y = df.iloc[:, 0:1]
# print(y)

le = LabelEncoder()
y = le.fit_transform(y)
#

# print(df['race/ethnicity'].value_counts())

df_Lunch = pd.get_dummies(df['lunch'])
df_new = pd.concat([df, df_Lunch], axis=1)
print(df_new)



