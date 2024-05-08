import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("AcademicPerformance.csv")
print(df)
print(df['math score'])
print(df['math score'].isnull())
print(df['reading score'])
print(df['reading score'].isnull())

missing_values = ["n/a", "na", "--"]
df = pd.read_csv(r"AcademicPerformance.csv", na_values=missing_values)
print(df['reading score'])
print(df['reading score'].isnull())


dataset = [11, 41, 20, 3, 101, 55, 68, 97, 99, 6]
print(sorted(dataset))
quantile1, quantile3 = np.percentile(dataset, [25, 75])
print(quantile1, quantile3)

iqr_value = (quantile3 - quantile1)
print(iqr_value)

lower_bound_value = quantile1 - (1.5*iqr_value)
upper_bound_value = quantile3 + (1.5*iqr_value)

print(lower_bound_value, upper_bound_value)

from datetime import date
df['age'] = date.today().year - df['Year_Birth']
df['Year'] = pd.DatetimeIndex(df['Dt_Admission']).year
df['E_L'] = date.today().year - df['Year']
print(df.head(5))

df['Fees$'] = df['College_Fees'].str.replace(',', '').str.replace('$', '').str.replace(')', '').str.replace('(', '-')
df['Fees_M$'] = df['Fees$'].apply(lambda x: round(float(x)/1000000))
print(df.head(5))


