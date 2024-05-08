import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split

from sklearn.datasets import fetch_california_housing
california_housing = fetch_california_housing(as_frame=True)
print(california_housing)

# x = pd.DataFrame(california_housing.data, columns=california_housing.feature_names)
# y = pd.DataFrame(california_housing.target)
#
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state = 4)
#
# reg = linear_model.LinearRegression()
# reg.fit(x_train, y_train)
#
# print(reg.coef_)
#
# y_pred = reg.predict(x_test)
# print(y_pred)