import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt

#read data
dataframe = pd.read_csv('challenge_dataset.txt')
x_values = dataframe[[0]]
y_values = dataframe[[1]]

#train model on data
model = linear_model.LinearRegression()
model.fit(x_values, y_values)

print(model.predict([[127], [248]]))

#visualize results
plt.scatter(x_values, y_values)
plt.plot(x_values, model.predict(x_values))
plt.show()
