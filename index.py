import numpy as np
import pandas as pd
import sklearn
from sklearn import model_selection
from sklearn import linear_model
from sklearn.utils import shuffle

# Importing data to file
data = pd.read_csv('student-mat.csv', sep=";")

# Getting attribute for training
data = data[['G1', 'G2', 'G3', 'age', 'studytime','absences']]

predict = 'G3'

# Placing attributes into an array
X = np.array(data.drop([predict], 1))
Y = np.array(data[predict])

# Splitting attributes for Testing
xTrain, xTest, yTrain, yTest = model_selection.train_test_split(X, Y, test_size=0.1)

# Getting the accuracy of the prediction
linear = linear_model.LinearRegression()
linear.fit(xTrain, yTrain)
accuracy = linear.score(xTest, yTest)
print(accuracy)