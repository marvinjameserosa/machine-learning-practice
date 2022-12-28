# Packages Used 
import numpy as np
import pandas as pd
import pickle as pk
from sklearn import model_selection
from sklearn import linear_model
from matplotlib import style
from matplotlib import pyplot

# Importing data to file
data = pd.read_csv('student-mat.csv', sep=";")

# Getting attribute for training
data = data[['G1', 'G2', 'G3', 'age', 'studytime','absences']]

# Data Point we want to predict
predict = 'G3'

# Placing attributes into an array
X = np.array(data.drop([predict], 1))
Y = np.array(data[predict])

iteration = int(input("Enter how may times you want to test the data:"))
currentScore = 0

# Splitting attributes for Testing

xTrain, xTest, yTrain, yTest = model_selection.train_test_split(X, Y, test_size=0.1)

for test in range(iteration):

    xTrain, xTest, yTrain, yTest = model_selection.train_test_split(X, Y, test_size=0.1)
    30
    # Getting the accuracy of the prediction
    ## Testing Data
    linear = linear_model.LinearRegression()
    linear.fit(xTrain, yTrain)
    accuracy = linear.score(xTest, yTest)
    print("curent accuracy: " + str(accuracy))

# Results File name
resultFileName = "resultLinearRegression.pickle"

if accuracy > currentScore:       
    currentScore = accuracy
    # Save Model
    with open(resultFileName, 'wb') as file:
        pk.dump(linear, file)

# Read Model
openFile = open(resultFileName, 'rb')
linear = pk.load(openFile)

# Predicting Untested Data
prediction = linear.predict(xTest)

for grades in range(len(prediction)):
    print(prediction[grades], xTest[grades], yTest[grades])

# Select data point you want to check
dataPoint = 'age'
# Graphing Function
style.use('ggplot')
pyplot.scatter(data['G3'], data[dataPoint])
pyplot.xlabel('final grade'.upper())
pyplot.ylabel(dataPoint.upper())
pyplot.show()