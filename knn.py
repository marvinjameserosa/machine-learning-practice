import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model, preprocessing, model_selection

# Reading from file
data = pd.read_csv('car.data')

# Preprocessing data [transforming str to int then placing them in a list]

label = preprocessing.LabelEncoder()

def labeler(head):
   return label.fit_transform(list(data[head]))

buying = labeler('buying')
maint = labeler('maint')
doors = labeler('doors')
persons = labeler('persons')
lug_boot = labeler('lug_boot')
safety = labeler('safety')

# Storing them in a class

predict = 'class'

x = list(zip(buying, maint, doors, persons, lug_boot, safety))
y = list(predict)

# Splitting attributes for Testing

xTrain, xTest, yTrain, yTest = model_selection.train_test_split(x, y, test_size=0.1)