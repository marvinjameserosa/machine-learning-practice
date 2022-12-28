import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model, preprocessing

# Reading from file
data = pd.read_csv('car.data')

# Preprocessing data [transforming str to int]

label = preprocessing.LabelEncoder()

def labeler(head):
   return label.fit_transform(list(data[head]))

buying = labeler('buying')
maint = labeler('maint')
doors = labeler('doors')
persons = labeler('persons')
lug_boot = labeler('lug_boot')
safety = labeler('safety')

print(safety)