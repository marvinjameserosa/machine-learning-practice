import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing, model_selection

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
cls = labeler('class')

# Storing them in a class

predict = 'class'

x = list(zip(buying, maint, doors, persons, lug_boot, safety))
y = list(cls)

# Splitting attributes for Testing

xTrain, xTest, yTrain, yTest = model_selection.train_test_split(x, y, test_size=0.1)

numOfNeighbors = 10

# Value for determining how many neigbors

knn = KNeighborsClassifier(n_neighbors = numOfNeighbors)

# Function for Testing Data

knn.fit(xTrain, yTrain)
knn.fit(xTrain, yTrain)
accuracy = knn.score(xTest, yTest)
print("curent accuracy: " + str(accuracy))

# Presentation of Results

result = knn.predict(xTest)

classLabel = ['unacc', 'acc', 'good', 'vgood']

for x in range(len(result)):
   # For additional information about the neighbors 

   neighborsInfo = knn.kneighbors([xTest[x]], numOfNeighbors, True)
   print('Predicted: ', classLabel[result[x]], 'Data: ', xTest[x], 'True Value: ', classLabel[yTest[x]], 'Additional Information: ', neighborsInfo)
