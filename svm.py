from sklearn import datasets, svm, metrics, model_selection
from sklearn.neighbors import KNeighborsClassifier


# Loading in datasets
cancerData = datasets.load_breast_cancer()

# Setting data into variables 
x = cancerData.data
y = cancerData.target

# Splitting attributes for Testing [storing them in a tuple]
xTrain, xTest, yTrain, yTest = model_selection.train_test_split(x, y, test_size=0.1)

classes = ['maglignant', 'benign']

# Training using SVM
## Setting up SVC
clf = svm.SVC(kernel='linear')
clf.fit(xTrain, yTrain)

## Testing and Checking accuracy
result = clf.predict(xTest)
accuracy = metrics.accuracy_score(yTest, result)

print('Result: ' + str(accuracy))


