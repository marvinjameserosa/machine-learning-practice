from sklearn import datasets, svm

cancerData = datasets.load_breast_cancer()

print(cancerData.feature_names)
