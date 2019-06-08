import sklearn as sk
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load cancer data from dataset.
cancer = load_breast_cancer()

# Define which data will be on the x and y axis.
x = cancer.data
y = cancer.target

# Use 80% of the data to train with and then test on the remaining 20%.
x_train, x_test, y_train, y_test = sk.model_selection.train_test_split(
    cancer.data, cancer.target, test_size=0.2)

# Fit the data using a linear kernal to the classifier.
clf = SVC(kernel="linear", C=2, gamma="auto")
clf.fit(x_train, y_train)

# Predict the y test values using our x test values.
y_prediction = clf.predict(x_test)

# Store and display the accuracy of this model.
accuracy = accuracy_score(y_test, y_prediction)
print(accuracy)
