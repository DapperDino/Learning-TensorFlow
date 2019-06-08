import pickle
import pandas as pd
import numpy as np
import sklearn as sk
from sklearn import linear_model
import matplotlib.pyplot as pyplot
from matplotlib import style

#Load student data from csv file.
data = pd.read_csv("StudentData.csv", sep=";")

#Filter only the data we wish to model.
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

#Define the label we wish to predict.
predict = "G3"

#Drop the prediction from the input data since we won't be using it as input data.
x = np.array(data.drop([predict], 1))
y = np.array(data[predict])

#Use 90% of the data to train with and then test on the remaining 10%.
x_train, x_test, y_train, y_test = sk.model_selection.train_test_split(x, y, test_size = 0.1)

"""
best_accuracy = 0
while(True):

    #Use 90% of the data to train with and then test on the remaining 10%.
    x_train, x_test, y_train, y_test = sk.model_selection.train_test_split(x, y, test_size = 0.1)

    #Create an instance of the Linear Regression class.
    linear = linear_model.LinearRegression()

    #Fit the training data to the model.
    linear.fit(x_train, y_train)
    accuracy = linear.score(x_test, y_test)

    #Check to see if this model is more accurate than our previous best.
    if accuracy > best_accuracy:

        #Store the new best accuracy to compare future models to.
        best_accuracy = accuracy
        print(best_accuracy)

        #Save the model to a file.
        with open("student_model.pickle", "wb") as f:
            pickle.dump(linear, f)
"""

#Load model from file.
pickle_in = open("student_model.pickle", "rb")
linear = pickle.load(pickle_in)

#Output statistics from the testing.
print("Co: {}".format(linear.coef_))
print("Intercept: {}".format(linear.intercept_))

#Output data from the testing.
predictions = linear.predict(x_test)
for i in range(len(predictions)):
    print(predictions[i], x_test[i], y_test[i])

#Display data as scatter graph.
p = "studytime"
style.use("ggplot")
pyplot.scatter(data[p], data[predict])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()