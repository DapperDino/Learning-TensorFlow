import os
import pandas as pd
import numpy as np
import sklearn as sk
from sklearn import linear_model, preprocessing
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier

#Load car data from file.
data = pd.read_csv("CarDataKNN/Car.data")

#Convert attribute names to int.
le = preprocessing.LabelEncoder()
buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
cls = le.fit_transform(list(data["class"]))

#Define the label we wish to predict.
preict = "class"

#Define which data will be on the x and y axis.
x = list(zip(buying, maint, door, persons, lug_boot, safety))
y = list(cls)

#Use 90% of the data to train with and then test on the remaining 10%.
x_train, x_test, y_train, y_test = sk.model_selection.train_test_split(
    x, y, test_size=0.1)

#Create and fit our model.
model = KNeighborsClassifier(n_neighbors=9)
model.fit(x_train, y_train)

#Store and display the accuracy of this model.
accuracy = model.score(x_test, y_test)
print(accuracy)

#Preict the y test values using our x test values.
predicted = model.predict(x_test)
names = ["unacc", "acc", "good", "vgood"]

#Display how accurate each prediction was.
for i in range(len(predicted)):
    print("Predicted: {}\nData: {}\nActual: {}".format(names[predicted[i]], x_test[i], names[y_test[i]]))
    n = model.kneighbors([x_test[i]], 9)
    print("N: {}".format(n))
