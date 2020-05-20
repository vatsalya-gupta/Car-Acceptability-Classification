# encoding: utf-8
# author: vatsalya-gupta

'''
Necessary import statements
'''
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

'''
Reading in the data using Pandas and output the first 5 rows to analyse it
'''
df = pd.read_csv("../data/car_cleaned.csv")
print("Dataset:\n", df.head())

'''
fit_transform method from LabelEncoder is used to fit and return encoded labels as KNN requires the
labels and features to be either of type int or float
'''
le = LabelEncoder()

buying = le.fit_transform(list(df["buying"]))
maint = le.fit_transform(list(df["maint"]))
doors = le.fit_transform(list(df["doors"]))
persons = le.fit_transform(list(df["persons"]))
lug_boot = le.fit_transform(list(df["lug_boot"]))
safety = le.fit_transform(list(df["safety"]))
decision = le.fit_transform(list(df["decision"]))

predict = "decision"
X = list(zip(buying, maint, doors, persons, lug_boot, safety))
y = list(decision)

'''
The data is split into training and testing samples
'''
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

'''
K-Nearest Neighbors Classifier (with K as 7) is implemented and training data is fit to it
'''
model = KNeighborsClassifier(n_neighbors = 7)
model.fit(X_train, y_train)

'''
Making the predictions using the default names for readability
'''
y_pred = model.predict(X_test)
names = ["unacc", "acc", "good", "vgood"]
print("\nPredictions:")

for x in range(len(X_test)):
    print("Predicted: ", names[y_pred[x]], ", Data: ", X_test[x], ", Actual: ", names[y_test[x]])

    '''
    Uncomment the following two lines to see the 7 nearest neighbors for each of the testing samples
    '''
    # n = model.kneighbors([X_test[x]], n_neighbors = 7, return_distance = True)
    # print("N: ", n)

'''
The accuracy is calculated using the accuracy_score metric from sklearn
'''
acc = accuracy_score(y_pred, y_test)
print("\nAccuracy: ", acc)
