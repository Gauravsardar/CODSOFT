import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

iris_data = pd.read_csv('iris.csv')

labelencoder = LabelEncoder()

iris_data['species'] = labelencoder.fit_transform(iris_data['species'])

X = iris_data.drop(columns='species')
Y = iris_data['species']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

def training_model():
    model = DecisionTreeClassifier()
    trained_model = model.fit(X_train, Y_train)
    return trained_model

