import numpy as np #contains math tools that is used to contain any math models in the code
import matplotlib.pyplot as plt #used for pretty and trendy graphs
import pandas as pd #the OG LIBRARY used to import data sets

#adjust all according to dataset
#importing the dataset
data = pd.read_csv('Data.csv')

#distinguish the matrix of features which are independant vs the dependant
#adjust according to dataset
X = data.iloc[:, :-1].values #all the lines, all the columns except the last one
y = data.iloc[:,-1].values #dependent variables

#splitting training and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#preprocessing complete!! now you dont have to do it from scratch ayra yay!!!
