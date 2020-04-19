import numpy as np #contains math tools that is used to contain any math models in the code
import matplotlib.pyplot as plt #used for pretty and trendy graphs
import pandas as pd #the OG LIBRARY used to import data sets

#adjust all according to dataset
#importing the dataset
data = pd.read_csv('Data.csv')

#distinguish the matrix of features which are independant vs the dependant
#adjust according to dataset
X = data.iloc[:, :-1].values #all the lines, all the columns except the last one
y = data.iloc[:,3].values #dependent variables

#Missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3 ])
#print(X)

#encoding categorical data ~ one hot encoding
#independant variables
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
#column transformer = transformer[( , , index you want encoded)], remainder always equals passthrough
ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [0])], remainder = 'passthrough')
X = np.array(ct.fit_transform(X))
#print(x)

#dependant variable using label encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y) #convert yes and no to numerical values
#print(y)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

#splitting training and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#preprocessing complete!! now you dont have to do it from scratch ayra yay!!!
