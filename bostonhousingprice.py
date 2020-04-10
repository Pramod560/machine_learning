# import required modules
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

#import boston housing data from sklearn.datasets
from sklearn.datasets import load_boston
boston = load_boston()

# transform data into dataframe using pandas DataFarame method 
bh_x = pd.DataFrame(boston.data, columns=boston.feature_names) #features
bh_y = pd.DataFrame(boston.target) #target value

#intilize classifier
clf = LinearRegression()
clf1 = DecisionTreeRegressor()

#get some statistics from the data
bh_x.describe()

#split data into training data and testing data
train_x, test_x, train_y, test_y = train_test_split(bh_x,bh_y, test_size= 0.25, random_state= 40)

#train our linear model
clf.fit(train_x,train_y)
clf1.fit(train_x,train_y)

#predict price of our testing data
y_pred = clf.predict(test_x)
y1_pred = clf1.predict(test_x)

#check accuracy of our model using r2_score
from sklearn.metrics import r2_score
print(r2_score(test_y,y_pred))
print(r2_score(test_y,y1_pred))

#check accuracy of our model using mean_squared_error 
from sklearn.metrics import mean_squared_error
print(mean_squared_error(test_y, y_pred))
print(mean_squared_error(test_y,y1_pred))
