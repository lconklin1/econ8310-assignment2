import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

data = pd.read_csv("https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3.csv")
data.drop(columns=['id'], inplace=True)
data["DateTime"] = pd.to_datetime(data["DateTime"])
data["hour_of_sale"] = data["DateTime"].dt.hour
data["dayofweek"] = data["DateTime"].dt.dayofweek
data = data.drop(columns=["DateTime"])

data_test = pd.read_csv('https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3test.csv')
data_test.drop(columns=['id'], inplace=True)
data_test["DateTime"] = pd.to_datetime(data_test["DateTime"])
data_test["hour_of_sale"] = data_test["DateTime"].dt.hour
data_test["dayofweek"] = data_test["DateTime"].dt.dayofweek
data_test = data_test.drop(columns=["DateTime"])
data_test['meal'] = 0


# Upper case before split, lower case after
Y = data['meal']
# make sure you drop a column with the axis=1 argument
X = data.drop('meal', axis=1) 

x, xt, y, yt = train_test_split(X, Y, test_size=0.1,random_state=42)

xgb = XGBClassifier(n_estimators=500, max_depth=7, learning_rate=0.3, objective='binary:logistic')
model = xgb
modelFit = xgb.fit(x, y)
prediction = xgb.predict(xt)

print("xgboost has an accuracy of: %s\n" 
	% str(accuracy_score(yt, prediction)*100))

X_test = data_test.drop('meal', axis=1) 
pred = modelFit.predict(X_test)

