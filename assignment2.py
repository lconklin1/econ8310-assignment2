from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import BaggingClassifier
import pandas as pd

data = pd.read_csv("https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3.csv")
data.drop(columns=['id'], inplace=True)
data["DateTime"] = pd.to_datetime(data["DateTime"])
data["month"] = data["DateTime"].dt.month
data["hour"] = data["DateTime"].dt.hour
data["dayofweek"] = data["DateTime"].dt.dayofweek
data = data.drop(columns=["DateTime"])

# Upper case before split, lower case after
Y = data['meal']
# make sure you drop a column with the axis=1 argument
X = data.drop('meal', axis=1) 

x, xt, y, yt = train_test_split(X, Y, test_size=0.1,
     random_state=42)

# Generate the bagging model
bag = BaggingClassifier(n_estimators=100, n_jobs = -1, 
	random_state=42)
model = bag
# Fit the model to the training data
baclf = bag.fit(x, y)
modelFit = baclf
