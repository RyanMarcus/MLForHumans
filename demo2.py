import sklearn.cross_validation
import sklearn.linear_model
import sklearn.ensemble
import numpy as np
import util

headers, data = util.read_csv("mpg_data.csv")

# the first column of the data is the MPG, 
# which we want to use as our label
data = np.asarray(data).astype(float)

labels = data[:,0]
features = data[:,1:]


linear_reg = sklearn.linear_model.LinearRegression()

# cv=3 for three fold cross validation
scores = sklearn.cross_validation.cross_val_score(
    linear_reg, features, labels,
    cv=3, scoring="mean_absolute_error")

print(scores.mean()) 
# sklearn switches the sign on ASE so that 
# larger numbers are better




