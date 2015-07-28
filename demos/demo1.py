import sklearn.linear_model
import numpy as np
import util

headers, data = util.read_csv("mpg_data.csv")

# the first column of the data is the MPG, 
# which we want to use as our label
data = np.asarray(data).astype(float)

labels = data[:,0]
features = data[:,1:]


linear_reg = sklearn.linear_model.LinearRegression()
linear_reg.fit(features, labels)


coeffs = zip(headers[1:], linear_reg.coef_)
for c in sorted(coeffs, key=lambda x:x[1]**2, reverse=True):
    print(c[0], c[1])
    
