from sklearn.cross_validation import cross_val_score
import sklearn.linear_model
import sklearn.tree
import sklearn.ensemble
import numpy as np
from timeit import default_timer as timer
import util

headers, data = util.read_csv("hw_data.csv")

# the last column of the data is the 
# failure status, which we want to use as our label
data = np.asarray(data)

labels = data[:,-1]
features = data[:,:-1].astype(float)


# basic learners
clf1 = sklearn.linear_model.LogisticRegression() 
# even though it says regression, 
# this is for classification!

clf2 = sklearn.tree.DecisionTreeClassifier()

# bagging ensemble methods
clf3 = sklearn.ensemble.BaggingClassifier(
    sklearn.linear_model.LogisticRegression()
)

clf4 = sklearn.ensemble.BaggingClassifier(
    sklearn.tree.DecisionTreeClassifier()
)

# other ensemble methods
clf5 = sklearn.ensemble.RandomForestClassifier(
    n_estimators=50
)

clf6 = sklearn.ensemble.GradientBoostingClassifier()
clf7 = sklearn.ensemble.GradientBoostingClassifier(
    n_estimators=500
)



learners = [ { "name": "logistic", "clf": clf1 },
             { "name": "dec tree", "clf": clf2 },
             { "name": "bag(log)", "clf": clf3 },
             { "name": "bag(tree)", "clf": clf4 },
             { "name": "random forest", "clf": clf5 },
             { "name": "gradient", "clf": clf6 },
             { "name": "gradientx5", "clf": clf7 } ]

print("algorithm", "acc", "time (s)", sep="\t")
for learner in learners:

    start_time = timer()
    # cv=3 for three fold cross validation
    scores = cross_val_score(learner["clf"], 
                             features,
                             labels, cv=3)

    elapsed_time = timer() - start_time
    print(learner["name"], 
          round(scores.mean()*100, 2), 
          round(elapsed_time, 2), 
          sep="\t")


