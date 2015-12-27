from sklearn import datasets
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import sklearn.ensemble as ensemble
import numpy as np

np.random.seed(123)

iris = datasets.load_iris()
X, Y = iris.data[:, 1:3], iris.target

lb = LogisticRegression(random_state=1)
gnb = GaussianNB()
rf = RandomForestClassifier(random_state=1)

votingClassifier = ensemble.VotingClassifier(estimators=[("lb", lb), ("rf", rf), ("gnb", gnb)], voting="hard")

print('5-fold cross validation:\n')

for clf, label in zip([lb, rf, gnb, votingClassifier], ['Logistic Regression', 'Random Forest', 'naive Bayes', 'Voting Classifier']):
    scores = cross_validation.cross_val_score(clf, X, Y, cv=5, scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))



