import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC

from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
import MNIST.DataClean as dc

trainFrame = dc.loadTrainData(describe=False)
trainData = dc.convertPandasDataFrameToNumpyArray(trainFrame)

pca = PCA(whiten=True)
svm = SVC(random_state=0)

estimators = [("pca", pca), ("svm", svm)]
pipe = Pipeline(estimators)


"""
n_components = [20, 35, 50, 100,]
Cs = [1, 10, 100, 1000]

tuningParams = [{"pca__n_components" : n_components, 'svm__kernel': ['rbf'], 'svm__gamma': [1e-3, 1e-4], 'svm__C': Cs},
                {"pca__n_components" : n_components, 'svm__kernel': ['linear'], 'svm__C': Cs}]

# Results :
if __name__ == "__main__":
    print("Starting GridSearch")
    clf = GridSearchCV(pipe, tuningParams, scoring="accuracy", n_jobs=4, cv=5, verbose=True)
    clf.fit(trainData[:, 1:], trainData[:, 0])

    print("GridSearch : \n", "Best Estimator : ", clf.best_estimator_,
            "\nBest Params : ", clf.best_params_, "\nBest Score", clf.best_score_)
"""

