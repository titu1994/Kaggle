import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import gc
import MNIST.DataClean as dc

trainData = dc.convertPandasDataFrameToNumpyArray(dc.loadTrainData(describe=False))

trainX = trainData[:, 1:]
trainY = trainData[:, 0]

pca = PCA(n_components=35, whiten=True)
svm = SVC(random_state=0,)
print("Loaded traindata")

pca.fit(trainX)
print("PCA : Finished fitting")
trainX = pca.transform(trainX)
print("PCA : Finished Transform of trainX")

svm.fit(trainX, trainY)
print("SVM : Finished fitting")

trainX = trainY = trainData = None
gc.collect()

testData = dc.convertPandasDataFrameToNumpyArray(dc.loadTestData())
testData = pca.transform(testData)
print("Finished transforming testX")

yPred = svm.predict(testData)

np.savetxt('mnist-svm.csv', np.c_[range(1,len(yPred)+1),yPred], delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')
print("Save predictions to file complete")


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

