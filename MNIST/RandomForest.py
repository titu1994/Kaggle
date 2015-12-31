import MNIST.DataClean as dc
import numpy as np
import sklearn.ensemble as ensemble

trainFrame = dc.loadTrainData(describe=False)
trainData = dc.convertPandasDataFrameToNumpyArray(trainFrame)
rf = ensemble.RandomForestClassifier(n_estimators=1000, n_jobs=4, verbose=True)
rf.fit(trainData[:, 1:], trainData[:, 0])


testFrame = dc.loadTestData()
testData = dc.convertPandasDataFrameToNumpyArray(testFrame)

testX = testData[:, 0:]
#print("Random Forest Accuracy : ", rf.score(trainX, trainY))
print("Beginning prediction now")
yPred = rf.predict(testX)
print("Prediction complete")

np.savetxt('mnist-rf.csv', np.c_[range(1,len(yPred)+1),yPred], delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')
print("Save predictions to file complete")




