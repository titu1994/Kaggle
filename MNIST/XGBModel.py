import xgboost as xgb
import MNIST.DataClean as dc
import numpy as np

trainFrame = dc.loadTrainData(describe=False)
trainData = dc.convertPandasDataFrameToNumpyArray(trainFrame)

rf = xgb.XGBClassifier(n_estimators=100, seed=0, max_depth=8,)
evalSet = [(trainData[:2000, 1:], trainData[:2000, 0])]
rf.fit(trainData[:, 1:], trainData[:, 0], eval_set=evalSet, verbose=True)


testFrame = dc.loadTestData()
testData = dc.convertPandasDataFrameToNumpyArray(testFrame)

testX = testData[:, 0:]
#print("Random Forest Accuracy : ", rf.score(trainX, trainY))
print("Beginning prediction now")
yPred = rf.predict(testX,)
print("Prediction complete")

np.savetxt('mnist-xgb.csv', np.c_[range(1,len(yPred)+1),yPred], delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')
print("Save predictions to file complete")
