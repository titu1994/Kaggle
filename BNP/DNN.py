import BNP.DataClean as dc
import keras.layers.core as core
import keras.models as models
import keras.utils.np_utils as kutils
import numpy as np
import csv

traindf, testdf = dc.loadTrain(), dc.loadTest()
traindf, testdf = dc.cleanData(traindf, testdf, describe=False)

trainData, testData = dc.convertPandasDataFrameToNumpyArray(traindf), dc.convertPandasDataFrameToNumpyArray(testdf)

trainX = trainData[:, 2:]
trainX -= np.mean(trainX)
trainX /= np.std(trainX)

trainY = trainData[:, 1]

testX = testData[:, 1:]
testX -= np.mean(testX)
testX /= np.std(testX)

model = models.Sequential()
model.add(core.Dense(200, input_shape=()))


model.summary()

"""
yPred = model.predict(testX)

min_y_pred = min(yPred)
max_y_pred = max(yPred)
min_y_train = min(trainY)
max_y_train = max(trainY)

for i in range(len(yPred)):
    yPred[i] = min_y_train + (((yPred[i] - min_y_pred)/(max_y_pred - min_y_pred))*(max_y_train - min_y_train))

predictions_file = open("xgboost_result.csv", "w", newline="")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["ID", "PredictedProb"])
open_file_object.writerows(zip(testData[:, 0].astype(int), yPred))
predictions_file.close()
"""