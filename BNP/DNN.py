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

nFeatures = trainX.shape[1]

trainY = trainData[:, 1]

testX = testData[:, 1:]
testX -= np.mean(testX)
testX /= np.std(testX)

epochs = 100

model = models.Sequential()
model.add(core.Dense(2000, init="uniform", input_shape=(nFeatures,), activation="relu"))
model.add(core.Dropout(0.2))
model.add(core.Dense(1000, activation="relu"))
model.add(core.Dropout(0.2))
model.add(core.Dense(1000, activation="relu"))
model.add(core.Dropout(0.2))
model.add(core.Dense(1000, activation="relu"))
model.add(core.Dense(1, activation="sigmoid"))

model.summary()

model.compile(optimizer="adamax", loss="binary_crossentropy", class_mode="binary")
model.fit(trainX, trainY, nb_epoch=epochs, validation_split=0.05, show_accuracy=True)

yPred = model.predict_proba(testX)[:,0]
#print(yPred)

min_y_pred = min(yPred)
max_y_pred = max(yPred)
min_y_train = min(trainY)
max_y_train = max(trainY)

for i in range(len(yPred)):
    yPred[i] = min_y_train + (((yPred[i] - min_y_pred)/(max_y_pred - min_y_pred))*(max_y_train - min_y_train))

predictions_file = open("dnn.csv", "w", newline="")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["ID", "PredictedProb"])
open_file_object.writerows(zip(testData[:, 0].astype(int), yPred))
predictions_file.close()
print("Finished")