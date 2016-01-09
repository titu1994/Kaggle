import keras.models as models
import keras.layers.core as core
import keras.layers.normalization as norm
import keras.utils.np_utils as kutils
import keras.callbacks as callbacks
import Homesite.DataClean as dc
import numpy as np
import sklearn.preprocessing as preproc
import csv

trainFrame, noOfClasses = dc.cleanDataNN(dc.loadTrainData(), describe=False)
testFrame, _ = dc.cleanDataNN(dc.loadTestData(), istest=True, describe=False)
trainFrame, testFrame = dc.postprocessObjects(trainFrame, testFrame)

trainData = dc.convertPandasDataFrameToNumpyArray(trainFrame)
print("Data loaded")
print("No of Classes : ", noOfClasses)

trainX = trainData[:, 1:].astype(np.float32)
trainY = kutils.to_categorical(trainData[:, 0])

noFeatures = trainX.shape[1]
scaler = preproc.StandardScaler()
trainX = scaler.fit_transform(trainX)

epochs = 8

nn = models.Graph()
nn.add_input(name="input", input_shape=(noFeatures,))
nn.add_node(norm.BatchNormalization(input_shape=(noFeatures,)), name="batchnormal", input="input")

nn.add_node(core.Dense(598, activation="relu"), name="d11", input="batchnormal")
nn.add_node(core.Dense(299, activation="relu"), name="d12", input="d11")
nn.add_node(core.Dropout(0.25), name="drop1", input="d12")

nn.add_node(core.Dense(299, activation="relu"), name="d21", input="batchnormal")
nn.add_node(core.Dense(598, activation="relu"), name="d22", input="d21")
nn.add_node(core.Dropout(0.25), name="drop2", input="d22")

nn.add_node(core.Dense(noOfClasses, activation="softmax"), name="output", inputs=["drop1", "drop2"], create_output=True)

print(nn.summary())

nn.compile(optimizer="adadelta", loss={"output" : "mean_squared_error"})
#                                                                       callbacks=[callbacks.EarlyStopping(patience=2, verbose=1)]
nn.fit({"input" : trainX, "output" : trainY}, nb_epoch=epochs, verbose=1, validation_split=0.01, )

testData = dc.convertPandasDataFrameToNumpyArray(testFrame)
testX = testData[:, 1:]
#print("No of test features : ", testX.shape[1])
testX = preproc.StandardScaler().fit_transform(testX)
yPred = nn.predict({"input" : testX},verbose=1)

f = open("graph_nn.csv", "w", newline="")
csvWriter = csv.writer(f)
csvWriter.writerow(["QuoteNumber", "QuoteConversion_Flag"])

for qn, qflag in zip(testData[:, 0], yPred):
    csvWriter.writerow([int(qn), float(qflag)])

f.close()
print("Output finished")
