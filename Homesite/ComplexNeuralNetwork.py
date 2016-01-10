import keras.models as models
import keras.layers.core as core
import keras.layers.normalization as norm
import keras.utils.np_utils as kutils
import keras.callbacks as callbacks
import keras.layers.advanced_activations as advact
import keras.optimizers as optm
from keras.layers import containers
import Homesite.DataClean as dc
import numpy as np
import sklearn.preprocessing as preproc
import csv

trainFrame, noOfClasses = dc.cleanDataCXNN(dc.loadTrainData(), describe=False)
testFrame, _ = dc.cleanDataCXNN(dc.loadTestData(), istest=True, describe=False)
trainFrame, testFrame = dc.postprocessObjects(trainFrame, testFrame)

trainData = dc.convertPandasDataFrameToNumpyArray(trainFrame)
print("Data loaded")
print("No of Classes : ", noOfClasses)

trainX = trainData[:, 1:].astype(np.float32)
trainY = kutils.to_categorical(trainData[:, 0])

noFeatures = trainX.shape[1]
scaler = preproc.StandardScaler()
trainX = scaler.fit_transform(trainX)

"""
Final Model
"""

epochs = 20

nn = models.Sequential()

nn.add(core.Dense(noFeatures, input_shape=(noFeatures,)))
nn.add(advact.PReLU())
nn.add(norm.BatchNormalization())
nn.add(core.Dropout(0.2))

nn.add(core.Dense(2*noFeatures,))
nn.add(advact.PReLU())
nn.add(norm.BatchNormalization())
nn.add(core.Dropout(0.25))

nn.add(core.Dense(noFeatures,))
nn.add(advact.PReLU())
nn.add(norm.BatchNormalization())
nn.add(core.Dropout(0.2))

nn.add(core.Dense(noOfClasses, activation="softmax"))

print(nn.summary())

opt = optm.Adadelta(lr=1, decay=0.995, epsilon=1e-5)
nn.compile(optimizer=opt, loss="binary_crossentropy")
#                                                                       callbacks=[callbacks.EarlyStopping(patience=2, verbose=1)]
nn.fit(trainX, trainY, nb_epoch=epochs, verbose=1, show_accuracy=True, validation_split=0.01, batch_size=256)

nn.save_weights("complex_nn_weights.h5", overwrite=True)

testData = dc.convertPandasDataFrameToNumpyArray(testFrame)
testX = testData[:, 1:]
#print("No of test features : ", testX.shape[1])
testX = preproc.StandardScaler().fit_transform(testX)
yPred = nn.predict_proba(testX)[:, 1]

f = open("complex_nn.csv", "w", newline="")
csvWriter = csv.writer(f)
csvWriter.writerow(["QuoteNumber", "QuoteConversion_Flag"])

for qn, qflag in zip(testData[:, 0], yPred):
    csvWriter.writerow([int(qn), float(qflag)])

f.close()
print("Output finished")

