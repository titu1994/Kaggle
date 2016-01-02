import MNIST.DataClean as dc
import numpy as np
import keras.layers.core as core
import keras.layers.convolutional as conv
import keras.optimizers as optm
import keras.models as models

batch_size = 128
nb_classes = 10
nb_epoch = 12

img_rows, img_cols = 28, 28
nb_filters = 32
nb_pool = 2
nb_conv = 3

trainData = dc.convertPandasDataFrameToNumpyArray(dc.loadTrainData(describe=False))
trainX = trainData[:, 1:].reshape(trainData.shape[0], 1, img_rows, img_cols)
trainX = trainX.astype(float)
trainX /= 255.0

cnn = models.Sequential()
cnn.add(conv.Convolution2D(nb_filters, nb_conv, nb_conv, border_mode="valid", input_shape=(1, 28, 28), activation="relu"))
cnn.add(conv.Convolution2D(nb_filters, nb_conv, nb_conv, activation="relu"))
cnn.add(conv.MaxPooling2D())
cnn.add(core.Dropout(0.25))
cnn.add(core.Flatten())
cnn.add(core.Dense(128, activation="relu"))
cnn.add(core.Dropout(0.5))
cnn.add(core.Dense(10, activation="relu"))
cnn.add(core.Dense(1, activation="softmax"))

cnn.compile(loss='categorical_crossentropy', optimizer='adadelta')

cnn.fit(trainX, trainData[:, 0], batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=1, )

testData = dc.convertPandasDataFrameToNumpyArray(dc.loadTestData())
testX = testData.reshape(testData.shape[0], 1, 28, 28)
testX = testX.astype(float)
testX /= 255.0

yPred = cnn.predict(testX)

np.savetxt('mnist-cnn.csv', np.c_[range(1,len(yPred)+1),yPred], delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')
print("Save predictions to file complete")