import MNIST.DataClean as dc
import numpy as np
import sklearn.metrics as metrics

from keras.layers import merge, Activation, Input
from keras.models import Model
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import MaxPooling2D, Convolution2D, ZeroPadding2D
import keras.callbacks as callbacks
import keras.utils.np_utils as kutils
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.visualize_util import plot
import pandas as pd

batch_size = 128 # 128
nb_epoch = 100 # 12
img_rows, img_cols = 28, 28

nb_filters_1 = 64
nb_filters_2 = 128
nb_filters_3 = 256
nb_conv = 3
nb_conv_mid = 4
nb_conv_init = 5

trainPath = r"D:\Users\Yue\PycharmProjects\Kaggle\MNIST\Data\mnist_train.csv"
validationPath = r"D:\Users\Yue\PycharmProjects\Kaggle\MNIST\Data\mnist_test.csv"

trainData = pd.read_csv(trainPath, header=0).values
trainX = trainData[:, 1:].reshape(trainData.shape[0], 1, img_rows, img_cols)
trainX = trainX.astype(float)
trainX /= 255.0

validationData = pd.read_csv(validationPath, header=0).values
validateX = validationData[:, 1:].reshape(validationData.shape[0], 1, img_rows, img_cols)
validateX = validateX.astype(float)
validateX /= 255.0

testData = dc.convertPandasDataFrameToNumpyArray(dc.loadTestData())
testX = testData.reshape(testData.shape[0], 1, 28, 28)
testX = testX.astype(float)
testX /= 255.0

trainY = kutils.to_categorical(trainData[:, 0])
validationY = kutils.to_categorical(validationData[:, 0])

generator = ImageDataGenerator(rotation_range=20,
                               width_shift_range=0.2,
                               height_shift_range=0.2,)

generator.fit(trainX, seed=0)

nb_classes = trainY.shape[1]

init = Input(shape=(1, 28, 28),)

c11 = Convolution2D(nb_filters_1, nb_conv_init, nb_conv_init,  activation="relu", border_mode='same')(init)
c12 = Convolution2D(nb_filters_1, nb_conv_init, nb_conv_init, activation="relu", border_mode='same')(init)
merge1 = merge([c11, c12], mode='concat', concat_axis=1)
maxpool1 = MaxPooling2D(strides=(2,2), border_mode='same')(merge1)

c21 = Convolution2D(nb_filters_2, nb_conv_mid, nb_conv_mid, activation="relu", border_mode='same')(maxpool1)
c22 = Convolution2D(nb_filters_2, nb_conv_mid, nb_conv_mid, activation="relu", border_mode='same')(maxpool1)
merge2 = merge([c21, c22, ], mode='concat', concat_axis=1)
maxpool2 = MaxPooling2D(strides=(2,2), border_mode='same')(merge2)

c31 = Convolution2D(nb_filters_3, nb_conv, nb_conv, activation="relu", border_mode='same')(maxpool2)
c32 = Convolution2D(nb_filters_3, nb_conv, nb_conv, activation="relu", border_mode='same')(maxpool2)
c33 = Convolution2D(nb_filters_3, nb_conv, nb_conv, activation="relu", border_mode='same')(maxpool2)
c34 = Convolution2D(nb_filters_3, nb_conv, nb_conv, activation="relu", border_mode='same')(maxpool2)
c35 = Convolution2D(nb_filters_3, nb_conv, nb_conv, activation="relu", border_mode='same')(maxpool2)
c36 = Convolution2D(nb_filters_3, nb_conv, nb_conv, activation="relu", border_mode='same')(maxpool2)
merge3 = merge([c31, c32, c33, c34, c35, c36], mode='concat', concat_axis=1)
maxpool3 = MaxPooling2D(strides=(2,2), border_mode='same')(merge3)

dropout = Dropout(0.5)(maxpool3)

flatten = Flatten()(dropout)
output = Dense(nb_classes, activation="softmax")(flatten)

model = Model(input=init, output=output)

model.summary()
plot(model, "DMCNN.png", show_shapes=False)

model.compile(loss="categorical_crossentropy", optimizer="adadelta", metrics=["accuracy"])

model.load_weights("DMCNN Weights.h5")
print("Model loaded.")

#model.fit_generator(generator.flow(trainX, trainY, batch_size=batch_size), samples_per_epoch=len(trainX), nb_epoch=nb_epoch,
#                    callbacks=[callbacks.ModelCheckpoint("DMCNN Weights.h5", monitor="val_acc", save_best_only=True)],
#                    validation_data=(validateX, validationY))

yPreds = model.predict(validateX)
yPred = np.argmax(yPreds, axis=1)
yTrue = validationData[:, 0]

accuracy = metrics.accuracy_score(yTrue, yPred) * 100
error = 100 - accuracy
print("Accuracy : ", accuracy)
print("Error : ", error)

import heapq
errorCount = 0
secondGuessCorrect = 0

for i in range(len(yPreds)):
    res = heapq.nlargest(2, range(len(yPreds[i])), yPreds[i].take)
    if yTrue[i] != res[0]:
        errorCount += 1

        if yTrue[i] == res[1]:
            secondGuessCorrect += 1

print("Error count : ", errorCount)
print("Second Guess Correct count : ", secondGuessCorrect)