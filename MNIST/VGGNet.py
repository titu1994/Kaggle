import MNIST.DataClean as dc
import numpy as np
import keras.layers.core as core
import keras.layers.convolutional as conv
import keras.models as models
import keras.callbacks as callbacks
import keras.utils.np_utils as kutils
import pandas as pd

batch_size = 128 # 128
nb_epoch = 20 # 12
img_rows, img_cols = 28, 28

nb_filters_1 = 64
nb_filters_2 = 128
nb_filters_3 = 256
nb_conv = 3

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

nb_classes = trainY.shape[1]

cnn = models.Sequential()

cnn.add(conv.ZeroPadding2D((1,1), input_shape=(1, 28, 28),))
cnn.add(conv.Convolution2D(nb_filters_1, nb_conv, nb_conv,  activation="relu"))
cnn.add(conv.ZeroPadding2D((1, 1)))
cnn.add(conv.Convolution2D(nb_filters_1, nb_conv, nb_conv, activation="relu"))
cnn.add(conv.MaxPooling2D(strides=(2,2)))

cnn.add(conv.ZeroPadding2D((1, 1)))
cnn.add(conv.Convolution2D(nb_filters_2, nb_conv, nb_conv, activation="relu"))
cnn.add(conv.ZeroPadding2D((1, 1)))
cnn.add(conv.Convolution2D(nb_filters_2, nb_conv, nb_conv, activation="relu"))
cnn.add(conv.MaxPooling2D(strides=(2,2)))

cnn.add(conv.ZeroPadding2D((1, 1)))
cnn.add(conv.Convolution2D(nb_filters_3, nb_conv, nb_conv, activation="relu"))
cnn.add(conv.ZeroPadding2D((1, 1)))
cnn.add(conv.Convolution2D(nb_filters_3, nb_conv, nb_conv, activation="relu"))
cnn.add(conv.ZeroPadding2D((1, 1)))
cnn.add(conv.Convolution2D(nb_filters_3, nb_conv, nb_conv, activation="relu"))
cnn.add(conv.ZeroPadding2D((1, 1)))
cnn.add(conv.Convolution2D(nb_filters_3, nb_conv, nb_conv, activation="relu"))
cnn.add(conv.MaxPooling2D(strides=(2,2)))

cnn.add(core.Flatten())
cnn.add(core.Dropout(0.2))
cnn.add(core.Dense(1024, activation="relu"))
cnn.add(core.Dense(nb_classes, activation="softmax"))

cnn.summary()
cnn.compile(loss="categorical_crossentropy", optimizer="adadelta", metrics=["accuracy"])

#cnn.load_weights("VGG_Temp.h5")
#print("Model loaded.")

cnn.fit(trainX, trainY, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_data=(validateX, validationY)) #validation_split=0.01,
        #callbacks=[callbacks.ModelCheckpoint("VGG_Best.h5", save_best_only=True)]
print("Model fit.")

cnn.save_weights("VGG_Temp.h5", overwrite=True)
print("Weights Saved.")

yPred = cnn.predict_classes(testX)

np.savetxt('mnist-vgg.csv', np.c_[range(1,len(yPred)+1),yPred], delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')
print("Save predictions to file complete")