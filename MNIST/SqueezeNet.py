from keras.models import Model
from keras.layers import Input, merge
from keras.layers.core import Flatten, Dense, Dropout, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
import keras.utils.np_utils as kutils
from keras.utils.visualize_util import plot, model_to_dot

import MNIST.DataClean as dc
import numpy as np

batch_size = 128 # 128
nb_epoch = 200 # 12
img_rows, img_cols = 28, 28

trainData = dc.convertPandasDataFrameToNumpyArray(dc.loadTrainData(describe=False))
trainX = trainData[:, 1:].reshape(trainData.shape[0], 1, img_rows, img_cols)
trainX = trainX.astype(float)
trainX /= 255.0

trainY = kutils.to_categorical(trainData[:, 0])
nb_classes = trainY.shape[1]

input_layer = Input(shape=(1, 28, 28), name="input")

#conv 1
conv1 = Convolution2D(96, 3, 3, activation='relu', init='glorot_uniform',subsample=(2,2),border_mode='valid')(input_layer)

#maxpool 1
maxpool1 = MaxPooling2D(pool_size=(2,2))(conv1)

#fire 1
fire2_squeeze = Convolution2D(16, 1, 1, activation='relu', init='glorot_uniform',border_mode='same')(maxpool1)
fire2_expand1 = Convolution2D(64, 1, 1, activation='relu', init='glorot_uniform',border_mode='same')(fire2_squeeze)
fire2_expand2 = Convolution2D(64, 3, 3, activation='relu', init='glorot_uniform',border_mode='same')(fire2_squeeze)
merge1 = merge(inputs=[fire2_expand1, fire2_expand2], mode="concat", concat_axis=1)
fire2 = Activation("linear")(merge1)

#fire 2
fire3_squeeze = Convolution2D(16, 1, 1, activation='relu', init='glorot_uniform',border_mode='same')(fire2)
fire3_expand1 = Convolution2D(64, 1, 1, activation='relu', init='glorot_uniform',border_mode='same')(fire3_squeeze)
fire3_expand2 = Convolution2D(64, 3, 3, activation='relu', init='glorot_uniform',border_mode='same')(fire3_squeeze)
merge2 = merge(inputs=[fire3_expand1, fire3_expand2], mode="concat", concat_axis=1)
fire3 = Activation("linear")(merge2)

#fire 3
fire4_squeeze = Convolution2D(32, 1, 1, activation='relu', init='glorot_uniform',border_mode='same')(fire3)
fire4_expand1 = Convolution2D(128, 1, 1, activation='relu', init='glorot_uniform',border_mode='same')(fire4_squeeze)
fire4_expand2 = Convolution2D(128, 3, 3, activation='relu', init='glorot_uniform',border_mode='same')(fire4_squeeze)
merge3 = merge(inputs=[fire4_expand1, fire4_expand2], mode="concat", concat_axis=1)
fire4 = Activation("linear")(merge3)

#maxpool 4
maxpool4 = MaxPooling2D((2,2))(fire4)

#fire 5
fire5_squeeze = Convolution2D(32, 1, 1, activation='relu', init='glorot_uniform',border_mode='same')(maxpool4)
fire5_expand1 = Convolution2D(128, 1, 1, activation='relu', init='glorot_uniform',border_mode='same')(fire5_squeeze)
fire5_expand2 = Convolution2D(128, 3, 3, activation='relu', init='glorot_uniform',border_mode='same')(fire5_squeeze)
merge5 = merge(inputs=[fire5_expand1, fire5_expand2], mode="concat", concat_axis=1)
fire5 = Activation("linear")(merge5)

#fire 6
fire6_squeeze = Convolution2D(48, 1, 1, activation='relu', init='glorot_uniform',border_mode='same')(fire5)
fire6_expand1 = Convolution2D(192, 1, 1, activation='relu', init='glorot_uniform',border_mode='same')(fire6_squeeze)
fire6_expand2 = Convolution2D(192, 3, 3, activation='relu', init='glorot_uniform',border_mode='same')(fire6_squeeze)
merge6 = merge(inputs=[fire6_expand1, fire6_expand2], mode="concat", concat_axis=1)
fire6 = Activation("linear")(merge6)

#fire 7
fire7_squeeze = Convolution2D(48, 1, 1, activation='relu', init='glorot_uniform',border_mode='same')(fire6)
fire7_expand1 = Convolution2D(192, 1, 1, activation='relu', init='glorot_uniform',border_mode='same')(fire7_squeeze)
fire7_expand2 = Convolution2D(192, 3, 3, activation='relu', init='glorot_uniform',border_mode='same')(fire7_squeeze)
merge7 = merge(inputs=[fire7_expand1, fire7_expand2], mode="concat", concat_axis=1)
fire7 =Activation("linear")(merge7)

#fire 8
fire8_squeeze = Convolution2D(64, 1, 1, activation='relu', init='glorot_uniform',border_mode='same')(fire7)
fire8_expand1 = Convolution2D(256, 1, 1, activation='relu', init='glorot_uniform',border_mode='same')(fire8_squeeze)
fire8_expand2 = Convolution2D(256, 3, 3, activation='relu', init='glorot_uniform',border_mode='same')(fire8_squeeze)
merge8 = merge(inputs=[fire8_expand1, fire8_expand2], mode="concat", concat_axis=1)
fire8 = Activation("linear")(merge8)

#maxpool 8
maxpool8 = MaxPooling2D((2,2))(fire8)

#fire 9
fire9_squeeze = Convolution2D(64, 1, 1, activation='relu', init='glorot_uniform',border_mode='same')(maxpool8)
fire9_expand1 = Convolution2D(256, 1, 1, activation='relu', init='glorot_uniform',border_mode='same')(fire9_squeeze)
fire9_expand2 = Convolution2D(256, 3, 3, activation='relu', init='glorot_uniform',border_mode='same')(fire9_squeeze)
merge8 = merge(inputs=[fire9_expand1, fire9_expand2], mode="concat", concat_axis=1)
fire9 = Activation("linear")(merge8)
fire9_dropout = Dropout(0.5)(fire9)

#conv 10
conv10 = Convolution2D(10, 1, 1, init='glorot_uniform',border_mode='valid')(fire9_dropout)

#avgpool 1
#avgpool10 = AveragePooling2D((13,13))(conv10)

flatten = Flatten()(conv10)

softmax = Dense(nb_classes, activation="softmax")(flatten)

model = Model(input=input_layer, output=softmax)

model.summary()
plot(model, "SqueezeNet.png", show_shapes=True)

model.compile(optimizer="adadelta", loss="categorical_crossentropy", metrics=["accuracy"])

#model.load_weights("SqueezeNet Weights.h5")
print("Model loaded")

model.fit(trainX,trainY, batch_size=batch_size, nb_epoch=nb_epoch)

model.save_weights("SqueezeNet Weights.h5", overwrite=True)
print("Model saved.")

testData = dc.convertPandasDataFrameToNumpyArray(dc.loadTestData())
testX = testData.reshape(testData.shape[0], 1, 28, 28)
testX = testX.astype(float)
testX /= 255.0

yPred = model.predict(testX, verbose=1)
yPred = np.argmax(yPred, axis=1)

np.savetxt('mnist-squeezenet.csv', np.c_[range(1,len(yPred)+1),yPred], delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')
