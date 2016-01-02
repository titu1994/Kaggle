import MNIST.DataClean as dc
import numpy as np
import keras.layers.core as core
import keras.layers.convolutional as conv
import keras.optimizers as optm
import keras.models as models

trainData = dc.convertPandasDataFrameToNumpyArray(dc.loadTrainData(describe=False))


"""
layers = [conv.Convolution2D(7, 3, 3, activation="relu", input_shape=(None, 1, 28, 28)),
          conv.MaxPooling2D(),
          conv.Convolution2D(12, 2, 2, activation="relu", ),
          conv.MaxPooling2D(),
          core.Dense(1000, activation="relu"),
          core.Flatten(),
          core.Dense(10, activation="softmax", )]
"""
cnn = models.Sequential()
cnn.add(core.Dense(output_dim=784, ))
cnn.add(conv.Convolution2D(7, 3, 3, activation="relu", input_shape=(1, 28, 28)))
cnn.add(conv.MaxPooling2D())
cnn.add(conv.Convolution2D(12, 2, 2, activation="relu"))
cnn.add(conv.MaxPooling2D())
cnn.add(core.Flatten())
cnn.add(core.Dense(1000, activation="relu"))
cnn.add(core.Dense(10, activation="softmax"))


sgd = optm.SGD(ir=0.01, momentum=0.9, nesterov=True, decay=1e-6)
cnn.compile(sgd, loss="rmse")

cnn.fit(trainData[:, 1:], trainData[:, 0], verbose=1, show_accuracy=True, batch_size=1)

testData = dc.convertPandasDataFrameToNumpyArray(dc.loadTestData())

yPred = cnn.predict(testData)

np.savetxt('mnist-cnn.csv', np.c_[range(1,len(yPred)+1),yPred], delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')
print("Save predictions to file complete")