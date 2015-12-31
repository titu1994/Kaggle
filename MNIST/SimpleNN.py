import MNIST.DataClean as dc
import numpy as np
import sknn.mlp as mlp
import pickle

try:
    nn = pickle.load(open("simplenn.pkl", "rb"))
    print("Model loaded")
except:
    nn = None

layers = [mlp.Convolution("Rectifier", channels=10,kernel_shape=(2,2)),
          mlp.Layer("Rectifier", units=1000),
          mlp.Layer("Softmax", units=10)]

if nn is None:
    trainFrame = dc.loadTrainData(describe=False)
    trainData = dc.convertPandasDataFrameToNumpyArray(trainFrame)

    nn = mlp.Classifier(layers=layers, learning_rate=0.00001, valid_size=0, random_state=0, n_iter=50, verbose=True, batch_size=1000, learning_rule="nesterov")
    nn.fit(trainData[:, 1:], trainData[:, 0])
    print("Model fitting complete")

    pickle.dump(nn, open("simplenn.pkl", "wb"))
    print("Model saved")

testFrame = dc.loadTestData()
testData = dc.convertPandasDataFrameToNumpyArray(testFrame)

testX = testData[:, 0:]
yPred = nn.predict(testX)

np.savetxt('mnist-rf-nn.csv', np.c_[range(1,len(yPred)+1),yPred], delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')
