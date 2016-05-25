from MLScripts.CleaningUtils import *

def loadTrainData(describe=False):
    return loadData(r"D:\Users\Yue\PycharmProjects\Kaggle\MNIST\Data\train.csv", describe=describe)

def loadTestData():
    return loadData(r"D:\Users\Yue\PycharmProjects\Kaggle\MNIST\Data\test.csv", )

def loadFullTrainData():
    return loadData(r"D:\Users\Yue\PycharmProjects\Kaggle\MNIST\Data\mnist_train.csv")

def loadFullTestData():
    return loadData(r"D:\Users\Yue\PycharmProjects\Kaggle\MNIST\Data\mnist_test.csv")
#loadTrainData(describe=True)
