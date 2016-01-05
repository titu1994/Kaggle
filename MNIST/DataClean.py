from MLScripts.CleaningUtils import *

def loadTrainData(describe=False):
    return loadData(r"C:\Users\Yue\PycharmProjects\Kaggle\MNIST\Data\train.csv", describe=describe)

def loadTestData():
    return loadData(r"C:\Users\Yue\PycharmProjects\Kaggle\MNIST\Data\test.csv")



