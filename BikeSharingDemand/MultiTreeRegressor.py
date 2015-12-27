from sklearn.grid_search import GridSearchCV
import BikeSharingDemand.DataClean as dataclean
import csv
import sklearn.ensemble as ensemble
import xgboost as xgb
import Metric.Metrics as metrics
import sklearn.tree as tree
import pandas as pd

trainFrame = dataclean.cleanDataset(dataclean.loadTrainData())
trainData = dataclean.convertPandasDataFrameToNumpyArray(trainFrame)

testFrame = dataclean.cleanDataset(dataclean.loadTestData(), True)
testData = dataclean.convertPandasDataFrameToNumpyArray(testFrame)

trainX = trainData[:, 1:]
trainY = trainData[:, 0]

testX = testData[:, 1:]

"""
Cross Validation
"""
xTrain, xTest, yTrain, yTest = metrics.traintestSplit(trainX, trainY, randomState=1)

xgbtree = xgb.XGBRegressor(n_estimators=500, learning_rate=0.01, max_depth=10, seed=1, nthread=4)
gbtree = ensemble.GradientBoostingRegressor(n_estimators=400, learning_rate=0.01, max_depth=6, random_state=1, presort=True)
randomforest = ensemble.RandomForestRegressor(n_estimators=500, n_jobs=4, random_state=1)
decisionTree = tree.DecisionTreeRegressor(presort=True)

xgbtree.fit(trainX, trainY)
gbtree.fit(trainX, trainY)
randomforest.fit(trainX, trainY)
decisionTree.fit(trainX, trainY)

xgbPredict = xgbtree.predict(testX)
gbPredict = gbtree.predict(testX)
randomforestPredict = randomforest.predict(testX)
decisionTreePredict = decisionTree.predict(testX)

nPredictions = len(xgbPredict)
yPredict = list()
"""
# Weight Calculation
nPredictions = len(xgbPredict)
yPredict = list()

holdFrame = pd.DataFrame(columns=("w1", "w2", "w3", "w4", "logloss"))
pos = 0


for w1 in range(1, 10):
    for w2 in range(1, 10):
        for w3 in range(1, 10):
            for w4 in range(1, 10):
                for i in range(nPredictions):
                    yPredict.append((w1 * xgbPredict[i] + w2 * gbPredict[i] + w3 * randomforestPredict[i] +
                                   w4 * decisionTreePredict[i]) / (w1 + w2 + w3 + w4))

                for i, x in enumerate(yPredict):
                    if yPredict[i] < 0:
                        print("yActual : ", yTest[i], " yPredicted : ", yPredict[i])
                        yPredict[i] = -yPredict[i]

                rmsle = metrics.rmsle(yTest, yPredict)

                holdFrame.loc[pos] = [w1, w2, w3, w4, rmsle]
                pos += 1

                yPredict.clear()
                #print("Finished (%d,%d,%d)" % (w1, w2, w3))

sortedHoldFrame = holdFrame.sort(["logloss"], ascending=True)
print(sortedHoldFrame, "\n\n")

#print(sortedHoldFrame.info(), "\n")
#print(sortedHoldFrame.describe(), "\n")

"""
for i in range(nPredictions):
        yPredict.append((4 * xgbPredict[i] + 1 * gbPredict[i] + 1 * randomforestPredict[i] +
                                   1 * decisionTreePredict[i]) / (7))

        for i, x in enumerate(yPredict):
            if yPredict[i] < 0:
                print("yActual : ", yTest[i], " yPredicted : ", yPredict[i])
                yPredict[i] = -yPredict[i]

f = open("bike-sharing-result.csv", "w", newline="")
csvWriter = csv.writer(f)
csvWriter.writerow(["datetime", "count"])

for datetime, count in zip(testData[:, 0], yPredict):
    csvWriter.writerow([datetime, int(count)])
f.close()

# Kaggle Score = 0.47119, loc = 1123.5