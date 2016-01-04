import csv
import BikeSharingDemand.Combine.Model as model
import MLScripts.Metrics as Metrics

trainFrame = model.cleanTrainset(model.loadTrainData())
trainData = model.convertPandasDataFrameToNumpyArray(trainFrame)

trainX = trainData[:, 3:]
trainYCasReg = trainData[:, 0:2] # [casual, registered]

xTrain, xTest, yTrain, yTest = Metrics.traintestSplit(trainX, trainYCasReg)

#xgboost = model.selectXGBoost()
#xgboost2 = model.selectXGBoost()

boostCount = 8
xgboosts = [model.selectXGBoost() for _ in range(boostCount)]

combinedRegressor = model.Combiner(xgboosts)
combinedRegressor.fit(xTrain, yTrain)

yPred = combinedRegressor.predict(xTest)

y = []
for i, x in enumerate(yTest):
    y.append(x[0] + x[1])

rmsle = Metrics.rmsle2(y, yPred)

print("RMSLE Score : ", rmsle)


"""
Final Model
"""

testFrame = model.cleanTestSet(model.loadTestData())
testData = model.convertPandasDataFrameToNumpyArray(testFrame)

testX = testData[:, 1:]

# Enable logging
model.enableLogs = True

combinedRegressor = model.Combiner(xgboosts)
combinedRegressor.fit(trainX, trainYCasReg)

#featurenames = model.getColNames(trainFrame)[3:]
#combinedRegressor.feature_importances_(trainFrame)

predictedY = combinedRegressor.predict(testX)


f = open("bike-sharing-result-combined.csv", "w", newline="")
csvWriter = csv.writer(f)
csvWriter.writerow(["datetime", "count"])

for datetime, count in zip(testData[:, 0], predictedY):
    csvWriter.writerow([datetime, int(count)])
f.close()

