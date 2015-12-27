import csv
import BikeSharingDemand.Combine.Model as model

trainFrame = model.cleanTrainset(model.loadTrainData(), describe=True)
trainData = model.convertPandasDataFrameToNumpyArray(trainFrame)

trainX = trainData[:, 3:]
trainYCasReg = trainData[:, 0:2] # [casual, registered]

#print(trainYCasReg[:, 0].dtype)

testFrame = model.cleanTestSet(model.loadTestData(), True)
testData = model.convertPandasDataFrameToNumpyArray(testFrame)

testX = testData[:, 1:]

# Enable logging
model.enableLogs = True

xgboost = model.selectXGBoost()
rf = model.selectRandomForest()

combinedRegressor = model.Combiner([xgboost, rf])
combinedRegressor.fit(trainX, trainYCasReg)

predictedY = combinedRegressor.predict(testX)


f = open("bike-sharing-result-combined.csv", "w", newline="")
csvWriter = csv.writer(f)
csvWriter.writerow(["datetime", "count"])

for datetime, count in zip(testData[:, 0], predictedY):
    csvWriter.writerow([datetime, int(count)])
f.close()

