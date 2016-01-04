import csv
import MLScripts.Metrics as Metrics
import sklearn.ensemble as ensemble

import BikeSharingDemand.DataClean as dataclean

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
crossvalidationTree = ensemble.GradientBoostingRegressor(n_estimators=400, learning_rate=0.01, max_depth=6, random_state=1, presort=True)
cvCount = 10
crossvalidation = Metrics.crossValidationScore(ensemble.GradientBoostingRegressor(random_state=1), trainX, trainY, cvCount=cvCount)

xTrain, xTest, yTrain, yTest = Metrics.traintestSplit(trainX, trainY, randomState=1)

"""
#{'n_estimators': 400, 'max_depth': 6, 'learning_rate': 0.01

if __name__ == "__main__":
    params = {"max_depth" : [3,4,5,6,7,8], "n_estimators" : [100, 200, 300, 400], "learning_rate" : [0.01, 0.05, 0.1, 0.2, 0.5, 1]}
    clf = GridSearchCV(crossvalidationTree, params, verbose=1, n_jobs=2, cv=10)
    clf.fit(trainX, trainY)

    print("GridSearch : \n", "Best Estimator : ", clf.best_estimator_,
        "\nBest Params : ", clf.best_params_, "\nBest Score", clf.best_score_)
"""

crossvalidationTree.fit(xTrain, yTrain)

yPredict = crossvalidationTree.predict(xTest)

#trainingAccuracy = metrics.trainingAccuracy(yTest, yPredict)
rmse = Metrics.rmse(yTest, yPredict)
nrmse = Metrics.nrmse(yTest, yPredict)
logloss = Metrics.rmsle(yTest, yPredict)

print("Max Cross Validation Score : ", crossvalidation.max(), "\nAverage Cross Validation Score : ", crossvalidation.mean(),
      "\nGradient Boosting Forest Score : ", crossvalidationTree.score(xTrain, yTrain),
      #"\nTraining Accuracy : ", trainingAccuracy,
      "\nRoot Mean Squared Error : ", rmse, "\nNormalized RMSE : ", nrmse,
      #"\nKFold Accuracy : ", kfoldAccuracy
      "\nLog Loss : ", logloss
      )

featureNames = dataclean.getColNames(trainFrame)[1:]
featureImportances = [(feature, importance) for feature, importance in zip(featureNames, crossvalidationTree.feature_importances_)]
featureImportances = sorted(featureImportances, key=lambda x: x[1], reverse=True)
print("Feature Importances : \n", featureImportances)

"""
Final Model
"""

finalModel = ensemble.GradientBoostingRegressor(n_estimators=400, learning_rate=0.01, max_depth=6, random_state=1, presort=True)
finalModel.fit(trainX, trainY)

finalPredicted = finalModel.predict(testX)
for i, x in enumerate(finalPredicted):
    finalPredicted[i] = finalPredicted[i] if finalPredicted[i] >= 0 else 0

f = open("bike-sharing-result.csv", "w", newline="")
csvWriter = csv.writer(f)
csvWriter.writerow(["datetime", "count"])

for datetime, count in zip(testData[:, 0], finalPredicted):
    csvWriter.writerow([datetime, int(count)])
f.close()