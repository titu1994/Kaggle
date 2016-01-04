import csv

import seaborn as sns
import sklearn.ensemble as ensemble
import xgboost as xgb

import BikeSharingDemand.DataClean as dataclean

sns.set_style("whitegrid")

trainFrame = dataclean.cleanDataset(dataclean.loadTrainData())
trainData = dataclean.convertPandasDataFrameToNumpyArray(trainFrame)

testFrame = dataclean.cleanDataset(dataclean.loadTestData(), True)
testData = dataclean.convertPandasDataFrameToNumpyArray(testFrame)

trainX = trainData[:, 1:]
trainY = trainData[:, 0]

testX = testData[:, 1:]

"""
Cross Validation
"""                                                                                     #colsample_bytree=5
crossvalidationTree = xgb.XGBRegressor(n_estimators=300, learning_rate=0.01, max_depth=10, seed=1, nthread=4)
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

def evalMetric(yPredicted, yTrainDMatrix):
    yT = yTrainDMatrix.get_label()
    return ("rmsle", Metrics.rmsle(yT, yPredicted))

evalSet = [(xTrain, yTrain), (xTest, yTest)]

crossvalidationTree.fit(xTrain, yTrain, eval_metric=evalMetric, verbose=False)

yPredict = crossvalidationTree.predict(xTest)

#trainingAccuracy = metrics.trainingAccuracy(yTest, yPredict)
rmse = Metrics.rmse(yTest, yPredict)
nrmse = Metrics.nrmse(yTest, yPredict)

for i, x in enumerate(yPredict):
    if yPredict[i] < 0:
        print("yActual : ", yTest[i], " yPredicted : ", yPredict[i])
        yPredict[i] = -yPredict[i]

logloss = Metrics.rmsle(yTest, yPredict)

print("Max Cross Validation Score : ", crossvalidation.max(), "\nAverage Cross Validation Score : ", crossvalidation.mean(),
      "\nGradient Boosting Forest Score : ", crossvalidationTree.score(xTrain, yTrain),
      #"\nTraining Accuracy : ", trainingAccuracy,
      "\nRoot Mean Squared Error : ", rmse, "\nNormalized RMSE : ", nrmse,
      #"\nKFold Accuracy : ", kfoldAccuracy
      "\nLog Loss : ", logloss
      )

featureNames = dataclean.getColNames(trainFrame)[1:]
featureImportances = [(feature, importance) for feature, importance in zip(featureNames, sorted(crossvalidationTree.booster().get_fscore(), key=lambda x: x[1]))]
#featureImportances = sorted(featureImportances, key=lambda x: x[1], reverse=True)
print("Feature Importances : \n", featureImportances)

#featureImportance = xgb.plot_importance(crossvalidationTree)
#sns.plt.show()

"""
Final Model
"""
finalModel = crossvalidationTree
#finalModel = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.01, max_depth=10, seed=1, nthread=4)
finalModel.fit(trainX, trainY)

finalPredicted = finalModel.predict(testX)
for i, x in enumerate(finalPredicted):
    finalPredicted[i] = finalPredicted[i] if finalPredicted[i] >= 0 else -finalPredicted[i]

f = open("bike-sharing-result.csv", "w", newline="")
csvWriter = csv.writer(f)
csvWriter.writerow(["datetime", "count"])

for datetime, count in zip(testData[:, 0], finalPredicted):
    csvWriter.writerow([datetime, int(count)])
f.close()

# Kaggle Score = 0.43068 loc = 577.5 n_estimators=300, learning_rate=0.01, max_depth=10, seed=1, nthread=4