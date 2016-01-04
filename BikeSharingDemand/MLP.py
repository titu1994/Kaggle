import csv
import MLScripts as Metrics
import seaborn as sns
import sknn.mlp as nn

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
"""

# Learning rules L: sgd, momentum, nesterov, adadelta, adagrad or rmsprop
mlp = nn.Regressor(layers=[nn.Layer("Rectifier", units=7),nn.Layer("Rectifier", units=8),
                           nn.Layer("Rectifier", units=9),
                           nn.Layer("Rectifier", units=8),nn.Layer("Rectifier", units=7),
                           nn.Layer("Linear", units=1)],
                   learning_rate=0.1, random_state=1, n_iter=100, verbose=True, learning_rule="adagrad",
                   valid_size=0.1, batch_size=500)
#cvCount = 10
#crossvalidation = metrics.crossValidationScore(ensemble.GradientBoostingRegressor(random_state=1), trainX, trainY, cvCount=cvCount)

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

mlp.fit(xTrain, yTrain)

yPredict = mlp.predict(xTest)

#trainingAccuracy = metrics.trainingAccuracy(yTest, yPredict)
rmse = Metrics.rmse(yTest, yPredict)
nrmse = Metrics.nrmse(yTest, yPredict)

for i, x in enumerate(yPredict):
    if yPredict[i] < 0:
        print("yActual : ", yTest[i], " yPredicted : ", yPredict[i])
        yPredict[i] = -yPredict[i]

logloss = Metrics.rmsle(yTest, yPredict)

print("\nGradient Boosting Forest Score : ", mlp.score(xTrain, yTrain),
      #"\nTraining Accuracy : ", trainingAccuracy,
      "\nRoot Mean Squared Error : ", rmse, "\nNormalized RMSE : ", nrmse,
      #"\nKFold Accuracy : ", kfoldAccuracy
      "\nRoot Mean Logarithmic Squared Error : ", logloss
      )

"""
Final Model
"""
finalModel = mlp
#finalModel = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.01, max_depth=10, seed=1, nthread=4)
finalModel.fit(trainX, trainY, w=finalModel.weights)

finalPredicted = finalModel.predict(testX)
for i, x in enumerate(finalPredicted):
    finalPredicted[i] = finalPredicted[i] if finalPredicted[i] >= 0 else -finalPredicted[i]

f = open("bike-sharing-result.csv", "w", newline="")
csvWriter = csv.writer(f)
csvWriter.writerow(["datetime", "count"])

for datetime, count in zip(testData[:, 0], finalPredicted):
    csvWriter.writerow([datetime, int(count)])
f.close()

