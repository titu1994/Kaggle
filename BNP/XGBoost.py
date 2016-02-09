import xgboost as xgb
import BNP.DataClean as dc
import csv

from sklearn.metrics import log_loss, make_scorer

traindf, testdf = dc.loadTrain(), dc.loadTest()
traindf, testdf = dc.cleanData(traindf, testdf, describe=False)

trainData, testData = dc.convertPandasDataFrameToNumpyArray(traindf), dc.convertPandasDataFrameToNumpyArray(testdf)

trainX = trainData[:, 2:]
trainY = trainData[:, 1]

testX = testData[:, 1:]

# Parameter : Number of Trees
numTree = 6000

model = xgb.XGBRegressor(max_depth=6, learning_rate=0.01, n_estimators=numTree, subsample=0.75, colsample_bytree=0.9)

model.fit(trainX, trainY, eval_metric="logloss", eval_set=[(trainX[:1000], trainY[:1000]),], early_stopping_rounds=10)

yPred = model.predict(testX)

min_y_pred = min(yPred)
max_y_pred = max(yPred)
min_y_train = min(trainY)
max_y_train = max(trainY)

for i in range(len(yPred)):
    yPred[i] = min_y_train + (((yPred[i] - min_y_pred)/(max_y_pred - min_y_pred))*(max_y_train - min_y_train))

predictions_file = open("xgboost_result.csv", "w", newline="")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["ID", "PredictedProb"])
open_file_object.writerows(zip(testData[:, 0].astype(int), yPred))
predictions_file.close()