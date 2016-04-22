import xgboost as xgb
import BNP.DataClean as dc
import csv
from MLScripts.Helpers import printXGBFeatureImportances

from sklearn.metrics import log_loss, make_scorer

traindf, testdf = dc.loadTrain(), dc.loadTest()
traindf, testdf = dc.cleanData(traindf, testdf, describe=False)

#best_feature_indices = [43, 95, 10, 8, 34, 28, 12, 19, 434, 61, 116, 53, 114, 115, 113, 433, 282, 432, 112, 274, 307, 273, 161, 306, 127, 303, 304, 300, 452, 301, 299, 302, 422, 298, 309, 308, 277, 436, 305, 272, 448, 119, 68, 439, 120, 98, 450, 283, 130, 83, 67, 30, 60, 73, 3, 92, 100, 290, 93, 24, 108, 59, 101, 84, 4, 122, 96, 65, 74, 49, 20, 48, 31, 105, 0, 85, 14, 111, 71, 75, 357, 16, 39, 40, 46, 7, 76, 107, 103, 6, 54, 82, 106, 1, 66, 70, 104, 58, 87, 90, 33, 38, 23, 29, 97, 5, 50, 135, 2, 94, 44, 36, 22, 99, 69, 9, 47, 21, 13, 102, 88, 86, 26, 45, 72, 79, 37, 52, 80, 62, 51, 18, 89, 77, 11, 27, 110, 42, 17, 63, 64, 78, 91, 57, 55, 35, 15, 56, 25, 41, 81, 125, 220, 109, 384, 281, 129, 453, 437, 137, 191, 147, 217, 196, 271, 126, 232, 131, 241, 133, 142, 278, 139, 136, 141, 254, 401, 132, 251, 296, 265, 150, 255, 230, 455, 250, 270, 207, 248, 123, 198, 163, 211, 183, 238, 138, 169, 117, 375, 344, 206, 395, 420, 360, 124, 151, 244, 180, 166, 259, 279, 218, 143, 235, 128, 194, 177, 402, 231, 269, 175, 327, 280, 228, 275, 454, 292, 331, 219, 186, 410, 345, 239, 201, 178, 247, 200, 261, 192, 257, 260, 264, 189, 237, 146, 240, 176, 156, 221, 205, 404, 134, 268, 262, 215, 253, 266]
#traindf = traindf[traindf.columns[best_feature_indices]]
#testdf = testdf[testdf.column[best_feature_indices]]

trainData, testData = dc.convertPandasDataFrameToNumpyArray(traindf), dc.convertPandasDataFrameToNumpyArray(testdf)

trainX = trainData[:, 2:]
trainY = trainData[:, 1]

testX = testData[:, 1:]

# Parameter : Number of Trees
numTree = 100

model = xgb.XGBClassifier(max_depth=7, learning_rate=0.1, n_estimators=numTree, subsample=0.75, colsample_bytree=0.68)
model.fit(trainX, trainY, eval_metric="auc", eval_set=[(trainX[:1000], trainY[:1000]),], early_stopping_rounds=100)

printXGBFeatureImportances(traindf.columns[2:], model, save_figure=True)

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
