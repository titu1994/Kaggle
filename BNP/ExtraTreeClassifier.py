import sklearn.ensemble as ensemble
import BNP.DataClean as dc
import csv

traindf, testdf = dc.loadTrain(), dc.loadTest()
traindf, testdf = dc.cleanData(traindf, testdf, describe=False)

trainData, testData = dc.convertPandasDataFrameToNumpyArray(traindf), dc.convertPandasDataFrameToNumpyArray(testdf)

trainX = trainData[:, 2:]
trainY = trainData[:, 1]

testX = testData[:, 1:]

# Parameter : Number of Trees
numTree = 700
max_depth = 60

print("Begin training")
model = ensemble.ExtraTreesClassifier(n_estimators=700, max_depth=55,  criterion="entropy", max_features=45,
                                      min_samples_split=3, min_samples_leaf=4, n_jobs=-1, verbose=2)
model.fit(trainX, trainY)

yPred = model.predict_proba(testX)[:, 1]

"""min_y_pred = min(yPred)
max_y_pred = max(yPred)
min_y_train = min(trainY)
max_y_train = max(trainY)

for i in range(len(yPred)):
    yPred[i] = min_y_train + (((yPred[i] - min_y_pred)/(max_y_pred - min_y_pred))*(max_y_train - min_y_train))
"""
predictions_file = open("extra_tree_result.csv", "w", newline="")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["ID", "PredictedProb"])
open_file_object.writerows(zip(testData[:, 0].astype(int), yPred))
predictions_file.close()
print("Finished")
