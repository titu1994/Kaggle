import csv
import seaborn as sns
import pickle
from sklearn.grid_search import GridSearchCV

sns.set_style("white")
import xgboost as xgb
from sklearn.cross_validation import cross_val_score

import Metric.Metrics as metric
import Titanic.DataClean as dataclean

trainFrame = dataclean.cleanDataSet(dataclean.loadTrainData())
#dataclean.displayRelationsRelations(trainSet)
trainData = dataclean.convertPandasDataFrameToNumpyArray(trainFrame)

testFrame = dataclean.cleanDataSet(dataclean.loadTestData())
testData = dataclean.convertPandasDataFrameToNumpyArray(testFrame)
#                                                                                  0.0960              random_state=600
#randomForest = ensemble.GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, warm_start=True)

randomForest = xgb.XGBClassifier(n_estimators=100, learning_rate=0.02, nthread=4, max_depth=6 )

trainX = trainData[:, 2:]
trainY = trainData[:, 1]

#randomForest.fit(trainX, trainY)

testX = testData[:, 1:]
#resultsY = randomForest.predict(testX)

"""
Cross Validation
"""
# Cross Validation
cvCount = 10
crossvalidation = cross_val_score(randomForest, trainX, trainY, cv = cvCount, scoring="accuracy")

xTrain, xTest, yTrain, yTest = metric.traintestSplit(trainX, trainY)

"""
accuracyScores = []
depths = range(1, 26)
for depth in depths:
    icvTree = xgb.XGBClassifier(max_depth=depth, n_estimators=100, nthread=4, seed=0)
    icvTree.fit(xTrain, yTrain)
    iyPreds = icvTree.predict(xTest)
    accuracyScores.append(metric.trainingAccuracy(yTest, iyPreds))
sns.plt.plot(depths, accuracyScores, alpha=0.7)
sns.plt.xlabel("Depth Values")
sns.plt.ylabel("Accuracy Scores")
sns.plt.show()
"""

crossvalidationTree = xgb.XGBClassifier(n_estimators=100, learning_rate=0.02, nthread=4, max_depth=6, gamma=1)

#GridSearch to find Best params for evaluating this model :
# {'max_depth': 5, 'n_estimators': 100, 'learning_rate': 0.01, 'gamma': 1}
# {'n_estimators': 350, 'learning_rate': 0.05, 'max_depth': 3, 'gamma': 0.5}
# {'gamma': 1, 'max_depth': 6, 'learning_rate': 0.02, 'n_estimators': 100}, seed = 1

"""
if __name__ == "__main__":
    params = {"max_depth" : [3,4,5,6,7,8], "n_estimators" : [50, 100, 150, 200, 250, 300, 350, 400], "learning_rate" : [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.5, 1], "gamma" : [0.1, 0.2, 0.5, 1]}
    clf = GridSearchCV(crossvalidationTree, params, verbose=1, n_jobs=4, cv=10)
    clf.fit(xTrain, yTrain)

    print("GridSearch : \n", "Best Estimator : ", clf.best_estimator_,
        "\nBest Params : ", clf.best_params_, "\nBest Score", clf.best_score_)
"""

evalSet = [(xTrain, yTrain), (xTest, yTest)]
crossvalidationTree.fit(xTrain, yTrain, eval_set=evalSet, eval_metric="auc")

predictedY = crossvalidationTree.predict(xTest)

trainingAccuracy = metric.trainingAccuracy(yTest, predictedY)
rmse = metric.rmse(yTest, predictedY)
nrmse = metric.nrmse(yTest, predictedY)

predictors = dataclean.getFeatureNames()[2:]
kfoldAccuracy = metric.measureKFoldAccuracy(trainFrame, crossvalidationTree, predictors, outputClass="Survived", outputClause="Survived", kFolds=10)

print("Max Cross Validation Score : ", crossvalidation.max(), "\nAverage Cross Validation Score : ", crossvalidation.mean(),
      "\nTraining Accuracy : ", trainingAccuracy,
      "\nRoot Mean Squared Error : ", rmse, "\nNormalized RMSE : ", nrmse,
      "\nKFold Accuracy : ", kfoldAccuracy)

"""
featureNames = dataclean.getFeatureNames()[2:]
featureImportances = [(feature, importance) for feature, importance in zip(featureNames, crossvalidationTree.)]
featureImportances = sorted(featureImportances, key=lambda x: x[1], reverse=True)
print("Feature Importances : \n", featureImportances)
"""

"""
Plots
"""

"""
featureNames = dataclean.getFeatureNames()[2:]
importances = xgb.plot_importance(randomForest)
sns.plt.show()

#tree = xgb.plot_tree(randomForest, num_trees=2)
#sns.plt.show()

"""

"""
with open("Titanic-Extreem.dot", "w") as file:
    val = xgb.to_graphviz(randomForest, num_trees=2)
    val.save("Titanic-Extreem.dot")

check_call(["dot", "-Tpdf", "Titanic-Extreem.dot", "-o", "Titanic-Extreem.pdf"])
"""

finalModel = xgb.XGBClassifier(n_estimators=100, learning_rate=0.02, nthread=4, max_depth=6, gamma=1)
finalModel.fit(trainX[400:, :], trainY[400:])
finalModel.fit(trainX[0:400], trainY[0:400])

finalPreditedY = finalModel.predict(testX)

f = open("titanic-result.csv", "w", newline="")
csvWriter = csv.writer(f)
csvWriter.writerow(["PassengerId", "Survived"])

for pid, survive in zip(testData[:, 0], finalPreditedY):
    csvWriter.writerow([int(pid), int(survive)])
f.close()


"""
Save And Load Model


with open("XGB.pkl", "wb") as f:
    pickle.dump(finalPreditedY, f)

with open("XGB.pkl", "rb") as f:
    model = pickle.load(f)
"""