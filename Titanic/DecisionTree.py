import csv
from subprocess import check_call

import seaborn as sns
import sklearn.tree as tree

sns.set_style("white")

import Titanic.DataClean as dataclean

trainFrame = dataclean.cleanDataSet(dataclean.loadTrainData())
#dataclean.displayRelationsRelations(trainSet)
trainData = dataclean.convertPandasDataFrameToNumpyArray(trainFrame)

testFrame = dataclean.cleanDataSet(dataclean.loadTestData())
testData = dataclean.convertPandasDataFrameToNumpyArray(testFrame)
#                                                                                  0.0960              random_state=600
decisionTree = tree.DecisionTreeClassifier(max_depth=4)
"""randomForest = ensemble.AdaBoostClassifier(ensemble.GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, max_depth=4,random_state=600, warm_start=True),
                                           n_estimators=100, learning_rate=0.1, random_state=600)"""

trainX = trainData[:, 2:]
trainY = trainData[:, 1]

decisionTree.fit(trainX, trainY)

testX = testData[:, 1:]
resultsY = decisionTree.predict(testX)

"""
Cross Validation
"""
# Cross Validation
cvCount = 10
crossvalidation = Metrics.crossValidationScore(decisionTree, trainX, trainY)

xTrain, xTest, yTrain, yTest = Metrics.traintestSplit(trainX, trainY, randomState=1)

"""
accuracyScores = []
depths = range(1, 26)
for randomState in range(0, 11):
    for depth in depths:
        icvTree = tree.DecisionTreeClassifier(max_depth=depth, random_state=randomState)
        icvTree.fit(xTrain, yTrain)
        iyPreds = icvTree.predict(xTest)
        accuracyScores.append(metrics.trainingAccuracy(yTest, iyPreds))
    sns.plt.plot(depths, accuracyScores, alpha=0.7)
    sns.plt.xlabel("Depth Values")
    sns.plt.ylabel("Accuracy Scores")
    print("Random State : ", randomState)
    sns.plt.show()
    accuracyScores.clear()
"""

decisionTree = tree.DecisionTreeClassifier(random_state=0, presort=True, max_features=6, max_depth=4)

""" Grid Search for best possible params
Best Params :  {'presort': True, 'max_features': 6, 'max_depth': 4}

if __name__ == "__main__":
    params = {"max_depth" : [2, 3, 4, 5, 6, 7, 8], "max_features" : [None, 2, 3, 4, 5, 6, 7, 8, 9, "auto", "log2", ], "presort" : [True, False]}
    clf = GridSearchCV(decisionTree, params, verbose=1, n_jobs=4, cv=10)
    clf.fit(xTrain, yTrain)

    print("GridSearch : \n", "Best Estimator : ", clf.best_estimator_,
        "\nBest Params : ", clf.best_params_, "\nBest Score", clf.best_score_)

"""

decisionTree.fit(xTrain, yTrain)

predictedY = decisionTree.predict(xTest)

trainingAccuracy = Metrics.trainingAccuracy(yTest, predictedY)
rmse = Metrics.rmse(yTest, predictedY)
nrmse = Metrics.nrmse(yTest, predictedY)

predictors = dataclean.getFeatureNames()[2:]
kfoldAccuracy = Metrics.measureKFoldAccuracy(trainFrame, decisionTree, predictors, outputClass="Survived", outputClause="Survived", kFolds=10)

print("Max Cross Validation Score : ", crossvalidation.max(), "\nAverage Cross Validation Score : ", crossvalidation.mean(),
      "\nExtraTreeCLassifier Score : ", decisionTree.score(xTrain, yTrain),
      "\nTraining Accuracy : ", trainingAccuracy,
      "\nRoot Mean Squared Error : ", rmse, "\nNormalized RMSE : ", nrmse,
      "\nKFold Accuracy : ", kfoldAccuracy)

featureNames = dataclean.getFeatureNames()[2:]
featureImportances = [(feature, importance) for feature, importance in zip(featureNames, decisionTree.feature_importances_)]
featureImportances = sorted(featureImportances, key=lambda x: x[1], reverse=True)
print("Feature Importances : \n", featureImportances)

#print(len(resultsY))

#print("Predicted Results : \n", resultsY)

with open("Titanic.dot", "w") as file:
    tree.export_graphviz(decisionTree, out_file=file,feature_names=dataclean.getFeatureNames()[2:], class_names=["Did Not Survive", "Survived"], filled=True,
                         rounded=True)

check_call(["dot", "-Tpdf", "Titanic.dot", "-o", "Titanic.pdf"])

decisionTree = tree.DecisionTreeClassifier(random_state=0, presort=True, max_features=6, max_depth=4)
decisionTree.fit(trainX, trainY)

finalPredictedY = decisionTree.predict(testX)

f = open("titanic-result.csv", "w", newline="")
csvWriter = csv.writer(f)
csvWriter.writerow(["PassengerId", "Survived"])

for pid, survive in zip(testData[:, 0], finalPredictedY):
    csvWriter.writerow([int(pid), int(survive)])
f.close()

