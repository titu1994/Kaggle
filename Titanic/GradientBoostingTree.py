import csv

import seaborn as sns
import sklearn.ensemble as ensemble

import Titanic.DataClean as dataclean

sns.set_style("whitegrid")


trainFrame = dataclean.cleanDataSet(dataclean.loadTrainData())
#dataclean.displayRelationsRelations(trainSet)
trainData = dataclean.convertPandasDataFrameToNumpyArray(trainFrame)

testFrame = dataclean.cleanDataSet(dataclean.loadTestData())
testData = dataclean.convertPandasDataFrameToNumpyArray(testFrame)
#                                                                                  0.0960              random_state=600
randomForest = ensemble.GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, warm_start=True)
"""randomForest = ensemble.AdaBoostClassifier(ensemble.GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, max_depth=4,random_state=600, warm_start=True),
                                           n_estimators=100, learning_rate=0.1, random_state=600)"""

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
crossvalidation = Metrics.crossValidationScore(randomForest, trainX, trainY, cvCount=cvCount)

xTrain, xTest, yTrain, yTest = Metrics.traintestSplit(trainX, trainY, randomState=1)
"""
accuracyScores = []
rngs = metrics.frange(0.01, 0.2)
for rng in rngs:
    icvTree = ensemble.GradientBoostingClassifier(n_estimators=100, learning_rate=rng, max_depth=4, random_state=1,
                                                  warm_start=True)
    icvTree.fit(xTrain, yTrain)
    iyPreds = icvTree.predict(xTest)
    accuracyScores.append(metrics.trainingAccuracy(yTest, iyPreds))
sns.plt.plot(rngs, accuracyScores, alpha=0.7)
sns.plt.xlabel("Learning Rate")
sns.plt.ylabel("Accuracy Score")
sns.plt.show()
"""

crossvalidationTree = ensemble.GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=1, warm_start=True)
crossvalidationTree.fit(xTrain, yTrain)

yPredict = crossvalidationTree.predict(xTest)

trainingAccuracy = Metrics.trainingAccuracy(yTest, yPredict)
rmse = Metrics.rmse(yTest, yPredict)
nrmse = Metrics.nrmse(yTest, yPredict)

predictors = dataclean.getFeatureNames()[2:]
kfoldAccuracy = Metrics.measureKFoldAccuracy(trainFrame, crossvalidationTree, predictors, outputClass="Survived", outputClause="Survived", kFolds=10)

print("Max Cross Validation Score : ", crossvalidation.max(), "\nAverage Cross Validation Score : ", crossvalidation.mean(),
      "\nGradient Boosting Forest Score : ", crossvalidationTree.score(xTrain, yTrain),
      "\nTraining Accuracy : ", trainingAccuracy,
      "\nRoot Mean Squared Error : ", rmse, "\nNormalized RMSE : ", nrmse,
      "\nKFold Accuracy : ", kfoldAccuracy)

featureNames = dataclean.getFeatureNames()[2:]
featureImportances = [(feature, importance) for feature, importance in zip(featureNames, crossvalidationTree.feature_importances_)]
featureImportances = sorted(featureImportances, key=lambda x: x[1], reverse=True)
print("Feature Importances : \n", featureImportances)

#print(len(resultsY))

#print("Predicted Results : \n", resultsY)

"""with open("Titanic.dot", "w") as file:
    tree.export_graphviz(decisionTree, out_file=file,feature_names=dataclean.getFeatureNames()[2:], class_names=["Did Not Survive", "Survived"], filled=True,
                         rounded=True)

check_call(["dot", "-Tpdf", "Titanic.dot", "-o", "Titanic.pdf"])"""

randomForest = ensemble.GradientBoostingClassifier(n_estimators=100, learning_rate=0.05,max_depth=4, random_state=1, warm_start=True)
randomForest.fit(trainX, trainY)

finalPredictedY = randomForest.predict(testX)

f = open("titanic-result.csv", "w", newline="")
csvWriter = csv.writer(f)
csvWriter.writerow(["PassengerId", "Survived"])

for pid, survive in zip(testData[:, 0], finalPredictedY):
    csvWriter.writerow([int(pid), int(survive)])
f.close()

