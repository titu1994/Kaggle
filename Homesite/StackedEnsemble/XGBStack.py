import MLScripts.Ensemble.Stacking as stack
import MLScripts.Helpers as helpers
import Homesite.DataClean as dc
import xgboost as xgb
import sklearn.linear_model as linear

trainFrame = dc.cleanData(dc.loadTrainData(), describe=False)
testFrame = dc.cleanData(dc.loadTestData(), istest=True, describe=False)
trainFrame, testFrame = dc.postprocessObjects(trainFrame, testFrame)

trainData = dc.convertPandasDataFrameToNumpyArray(trainFrame)
print("Data loaded")

xgb1 = xgb.XGBClassifier(max_depth=10, n_estimators=50, seed=0, learning_rate=0.025, subsample=0.8, colsample_bytree=0.8)
xgb2 = xgb.XGBClassifier(max_depth=10, n_estimators=100, seed=0, learning_rate=0.1, subsample=0.9, colsample_bytree=0.8)
xgb3 = xgb.XGBClassifier(max_depth=6, n_estimators=6000, seed=0, learning_rate=0.01, subsample=0.83, colsample_bytree=0.77)

#xgbBlend = xgb.XGBClassifier(max_depth=6, n_estimators=6000, seed=0, learning_rate=0.1, subsample=0.83, colsample_bytree=0.77)
blend = linear.LogisticRegression(random_state=0, verbose=True, tol=1e-6)
clfs = [xgb1, xgb2, xgb3]

ensemble = stack.StackedClassifier(clfs, blend, verbose=True)

trainX = trainData[:, 1:]
trainY = trainData[:, 0]

ensemble.fit(trainX, trainY, xgb_eval_metric="auc", xgb_eval_set=[(trainData[:1000, 1:], trainData[:1000, 0])])

testData = dc.convertPandasDataFrameToNumpyArray(testFrame)
preds = ensemble.predict(testData[:, 1:])[:, 1]

helpers.writeOutputFile("stacked_ensemble.csv", headerColumns=["QuoteNumber", "QuoteConversion_Flag"],
                        submissionRowsList=[testData[:, 0], preds], dtypes=[int, float])
