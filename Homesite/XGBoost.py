import xgboost as xgb
import Homesite.DataClean as dc
import MLScripts.Helpers as helpers

trainFrame = dc.cleanData(dc.loadTrainData(), describe=False)

testFrame = dc.cleanData(dc.loadTestData(), istest=True, describe=False)
trainFrame, testFrame = dc.postprocessObjects(trainFrame, testFrame)

trainData = dc.convertPandasDataFrameToNumpyArray(trainFrame)
print("Data loaded")

# max_depth=10, n_estimators=25, seed=0, learning_rate=0.025, silent=True, subsample=0.8, colsample_bytree=0.8
# max_depth=10, n_estimators=100, seed=0, learning_rate=0.1, silent=True, subsample=0.9, colsample_bytree=0.8

xgbtree = xgb.XGBClassifier(max_depth=10, n_estimators=300, seed=0, learning_rate=0.05, silent=True, subsample=0.9, colsample_bytree=0.9)

print("Training")
xgbtree.fit(trainData[:, 1:], trainData[:, 0], eval_metric="auc", verbose=True,
            eval_set=[(trainData[:1000, 1:], trainData[:1000, 0])], early_stopping_rounds=5)
print("Training finished")

testData = dc.convertPandasDataFrameToNumpyArray(testFrame)

preds = xgbtree.predict_proba(testData[:, 1:])[:, 1]

helpers.writeOutputFile("xgboost.csv", headerColumns=["QuoteNumber", "QuoteConversion_Flag"],
                        submissionRowsList=[testData[:, 0], preds], dtypes=[int, float])
