import xgboost as xgb
import Homesite.DataClean as dc
import csv

trainFrame = dc.cleanData(dc.loadTrainData(), describe=False)
trainData = dc.convertPandasDataFrameToNumpyArray(trainFrame)
print("Data loaded")

xgbtree = xgb.XGBClassifier(max_depth=10, n_estimators=25, seed=0, learning_rate=0.025, silent=True)

print("Training")
xgbtree.fit(trainData[:, 1:], trainData[:, 0], eval_metric="auc", verbose=True,)
print("Training finished")

testFrame = dc.cleanData(dc.loadTestData(), istest=True, describe=False)
testData = dc.convertPandasDataFrameToNumpyArray(testFrame)

preds = xgbtree.predict_proba(testData[:, 1:])[:, 1]

f = open("xgboost.csv", "w", newline="")
csvWriter = csv.writer(f)
csvWriter.writerow(["QuoteNumber", "QuoteConversion_Flag"])

for qn, qflag in zip(testData[:, 0], preds):
    csvWriter.writerow([int(qn), float(qflag)])

f.close()


