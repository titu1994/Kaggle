import sklearn.ensemble as ensemble
import BNP.DataClean as dc
import csv
import numpy as np

from sklearn.metrics import log_loss, make_scorer

if __name__ == "__main__":
    traindf, testdf = dc.loadTrain(), dc.loadTest()
    traindf, testdf = dc.cleanData(traindf, testdf, describe=False)

    trainData, testData = dc.convertPandasDataFrameToNumpyArray(traindf), dc.convertPandasDataFrameToNumpyArray(testdf)

    trainX = trainData[:, 2:]
    trainY = trainData[:, 1]

    testX = testData[:, 1:]

    print("Data loaded")

    LL  = make_scorer(log_loss, greater_is_better=False)

    # Averaging Random Fores { n_estimators=[25,25,25,50,75], max_depth=[11,12,10,11,11] }
    # {'max_depth': 11, 'n_estimators': 75}
    model1 = ensemble.RandomForestRegressor(n_estimators=25, max_depth=11, verbose=1, n_jobs=-1)
    model2 = ensemble.RandomForestRegressor(n_estimators=25, max_depth=12, verbose=1, n_jobs=-1)
    model3 = ensemble.RandomForestRegressor(n_estimators=25, max_depth=10, verbose=1, n_jobs=-1)
    model4 = ensemble.RandomForestRegressor(n_estimators=50, max_depth=11, verbose=1, n_jobs=-1)
    model5 = ensemble.RandomForestRegressor(n_estimators=75, max_depth=11, verbose=1, n_jobs=-1)

    models = [model1, model2, model3, model4, model5]
    weights = [2, 2, 2, 1, 1]

    assert(len(models) == len(weights))

    print("Begin fitting")

    for i, model in enumerate(models):
        model.fit(trainX, trainY)
        print("Finished fitting %d classifier" % (i+1))

    ypredTemp = np.zeros((len(models), testX.shape[0]))
    yPred = np.zeros((testX.shape[0],))

    for i, model in enumerate(models):
        ypredTemp[i] = model.predict(testX)
        print("Finished predicting %d classifier" % (i+1))

    for i in range(testX.shape[0]):
        for j in range(len(models)):
            yPred[i] += weights[j] * ypredTemp[j][i]

        yPred[i] = yPred[i] / sum(weights)

    print("Finished predicting values")

    """
    min_y_pred = min(yPred)
    max_y_pred = max(yPred)
    min_y_train = min(trainY)
    max_y_train = max(trainY)

    for i in range(len(yPred)):
        yPred[i] = min_y_train + (((yPred[i] - min_y_pred)/(max_y_pred - min_y_pred))*(max_y_train - min_y_train))

    print("Finished scaling predictions")
    """

    predictions_file = open("averageing_random_forest.csv", "w", newline="")
    open_file_object = csv.writer(predictions_file)
    open_file_object.writerow(["ID", "PredictedProb"])
    open_file_object.writerows(zip(testData[:, 0].astype(int), yPred))
    predictions_file.close()