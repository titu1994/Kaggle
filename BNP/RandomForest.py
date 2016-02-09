import sklearn.ensemble as ensemble
import BNP.DataClean as dc
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
import csv

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

    # {'max_depth': 11, 'n_estimators': 75}
    model = ensemble.RandomForestRegressor(n_estimators=75, max_depth=11, random_state=0)
    """
    params = {"max_depth" : [9,10,11,12,13], "n_estimators" : [25, 35, 40, 50, 75], }

    grid = GridSearchCV(model, param_grid=params, scoring=LL, n_jobs=-1, cv=3, verbose=20)
    grid.fit(trainX, trainY)

    print("Best parameters found by grid search:")
    print(grid.best_params_)
    print("Best CV score:")
    print(grid.best_score_)
    """
    yPred = model.predict(testX)

    min_y_pred = min(yPred)
    max_y_pred = max(yPred)
    min_y_train = min(trainY)
    max_y_train = max(trainY)

    for i in range(len(yPred)):
        yPred[i] = min_y_train + (((yPred[i] - min_y_pred)/(max_y_pred - min_y_pred))*(max_y_train - min_y_train))

    predictions_file = open("random_forest.csv", "w", newline="")
    open_file_object = csv.writer(predictions_file)
    open_file_object.writerow(["ID", "PredictedProb"])
    open_file_object.writerows(zip(testData[:, 0].astype(int), yPred))
    predictions_file.close()