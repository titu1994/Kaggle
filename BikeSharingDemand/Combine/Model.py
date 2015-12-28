import math
import numpy as nmp
import xgboost as xgb
import sklearn.ensemble as ensemble
from sklearn.grid_search import GridSearchCV
import Metric.Metrics as metrics

from BikeSharingDemand.DataClean import *

enableLogs = False

class Regressor:

    def __init__(self, regCas, regRegistered):
        """
        Regressor of type RF or XGBT which will separately learn CAS and REGISTERED vals

        :param regCas: XGBoost or RF
        :param regRegistered: XGBoost or RF
        """
        self.regCas = regCas
        self.regRegistered = regRegistered

    def fit(self, xs, ys):
        """
        Fits the two predictors with the given outputs
        :param xs: All rows other than 'count' and ['casual', 'registered']
        :param ys:  [all rows of casual, all rows of registered]
        """
        ys = ys[:, 0:2].astype(int)
        self.regCas.fit(xs, nmp.log(ys[:, 0] + 1))
        self.regRegistered.fit(xs, nmp.log(ys[:, 1] + 1))

    def predict(self, xs):
        """
        Predicts the rounded value of 'casual' and 'registered' as predicted by the predictors
        :param xs: All rows other than 'count' and ['casual', 'registered']
        :return:
        """
        yCas = nmp.exp(self.regCas.predict(xs)) - 1
        yRegistered = nmp.exp(self.regRegistered.predict(xs))
        yRegressor = nmp.around(yCas + yRegistered)
        yRegressor[yRegressor < 0] = 0
        return yRegressor


class Combiner:

    def __init__(self, regressors):
        self.regressors = regressors

        if enableLogs: print("Combined model created")

    def fit(self, xs, ys):
        """
        Fits the four predictors with the given outputs
        :param xs: All rows other than 'count' and ['casual', 'registered']
        :param ys:  [all rows of casual, all rows of registered]
        """
        if enableLogs: print("Combined model: Started fitting")

        for i, regressor in enumerate(self.regressors):
            if enableLogs: print("Combined Model: Began training model %d" % ((i+1)))

            regressor.fit(xs, ys)

            if enableLogs: print("Combined Model: Finished training model %d" % ((i+1)))


        if enableLogs: print("Combined model: Finished fitting")

    def predict(self, xs):
        """
        Predicts the count for given xs using all internal regressors
        :param xs:
        :return: list of predicted counts
        """
        if enableLogs: print("Combined model: Began predicting")

        ys = nmp.zeros(xs.shape[0])
        for  regressor in self.regressors:
            ys += regressor.predict(xs)

        ys *= 1.0 / len(self.regressors)
        ys = (nmp.around(ys))

        ys = [ys[i] if ys[i] >= 0 else 0 for i, _ in enumerate(ys)]

        if enableLogs: print("Combined model: Finished predicting")

        return ys

def selectXGBoost():
    regCas = xgb.XGBRegressor(max_depth=6, seed=0, n_estimators=100,)
    regRegistered = xgb.XGBRegressor(max_depth=6, seed=0, n_estimators=100, )
    regressor1 = Regressor(regCas, regRegistered)

    if enableLogs: print("XGBoost model creted")

    return regressor1

def selectRandomForest():                                       # min_samples_split=11
    regCas = ensemble.RandomForestRegressor(1000, random_state=0, min_samples_split=11, oob_score=False, n_jobs=-1)
    regRegistered = ensemble.RandomForestRegressor(1000, random_state=0, min_samples_split=11, oob_score=False, n_jobs=-1)
    regressor2 = Regressor(regCas, regRegistered)

    if enableLogs: print("Random Forest model created")

    return regressor2

def cleanTrainset(df, isRF=False, describe=False):
    traindf = df
    cols = traindf.columns.tolist()
    cols = cols[-3:] + cols[:-3]
    traindf = traindf[cols]

    traindf = factorizeColumns(traindf)

    addTimeColumn(traindf)
    addDateColumn(traindf)
    addSundayColumn(traindf)
    addHoursAndDaypartColumns(traindf)
    #addIdeal(traindf)
    #addPeakColumn(traindf)
    #addSticky(traindf)
    #if not isRF:

    #else:

    traindf = traindf.drop(["datetime", "time"], axis=1)
    if describe: describeDataframe(traindf)
    return traindf

def cleanTestSet(df, describe=False):
    testdf = df
    addTimeColumn(testdf)
    addDateColumn(testdf)
    addSundayColumn(testdf)
    addHoursAndDaypartColumns(testdf)
    #addIdeal(testdf)
    #addPeakColumn(testdf)
    #addSticky(testdf)

    testdf = dropUnimportantFeatures(testdf, True)
    if describe: describeDataframe(testdf)
    return testdf

"""
GridSearchParams for XGBoost
"""

#trainFrame = cleanTrainset(loadTrainData())
#trainData = convertPandasDataFrameToNumpyArray(trainFrame)

#trainX = trainData[:, 3:]
#trainYCasReg = trainData[:, 0:2] # [casual, registered]

def evalMetric(estimator, X, y):
    xTrain, xTest, yTrain, yTest = metrics.traintestSplit(X, y, randomState=1)
    estimator.fit(xTrain, yTrain)
    yPredicted = estimator.predict(xTest)
    return  metrics.rmsle2(yTest, yPredicted)

"""
{max_depth = 3, n_estimators=100, learning_rate=0.01}
if __name__ == "__main__":
    params = {"max_depth" : [3,4,5,6,7,8,9,10], "n_estimators" : [100, 200, 300, 400, 500], "learning_rate" : [0.01, 0.1, 0.2, 0.5]}
    tree = xgb.XGBRegressor(seed=0, nthread=2)
    clf = GridSearchCV(tree, params, verbose=1, n_jobs=2, cv=5, scoring=evalMetric)

    ys = trainYCasReg[:, 0].astype(int)
    clf.fit(trainX, ys)

    print("GridSearch : \n", "Best Estimator : ", clf.best_estimator_,
        "\nBest Params : ", clf.best_params_, "\nBest Score", clf.best_score_)

"""

"""
{min_samples_split=10, max_depth=3}

if __name__ == "__main__":
    params = {"max_depth" : [3,4,5,6,7,8,9,10],"min_samples_split" : [10,11,12,13,14,15,16]}
    tree = ensemble.RandomForestRegressor(1000, random_state=0, n_jobs=-1)
    clf = GridSearchCV(tree, params, verbose=1, n_jobs=2, cv=5, scoring=evalMetric)

    ys = trainYCasReg[:, 0].astype(int)
    clf.fit(trainX, ys)

    print("GridSearch : \n", "Best Estimator : ", clf.best_estimator_,
        "\nBest Params : ", clf.best_params_, "\nBest Score", clf.best_score_)
"""
