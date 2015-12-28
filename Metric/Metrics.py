import math
import numpy as np
import sklearn.metrics as metrics
import sklearn.cross_validation as crossVal
import ml_metrics as mlmetrics


def rmse(yTrue, yPredicted):
    """
    Calculates Root Mean Squared Error

    :param yTrue: list (int, float)
    :param yPredicted: list (int, float)
    :return: RMSE score
    """
    return metrics.mean_squared_error(yTrue, yPredicted)**0.5

def nrmse(yTrue, yPredicted):
    """
    Calculates Normalized Root Mean Squared Error

    :param yTrue: list (int, float)
    :param yPredicted: list (int, float)
    :return: Normalized RMSE score
    """
    return rmse(yTrue, yPredicted) / (max(yTrue) - min(yTrue))

def rmsle(y, y_pred):
    """
    Calculate Root Mean Squared Logarithmic Error

    :param y: list (int, float)
    :param y_pred: list (int, float)
    :return: RMSLE score
    """
    terms_to_sum = [(math.log(y_pred[i] + 1) - math.log(y[i] + 1)) ** 2.0 for i,pred in enumerate(y_pred)]
    return (sum(terms_to_sum) * (1.0/len(y))) ** 0.5

def rmsle2(y, ypred):
    """
    Calculate Root Mean Squared Logarithmic Error/
    Note : Uses external library - ml_metrics - which is more stable than above version

    :param y: list (int, float)
    :param y_pred: list (int, float)
    :return: RMSLE score
    """
    return mlmetrics.rmsle(y, ypred)

def trainingAccuracy(yTest, yPredicted):
    """
    Calculates Accuracy Score

    :param yTest: list (int, float)
    :param yPredicted: list (int, float)
    :return: accuracy score
    """
    return metrics.accuracy_score(yTest, yPredicted)

def traintestSplit(X, Y, trainPercent=0.75, randomState=None):
    """
    Splits input and output class data sets into separate train and test sets.

    :param X: array (int, float), list (int, float)
    :param Y: array (int, float), list (int, float)
    :param trainPercent (Optional, Default = 0.75) : Percentage of data set that will be used to create training set.
    :param randomState (Optional, Default = False) : Random state is used to determine how the data set is split
    :return: (xTrain, xTest, yTrain, yTest)
    """
    xTrain, xTest, yTrain, yTest = crossVal.train_test_split(X, Y, train_size=trainPercent, random_state=randomState)
    return (xTrain, xTest, yTrain, yTest)

def frange(start, end, noOfPartitions=-1):
    """
    range() for floating point numbers, dividing the range into floating point increments

    :param start: starting value (int, float)
    :param end: ending value (int, float)
    :param noOfPartitions (Optional, Default =  -1) : Decides the no of partitions to split the range. -1 indicates even split
    :return: floating point range
    """
    if noOfPartitions == -1:
        noOfPartitions = end / start
        return np.linspace(start, stop=end, num=noOfPartitions)
    else:
        return np.linspace(start, stop=end, num=noOfPartitions)

def crossValidationScore(algo, trainX, trainY, cvCount = 10):
    """
    :param algo: Classification / Regression Algorithm
    :param trainX: numpy.array of trainX values
    :param trainY: numpy.array of trainY values
    :param cvCount: Cross validation count
    :return: cross_validation_score.
    """
    return crossVal.cross_val_score(algo, trainX, trainY, cv=cvCount)

def crossValidationScoreValues(cv):
    """
    Computes the max and mean values of cross validation
    :param cv: cross_validation_score result
    :return: (Maximum value of cv, Average value of cv)
    """
    return cv.max(), cv.mean()

def __kFold(dframe, algo, predictors, outputClass, nFolds = 3, randomState = 1):
    """
    Performs KFold Evaluation and testing on the dataset
    :param dframe: Pandas Dataset. eg traindf
    :param algo: Machine learning algorithm
    :param predictors: list of input variables
    :param outputClasses: output class lable
    :param nFolds: no of folds
    :param randomState: initial random setting
    :return: list of predictions
    """
    predictions = []
    kf = crossVal.KFold(dframe.shape[0], nFolds, random_state=randomState)

    for train, test in kf:
        trainPredictors = dframe[predictors].iloc[train, :]
        trainTargetOutputs = dframe[outputClass].iloc[train]
        try:
            algo.fit(trainPredictors, trainTargetOutputs)
        except ValueError:
            trainPredictors = np.copy(trainPredictors, order="C")
            algo.fit(trainPredictors, trainTargetOutputs)
        testPredictions = algo.predict(dframe[predictors].iloc[test, :])
        predictions.append(testPredictions)

    return predictions

def __concatenatePredictions(predictions, ax=0):
    """
    Concatenates list of predictions into a single list
    :param predictions: list of predictions from kFolds
    :param ax: axis : 0 = Column, 1 = Rows
    :return: concatenated list
    """
    return np.concatenate(predictions, axis=ax)


def __mapPredictionsValues(predictions, mapperFn, ):
    np.apply_along_axis(mapperFn, 1, predictions)

def __accuracyOfPredictedValues(predictions, dframe, outputClause):
    """
    Determines accuracy of the predictions
    :param predictions: concatenated prediction list which has been mapped to discrete values
    :param dframe: pandas dataframe, eg traindf
    :param outputClause: outputclass where accuracy will be determined from
    :return: accuracy as a floating point value
    """
    accuracy = sum(predictions[predictions == dframe[outputClause]]) / len(predictions)
    return accuracy

def thresholdMapper(predictions, theta=0.5):
    """
    Mapping function with [theta (threshold) = 0.5]

    :param predictions: list of predicted values
    :param theta (Optional, Default = 0.5) : Min value that the prediction must have to be accepted to "high" output
    :return: mapped prediction list
    """
    for  prediction in (predictions):
        yield 1 if prediction > theta else 0

def measureKFoldAccuracy(dframe, algo, predictors, outputClass, outputClause, mapperFn=None, kFolds = 3, randomState = 1):
    """
    Performs K-Fold analysis on given data and algorithm

    :param dframe: pandas dataframe
    :param algo: scikit algorithm (Machine Learning)
    :param predictors: list of labels that are input classes in dframe
    :param outputClass: string name of output class
    :param outputClause: string name of output class (untested supported for multiple output classes)
    :param mapperFn (Optional, Default = None) : MapperFunc of the type func(predictions), returns a list of mapped predictions
    :param kFolds (Optional, Default = 3) : Number of folds to generate
    :param randomState (Optional, Default = 1) : Seed value for randomness. Default behaviour is to get consistent samples
    :return: K-Fold accuracy
    """
    predictions = __kFold(dframe, algo, predictors, outputClass, nFolds=kFolds, randomState=randomState)
    predictions = __concatenatePredictions(predictions)
    if mapperFn is not None:
        __mapPredictionsValues(predictions, mapperFn)
    return __accuracyOfPredictedValues(predictions, dframe, outputClause)

def printAllScores(crossvalidationScore, learningAlgo, trainX, trainY, trainAccuracy, rmse, nrmse, kFoldAccuracy):
    print("Max Cross Validation Score : ", crossvalidationScore.max(), "\nAverage Cross Validation Score : ", crossvalidationScore.mean(),
      "\nGradient Boosting Forest Score : ", learningAlgo.score(trainX, trainY),
      "\nTraining Accuracy : ", trainAccuracy,
      "\nRoot Mean Squared Error : ", rmse, "\nNormalized RMSE : ", nrmse,
      "\nKFold Accuracy : ", kFoldAccuracy)




def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)

def mapk(actual, predicted, k=10):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])

