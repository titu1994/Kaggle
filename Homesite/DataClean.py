from MLScripts.CleaningUtils import *
import pandas as pd
import numpy as np
import sklearn.preprocessing as preproc


def loadTrainData(describe=False):
    return loadData(r"C:\Users\Yue\PycharmProjects\Kaggle\Homesite\Data\train.csv", describe=describe)

def loadTestData(describe=False):
    return loadData(r"C:\Users\Yue\PycharmProjects\Kaggle\Homesite\Data\test.csv", describe=describe)

def encodeQuoteFlag(df):
    y = df.QuoteConversion_Flag.values
    encoder = preproc.LabelEncoder()
    y = encoder.fit_transform(y).astype(np.int32)
    df["QuoteConversion_Flag"] = y
    return df, len(encoder.classes_)

def addDateColumn(df):
    df['Date'] = pd.to_datetime(pd.Series(df['Original_Quote_Date']))
    df = df.drop('Original_Quote_Date', axis=1)

    df['Year'] = df['Date'].apply(lambda x: int(str(x)[:4]))
    df['Month'] = df['Date'].apply(lambda x: int(str(x)[5:7]))
    df['Weekday'] = df['Date'].dt.dayofweek

    df = df.drop('Date', axis=1)
    df = df.fillna(-1)
    return df


def postprocessObjects(dfTrain, dfTest):
    for f in dfTrain.columns:
        if dfTrain[f].dtype=='object':
            #print(f)
            lbl = preproc.LabelEncoder()
            lbl.fit(list(dfTrain[f].values) + list(dfTest[f].values))
            dfTrain[f] = lbl.transform(list(dfTrain[f].values))
            dfTest[f] = lbl.transform(list(dfTest[f].values))
    return (dfTrain, dfTest)

def cleanData(df, istest=False, describe=False, ):
    df = addDateColumn(df)

    if not istest: df = dropUnimportantFeatures(df, ['QuoteNumber'])
    if describe: describeDataframe(df)
    return df

def cleanDataNN(df, istest=False, describe=False, ):
    noOfClasses = 0
    df = addDateColumn(df)

    if not istest:
        df = dropUnimportantFeatures(df, ['QuoteNumber'])
        df, noOfClasses = encodeQuoteFlag(df)

    if describe: describeDataframe(df)
    return df, noOfClasses



