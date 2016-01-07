from MLScripts.CleaningUtils import *
import pandas as pd
import numpy as np
import sklearn.preprocessing as preproc


def loadTrainData(describe=False):
    return loadData(r"C:\Users\Yue\PycharmProjects\Kaggle\Homesite\Data\train.csv", describe=describe)

def loadTestData(describe=False):
    return loadData(r"C:\Users\Yue\PycharmProjects\Kaggle\Homesite\Data\test.csv", describe=describe)

def addDateColumn(df):
    df['Date'] = pd.to_datetime(pd.Series(df['Original_Quote_Date']))
    df = df.drop('Original_Quote_Date', axis=1)

    df['Year'] = df['Date'].apply(lambda x: int(str(x)[:4]))
    df['Month'] = df['Date'].apply(lambda x: int(str(x)[5:7]))
    df['Weekday'] = df['Date'].dt.dayofweek

    df = df.drop('Date', axis=1)
    df = df.fillna(-1)
    return df

def preprocessObjects(df):
    for f in df.columns:
        if df[f].dtype=='object':
            #print(f)
            lbl = preproc.LabelEncoder()
            lbl.fit(list(df[f].values))
            df[f] = lbl.transform(list(df[f].values))

def cleanData(df, istest=False, describe=False):
    df = addDateColumn(df)
    preprocessObjects(df)

    if not istest: df = dropUnimportantFeatures(df, ['QuoteNumber'])
    if describe: describeDataframe(df)
    return df



