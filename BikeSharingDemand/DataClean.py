import pandas as pd
import numpy as np
from datetime import datetime
import seaborn as sns
sns.set_style("white")

def loadTrainData():
    df = pd.read_csv(r"C:\Users\Yue\PycharmProjects\Kaggle\BikeSharingDemand\Data\train.csv", header=0)
    return df

def loadTestData():
    df = pd.read_csv(r"C:\Users\Yue\PycharmProjects\Kaggle\BikeSharingDemand\Data\test.csv", header=0)
    return df

def getColNames(df):
    return df.columns.tolist()

def describeDataframe(df):
    print(df.info(), "\n")
    print(df.describe(), "\n")
    print(df.dtypes, "\n")

def factorizeColumns(df):
    df["weather"] = df["weather"].astype(int)
    df["holiday"] = df["holiday"].astype(int)
    df["workingday"] = df["workingday"].astype(int)
    df["season"] = df["season"].astype(int)
    return df

def addTimeColumn(df):
    def splitter(df):
        return df["datetime"].split(" ")[1]
    df["time"] = df[["datetime"]].apply(splitter, axis=1)
    df["time"] = df["time"].astype("category")

def addDateColumn(df):
    def strpDate(df):
        date = datetime.strptime(df["datetime"], "%Y-%m-%d %H:%M:%S")
        return date.weekday()
    df["day"] = df[["datetime"]].apply(strpDate, axis=1)
    df["day"] = df["day"].astype(int)

def addSundayColumn(df):
    df.loc[df.day == 6, "sunday"] = 1
    df.loc[df.sunday != 1, "sunday"] = 0
    df["sunday"] = df["sunday"].astype(int)

def addHoursAndDaypartColumns(df):
    def timeToHours(df):
        time = df["time"]
        return int(time.split(":")[0])
    df["hour"] = df[["time"]].apply(timeToHours, axis=1)
    df["daypart"] = 4
    df.loc[(df.hour < 10) & (df.hour > 3), "daypart"] = 1
    df.loc[(df.hour < 16) & (df.hour > 9), "daypart"] = 2
    df.loc[(df.hour < 22) & (df.hour > 15), "daypart"] = 3

    df["hour"] = df["hour"].astype("category")
    df["daypart"] = df["daypart"].astype("category")

    dt = pd.DatetimeIndex(df["datetime"])
    df.set_index(dt, inplace=True)

    df['month'] = dt.month
    df['year'] = dt.year
    df['dow'] = dt.dayofweek
    df['woy'] = dt.weekofyear

def addPeakColumn(df):
    def peaks(x):
        return (x['workingday'] == 1 and  ( x['hour'] == 8 or 17 <= x['hour'] <= 18 or 12 <= x['hour'] <= 12)) or (x['workingday'] == 0 and  10 <= x['hour'] <= 19)
    df["peak"] = df[["hour", "workingday"]].apply(peaks, axis=1)

def addIdeal(df):
    def ideals(x):
        return x['temp'] > 27 and x['windspeed'] < 30
    df['ideal'] = df[['temp', 'windspeed']].apply(ideals, axis=1)

def addSticky(df):
    def sticky(x):
        return x['workingday'] == 1 and x['humidity'] >= 60
    df['sticky'] = df[['humidity', 'workingday']].apply(sticky, axis=1)

def dropUnimportantFeatures(df, istest=False):
    if not istest:                                            # "day"
        df = df.drop(["datetime", "casual", "registered", "time",  ], axis=1)
    else:                   # "day"
        df = df.drop(["time", ], axis=1)
    df = df.dropna()
    return df

def convertPandasDataFrameToNumpyArray(df):
    return df.values

def cleanDataset(df, istest=False, describe=False):
    traindf = df

    if not istest:
        cols = df.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        traindf = traindf[cols]

    traindf = factorizeColumns(traindf)

    addTimeColumn(traindf)
    addDateColumn(traindf)
    addSundayColumn(traindf)
    addHoursAndDaypartColumns(traindf)

    if not istest:
        traindf = dropUnimportantFeatures(traindf)
    else:
        traindf = dropUnimportantFeatures(traindf, True)

    if describe: describeDataframe(traindf)
    return traindf


"""
Split Data based on registered or experienced


def loadCasualTrainingData(df):
    dfC = df[df.casual > 0].copy()
    return dfC

def loadRegisteredTrainingData(df):
    dfR = df[df.registered > 0].copy()
    return dfR

traindf = loadTrainData()

casdata = loadCasualTrainingData(traindf)
regdata = loadRegisteredTrainingData(traindf)

describeDataframe(casdata)
describeDataframe(regdata)
"""

"""
#Visualise weakest day : Sunday

# Avg no of people on days of the week
days=[i for i in range(7)]
means = traindf.groupby("day").mean()

meanTravels = []
for x in means["count"]:
    meanTravels.append(x)

sns.plt.subplot(1, 2, 1)
sns.barplot(x=days, y=meanTravels)

sns.plt.subplot(1,2,2)
minValue = min(means["count"])
means["count"] = means["count"] - minValue

meanTravels.clear()
for x in means["count"]:
    meanTravels.append(x)

sns.barplot(x=days, y=meanTravels)
sns.plt.show()
"""