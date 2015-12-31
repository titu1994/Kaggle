import pandas as pd
import numpy as np

def loadData(filePath, header=0, describe=False):
    df = pd.read_csv(filePath, header=header)
    if describe: describeDataframe(df)
    return df

def getColNames(df):
    return df.columns.tolist()

def describeDataframe(df):
    print(df.info(), "\n")
    print(df.describe(), "\n")
    print(df.dtypes, "\n")

def convertPandasDataFrameToNumpyArray(df):
    return df.values