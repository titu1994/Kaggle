from MLScripts.CleaningUtils import *
import pandas as pd
from sklearn.feature_extraction import DictVectorizer as DV

objectCols = ['v3', 'v24', 'v30', 'v31', 'v47', 'v52', 'v56', 'v66', 'v71',
       'v74', 'v75', 'v79', 'v107', 'v110', 'v112', 'v113', 'v125']

def compute_most_freq_value(df,colname):
    c = df[colname].value_counts()
    return c.index[0]

def loadTrain() -> pd.DataFrame:
    return loadData(r"D:\Users\Yue\PycharmProjects\Kaggle\BNP\Data\train.csv")

def loadTest() -> pd.DataFrame:
    return loadData(r"D:\Users\Yue\PycharmProjects\Kaggle\BNP\Data\test.csv")

def cleanData(traindf: pd.DataFrame, testdf: pd.DataFrame, describe=False) -> (pd.DataFrame, pd.DataFrame):

    traindf.drop(['v22', 'v91'], axis=1, inplace=True)
    testdf.drop(['v22', 'v91'], axis=1, inplace=True)

    nas = {}
    for colname in objectCols:
        nas[colname] = compute_most_freq_value(traindf,colname)

    for colname in objectCols:
        traindf[colname].fillna(nas[colname],inplace=True)

    for colname in objectCols:
        testdf[colname].fillna(nas[colname],inplace=True)

    cat_train = traindf[objectCols]
    cat_test = testdf[objectCols]

    traindf.drop(objectCols, axis=1, inplace=True)
    testdf.drop(objectCols, axis=1, inplace=True)

    dict_train_data = cat_train.T.to_dict().values()
    dict_test_data = cat_test.T.to_dict().values()

    #vectorize
    vectorizer = DV(sparse = False)
    features = vectorizer.fit_transform(dict_train_data)
    vec_data = pd.DataFrame(features)
    vec_data.columns = vectorizer.get_feature_names()
    vec_data.index = traindf.index
    traindf = traindf.join(vec_data)

    features = vectorizer.transform(dict_test_data)
    vec_data = pd.DataFrame(features)
    vec_data.columns = vectorizer.get_feature_names()
    vec_data.index = testdf.index
    testdf = testdf.join(vec_data)

    traindf.fillna(traindf.mean(), inplace=True)
    testdf.fillna(testdf.mean(), inplace=True)

    if describe: describeDataframe(traindf)
    return traindf, testdf


if __name__ == "__main__":
    traindf, testdf = loadTrain(), loadTest()
    traindf, testdf = cleanData(traindf, testdf, describe=False)

    #print("\nObject Cols")
    #print(traindf.select_dtypes(include=["object"]).columns)

    #print(traindf.isnull().any(axis=1))
    #print("\n\n", testdf.isnull().any(axis=1))