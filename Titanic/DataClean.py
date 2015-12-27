import pandas as pd
import numpy as np
from scipy.stats import mode
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

def loadTrainData():
    df = pd.read_csv(r"C:\Users\Yue\PycharmProjects\Kaggle\Titanic\Data\train.csv", header=0)
    return df

def loadTestData():
    df = pd.read_csv(r"C:\Users\Yue\PycharmProjects\Kaggle\Titanic\Data\test.csv", header=0)
    #describeAndProvideInfo(df)
    return df

def getFeatureNames():                          #"SibSp"        "Cabin"   "Embarked"                                                        "Class*Gender"
    return ["PassengerId", "Survived", "Pclass", "Parch", "Fare", "Gender", "AgeFill", "AgeIsNull", "FamilySize", "Age*Class", "FarePerPerson"]

def getTestFeatureNames():                          #"SibSp"        "Cabin"   "Embarked"                                                        "Class*Gender"
    return ["PassengerId", "Pclass", "Parch", "Fare", "Gender", "AgeFill", "AgeIsNull", "FamilySize", "Age*Class", "FarePerPerson"]

def describeAndProvideInfo(df):
    print(df.info(), "\n")
    print(df.describe(), "\n")
    print(df.dtypes, "\n")

def describeStringFeatures(df):
    print(df.dtypes[df.dtypes.map(lambda x: x == "object")], "\n")

def getMissingRowsOfCol(df, colName, attributes=[]):
    if len(attributes) > 0:
        return df[df[colName].isnull()][attributes]
    else:
        return df[df[colName].isnull()]

def getPerson(df):
    sex, age = df
    return "child" if age < 16 else sex

def findChildrenInSex(df):
    df["Sex"] = df[["Sex", "Age"]].apply(getPerson, axis=1)
    #sns.countplot(x="Sex", data=df, alpha=0.7)
    #sns.plt.show()

def mapGenderToIntegers(df):
    #df["Gender"] = df["Sex"].map( {"female" : 0, "male" : 1}).astype(int)
    df["Gender"] = df["Sex"].map( {"female" : 0, "male" : 1, "child" : 2}).astype(int)

def mapEmbarkedToInts(df):
    df.loc[(df.Embarked.isnull()), "Embarked"] = "Q"
    df["Embarked"] = df["Embarked"].map({"C" : 0, "S" : 1, "Q" : 2})

def calculateMedianAges(df):
    medianAges = np.zeros((2, 3))
    for i in range(2):
        for j in range(3):
            medianAges[i,j] = df[(df["Gender"] == i) & (df["Pclass"] == (j+1))]["Age"].dropna().mean()
    return medianAges

def copyFeatures(df, oldfeatureName, newfeatureName):
    df[newfeatureName] = df[oldfeatureName]

def calculateAgeFillValues(df, medianAges):
    for i in range(2):
        for j in range(3):
            df.loc[(df.Age.isnull()) & (df.Gender == i) & (df.Pclass == (j+1)), "AgeFill"] = medianAges[i,j]

def calculateIfAgeWasMissing(df):
    df["AgeIsNull"] = pd.isnull(df.Age).astype(int)

def calculateIfFareWasMissing(df):
    meanFares = [df[df["Pclass"] == (i + 1)]["Fare"].dropna().mean() for i in range(3)]
    for i in range(3):
        df.loc[((df.Fare.isnull()) | (df["Fare"] == 0.0)) & (df["Pclass"] == (i+1)), "Fare"] = meanFares[i]

def calculateCabinToIntegers(df):
    df["Cabin"] = pd.notnull(df.Cabin).astype(int)


"""
Feature Engineering
"""

def addFamilySizeAsFeature(df):
    df["FamilySize"] = df["SibSp"] + df["Parch"]
    # Normalize whether the person was alone or with family
    #df["FamilySize"].loc[df["FamilySize"] > 0] = 1


def addAgeIntoClassAsFeature(df):
    df["Age*Class"] = df["AgeFill"] * df["Pclass"]

def addFairPerPerson(df):
    df["FarePerPerson"] = df["Fare"] / ((df["FamilySize"] + 1))

def addClassIntoGenderAsFeature(df):
    df["Class*Gender"] = df["Pclass"] * (df["Gender"] + 1)

def addTitleAsFeature(df):
    df['Title']=df.apply(replaceTitles, axis=1)


def dropUnimportantFeatures(df):            #'Embarked',
    df = df.drop(['Name', 'Sex', 'Ticket', 'Embarked',"Cabin", "Age", "SibSp"], axis=1)
    df = df.dropna()
    return df

def convertPandasDataFrameToNumpyArray(df):
    return df.values

def replaceTitles(x):
    title=x["Title"]
    if title in ['Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col']:
        return 1 #'Mr'
    elif title in ['Countess', 'Mme']:
        return 3 #'Mrs'
    elif title in ['Mlle', 'Ms']:
        return 2 #'Miss'
    elif title =='Dr':
        if x['Sex']=='Male':
            return 1 #'Mr'
        else:
            return 2 #'Mrs'
    else:
        return 0 #title



def substringsInString(big_string, substrings):
    for substring in substrings:
        if substring in big_string:
            return substring
    return np.nan

title_list=['Mrs', 'Mr', 'Master', 'Miss', 'Major', 'Rev',
                    'Dr', 'Ms', 'Mlle','Col', 'Capt', 'Mme', 'Countess',
                    'Don', 'Jonkheer']

def cleanDataSet(df):
    """
    @rtype: numpy.array
    """
    traindf = df
    #findChildrenInSex(traindf)
    mapGenderToIntegers(traindf)
    #mapEmbarkedToInts(traindf)
    medianAges = calculateMedianAges(traindf)
    copyFeatures(traindf, "Age", "AgeFill")
    calculateAgeFillValues(traindf, medianAges)
    calculateIfAgeWasMissing(traindf)
    calculateIfFareWasMissing(traindf)

    #calculateIfEmbarked(traindf)
    #calculateCabinToIntegers(traindf)
    #df['Title']=df['Name'].map(lambda x: substringsInString(x, title_list))

    addFamilySizeAsFeature(traindf)
    addAgeIntoClassAsFeature(traindf)
    addFairPerPerson(traindf)
    #addClassIntoGenderAsFeature(traindf)


    traindf = dropUnimportantFeatures(traindf)
    #describeAndProvideInfo(traindf)
    #input()
    return traindf

def displayRelationsRelations(df):
    #sns.factorplot(x="Pclass", y="Survived", data=df)
    plt.subplot(1, 2, 1)
    sns.countplot(x="Survived", hue="Pclass", data=df, order=[0, 1], hue_order=[1, 2, 3])
    plt.subplot(1, 2, 2)
    sns.countplot(x="Survived", hue="Gender", data=df, order=[0, 1], hue_order=[0, 1])
    sns.plt.show()

    g = sns.FacetGrid(df, col="Pclass", hue="Gender")
    g.map(plt.scatter, "Fare", "FarePerPerson", alpha=0.5)
    g.add_legend()
    sns.plt.show()




"""

traindf = loadTrainData()
#describeAndProvideInfo(traindf)

#print(getMissingRowsOfCol(traindf, "Age", ["Sex", "Pclass", "Age"]))

mapGenderToIntegers(traindf)
medianAges = calculateMedianAges(traindf)

#print("Median Ages : " , medianAges)

copyFeatures(traindf, "Age", "AgeFill")
calculateAgeFillValues(traindf, medianAges)

#print(traindf[traindf["Age"].isnull()][["Gender", "Pclass", "Age", "AgeFill"]].head(10))

calculateIfAgeWasMissing(traindf)

'''
Feature Engineering
'''

addFamilySizeAsFeature(traindf)
addAgeIntoClassAsFeature(traindf)
addFairPerPerson(traindf)

traindf = dropUnimportantFeatures(traindf)

describeAndProvideInfo(traindf)
#describeStringFeatures(traindf)

#print(convertPandasDataFrameToNumpyArray(traindf))

"""




