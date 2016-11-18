import pandas as pd
from sklearn.preprocessing.label import LabelEncoder

group1 = LabelEncoder()
char1 = LabelEncoder()
char2 = LabelEncoder()
char3 = LabelEncoder()
char4 = LabelEncoder()
char5 = LabelEncoder()
char6 = LabelEncoder()
char7 = LabelEncoder()
char8 = LabelEncoder()
char9 = LabelEncoder()

people_path = r"D:\Users\Yue\PycharmProjects\Kaggle\Redhat\Data\people.csv"

act_train_path = r"D:\Users\Yue\PycharmProjects\Kaggle\Redhat\Data\act_train.csv"
test_path = r"D:\Users\Yue\PycharmProjects\Kaggle\Redhat\Data\act_test.csv"

def load_train() -> pd.DataFrame:
    print("Loading training data")
    train = pd.read_csv(act_train_path, usecols=['people_id', 'outcome'])
    people = pd.read_csv(people_path)

    # Preprocess
    addDates(people)
    convertCategorical(people)
    convertBooleanToInt(people)

    full_train = pd.merge(train, people, on='people_id', how='left', left_index=True)
    print("Finished loading training data")
    return full_train

def load_test() -> pd.DataFrame:
    print("Loading testing data")
    test = pd.read_csv(test_path, usecols=['activity_id', 'people_id'])
    people = pd.read_csv(people_path)

    # Preprocess
    addDates(people)
    convertCategorical(people)
    convertBooleanToInt(people)

    full_test = pd.merge(test, people, how='left', on='people_id', left_index=True)
    print("Finished loading testing data")
    return full_test

def fillna(df:pd.DataFrame) -> pd.DataFrame:
    df.fillna(-1, inplace=True)

def leave_one_out(data1, data2, columnName, useLOO=False):
    grpOutcomes = data1.groupby(columnName).mean().reset_index()
    outcomes = data2['outcome'].values
    x = pd.merge(data2[[columnName, 'outcome']], grpOutcomes,
                 suffixes=('x_', ''),
                 how='left',
                 on=columnName,
                 left_index=True)['outcome']
    if(useLOO):
        x = ((x*x.shape[0])-outcomes)/(x.shape[0]-1)
    return x.fillna(x.mean())

def create_leave_one_out_set(df:pd.DataFrame, df2:pd.DataFrame, useLOO=False) -> pd.DataFrame:
    loodf = pd.DataFrame()

    for col in df.columns:
        if (col != 'outcome' and col != 'people_id'):
            print("Processing column : %s" % (col))
            loodf[col] = leave_one_out(df, df2, col, useLOO).values

    return loodf

def addDates(df):
    # dt = pd.DatetimeIndex(df["date"])
    # df.set_index(dt, inplace=True)
    #
    # df["month"] = dt.month
    # df["year"] = dt.year
    # df["day"] = dt.day
    # df["dow"] = dt.dayofweek
    # df["weekday"] = dt.weekday
    df.drop(["date"], axis=1, inplace=True)

def convertCategorical(df, isTest=False):
    if not isTest:
        df.group_1 = group1.fit_transform(df.group_1)
        df.char_1 = char1.fit_transform(df.char_1)
        df.char_2 = char2.fit_transform(df.char_2)
        df.char_3 = char3.fit_transform(df.char_3)
        df.char_4 = char4.fit_transform(df.char_4)
        df.char_5 = char5.fit_transform(df.char_5)
        df.char_6 = char6.fit_transform(df.char_6)
        df.char_7 = char7.fit_transform(df.char_7)
        df.char_8 = char8.fit_transform(df.char_8)
        df.char_9 = char9.fit_transform(df.char_9)
    else:
        df.group_1 = group1.transform(df.group_1)
        df.char_1 = char1.transform(df.char_1)
        df.char_2 = char2.transform(df.char_2)
        df.char_3 = char3.transform(df.char_3)
        df.char_4 = char4.transform(df.char_4)
        df.char_5 = char5.transform(df.char_5)
        df.char_6 = char6.transform(df.char_6)
        df.char_7 = char7.transform(df.char_7)
        df.char_8 = char8.transform(df.char_8)
        df.char_9 = char9.transform(df.char_9)

def convertBooleanToInt(df):
    cols = ['char_%d' % (i) for i in range(10, 38)]

    for col in cols:
        df[col] = df[col].astype(int)

def cleanData(df, test=False):
    fillna(df)

    if test:
        df['outcome'] = 0

    return df
def standardize(df:pd.DataFrame):
    from sklearn.preprocessing.data import StandardScaler
    cols = ['group_1','char_1', 'char_2', 'char_3', 'char_4', 'char_5', 'char_6', 'char_7', 'char_8', 'char_9', 'char_38']

    for col in cols:
        df[col] = df[col].astype(float)

    df[cols] = df[cols].apply(lambda x: StandardScaler().fit_transform(x))


if __name__ == "__main__":
    df = load_train()
    df = cleanData(df)
    print(df.head(5))
    print(df.info())
    print(df.describe())

    print('\nBreak to continue...')

    df = load_test()
    print(df.info())
    print("Done.")
