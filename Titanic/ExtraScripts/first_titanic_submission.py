# -*- coding: utf-8 -*-
import random
import Titanic.ExtraScripts.ticart
import pandas as pd
import Titanic.ExtraScripts.cleantitanic as ct
from Titanic.ExtraScripts import ticart


def cross_validate(no_folds, data, resample=False):
    rows = list(data.index)
    random.shuffle(rows)
    N=len(data)
    len_fold = int(N/no_folds)
    start=0
    for i in range(no_folds):
        if i==no_folds-1:
            stop =N
        else:
            stop = start +len_fold
        test = data.ix[rows[start:stop]]
        train = data.ix[rows[:start]+rows[stop:]]
        if resample:
            train_len=start+N-stop
            no_resamples = N-train_len
            train_rows = list(train.index)
            random_extra_rows =[random.choice(train_rows) for row in range(no_resamples)]
            train_rows = train_rows+random_extra_rows
            train=train.ix[train_rows]
        yield {'test':test, 'train':train}
        start=stop


df=ct.cleaneddf(no_bins=7)[0]
df2=ct.cleaneddf(no_bins=7)[1]

df=df[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 
       'Fare', 'Embarked']]
data_type_dict={'Survived':'nominal', 'Pclass':'ordinal', 'Sex':'nominal', 
                'Age':'ordinal', 'SibSp':'ordinal', 'Parch':'ordinal', 
                'Fare':'ordinal', 'Embarked':'nominal'}

       
def tree_train(data_type_dict, train_data,test_data, response, no_folds,
                   min_node_size, max_depth, no_iter):
        parameters={'min_node_size':min_node_size, 'max_node_depth':max_depth, 
                    'threshold':0, 'metric_kind':'Gini', 'alpha':0,
                    'response':response}
        model=ticart.ClassificationTree()
        predictions=[]
        for i in range(no_iter):
            for fold in cross_validate(no_folds, train_data):
                model=ticart.ClassificationTree()
                model.train(fold['train'], data_type_dict, parameters, prune=False)
                model.load_new_data(fold['test'])
                model.prune_tree(alpha=0, new_data=True)
                predictions.append(test_data.apply(model.predict, axis=1))
        return predictions          


def combine_predictions(predictions):
    data_dict ={i:predictions[i] for i in range(len(predictions))}
    d=pd.DataFrame(data_dict)
    def mode(x):
        key,value = max(x.value_counts().iteritems(), key=lambda x:x[1])
        return key
    pred=d.apply(mode, axis=1)
    return pred

predictions=tree_train(data_type_dict=data_type_dict, train_data=df,
                           test_data=df2, response='Survived', no_folds=2,
                           max_depth=50, min_node_size=5, no_iter=2)
predictions = combine_predictions(predictions)
prediction_path = r'C:\Users\Yue\PycharmProjects\Kaggle\Titanic\ExtraScripts\prediction.csv'
prediction_csv=pd.read_csv(prediction_path)
prediction_csv['Survived']=predictions
prediction_csv.to_csv('C:/Users/Yue/PycharmProjects/Kaggle/Titanic/ExtraScripts/my_first_submission2.csv',
                      index =False)


