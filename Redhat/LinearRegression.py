import Redhat.DataClean as dc
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import pandas as pd

traindf = dc.load_train()
testdf = dc.load_test()
testdf['outcome'] = 0

all_features = ['people_id', 'group_1', 'char_2', 'char_38', 'outcome']
features = ['group_1', 'char_2', 'char_38']

lootrain = dc.cleanData(traindf)[all_features]

lootrain = dc.create_leave_one_out_set(lootrain, lootrain, useLOO=True)
lootest = dc.create_leave_one_out_set(traindf, testdf, useLOO=False)

# Create and fit model
lr = LogisticRegression(C=100000.0)
lr.fit(lootrain[features], traindf['outcome'])

preds = lr.predict_proba(lootrain[features])[:, 1]
print('roc', roc_auc_score(traindf.outcome, preds))

activity_id = testdf.activity_id.values
testdf.drop('activity_id', inplace=True, axis=1)

# Output
preds = lr.predict_proba(lootest[features])[:, 1]
submission = pd.DataFrame()
submission['activity_id'] = activity_id
submission['outcome'] = preds
submission.to_csv('linear regression.csv', index=False, float_format='%.3f')