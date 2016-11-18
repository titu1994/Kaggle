import Redhat.DataClean as dc
import pandas as pd
from keras.models import Model, load_model
from keras.layers import Input, Dense, Dropout, merge
from keras.callbacks import ModelCheckpoint
from keras.utils.np_utils import to_categorical

traindf = dc.load_train()
testdf = dc.load_test()

features = ['group_1', 'char_2', 'char_38']

activity_id = testdf.activity_id.values
testdf.drop(['activity_id', 'people_id'], inplace=True, axis=1)

lootrain = dc.cleanData(traindf)
testdf = dc.cleanData(testdf)

dc.standardize(lootrain)
dc.standardize(testdf)

trainY = lootrain["outcome"].values
lootrain.drop(["people_id", "outcome"], axis=1, inplace=True)

trainX = lootrain.values
trainY = to_categorical(trainY)

testX = testdf.values

# Create and fit model
nb_features = trainX.shape[1]
nb_classes = trainY.shape[1]

batchSize = 128
nbEpochs = 100

init = Input(shape=(nb_features,))

x = Dense(256, activation='relu')(init)
x = Dense(384, activation='relu')(x)

x = Dense(512, activation='relu')(x)
x = Dense(1024, activation='relu')(x)

d = Dropout(0.2)(x)

out = Dense(nb_classes, activation='softmax')(d)

model = Model(init, out)
model.summary()

model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['acc'])

#model.load_weights('NN Weights.h5')

model.fit(trainX, trainY, batch_size=batchSize, nb_epoch=nbEpochs, validation_data=(trainX, trainY),
          callbacks=[ModelCheckpoint('NN Weights.h5', monitor='val_acc', save_weights_only=True, save_best_only=True)])

preds = model.predict(testX, batchSize)[:, 1]

# Predict
submission = pd.DataFrame()
submission['activity_id'] = activity_id
submission['outcome'] = preds
submission.to_csv('dnn.csv', index=False, float_format='%.6f')