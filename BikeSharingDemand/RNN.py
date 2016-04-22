import csv
import keras.models as models
import keras.layers.core as core
import keras.layers.embeddings as embed
import keras.layers.recurrent as recurrent

import BikeSharingDemand.DataClean as dataclean

trainFrame = dataclean.cleanDataset(dataclean.loadTrainData())
trainData = dataclean.convertPandasDataFrameToNumpyArray(trainFrame)

testFrame = dataclean.cleanDataset(dataclean.loadTestData(), True)
testData = dataclean.convertPandasDataFrameToNumpyArray(testFrame)

trainX = trainData[:, 1:]
trainY = trainData[:, 0]

testX = testData[:, 1:]
nFeatures = trainX.shape[1]


model = models.Sequential()
model.add(embed.Embedding(1, 256, input_length=nFeatures))
model.add(recurrent.LSTM(output_dim=128, activation="sigmoid"))
model.add(core.Dropout(0.2))
model.add(core.Dense(1))

model.summary()
model.compile(optimizer="sgd", loss="mse")

model.fit(trainX, trainY, nb_epoch=100, verbose=1, )

finalPredicted = model.predict(testX)
for i, x in enumerate(finalPredicted):
    finalPredicted[i] = finalPredicted[i] if finalPredicted[i] >= 0 else -finalPredicted[i]

f = open("bike-sharing-result-nn.csv", "w", newline="")
csvWriter = csv.writer(f)
csvWriter.writerow(["datetime", "count"])

for datetime, count in zip(testData[:, 0], finalPredicted):
    csvWriter.writerow([datetime, int(count)])
