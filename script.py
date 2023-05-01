import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow	import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.metrics import r2_score

dataset = pd.read_csv('admissions_data.csv')

dataset = dataset.drop(['Serial No.'], axis = 1)
#dataset = dataset.drop(['University Rating'], axis = 1)
labels = dataset.iloc[:,-1]
features = dataset.iloc[:,0:-1]
#features = pd.get_dummies(features)
#print(dataset.head())

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2, random_state=42)

numerical_features = features.select_dtypes(include=['float64', 'int64'])
numerical_columns = numerical_features.columns
 
ct = ColumnTransformer([("only numeric", StandardScaler(), numerical_columns)], remainder='passthrough')

features_train = ct.fit_transform(features_train)
features_test = ct.transform(features_test)

model = Sequential()
num_features = features.shape[1]
input = layers.InputLayer(input_shape=(num_features,))
model.add(input)
model.add(Dense(128, activation='relu'))
model.add(Dense(1))

learning_rate = 0.001 
num_epochs = 50
batch_size = 16

opt = Adam(learning_rate=learning_rate)
model.compile(loss='mse',  metrics=['mae'], optimizer=opt)

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience = 5)
history = model.fit(features_train, labels_train, epochs=num_epochs, batch_size= batch_size, verbose=0, validation_split = 0.2, callbacks = [es])
#val_mse, val_mae = model.evaluate(features_test, labels_test, verbose = 0)

predicted_values = model.predict(features_test) 
print('r2 = ' + str(r2_score(labels_test, predicted_values))) 

fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax1.plot(history.history['mae'])
ax1.plot(history.history['val_mae'])
ax1.set_title('Model MAE')
ax1.set_ylabel('MAE')
ax1.set_xlabel('epoch')
ax1.legend(['train', 'validation'], loc='upper left')
 
ax2 = fig.add_subplot(2, 1, 2)
ax2.plot(history.history['loss'])
ax2.plot(history.history['val_loss'])
ax2.set_title('Model loss')
ax2.set_ylabel('loss')
ax2.set_xlabel('epoch')
ax2.legend(['train', 'validation'], loc='upper left')

fig.tight_layout()
plt.show()
