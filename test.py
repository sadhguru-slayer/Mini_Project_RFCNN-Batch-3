import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential
from sklearn.metrics import accuracy_score
from keras.callbacks import ModelCheckpoint
import os

dataset = pd.read_csv("Dataset/US_Accidents_Dec21_updated.csv")
'''
missing_value = dataset.isnull().sum()
missing_value.plot(kind='barh')
plt.show()
'''

column = dataset.columns.ravel()
label_encoder = []

for i in range(len(column)):
    if str(dataset.dtypes[column[i]]) == 'object':
        le = LabelEncoder()
        dataset[column[i]] = pd.Series(le.fit_transform(dataset[column[i]].astype(str)))
        label_encoder.append(le)
    if str(dataset.dtypes[column[i]]) == 'bool':
        dataset[column[i]] = dataset[column[i]].astype(int)

selected_features = ['Distance(mi)', 'Temperature(F)', 'Wind_Chill(F)', 'Humidity(%)', 'Pressure(in)', 'Visibility(mi)', 'Wind_Direction', 'Wind_Speed(mph)',
                     'Precipitation(in)', 'Weather_Condition','Amenity','Bump','Crossing','Give_Way','Junction','No_Exit','Railway','Roundabout',
                     'Station', 'Stop', 'Traffic_Calming', 'Traffic_Signal', 'Turning_Loop']

Y = dataset['Severity'].ravel()
dataset.drop(['Severity'], axis = 1,inplace=True)

dataset = dataset.apply(lambda x: x.fillna(x.mean()))
selected_df = dataset[selected_features]

scaler1 = StandardScaler()
scaler2 = StandardScaler()

full_X = dataset.values
selected_X = selected_df.values


full_X = scaler1.fit_transform(full_X)
selected_X = scaler2.fit_transform(selected_X)


full_X_train, full_X_test, full_y_train, full_y_test = train_test_split(full_X, Y, test_size = 0.2)
selected_X_train, selected_X_test, selected_y_train, selected_y_test = train_test_split(selected_X, Y, test_size = 0.2)

full_X_train = np.reshape(full_X_train, (full_X_train.shape[0], full_X_train.shape[1], 1, 1))
full_X_test = np.reshape(full_X_test, (full_X_test.shape[0], full_X_test.shape[1], 1, 1))
selected_X_train = np.reshape(selected_X_train, (selected_X_train.shape[0], selected_X_train.shape[1], 1, 1))
selected_X_test = np.reshape(selected_X_test, (selected_X_test.shape[0], selected_X_test.shape[1], 1, 1))

selected_X = np.reshape(selected_X, (selected_X.shape[0], selected_X.shape[1], 1, 1))
Y = to_categorical(Y)


full_y_train = to_categorical(full_y_train)
full_y_test = to_categorical(full_y_test)

selected_y_train = to_categorical(selected_y_train)
selected_y_test = to_categorical(selected_y_test)
print(full_X_train.shape)
print(selected_X_train.shape)


classifier = Sequential()
#classifier.add(densenet)
classifier.add(Convolution2D(32, 1, 1, input_shape = (full_X_train.shape[1], full_X_train.shape[2], full_X_train.shape[3]), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (1, 1)))
classifier.add(Convolution2D(32, 1, 1, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (1, 1)))
classifier.add(Flatten())
classifier.add(Dense(output_dim = 256, activation = 'relu'))
classifier.add(Dense(output_dim = full_y_train.shape[1], activation = 'softmax'))
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
if os.path.exists("model/full_weights.hdf5") == False:
    model_check_point = ModelCheckpoint(filepath='model/full_weights.hdf5', verbose = 1, save_best_only = True)
    hist = classifier.fit(full_X_train, full_y_train, batch_size = 16, epochs = 25, validation_data=(full_X_test, full_y_test), callbacks=[model_check_point], verbose=1)      
else:
    classifier.load_weights("model/full_weights.hdf5")


predict = classifier.predict(full_X_test)
predict = np.argmax(predict, axis=1)
y_train = np.argmax(full_y_test, axis=1)
acc = accuracy_score(y_train, predict)
print(acc)

classifier = Sequential()
#classifier.add(densenet)
classifier.add(Convolution2D(32, 1, 1, input_shape = (selected_X_train.shape[1], selected_X_train.shape[2], selected_X_train.shape[3]), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (1, 1)))
classifier.add(Convolution2D(32, 1, 1, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (1, 1)))
classifier.add(Flatten())
classifier.add(Dense(output_dim = 256, activation = 'relu'))
classifier.add(Dense(output_dim = selected_y_train.shape[1], activation = 'softmax'))
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
if os.path.exists("model/selected_weights.hdf5") == False:
    model_check_point = ModelCheckpoint(filepath='model/selected_weights.hdf5', verbose = 1, save_best_only = True)
    hist = classifier.fit(selected_X_train, selected_y_train, batch_size = 16, epochs = 25, validation_data=(selected_X_test, selected_y_test), callbacks=[model_check_point], verbose=1)      
else:
    classifier.load_weights("model/selected_weights.hdf5")


predict = classifier.predict(selected_X_test)
predict = np.argmax(predict, axis=1)
y_train = np.argmax(selected_y_test, axis=1)
acc = accuracy_score(y_train, predict)
print(acc)









