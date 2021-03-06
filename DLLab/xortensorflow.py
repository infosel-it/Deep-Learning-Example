import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
from keras.utils.all_utils import plot_model


# the four different states of the XOR gate
training_data = np.array([[0,0],[0,1],[1,0],[1,1]], "float32")

# the four expected results in the same order
target_data = np.array([[0],[1],[1],[0]], "float32")

model = Sequential()
model.add(Dense(16, input_dim=2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='mean_squared_error',
              optimizer='NAdam',
              metrics=['binary_accuracy'])

model.fit(training_data, target_data, epochs=200, verbose=2)

print(model.predict(training_data).round())

dot_img_file = 'model_1.png'
plot_model(model, to_file=dot_img_file, show_shapes=True)


model.compile(loss='mean_squared_error',
              optimizer='Adam',
              metrics=['binary_accuracy'])
