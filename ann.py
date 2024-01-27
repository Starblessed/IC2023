from tensorflow import keras

# Basic Model (16-1)
def basic(input_shape):
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(16, input_shape=input_shape, activation='elu'))
    model.add(keras.layers.Dense(1, activation='relu'))

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss='mse')

    return model


# Deep Model (64-16-4-1)
def deep(input_shape):
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(128, input_shape=input_shape, activation='tanh'))
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.Dense(64, activation='elu'))
    model.add(keras.layers.Dense(16, activation='selu'))
    model.add(keras.layers.Dense(1, activation='relu'))

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss='mse')
    
    return model

