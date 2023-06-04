from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, ELU, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.losses import Huber

def VanillaCNN(input_shape, action_space):
    X_input = Input(input_shape)
    # position_input = Input((2,))

    X = Conv2D(32, (3, 3), strides=(2, 2), padding="same", kernel_initializer='he_uniform')(X_input)
    X = ELU()(X)
    X = Conv2D(32, (3, 3), strides=(2, 2), padding="same", kernel_initializer='he_uniform')(X)
    X = ELU()(X)
    X = Conv2D(32, (3, 3), strides=(2, 2), padding="same", kernel_initializer='he_uniform')(X)
    X = ELU()(X)
    X = Conv2D(32, (3, 3), strides=(2, 2), padding="same", kernel_initializer='he_uniform')(X)
    X = ELU()(X)
    X = Flatten()(X)

    # X = Concatenate()([X, position_input])

    X = Dense(512, activation="elu", kernel_initializer='he_uniform')(X)
    X = Dense(256, activation="elu", kernel_initializer='he_uniform')(X)
    X = Dense(128, activation="elu", kernel_initializer='he_uniform')(X)
    X = Dense(action_space, activation="linear", kernel_initializer='he_uniform')(X)

    # model = Model(inputs = [X_input, position_input], outputs = X, name='BrainWorld-CNN')
    model = Model(inputs = X_input, outputs = X, name='BrainWorld-CNN')
    model.compile(loss=Huber(), optimizer=Adam(lr=1e-4), metrics=["accuracy"])

    return model

