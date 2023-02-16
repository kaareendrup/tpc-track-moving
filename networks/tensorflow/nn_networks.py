
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout, Flatten, Activation, BatchNormalization, Concatenate, Reshape
from tensorflow.keras.layers import Reshape, RepeatVector

# class tf_MLP(tf.keras.Model):

#     def __init__(self, inputArray, n_output=5, dropoutRate=0.1):
#         super(tf_MLP, self).__init__()

#         self.layer_1 = Dense(100, activation='relu')
#         self.layer_2 = Dense(50, activation='relu')
#         self.layer_3 = Dense(20, activation='relu')
#         self.layer_4 = Dense(20, activation='relu')
#         self.output = Dense(n_output, activation='linear')

#         self.bacthnorm = BatchNormalization()
#         self.dropout = Dropout(dropoutRate)
#         self.concat = Concatenate()

#     def call(self, inputArray):
#         x = self.layer_1(inputArray)
#         x = self.dropout(x)


#         return self.output(x)

def get_tfMLP_model(input_shape, config):
    dropoutRate = config.HYPER_PARAMS.DROPOUT_RATE

    inputArray = Input(shape=(input_shape,))

    x = Dense(100, activation='relu')(inputArray)
    x = Dropout(dropoutRate)(x)

    x = BatchNormalization()(x)
    x = Dense(100, activation='relu')(x)
    x = Dropout(dropoutRate)(x)

    x = Dense(50)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(dropoutRate)(x)

    x = Dense(20)(x) 
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(dropoutRate)(x)

    x = BatchNormalization()(x)
    x = Dense(20, activation='relu')(x)
    x = Dropout(dropoutRate)(x)

    # Implement input propagation
    x = Concatenate()([x, inputArray])

    output = Dense(5, activation='linear')(x)

    model = Model(inputs=inputArray, outputs=output)
    model.compile(loss=config.HYPER_PARAMS.LOSS_FUNCTION, optimizer=config.HYPER_PARAMS.OPTIMIZER)

    return model


def get_GNNprox_model(input_shape, config):
    dropoutRate = config.HYPER_PARAMS.DROPOUT_RATE

    inputArray = Input(shape=(input_shape,))

    x_graph = inputArray[:,5:]
    x_graph = Reshape((-1,3))(x_graph)

    x_vec = inputArray[:,:5]
    x_repeat = RepeatVector(x_graph.shape[1])(x_vec)

    x = Concatenate()([x_graph, x_repeat])

    x = Dense(50)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(dropoutRate)(x)

    x = Dense(50)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(dropoutRate)(x)

    x = Dense(50)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(dropoutRate)(x)

    x = Flatten()(x)

    output = Dense(5, activation='linear')(x)

    model = Model(inputs=inputArray, outputs=output)
    model.compile(loss=config.HYPER_PARAMS.LOSS_FUNCTION, optimizer=config.HYPER_PARAMS.OPTIMIZER)

    return model