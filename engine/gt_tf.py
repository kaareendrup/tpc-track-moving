import sys

from tpcutils.data import getAllData
from tpcutils.training import LinearScheduler

from networks.tensorflow.nn_networks import get_GNNprox_model, get_tfMLP_model
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve, auc
import glob

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, TerminateOnNaN, ModelCheckpoint

import yaml
import io

from dotmap import DotMap

from config.paths import dpaths as dp

def generalised_trainer_PT_clusters(**kwargs):


    # config = Struct(**yaml.safe_load(open(dp['config1'])))
    config = DotMap(yaml.safe_load(open(dp['config1'])))

    files = glob.glob(config.PATHS.DATA_PATH + '/*.txt')

    X_train, X_val, y_train, y_val = getAllData(files[0], files[2], test_size=config.DATA_PARAMS.TEST_SIZE, random_state=config.DATA_PARAMS.RANDOM_STATE)

    #input shape: 7+nClustersSelected*3
    model = get_tfMLP_model(X_train.shape[1], config)

    LRScheduler = LinearScheduler(config.HYPER_PARAMS.MAX_EPOCHS, config.HYPER_PARAMS.LEARNING_RATE)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=config.PATHS.SAVE_PATH + '/' + config.PATHS.MODEL_DIR + '/log', histogram_freq=1)

    history = model.fit(X_train, y_train, epochs=config.HYPER_PARAMS.MAX_EPOCHS, batch_size=config.HYPER_PARAMS.BATCH_SIZE, verbose = 2,
                    validation_data=(X_val, y_val),
                    callbacks = [
                        EarlyStopping(monitor='val_loss', patience=15, verbose=1),
                        LRScheduler,
                        TerminateOnNaN(),
                        tensorboard_callback,
                        # ModelCheckpoint(
                        #     config.PATHS.SAVE_PATH + '/' + config.PATHS.MODEL_DIR,
                        #     verbose=0,
                        #     monitor = 'val_loss',
                        #     mode = 'min',
                        # )
                    ])

    # predict_val = model.predict(X_val)
    model.save(config.PATHS.SAVE_PATH + '/' + config.PATHS.MODEL_DIR)

    with io.open(config.PATHS.SAVE_PATH + '/' + config.PATHS.MODEL_DIR + '/hyperparams.yml', 'w', encoding='utf8') as outfile:
        yaml.dump(config.toDict(),outfile)



if __name__=='__main__':

    generalised_trainer_PT_clusters()
