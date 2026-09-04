import random
from logging import Logger

import numpy as np
import pandas as pd
import tensorflow as tf


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Bidirectional, Input

from sklearn.utils.class_weight import compute_class_weight

from src.utils.utils import get_path
from src.config.data_class import LSTMTrainingParameters





def train_lstm(train_param:LSTMTrainingParameters, logger:Logger):

  try:
    target_train = train_param.target_train
    feature_train_3d = train_param.feature_train_3d
    feature_validation_3d = train_param.feature_validation_3d
    target_validation = train_param.target_validation
    output_path = train_param.model_output_path
    random.seed( 42 )
    np.random.seed( 42 )
    tf.random.set_seed( 42 )

    # handling class imbalance to prevent the network from ignoring rare shocks
    logger.info("calculating weight class to handle class imbalance")
    classes = np.unique(target_train)
    weights = compute_class_weight(class_weight = 'balanced', classes = classes, y = target_train)
    class_weight_dict = {classes[0]:weights[0], classes[1]:weights[1]}
    logger.info(f"class_weight:{class_weight_dict}")

    # configures the recurrent lstm architecture
    logger.info("configuring LSTM model")
    lstm_model = Sequential([
      # Input layer
      Input(shape = (
        feature_train_3d.shape[ 1 ] ,
        feature_train_3d.shape[ 2 ]
      )),
      # layer 1: lstm to process temporal patterns forward
      # LSTM(units = 64, return_sequences = True),
      # # stops neuron co-dependency and overfitting
      # Dropout(0.2) ,

      #layer 2: standard layer focusing on compressed representations
      LSTM(units = 32, return_sequences = False),
      Dropout(0.2) ,

      # layer 3: dense layer for mapping complex non-linear combinations
      Dense(units = 16, activation = 'relu'),

      #layer 4: output layer returning a continuous probability between 0 and 1
      Dense(units = 1, activation = 'sigmoid')
    ])


    lstm_model.compile(

      optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001),
      loss ='binary_crossentropy',
      metrics = [
        tf.keras.metrics.AUC( name = 'auc' ) ,
        tf.keras.metrics.AUC( curve = 'PR' , name = 'auc_pr' ) ,
        'accuracy' ,
        tf.keras.metrics.Precision() ,
        tf.keras.metrics.Recall() ,
      ]
    )
    logger.info("LSTM model configuration done")
  except Exception as e:
    raise e

  # early stopping to stop training if the network  starts memorizing noise
  early_stop = tf.keras.callbacks.EarlyStopping(
    monitor = 'val_auc_pr' , mode = 'max' , patience = 5 , restore_best_weights = True
  )
  # shrink the learning rate before early stopping gives up entirely,
  # rather than jumping straight from "improving" to "stopped"
  reduce_learning_rate = tf.keras.callbacks.ReduceLROnPlateau(
    monitor = 'val_auc_pr' , mode = 'max' , factor = 0.5 , patience = 2 , min_lr = 1e-5
  )

  # persist the best-validation-epoch model as an explicit artifact,
  # not just restored in-memory weights
  checkpoint = tf.keras.callbacks.ModelCheckpoint(
    output_path , monitor = 'val_auc_pr' , mode = 'max' , save_best_only = True
  )

  # train model
  try:
    logger.info("Training LSTM model")
    lstm_model.fit(
      feature_train_3d , target_train ,
      validation_data = (feature_validation_3d, target_validation),
      epochs = 50 ,
      batch_size = 32 ,
      class_weight = class_weight_dict ,
      callbacks = [ early_stop, reduce_learning_rate, checkpoint ] ,
      verbose = 1 ,
    )
    logger.info("Training LSTM model done")


  except Exception as e:
    raise e


def save_lstm_model(lstm_model:Sequential, output_file:str, logger:Logger):
  try:
    logger.info("Saving LSTM model")
    output_path = get_path( output_file )
    lstm_model.save( output_path )
    logger.info("Saving LSTM model done")
  except Exception as e:
    raise e
