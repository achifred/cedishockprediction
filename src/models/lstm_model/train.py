import random
from logging import Logger

import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input
from sklearn.utils.class_weight import compute_class_weight

from src.data.feature_importance import generate_lstm_feature_importance
from src.config.data_class import LSTMTrainingParameters
from src.config.config import BEST_LSTM_THRESHOLD_FILE , EARLY_STOP_PATIENCE , LSTM_RANDOM_SEED , LSTM_UNITS , \
  LSTM_DROPOUT_RATE , \
  LSTM_RELU_DENSE_UNITS , LSTM_RELU_DENSE_DROPOUT_RATE , LSTM_LEARNING_RATE , LSTM_EPOCHS , LSTM_BATCH_SIZE , \
  REDUCE_LEARNING_PATIENCE,LSTM_VALIDATION_PREDICTION_RESULT_FILE
from src.utils.utils import save_to_csv,get_best_threshold


def train_lstm(train_param:LSTMTrainingParameters, logger:Logger):

  try:
    target_train = train_param.target_train
    feature_train_3d = train_param.feature_train_3d
    feature_validation_3d = train_param.feature_validation_3d
    target_validation = train_param.target_validation
    validation_dates = train_param.validation_dates
    output_path = train_param.model_output_path
    feature_names = train_param.feature_names



    random.seed( LSTM_RANDOM_SEED )
    np.random.seed( LSTM_RANDOM_SEED )
    tf.random.set_seed( LSTM_RANDOM_SEED )

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

      #layer 1: LSTM layer with 12 units and a dropout of 0.4 to prevent overfitting
      # tf.keras.layers.BatchNormalization() ,
      LSTM(units = LSTM_UNITS, return_sequences = False),
      Dropout(LSTM_DROPOUT_RATE) ,

      # layer 2: dense layer for mapping complex non-linear combinations
      Dense(units = LSTM_RELU_DENSE_UNITS, activation = 'relu'),
      Dropout(LSTM_RELU_DENSE_DROPOUT_RATE),

      #layer 3: output layer returning a continuous probability between 0 and 1
      Dense(units = 1, activation = 'sigmoid')
    ])


    lstm_model.compile(

      optimizer = tf.keras.optimizers.Adam(learning_rate = LSTM_LEARNING_RATE),
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
    monitor = 'val_auc_pr' , mode = 'max' , patience = EARLY_STOP_PATIENCE , restore_best_weights = True
  )
  # shrink the learning rate before early stopping gives up entirely,
  # rather than jumping straight from "improving" to "stopped"
  reduce_learning_rate = tf.keras.callbacks.ReduceLROnPlateau(
    monitor = 'val_auc_pr' , mode = 'max' , factor = 0.5 , patience = REDUCE_LEARNING_PATIENCE , min_lr = 1e-5
  )

  # persist the best-validation-epoch model as an explicit artifact,
  # not just restored in-memory weights
  checkpoint = tf.keras.callbacks.ModelCheckpoint(
    output_path , monitor = 'val_auc_pr' , mode = 'max' , save_best_only = True, save_weights_only = False
  )

  # train model
  try:
    logger.info("Training LSTM model")
    lstm_model.fit(
      feature_train_3d , target_train ,
      validation_data = (feature_validation_3d, target_validation),
      epochs = LSTM_EPOCHS ,
      batch_size = LSTM_BATCH_SIZE ,
      class_weight = class_weight_dict ,
      callbacks = [ early_stop, reduce_learning_rate, checkpoint ] ,
      verbose = 1 ,
    )
    logger.info("Training LSTM model done")
    logger.info( "Reloading optimal checkpoint file from disk to overwrite overfitted tail weights" )

    # best_saved_model = tf.keras.models.load_model( output_path )

    # Save the peak model architecture directly back to the final output file
    # best_saved_model.save( output_path )
    # logger.info( f"Optimal point-in-time model successfully locked into: {output_path}" )
    val_probabilities = lstm_model.predict( feature_validation_3d ).flatten()
    lstm_val_prediction_result = pd.DataFrame( {
      "actual_shock_window" : target_validation ,
      "lstm_shock_probability" : val_probabilities
    }, index = pd.DatetimeIndex(validation_dates, name = "date") )

    threshold = get_best_threshold(target_validation,val_probabilities)
    threshold_df = pd.DataFrame( { "threshold" : threshold } , index = [ 0 ] )

    logger.info( f"📈 Mathematically optimized LSTM threshold found at validation: {threshold:.4f}" )
    save_to_csv( data = threshold_df , output_file = BEST_LSTM_THRESHOLD_FILE , logger = logger )
    save_to_csv(data = lstm_val_prediction_result, output_file = LSTM_VALIDATION_PREDICTION_RESULT_FILE, logger = logger, set_index = True )

    generate_lstm_feature_importance( lstm_model = lstm_model , feature_names = feature_names , X_lstm = feature_train_3d , y_lstm = target_train )
    return lstm_model


  except Exception as e:
    raise e

