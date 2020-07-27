import os
import sys
import pdb
import numpy as np

sys.path.insert(0, './')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from model_scannet import PointConvModel
from tensorflow import keras
import tensorflow as tf
import tensorflow_datasets as tfds
from loadData import getG4Arrays
from plotUtils import Plot
from datetime import datetime

timestamp = str(datetime.now().year)+str("{:02d}".format(datetime.now().month))+str("{:02d}".format(datetime.now().day))+str("{:02d}".format(datetime.now().hour))+str("{:02d}".format(datetime.now().minute))

# limit full GPU memory allocation by gradually allocating memory as needed
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

tf.random.set_seed(1234)
tf.keras.backend.set_floatx('float32')


def load_dataset(in_file, batch_size):

    assert os.path.isfile(in_file), '[error] dataset path not found'

    n_points = 8192
    shuffle_buffer = 1000

    def _extract_fn(data_record):

        in_features = {
            'points': tf.io.FixedLenFeature([n_points * 3], tf.float32),
            'labels': tf.io.FixedLenFeature([n_points], tf.int64)
        }

        return tf.io.parse_single_example(data_record, in_features)

    def _preprocess_fn(sample):

        points = sample['points']
        labels = sample['labels']

        points = tf.reshape(points, (n_points, 3))
        labels = tf.reshape(labels, (n_points, 1))

        shuffle_idx = tf.range(points.shape[0])
        points = tf.gather(points, shuffle_idx)
        labels = tf.gather(labels, shuffle_idx)

        return points, labels

    dataset = tf.data.TFRecordDataset(in_file)
    dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.map(_extract_fn)
    dataset = dataset.map(_preprocess_fn)
    dataset = dataset.batch(batch_size, drop_remainder=True)

    return dataset

def load_G4_dataset(data_input, data_output, n_points=1000, batch_size=2048, shuffle_buffer_size=1000):
    '''
    From np.array make point-clouds datasets with n_points points each
    '''

    def _convertType_fn(x, y):
        outType=tf.float32
        return tf.cast(x, outType), tf.cast(y, outType)

    dataset = tf.data.Dataset.from_tensor_slices((data_input,data_output))
    dataset = dataset.map(_convertType_fn)
    # slice point-clouds
    dataset = dataset.batch(n_points, drop_remainder=True)
    dataset = dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.batch(batch_size, drop_remainder=True)

    return dataset

def train():

    global model

    model = PointConvModel(config['batch_size'], config['bn'], num_classes=1)
    
    callbacks = [
        # keras.callbacks.EarlyStopping(
            # 'val_mean_absolute_error', min_delta=0.1, patience=3),
        keras.callbacks.TensorBoard(
            './logs/{}'.format(config['log_dir']), update_freq=50),
        keras.callbacks.ModelCheckpoint(
            './logs/{}/model/weights'.format(config['log_dir']), 'val_mean_absolute_error', save_best_only=True)
    ]

    model.build((config['batch_size'], 1000, 6))
    print(model.summary())

    model.compile(
        optimizer=keras.optimizers.Adam(config['lr']),
        loss=keras.losses.MeanSquaredError(),
        metrics=[keras.losses.MeanSquaredError(),keras.losses.MeanAbsoluteError()]
    )

    model.fit(
        train_G4_ds,
        validation_data=validate_G4_ds,
        validation_steps=10,
        validation_freq=1,
        callbacks=callbacks,
        epochs=50,
        verbose=1
    )


if __name__ == '__main__':

    config = {
        'train_G4_data': '/users/ekourlitis/distanceRegression/data/sphere/G4Sphere_train.csv',
        'validate_G4_data': '/users/ekourlitis/distanceRegression/data/sphere/G4Sphere_validate.csv',
        'batch_size': 4,
        'lr': 5e-3,
        'bn': False,
        'log_dir': 'scannet_2'
    }

    # my G4 data
    trainX, trainY  = getG4Arrays(config['train_G4_data'])
    train_G4_ds = load_G4_dataset(trainX, trainY, batch_size=config['batch_size'])
    validateX, validateY  = getG4Arrays(config['validate_G4_data'])
    validate_G4_ds = load_G4_dataset(validateX, validateY, batch_size=config['batch_size'])
    validate_G4_ds_lengths = validate_G4_ds.map(lambda x,y : y) # take the lengths only
    # validate_G4_ds_lengths_numpy = tfds.as_numpy(validate_G4_ds_lengths) # python generator
    truth_validateY = np.array(list(tfds.as_numpy(validate_G4_ds_lengths))).flatten() # flat np.array

    train()

    # make some predictions
    print("Calculating validation predictions for %i points..." % len(validateX))
    pred_validateY = model.predict(validate_G4_ds)
    pred_validateY = pred_validateY.flatten()

    # plot
    validationPlots = Plot('validation', timestamp, truth=truth_validateY, prediction=pred_validateY)
    validationPlots.plotPerformance()

    print("Done!")
