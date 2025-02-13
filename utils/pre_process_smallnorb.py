# Copyright 2021 Vittorio Mazzia & Francesco Salvetti. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np
import tensorflow as tf
import os
from tqdm.notebook import tqdm


# constants
SAMPLES = 24300
INPUT_SHAPE = 96
PATCH_SMALLNORB = 48
N_CLASSES = 5
MAX_DELTA = 2.0
LOWER_CONTRAST = 0.5
UPPER_CONTRAST = 1.5
PARALLEL_INPUT_CALLS = 16

# This function separates labels from data into two numpy arrays, which can be 
# helpful for pre-processing if the dataset fits entirely in memory. 
# While I don't use it personally, as I rely on TensorFlow's data pipeline 
# to manage the entire process (due to my small GPU), I am keeping this function 
# since it could be useful for others who might not have the same constraints.
def pre_process(ds):
    num_elements = tf.data.experimental.cardinality(ds).numpy()
    X = np.empty((num_elements, INPUT_SHAPE, INPUT_SHAPE, 2))
    y = np.empty((num_elements,))
        
    for index, d in tqdm(enumerate(ds.batch(1))):
        X[index, :, :, 0:1] = d['image']
        X[index, :, :, 1:2] = d['image2']
        y[index] = d['label_category']
    return X, y


def calculate_mean_and_std(dataset, feature, batch_size=1):
    total_sum = 0.0
    total_squared_sum = 0.0
    total_count = 0

    for batch in dataset.batch(batch_size):
        # Flatten the batch to handle any shape (e.g., images)
        batch = tf.reshape(batch[feature], [-1])

        # tf.reduce sum uses the same dtype as the input (which is uint8).
        # Since the sum gets large, we cast to float64 te prevent overflowing
        batch = tf.cast(batch, dtype=tf.float64)
        
        total_sum += tf.reduce_sum(batch)
        total_squared_sum += tf.reduce_sum(tf.square(batch))
        
        total_count += tf.size(batch, out_type=tf.float64)

    # Cast everything to float64 to be sure
    total_sum = tf.cast(total_sum, dtype=tf.float64)
    total_squared_sum = tf.cast(total_squared_sum, tf.float64)

    mean = total_sum / total_count

    # Calculate the variance
    variance = (total_squared_sum / total_count) - tf.square(mean)

    # Calculate the standard deviation
    std = tf.sqrt(variance)

    # .numpy() just ensures we get a number, not tensor
    return mean.numpy(), std.numpy()


def standardize_sample(sample, image1_mean_std, image2_mean_std):
    """Standardize one example
    
    mean and std are passed as arguments so they are only computed once"""
    statistics = [image1_mean_std, image2_mean_std]
    for image, stats in zip(["image", "image2"], statistics):
        mean, std = stats
        # dtype is uint8, which underflows when subtracting the mean
        sample[image] = tf.cast(sample[image], tf.float16)
        sample[image] = (sample[image] - mean) / std
    return sample


def standardize(x, y):
    x[...,0] = (x[...,0] - x[...,0].mean()) / x[...,0].std()
    x[...,1] = (x[...,1] - x[...,1].mean()) / x[...,1].std()
    return x, tf.one_hot(y, N_CLASSES)

def rescale_sample(sample, config):
    size = (config['scale_smallnorb'], config['scale_smallnorb'])
    sample["image"] = tf.image.resize(sample["image"], size)
    sample["image2"] = tf.image.resize(sample["image2"], size)
    return sample

def rescale(x, y, config):
    with tf.device("/cpu:0"):
        x = tf.image.resize(x , [config['scale_smallnorb'], config['scale_smallnorb']])
    return x, y

def test_patch_sample(sample, res):
    sample["image"] = sample["image"][res:-res, res:-res, :]
    sample["image2"] = sample["image2"][res:-res, res:-res, :]
    return sample


def test_patches(x, y, config):
    res = (config['scale_smallnorb'] - config['patch_smallnorb']) // 2
    return x[:,res:-res,res:-res,:], y


def generator(image, label):
    return (image, label), (label, image)

def random_patches(x, y):
    return tf.image.random_crop(x, [PATCH_SMALLNORB, PATCH_SMALLNORB, 2]), y

def random_brightness(x, y):
    return tf.image.random_brightness(x, max_delta=MAX_DELTA), y

def random_contrast(x, y):
    return tf.image.random_contrast(x, lower=LOWER_CONTRAST, upper=UPPER_CONTRAST), y


def generate_tf_data_stream(dataset_train, dataset_test, batch_size):
    # This is the same function as the one below, just changed the way the
    # data is passed (was: numpy arrays, is now: tf Dataset)
    

    dataset_train = dataset_train.map(random_patches,
        num_parallel_calls=PARALLEL_INPUT_CALLS)
    dataset_train = dataset_train.map(random_brightness,
        num_parallel_calls=PARALLEL_INPUT_CALLS)
    dataset_train = dataset_train.map(random_contrast,
        num_parallel_calls=PARALLEL_INPUT_CALLS)
    dataset_train = dataset_train.map(generator,
        num_parallel_calls=PARALLEL_INPUT_CALLS)
    dataset_train = dataset_train.batch(batch_size)
    dataset_train = dataset_train.prefetch(-1)


    dataset_test = dataset_test.cache()
    dataset_test = dataset_test.map(generator,
        num_parallel_calls=PARALLEL_INPUT_CALLS)
    dataset_test = dataset_test.batch(1)
    dataset_test = dataset_test.prefetch(-1)


    return dataset_train, dataset_test


def generate_tf_data(X_train, y_train, X_test_patch, y_test, batch_size):
    dataset_train = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    # dataset_train = dataset_train.shuffle(buffer_size=SAMPLES) not needed if imported with tfds
    dataset_train = dataset_train.map(random_patches,
        num_parallel_calls=PARALLEL_INPUT_CALLS)
    dataset_train = dataset_train.map(random_brightness,
        num_parallel_calls=PARALLEL_INPUT_CALLS)
    dataset_train = dataset_train.map(random_contrast,
        num_parallel_calls=PARALLEL_INPUT_CALLS)
    dataset_train = dataset_train.map(generator,
        num_parallel_calls=PARALLEL_INPUT_CALLS)
    dataset_train = dataset_train.batch(batch_size)
    dataset_train = dataset_train.prefetch(-1)

    dataset_test = tf.data.Dataset.from_tensor_slices((X_test_patch, y_test))
    dataset_test = dataset_test.cache()
    dataset_test = dataset_test.map(generator,
        num_parallel_calls=PARALLEL_INPUT_CALLS)
    dataset_test = dataset_test.batch(1)
    dataset_test = dataset_test.prefetch(-1)

    return dataset_train, dataset_test
