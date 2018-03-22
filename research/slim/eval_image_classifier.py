# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Generic evaluation script that evaluates a model using a given dataset."""

from __future__ import division, print_function

import math

import tensorflow as tf
import tensorflow.contrib.slim as slim

import common
from nets import nets_factory
from preprocessing import preprocessing_factory

tf.app.flags.DEFINE_integer(
    'eval_every_sec', 180, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'batch_size', 50, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'max_num_batches', None,
    'Max number of batches to evaluate by default use all.')

tf.app.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.app.flags.DEFINE_string(
    'checkpoint_path', None, '')

tf.app.flags.DEFINE_string(
    'eval_dir', '/tmp/tfmodel/', 'Directory where the results are saved to.')

tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 4,
    'The number of threads used to create the batches.')

tf.app.flags.DEFINE_string(
    'dataset_name', 'imagenet', 'The name of the dataset to load.')

tf.app.flags.DEFINE_string(
    'dataset_split_name', 'test', 'The name of the train/test split.')

tf.app.flags.DEFINE_string(
    'dataset_dir', None, 'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

tf.app.flags.DEFINE_string(
    'model_name', 'inception_v3', 'The name of the architecture to evaluate.')

tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
                                'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

tf.app.flags.DEFINE_integer(
    'eval_image_size', None, 'Eval image size')

FLAGS = tf.app.flags.FLAGS

import util
import os

NUM_CLASSES = 2
num_samples = 1700


def main(_):
    if not FLAGS.dataset_dir:
        raise ValueError('You must supply the dataset directory with --dataset_dir')

    dataset_dir = os.path.join(FLAGS.dataset_dir)

    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Graph().as_default():
        tf_global_step = slim.get_or_create_global_step()  # ####################
        # # Select the model #
        # ####################
        network_fn = nets_factory.get_network_fn(
            FLAGS.model_name,
            num_classes=NUM_CLASSES,
            is_training=False)

        image, label = util.load_dataset(dataset_dir)

        #####################################
        # Select the preprocessing function #
        #####################################

        image = util.preprocessing(image, network_fn, FLAGS)

        images, labels = tf.train.batch(
            [image, label],
            batch_size=FLAGS.batch_size,
            num_threads=FLAGS.num_preprocessing_threads,
            capacity=5 * FLAGS.batch_size)

        ####################
        # Define the model #
        ####################
        logits, endpoints = network_fn(images)

        if FLAGS.moving_average_decay:
            variable_averages = tf.train.ExponentialMovingAverage(
                FLAGS.moving_average_decay, tf_global_step)
            variables_to_restore = variable_averages.variables_to_restore(
                slim.get_model_variables())
            variables_to_restore[tf_global_step.op.name] = tf_global_step
        else:
            variables_to_restore = slim.get_variables_to_restore()

        predictions = tf.argmax(logits, 1)
        labels = tf.squeeze(labels)

        one_hot_labels = slim.one_hot_encoding(labels, NUM_CLASSES)
        softmax = tf.losses.softmax_cross_entropy(one_hot_labels, logits, loss_collection=None)
        softmax2 = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_labels, logits=logits)

        evals = []
        evals.append(tf.Print(images, util.image_info(images), "images_info"))
        evals.append(tf.Print(images, [images], "images"))
        # evals.append(tf.Print(predictions, [labels], 'labels', summarize=30))
        # evals.append(tf.Print(predictions, [predictions], 'predictions', summarize=30))
        # evals.append(tf.Print(one_hot_labels, [one_hot_labels], 'one hot', summarize=30))
        # evals.append(tf.Print(logits, [logits], 'logits', summarize=30))
        # evals.append(tf.Print(softmax, [softmax], 'softmax', summarize=30))
        # mean = util.get_var(u'InceptionV4/Mixed_6g/Branch_2/Conv2d_0c_1x7/BatchNorm/moving_mean:0')
        # variance = util.get_var(u'InceptionV4/Mixed_6g/Branch_2/Conv2d_0c_1x7/BatchNorm/moving_variance:0')
        # beta = util.get_var(u'InceptionV4/Mixed_6g/Branch_2/Conv2d_0c_1x7/BatchNorm/beta:0')
        # evals.append(tf.Print(logits, [mean, variance, beta]))
        # evals.append(tf.Print(predictions, [softmax2], 'softmax2', summarize=30))

        # for name, v in endpoints.items():
        #     evals.append(tf.Print(v, [v], name, summarize=common.SUMMARIZE_COUNT))

        # Define the metrics:
        names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
            'TestAccuracy': slim.metrics.streaming_accuracy(predictions, labels),
            'TestLoss': slim.metrics.streaming_mean_tensor(softmax),
        })

        # Print the summaries to screen.
        for name, value in names_to_values.items():
            summary_name = 'eval/%s' % name
            op = tf.summary.scalar(summary_name, value, collections=[])
            op = tf.Print(op, [value], summary_name)
            tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)

        # TODO(sguada) use num_epochs=1
        if FLAGS.max_num_batches:
            num_batches = FLAGS.max_num_batches
        else:
            # This ensures that we make a single pass over all of the data.
            num_batches = math.ceil(num_samples / float(FLAGS.batch_size))

        checkpoint_path = FLAGS.checkpoint_path

        tf.logging.info('Evaluating %s' % checkpoint_path)

        slim.evaluation.evaluation_loop(
            master=FLAGS.master,
            checkpoint_dir=checkpoint_path,
            logdir=FLAGS.eval_dir,
            num_evals=num_batches,
            eval_op=list(names_to_updates.values()) + evals,
            variables_to_restore=variables_to_restore
        )


if __name__ == '__main__':
    tf.app.run()
