from __future__ import division
from __future__ import print_function

import os

import numpy as np
import tensorflow as tf

import preprocessing.preprocessing_factory as preprocessing_factory


def load_images(dir_path):
    image_names, image_paths = get_images_names_and_paths(dir_path)
    print(len(image_paths))
    image = tf.train.slice_input_producer(
        [image_paths],
        shuffle=False)
    # image_paths_ph = tf.placeholder(tf.string, [len(image_paths)])
    image = load_img(tf.cast(image[0], tf.string))
    return image_names, image


def load_img(img_path):
    img_contents = tf.read_file(img_path)
    return tf.image.decode_png(img_contents, channels=3)
    # img_r, img_g, img_b = tf.split(axis=2, num_or_size_splits=3, value=img)
    # img = tf.cast(tf.concat(axis=2, values=[img_b, img_g, img_r]), dtype=tf.float32)


def get_images_names_and_paths(dir_path):
    image_names = [f for f in sorted(os.listdir(dir_path)) if f.endswith(".png")]
    image_paths = [os.path.join(dir_path, f) for f in image_names]
    return image_names, image_paths


def load_dataset(root_dir):
    labels_folder = os.path.join(root_dir, 'labels')
    res_img_paths = []
    res_labels = []

    for dir in sorted(os.listdir(root_dir)):
        dir_path = os.path.join(root_dir, dir)

        if not os.path.isdir(dir_path):
            continue

        _, image_paths = get_images_names_and_paths(dir_path)
        if not image_paths:
            continue

        labels = np.load(os.path.join(labels_folder, dir + '.npy'))

        res_img_paths.extend(image_paths)
        res_labels.extend(np.uint8(labels))

        break

    print(len(res_img_paths))

    # todo remove (debug)
    # res_img_paths = res_img_paths[:20]
    # res_labels= res_labels[:20]

    res_img_paths = tf.constant(res_img_paths, dtype=tf.string)
    result = tf.train.slice_input_producer(
        [res_img_paths, res_labels],
        shuffle=True)

    img = result[0]
    # print(img.dtype)

    label = result[1]
    label = tf.cast(label, tf.uint8)

    img = tf.cast(img, tf.string)
    # img = tf.Print(img, [img, label]) # todo remove (debug)
    img = load_img(img)
    # print(img.dtype)
    img = tf.cast(img, tf.float32)

    return img, label


def get_var(name):
    return [v for v in tf.global_variables() if v.name == name][0]

def image_info(image):
    return tf.reduce_mean(image), tf.reduce_max(image), tf.reduce_min(image)


def preprocessing(image, network_fn, FLAGS):
    train_image_size = FLAGS.train_image_size or network_fn.default_image_size

    preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
    image_preprocessing_fn = preprocessing_factory.get_preprocessing(
        preprocessing_name,
        is_training=True)
    return image_preprocessing_fn(image, train_image_size, train_image_size)
