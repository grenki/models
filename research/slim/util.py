import os

import tensorflow as tf
from preprocessing import preprocessing_factory



def load_images(dir_path, FLAGS, img_size):
    def load_img(img_path):
        img_contents = tf.read_file(img_path)
        return tf.image.decode_png(img_contents, channels=3)
        # img_r, img_g, img_b = tf.split(axis=2, num_or_size_splits=3, value=img)
        # img = tf.cast(tf.concat(axis=2, values=[img_b, img_g, img_r]), dtype=tf.float32)

    image_names = [f for f in os.listdir(dir_path) if f.endswith(".png")]
    image_paths = [os.path.join(dir_path, f) for f in image_names]
    print(len(image_paths))
    # image_paths_ph = tf.placeholder(tf.string, [len(image_paths)])
    img = map(load_img, image_paths)

    #####################################
    # Select the preprocessing function #
    #####################################
    preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
    image_preprocessing_fn = preprocessing_factory.get_preprocessing(
        preprocessing_name,
        is_training=False)

    eval_image_size = img_size

    img = [image_preprocessing_fn(i, eval_image_size, eval_image_size) for i in img]
    return image_names, tf.stack(img)
