from multiprocessing import Condition

import tensorflow as tf


def filter_boxes(boxes):
    argmaxes = tf.argmax(boxes[..., 4:], axis=-1)
    return tf.gather(
        boxes[..., :4], tf.squeeze(tf.where(argmaxes != 0)), axis=0
    ), tf.gather(argmaxes - 1, tf.squeeze(tf.where(argmaxes != 0)), axis=0)


if __name__ == "__main__":

    a = tf.random.uniform((7, 9))

    print(a)

    print(filter_boxes(a))
