import tensorflow as tf


def fillEmptyBoxes(A, B):
    B = tf.concat(
        [
            B,
            tf.cast(
                tf.tile(
                    [tf.tile(
                        [tf.concat(
                            [
                                tf.constant([0, 0, 0, 0, 1]),
                                tf.repeat([0], tf.shape(A)[-1] - 5)
                            ],
                            axis=0
                        )],
                        [tf.shape(A)[1] - tf.shape(B)[1], 1]
                    )],
                    [tf.shape(A)[0], 1, 1]
                ),
                tf.float32
            )
        ],
        axis = -2
    )
    return B
