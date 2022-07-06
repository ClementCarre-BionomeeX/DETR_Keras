import tensorflow as tf


class PackedOutput(tf.experimental.BatchableExtensionType):
    out1: tf.Tensor
    out2: tf.Tensor

    shape = property(lambda self: self.out2.shape)
    dtype = property(lambda self: self.out2.dtype)

    class Spec:
        def __init__(self, shape, dtype=tf.float32):
            self.out1 = tf.TensorSpec(shape, dtype)
            self.out2 = tf.TensorSpec(shape, dtype)

        # shape and dtype hold no meaning in this context, so we use a dummy
        # to stop Keras from complaining
        shape: tf.TensorShape = tf.constant(1.0).shape
        dtype: tf.DType = tf.constant(1.0).dtype


# these two functions have no meaning, but need dummy implementations
# to stop Keras from complaining
@tf.experimental.dispatch_for_api(tf.shape)
def packed_shape(input: PackedOutput, out_type=tf.int32, name=None):
    return tf.shape(input.out1)


@tf.experimental.dispatch_for_api(tf.cast)
def packed_cast(x: PackedOutput, dtype: str, name=None):
    return x


class PackingLayer(tf.keras.layers.Layer):
    def call(self, inputs, training=None):
        first_out, second_out = inputs
        packed_out = PackedOutput(first_out, second_out)
        return packed_out


inp = tf.keras.layers.Input((7, 11))
a = tf.keras.layers.Dense(3)(inp)
b = tf.keras.layers.Dense(5)(inp)
packed = PackingLayer()([a, b])

model = tf.keras.models.Model([inp], [packed])

model.summary(150)


def customLoss(ytrue, ypred):
    # tf.print(ypred)
    # tf.print(ytrue)

    return tf.math.reduce_mean((ypred.out1 - ytrue[0]) ** 2) + tf.math.reduce_mean(
        (ypred.out2 - ytrue[1]) ** 2
    )


import numpy as np

data = np.random.uniform(0, 1, (1, 7, 11))

pred = model(data)

print(pred)


dataX = np.random.uniform(0, 1, (200, 7, 11))
dataY = [
    #[
        np.random.uniform(0, 1, (1, 7, 3)),
        np.random.uniform(0, 1, (1, 7, 5)),
    #],
    #np.random.uniform(0, 1, (1,)),
]

print(dataY)

print(customLoss(dataY, pred))

# model.compile(tf.keras.optimizers.SGD(), loss=customLoss)


# model.fit(dataX, dataY, epochs=1)
