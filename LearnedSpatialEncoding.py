#%%
import tensorflow as tf

# %%

# Use the _build_from_signature tricks:
# https://github.com/keras-team/keras/blob/v2.9.0/keras/layers/attention/multi_head_attention.py#L306
class LearnedSpatialEncoding(tf.keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._built_from_signature = False

    def _build_from_signature(self, features):
        self._built_from_signature = True

        self.w = self.add_weight(
            name="encodings",
            shape=(features.shape[-2], features.shape[-1]),
            initializer="orthogonal",
            trainable=True,
        )

    def get_config(self):
        config = super().get_config()
        config.update({})
        return config

    def call(self, features):
        if not self._built_from_signature:
            self._build_from_signature(features)

        return tf.repeat([self.w], [tf.shape(features)[0]], axis=0)

# %%
if __name__ == "__main__":
    dim = 16
    features = tf.keras.layers.Input(shape=(11, dim), name="features")
    spatial_enc = LearnedSpatialEncoding()(features)

    model = tf.keras.models.Model([features], [spatial_enc])
    model.summary(150)

    import numpy as np

    data_f = np.random.uniform(0, 1, (3, 11, dim))

    pred = model([data_f])
    print(pred)

    model.save("test.hdf5")

    model2 = tf.keras.models.load_model(
        "test.hdf5",
        custom_objects={"LearnedSpatialEncoding": LearnedSpatialEncoding},
    )

    pred2 = model2([data_f])
    print(pred2)

    print(np.sum((pred - pred2) ** 2))

    model.compile(
        tf.keras.optimizers.SGD(),
        tf.keras.losses.MeanSquaredError(),
    )

    data_X = [
        np.random.uniform(0, 1, (200, 11, dim)),
    ]
    data_Y = np.random.uniform(0, 1, (200, 11, dim))

    model.fit(data_X, data_Y, epochs=5, batch_size=20)

    pred3 = model([data_f])

    print(np.sum((pred - pred3) ** 2))

    model.save("test2.hdf5")

    model2 = tf.keras.models.load_model(
        "test2.hdf5",
        custom_objects={"LearnedSpatialEncoding": LearnedSpatialEncoding},
    )

    pred4 = model2([data_f])

    print(np.sum((pred4 - pred3) ** 2))
