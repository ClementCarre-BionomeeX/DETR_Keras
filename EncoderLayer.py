# %%
import tensorflow as tf

# %%

# Use the _build_from_signature tricks:
# https://github.com/keras-team/keras/blob/v2.9.0/keras/layers/attention/multi_head_attention.py#L306
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, nheads, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nheads = nheads
        self._built_from_signature = False

    def _build_from_signature(self, inputs):
        self._built_from_signature = True

        self.attention = tf.keras.layers.MultiHeadAttention(
            num_heads=self.nheads, key_dim=inputs.shape[-1] // self.nheads
        )
        self.norm1 = tf.keras.layers.BatchNormalization()

        self.ffn = tf.keras.layers.Dense(inputs.shape[-1])
        self.norm2 = tf.keras.layers.BatchNormalization()

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "nheads": self.nheads,
            }
        )
        return config

    def call(self, inputs, spatial_encodings):
        if not self._built_from_signature:
            self._build_from_signature(inputs)

        tmp = self.norm1(
            self.attention(
                query=inputs + spatial_encodings,
                key=inputs + spatial_encodings,
                value=inputs,
            )
            + inputs
        )

        return self.norm2(tmp + self.ffn(tmp))


# %%
if __name__ == "__main__":

    inputs = tf.keras.layers.Input(shape=(11, 8))
    spatial_enc = tf.keras.layers.Input(shape=(11, 8))

    test_layer = EncoderLayer(nheads=2)(
        inputs=inputs,
        spatial_encodings=spatial_enc,
    )

    model = tf.keras.models.Model(
        [inputs, spatial_enc], [test_layer]
    )

    model.summary(150)

    import numpy as np

    data_i = np.random.uniform(0, 1, (3, 11, 8))
    data_s = np.random.uniform(0, 1, (3, 11, 8))

    pred = model([data_i, data_s])
    print(pred.shape)

    model.save("test.hdf5")

    model2 = tf.keras.models.load_model(
        "test.hdf5", custom_objects={"EncoderLayer": EncoderLayer}
    )

    pred2 = model2([data_i, data_s])
    print(pred2.shape)

    print(np.sum((pred - pred2) ** 2))

    model.compile(
        tf.keras.optimizers.SGD(),
        tf.keras.losses.MeanSquaredError(),
    )

    data_X = [
        np.random.uniform(0, 1, (200, 11, 8)),
        np.random.uniform(0, 1, (200, 11, 8)),
    ]

    data_Y = np.random.uniform(0, 1, (200, 11, 8))

    model.fit(data_X, data_Y, epochs=5, batch_size=20)

    pred3 = model([data_i, data_s])
    print(pred3.shape)

    model.save("test2.hdf5")

    model2 = tf.keras.models.load_model(
        "test2.hdf5", custom_objects={"EncoderLayer": EncoderLayer}
    )

    pred4 = model([data_i, data_s])
    print(pred4.shape)

    print(np.sum((pred - pred3) ** 2))
    print(np.sum((pred4 - pred3) ** 2))
# %%
