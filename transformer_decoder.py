# %%
import tensorflow as tf

# %%

# use the _build_from_signature tricks:
# https://github.com/keras-team/keras/blob/v2.9.0/keras/layers/attention/multi_head_attention.py#L306
class transformer_decoder(tf.keras.layers.Layer):
    def __init__(self, nheads, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nheads = nheads
        self._built_from_signature = False

    def _build_from_signature(self, query, key, value):
        self._built_from_signature = True
        self.attention = tf.keras.layers.MultiHeadAttention(
            num_heads=self.nheads, key_dim=query.shape[-1] // self.nheads
        )
        self.proj = tf.keras.layers.Dense(value.shape[-1])

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "nheads": self.nheads,
            }
        )
        return config

    def call(self, query, key, value):
        if not self._built_from_signature:
            self._build_from_signature(query, key, value)
        tmp = (
            self.attention(query=query, key=key, value=value) + query
        )
        return tmp + self.proj(tmp)


# %%
if __name__ == "__main__":

    inp1 = tf.keras.layers.Input(shape=(11, 8))
    inp2 = tf.keras.layers.Input(shape=(7, 8))

    test_layer = transformer_decoder(nheads=2)(query=inp1, key=inp2, value=inp2)

    model = tf.keras.models.Model([inp1, inp2], [test_layer])

    model.summary(150)

    import numpy as np

    data1 = np.random.uniform(0, 1, (3, 11, 8))
    data2 = np.random.uniform(0, 1, (3, 7, 8))

    pred = model([data1, data2])
    print(pred.shape)

    model.save("test.hdf5")

    model2 = tf.keras.models.load_model(
        "test.hdf5", custom_objects={"transformer_decoder": transformer_decoder}
    )

    pred2 = model2([data1, data2])
    print(pred2.shape)

    print(np.sum((pred - pred2) ** 2))

    model.compile(
        tf.keras.optimizers.SGD(),
        tf.keras.losses.MeanSquaredError(),
    )

    data_X = [np.random.uniform(0, 1, (200, 11, 8)), np.random.uniform(0, 1, (200, 7, 8))]
    data_Y = np.random.uniform(0, 1, (200, 11, 8))

    model.fit(data_X, data_Y, epochs=200, batch_size=20)

    pred3 = model([data1, data2])
    print(pred3.shape)

    model.save("test2.hdf5")

    model2 = tf.keras.models.load_model(
        "test2.hdf5", custom_objects={"transformer_decoder": transformer_decoder}
    )

    pred4 = model([data1, data2])
    print(pred4.shape)

    print(np.sum((pred - pred3) ** 2))
    print(np.sum((pred4 - pred3) ** 2))
# %%
