# %%
import tensorflow as tf


# %%

# Use the _build_from_signature tricks:
# https://github.com/keras-team/keras/blob/v2.9.0/keras/layers/attention/multi_head_attention.py#L306
class DecoderFirstLayer(tf.keras.layers.Layer):
    def __init__(self, nqueries, nheads, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nqueries = nqueries
        self.nheads = nheads
        self._built_from_signature = False

    def _build_from_signature(self, features):
        self._built_from_signature = True

        self.w = self.add_weight(
            name="queries",
            shape=(self.nqueries, self.nqueries),
            initializer="orthogonal",
            trainable=True,
        )
        self.proj = tf.keras.layers.Dense(features.shape[-1])
        # proj is the equivalent of first attention
        self.attention = tf.keras.layers.MultiHeadAttention(
            num_heads=self.nheads, key_dim=features.shape[-1] // self.nheads
        )
        self.norm1 = tf.keras.layers.BatchNormalization()
        self.ffn = tf.keras.layers.Dense(features.shape[-1])
        self.norm2 = tf.keras.layers.BatchNormalization()

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "nqueries": self.nqueries,
                "nheads": self.nheads,
            }
        )
        return config

    def call(self, features, spatial_encodings):
        if not self._built_from_signature:
            self._build_from_signature(features)
        queries = tf.repeat([self.proj(self.w)], [tf.shape(features)[0]], axis=0)
        tmp = self.norm1(
            self.attention(
                query=queries,
                key=features + spatial_encodings,
                value=features,
            )
            + queries
        )
        return self.norm2(tmp + self.ffn(tmp)), queries


# %%
if __name__ == "__main__":
    dim = 16
    features = tf.keras.layers.Input(shape=(11, dim), name="features")
    spatial_enc = tf.keras.layers.Input(shape=(11, dim), name="spatial_encodings")
    decoder, queries = DecoderFirstLayer(nqueries=17, nheads=2)(
        features=features, spatial_encodings=spatial_enc
    )
    output = tf.keras.layers.Dense(units=1)(decoder)
    model = tf.keras.models.Model([features, spatial_enc], [output, queries])
    model.summary(150)

    import numpy as np

    data_f = np.random.uniform(0, 1, (3, 11, dim))
    data_s = np.random.uniform(0, 1, (3, 11, dim))

    pred = model([data_f, data_s])[0]
    print(pred)

    model.save("test.hdf5")

    model2 = tf.keras.models.load_model(
        "test.hdf5",
        custom_objects={"DecoderFirstLayer": DecoderFirstLayer},
    )

    pred2 = model2([data_f, data_s])[0]
    print(pred2)

    print(np.sum((pred - pred2) ** 2))

    model.compile(
        tf.keras.optimizers.SGD(),
        tf.keras.losses.BinaryCrossentropy(),
    )

    data_X = [
        np.random.uniform(0, 1, (200, 11, dim)),
        np.random.uniform(0, 1, (200, 11, dim)),
    ]
    data_Y = [
        np.random.uniform(0, 1, (200, 17, 1)),
        np.random.uniform(0, 1, (200, 17, dim)),
    ]

    model.fit(data_X, data_Y, epochs=5, batch_size=20)

    pred3 = model([data_f, data_s])[0]

    print(np.sum((pred - pred3) ** 2))

    model.save("test2.hdf5")

    model2 = tf.keras.models.load_model(
        "test2.hdf5",
        custom_objects={"DecoderFirstLayer": DecoderFirstLayer},
    )

    pred4 = model2([data_f, data_s])[0]

    print(np.sum((pred4 - pred3) ** 2))

    # %%
