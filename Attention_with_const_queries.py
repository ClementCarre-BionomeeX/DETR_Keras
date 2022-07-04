# %%
import tensorflow as tf

# %%
class Attention_with_const_queries(tf.keras.layers.Layer):
    def __init__(self, nqueries, nheads, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nqueries = nqueries
        self.nheads = nheads

    def build(self, input_shape):
        super().build(input_shape)
        self.w = self.add_weight(
            name="queries",
            shape=(self.nqueries, self.nqueries),
            initializer="orthogonal",
            trainable=True
        )
        self.proj = tf.keras.layers.Dense(input_shape[-1])
        self.proj.build((self.nqueries, self.nqueries))
        self.attention = tf.keras.layers.MultiHeadAttention(
            num_heads=self.nheads,
            key_dim=input_shape[-1] // self.nheads
        )
        self.attention._build_from_signature(
            query=(None, self.nqueries, input_shape[-1]),
            key=input_shape,
            value=input_shape
        )

    def get_config(self):
        config = super().get_config()
        config.update({
            "nqueries" : self.nqueries,
            "nheads" : self.nheads,
        })
        return config

    def call(self, inputs):
        return self.attention(
            query=tf.repeat([self.proj(self.w)], [tf.shape(inputs)[0]], axis=0), key=inputs, value=inputs
        )

# %%
if __name__ == "__main__":

    input_model = tf.keras.layers.Input(shape=(11, 8), name="input")
    constant_input = Attention_with_const_queries(nqueries=17, nheads=2)(input_model)
    output = tf.keras.layers.Dense(units=1)(constant_input)
    model = tf.keras.models.Model([input_model], [output])
    model.summary(150)

    # %%
    import numpy as np

    data = np.random.uniform(0, 1, (3, 11, 8))

    # %%
    pred = model(data)
    print(pred)

    # %%
    model.save("test.hdf5")

    # %%
    model2 = tf.keras.models.load_model("test.hdf5", custom_objects={
        "Attention_with_const_queries": Attention_with_const_queries
    })

    # %%
    pred2 = model2(data)
    print(pred2)

    # %%
    print(np.sum((pred - pred2) ** 2))

    # %%
    model.compile(
        tf.keras.optimizers.SGD(),
        tf.keras.losses.BinaryCrossentropy(),
    )

    # %%

    data_X = np.random.uniform(0, 1, (200, 11, 8))
    data_Y = np.random.uniform(0, 1, (200, 17, 1))


    # %%
    model.fit(
        data_X, data_Y, epochs=200, batch_size=20
    )
    # %%

    pred3 = model(data)

    # %%
    print(np.sum((pred - pred3) ** 2))

    # %%
    model.save("test2.hdf5")
    # %%

    model2 = tf.keras.models.load_model("test2.hdf5", custom_objects={
        "Attention_with_const_queries": Attention_with_const_queries
    })
    # %%

    pred4 = model2(data)

    # %%
    print(np.sum((pred4 - pred3) ** 2))

    # %%
