import tensorflow as tf


class ImageEncoding(tf.keras.layers.Layer):
    def __init__(self, grid_dim, nqueries, margin: int = 0, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.grid_dim = grid_dim
        self.nqueries = nqueries
        self.positions = tf.cast(
            tf.reshape(
                tf.transpose(
                    tf.meshgrid(
                        tf.linspace(0, 1, grid_dim + 1)[:-1],
                        tf.linspace(0, 1, grid_dim + 1)[:-1],
                    )
                ),
                (grid_dim * grid_dim, 2),
            ),
            tf.float32,
        )
        self.margin = margin
        self._built_from_signature = False

    def _build_from_signature(self, images):
        self._built_from_signature = True

        self.queries = self.add_weight(
            name="queries",
            shape=(self.nqueries, images.shape[-1]),
            initializer="orthogonal",
            trainable=True,
        )

        image = tf.keras.layers.Input((None, images.shape[-1]))

        self.encodingmodel = tf.keras.models.Model(
            [image], [tf.keras.layers.Attention()([self.queries, image, image])]
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "nqueries": self.nqueries,
                "grid_dim": self.grid_dim,
                "margin": self.margin,
            }
        )
        return config

    @staticmethod
    def _find_pos(image, pos, step, margin, encoding):
        image_h = tf.cast(tf.shape(image)[0], tf.float32)
        image_w = tf.cast(tf.shape(image)[1], tf.float32)

        h_start = tf.math.maximum(tf.cast(image_h * pos[0], tf.int32) - margin, 0)
        h_end = tf.math.minimum(
            tf.cast(image_h * (pos[0] + step), tf.int32) + margin, tf.shape(image)[0]
        )
        w_start = tf.math.maximum(tf.cast(image_w * pos[1], tf.int32) - margin, 0)
        w_end = tf.math.minimum(
            tf.cast(image_w * (pos[1] + step), tf.int32) + margin, tf.shape(image)[1]
        )

        # add a batch dimension
        region = tf.reshape(
            image[h_start:h_end, w_start:w_end, :], [1, -1, tf.shape(image)[-1]]
        )

        # remove batch dimension
        return tf.squeeze(encoding(region), axis=0)

    @staticmethod
    def _find_all_pos(image, grid_dim, positions, margin, encoding):
        def curried_find_pos(pos):
            return ImageEncoding._find_pos(image, pos, 1 / grid_dim, margin, encoding)

        regions = tf.map_fn(curried_find_pos, positions, dtype=tf.float32)

        return regions

    def call(self, images):
        if not self._built_from_signature:
            self._build_from_signature(images)

        def curried_find_all_pos(image):
            return ImageEncoding._find_all_pos(
                image, self.grid_dim, self.positions, self.margin, self.encodingmodel
            )

        regions = tf.map_fn(curried_find_all_pos, images, dtype=tf.float32)

        # return positions ??
        return regions  # , self.positions


if __name__ == "__main__":

    grid = 16
    nq = 64
    margin = 2
    image = tf.keras.layers.Input((None, None, 3))
    encoding, positions = ImageEncoding(grid, nq, margin=margin)(image)

    model = tf.keras.models.Model([image], [encoding, positions])

    model.summary(150)

    import numpy as np

    data = np.random.uniform(0, 1, (1, 1920, 1080, 3))

    pred = model(data)[0]

    print(pred)

    model.save("test.hdf5")

    model2 = tf.keras.models.load_model(
        "test.hdf5",
        custom_objects={"ImageEncoding": ImageEncoding},
    )

    pred2 = model2(data)[0]
    print(pred2)

    print(np.sum((pred - pred2) ** 2))

    model.compile(
        tf.keras.optimizers.SGD(),
        tf.keras.losses.MeanSquaredError(),
    )

    data_X = np.random.uniform(0, 1, (200, 128, 128, 3))
    data_Y = [
        np.random.uniform(0, 1, (200, grid * grid, nq, 3)),
        np.random.uniform(0, 1, (200, grid * grid, 2)),
    ]

    model.fit(data_X, data_Y, epochs=5, batch_size=20)

    pred3 = model(data)[0]
    print(pred3.shape)

    model.save("test2.hdf5")

    model2 = tf.keras.models.load_model(
        "test2.hdf5", custom_objects={"ImageEncoding": ImageEncoding}
    )

    pred4 = model(data)[0]
    print(pred4.shape)

    print(np.sum((pred - pred3) ** 2))
    print(np.sum((pred4 - pred3) ** 2))

    print(model(data)[0])
