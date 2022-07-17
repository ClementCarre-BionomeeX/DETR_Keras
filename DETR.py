import unittest
from warnings import filters

import tensorflow as tf

from DecoderFirstLayer import DecoderFirstLayer
from DecoderLayer import DecoderLayer
from EncoderLayer import EncoderLayer
from ImageEncoding import ImageEncoding
from SineSpatialEncoding import SineSpatialEncoding


def DETR(
    nqueries=128,
    nclasses_plus_none=2,
    backbone_dim=16,
    backbone_depth=4,
    encoding_grid_dim=32,
    encoding_nqueries=64,
    encoding_margin=4,
    encoder_depth=4,
    encoder_nheads=8,
    decoder_nheads=8,
    decoder_depth=4,
):
    """DETR struct
    INPUT -> BACKBONE -> TRANSFORMER -> bbox  -> Concat
                                   |--> class   -^
    """

    assert encoder_depth >= 2, f"encoder_depth must be greater or equal to 2, {encoder_depth} given"

    # input layer => images <None, None, 3>
    input_layer = tf.keras.layers.Input((None, None, 3), name="input_image")

    # Backbone : conv2D + residuals
    backbone = tf.keras.layers.Conv2D(
        filters=backbone_dim,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding="same",
        name="Backbone_start",
    )(input_layer)
    backbone = tf.keras.layers.Activation("swish")(backbone)
    backbone = tf.keras.layers.BatchNormalization()(backbone)

    for _ in range(backbone_depth):
        conv = tf.keras.layers.Conv2D(
            filters=backbone_dim, kernel_size=(5, 5), strides=(1, 1), padding="same"
        )(backbone)
        conv = tf.keras.layers.Activation("swish")(conv)
        backbone = tf.keras.layers.BatchNormalization()(conv + backbone)

        backbone_dim *= 2
        backbone = tf.keras.layers.Conv2D(
            filters=backbone_dim, kernel_size=(5, 5), strides=(2, 2), padding="same"
        )(backbone)
        backbone = tf.keras.layers.Activation("swish")(backbone)

    backbone = tf.keras.layers.Conv2D(
        filters=backbone_dim, kernel_size=(1, 1), strides=(1, 1), name="Backbone_end"
    )(backbone)

    # features, positions (not used)
    #features, _ = ImageEncoding(
    features = ImageEncoding(
        encoding_grid_dim, encoding_nqueries, encoding_margin, name="Encoding"
    )(backbone)

    spatialEncoding = SineSpatialEncoding(name="Spatial_encoding")(features)

    while features.shape[-2] > 1:
        features = tf.keras.layers.Conv2D(
            filters=backbone_dim, kernel_size=(1, 1), strides=(1, 2), padding="same"
        )(features)
        features = tf.keras.layers.Activation("swish")(features)
        features = tf.keras.layers.BatchNormalization()(features)

    features = tf.keras.layers.Reshape([-1, backbone_dim], name="Features")(features)

    encoding = features

    encoding = EncoderLayer(encoder_nheads, name="Encoding_start")(
        encoding, spatialEncoding
    )
    for _ in range(encoder_depth - 2):
        encoding = EncoderLayer(encoder_nheads)(encoding, spatialEncoding)
    encoding = EncoderLayer(encoder_nheads, name="Encoding_end")(
        encoding, spatialEncoding
    )

    encoding, queries = DecoderFirstLayer(
        nqueries, decoder_nheads, name="Decoding_start"
    )(encoding, spatialEncoding)

    for _ in range(decoder_depth - 1):
        encoding = DecoderLayer(decoder_nheads)(
            encoding, queries, features, spatialEncoding
        )

    encoding = DecoderLayer(decoder_nheads, name="Decoding_end")(
        encoding, queries, features, spatialEncoding
    )

    # First output is bbox
    bbox = tf.keras.layers.Dense(units=4)(encoding)
    bbox = tf.keras.layers.Activation("sigmoid", name="Bbox")(bbox)

    # Second one is class
    classes = tf.keras.layers.Dense(units=nclasses_plus_none)(encoding)
    classes = tf.keras.layers.Activation("softmax", name="Classes")(classes)

    # Final Output is concatenation
    output = tf.keras.layers.Concatenate(name="Output")([bbox, classes])

    return tf.keras.models.Model([input_layer], [output])


if __name__ == "__main__":

    model = DETR(nclasses_plus_none=11)

    model.summary(250)

    model.save("test.hdf5")

    model2 = tf.keras.models.load_model(
        "test.hdf5",
        custom_objects={
            "ImageEncoding": ImageEncoding,
            "SineSpatialEncoding": SineSpatialEncoding,
            "EncoderLayer": EncoderLayer,
            "DecoderFirstLayer": DecoderFirstLayer,
            "DecoderLayer": DecoderLayer,
        },
    )

    model2.summary(250)

    import numpy as np

    image = np.random.uniform(0, 1, (1, 1920, 1080, 3))

    boxes = model(image)

    print(boxes)

    from output_filtering import filter_boxes

    print(filter_boxes(tf.squeeze(boxes)))
