from builtins import print

import tensorflow as tf

from giou import _calculate_giou
from HungarianMatcher import hungarian_matcher


def hungarianLoss(cost_class=1, cost_bbox=1, cost_giou=1):
    @tf.function
    def _internal_computation(ytrue, ypred):
        indices = tf.stop_gradient(
            hungarian_matcher(cost_class, cost_bbox, cost_giou)(ypred, ytrue)
        )
        outputs = tf.gather(ypred, indices[:, 0, :], batch_dims=1)
        targets = tf.gather(ytrue, indices[:, 1, :], batch_dims=1)
        # hypothesis => same number of boxes for outputs and targets
        # Should be done in datagenerator
        # -> targets = fillEmptyBoxes(outputs, targets)

        # giou loss between outputs and targets
        giou_loss = tf.math.reduce_mean(
            _calculate_giou(outputs[..., :4], targets[..., :4]), axis=-1
        )

        # cross_entropy loss between outputs and targets
        labelossfunction = lambda i: tf.keras.losses.CategoricalCrossentropy()(
            outputs[i, :, 4:], targets[i, :, 4:]
        )
        label_loss = tf.map_fn(
            labelossfunction,
            tf.range(tf.shape(outputs)[0]),
            fn_output_signature=tf.float32,
        )

        # L1 box loss
        l1lossfunction = lambda i: tf.reduce_sum(
            tf.abs(outputs[i, ..., :4] - targets[i, ..., :4]), axis=-1
        )
        l1_loss = tf.reduce_mean(
            tf.map_fn(
                l1lossfunction,
                tf.range(tf.shape(outputs)[0]),
                fn_output_signature=tf.float32,
            ),
            axis=-1,
        )

        return giou_loss * cost_giou + label_loss * cost_class + l1_loss * cost_bbox

    return _internal_computation


if __name__ == "__main__":

    from fillEmptyBoxes import fillEmptyBoxes
    from giou import _calculate_giou

    A = tf.random.uniform((7, 5, 4 + 8))
    B = tf.random.uniform((7, 3, 4 + 8))
    B = fillEmptyBoxes(A, B)

    print(hungarianLoss()(A, B))
