from builtins import print

import tensorflow as tf

from cdist import cdist
from giou import box_giou


@tf.function
def _class_approx(output_probs, target_probs):
    return -tf.gather(output_probs, tf.argmax(target_probs, -1), batch_dims=1, axis=-1)


@tf.function
def _bbox(output_bbox, target_bbox):
    return cdist(output_bbox, target_bbox)


@tf.function
def _giou(output_bbox, target_bbox):
    return box_giou(output_bbox, target_bbox)


def hungarian_matcher(cost_class=1, cost_bbox=1, cost_giou=1):
    from scipy.optimize import linear_sum_assignment

    @tf.function
    def tf_linear_sum_assignement(cost_matrix):
        return tf.numpy_function(
            func=linear_sum_assignment, inp=[cost_matrix], Tout=[tf.int64, tf.int64]
        )

    @tf.function
    def _internal_computation(output, target):
        cclap = _class_approx(output[..., 4:], target[..., 4:])
        cbbox = _bbox(output[..., :4], target[..., :4])
        cgiou = _giou(output[..., :4], target[..., :4])
        cost = cclap * cost_class + cbbox * cost_bbox + cgiou * cost_giou

        def stacking(matrix):
            return tf.stack(tf_linear_sum_assignement(matrix))

        return tf.map_fn(stacking, cost, fn_output_signature=tf.int64)

    return _internal_computation


if __name__ == "__main__":
    output = tf.random.uniform((3, 5, 4 + 7))
    target = tf.random.uniform((3, 5, 4 + 7))

    indices = hungarian_matcher(1, 1, 1)(output, target)
    print(indices)

    out = tf.gather(output, indices[:, 0, :], batch_dims=1)
    ta = tf.gather(target, indices[:, 1, :], batch_dims=1)

    print(target)
    print(ta)

    # print(output)
    # print(target)

    # C = tf.argmax(target[..., 4:], -1)

    # print(C)

    # cclap = _class_approx(output[..., 4:], target[..., 4:])
    # cbbox = _bbox(output[..., :4], target[..., :4])
    # cgiou = _giou(output[..., :4], target[..., :4])
    # print(cclap)
    # print(cbbox)
    # print(cgiou)

    # COST = cclap + cbbox + cgiou
    # print(COST)

    # from scipy.optimize import linear_sum_assignment

    # indices = tf.stack([tf.stack(linear_sum_assignment(C)) for C in COST])

    # print(indices)


# def hungarianLoss(
#     ytrue,
#     ypred,
#     cost_class: float = 1,
#     cost_bbox: float = 1,
#     cost_giou: float = 1,
# ):
#     truebbox, trueclasses = ytrue
#     predbbox, predclasses = ypred

#     def curried_hungarian_loss(ytrue,
#     ypred,
#     cost_class: float = 1,
#     cost_bbox: float = 1,
#     cost_giou: float = 1,):
#         return _hungarianLoss()


# def _hungarianLoss(
#     ytrue,
#     ypred,
#     cost_class: float = 1,
#     cost_bbox: float = 1,
#     cost_giou: float = 1,
# ):

#     """Hungarian matching Loss function

#     Params:
#         - ypred:
#             * <b, N, 4 "bbox tensor" + C + 1 "classe probs tensor">
#         - ytrue:
#             * <b, N, 4 "bbox tensor" + C + 1 "classe probs tensor">

#         b = batchsize
#         N = number of queries:
#             For targets, we fill the missing values with 0 sized boxes with class 0
#         C = number of classes:
#             We add a zero-th class : the no-object class

#     """
#     pass


# def loss_label(ytrue, ypred, indices, num_boxes):

#     indices = tf.stop_gradient(_hungarian_matcher(ytrue, ypred))


# def _hungarian_matcher_on_batch(
#     targets,
#     outputs,
# ):
#     """
#     targets :
#         <N, 4 "bbox tensor" + C + 1 "classe probs tensor">
#     outputs :
#         <N, 4 "bbox tensor" + C + 1 "classe probs tensor">
#     """
#     ta_bbox = targets[:, :4]
#     ta_clss = targets[:, 4:]
#     ou_bbox = outputs[:, :4]
#     ou_clss = outputs[:, 4:]

#     # We want to construct a cost matrix of size N, N
