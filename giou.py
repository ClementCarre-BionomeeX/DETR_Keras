import tensorflow as tf

# from tensorflow_addons.losses import GIoULoss

@tf.function
def _calculate_giou(b1, b2):
    zero = tf.convert_to_tensor(0.0, b1.dtype)
    b1_ymin, b1_xmin, b1_ymax, b1_xmax = tf.unstack(b1, 4, axis=-1)
    b2_ymin, b2_xmin, b2_ymax, b2_xmax = tf.unstack(b2, 4, axis=-1)
    b1_width = tf.maximum(zero, b1_xmax - b1_xmin)
    b1_height = tf.maximum(zero, b1_ymax - b1_ymin)
    b2_width = tf.maximum(zero, b2_xmax - b2_xmin)
    b2_height = tf.maximum(zero, b2_ymax - b2_ymin)
    b1_area = b1_width * b1_height
    b2_area = b2_width * b2_height

    intersect_ymin = tf.maximum(b1_ymin, b2_ymin)
    intersect_xmin = tf.maximum(b1_xmin, b2_xmin)
    intersect_ymax = tf.minimum(b1_ymax, b2_ymax)
    intersect_xmax = tf.minimum(b1_xmax, b2_xmax)
    intersect_width = tf.maximum(zero, intersect_xmax - intersect_xmin)
    intersect_height = tf.maximum(zero, intersect_ymax - intersect_ymin)
    intersect_area = intersect_width * intersect_height

    union_area = b1_area + b2_area - intersect_area
    iou = tf.math.divide_no_nan(intersect_area, union_area)

    enclose_ymin = tf.minimum(b1_ymin, b2_ymin)
    enclose_xmin = tf.minimum(b1_xmin, b2_xmin)
    enclose_ymax = tf.maximum(b1_ymax, b2_ymax)
    enclose_xmax = tf.maximum(b1_xmax, b2_xmax)
    enclose_width = tf.maximum(zero, enclose_xmax - enclose_xmin)
    enclose_height = tf.maximum(zero, enclose_ymax - enclose_ymin)
    enclose_area = enclose_width * enclose_height
    return 1 - tf.squeeze(iou - tf.math.divide_no_nan((enclose_area - union_area), enclose_area))


@tf.function
def _box_giou_per_batch(box1, boxes2):

    @tf.function
    def curried_calculate_loss(box):
        return _calculate_giou(box, boxes2)

    return tf.map_fn(curried_calculate_loss, box1)


@tf.function
def box_giou(boxes1, boxes2):

    @tf.function
    def curried_giou_per_batch(i):
        return _box_giou_per_batch(boxes1[i, ], boxes2[i, ])

    return tf.map_fn(curried_giou_per_batch, tf.range(tf.shape(boxes1)[0], dtype=tf.int64), fn_output_signature=boxes1.dtype)



if __name__ == "__main__":
    A = tf.random.uniform((3, 5, 4))
    B = tf.random.uniform((3, 5, 4))

    C = box_giou(A, B)

    print(C)
