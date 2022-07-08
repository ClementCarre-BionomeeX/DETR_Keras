import tensorflow as tf


bbox = tf.random.uniform((3, 5, 4))
classes = tf.random.uniform((3, 5, 7))


pack = tf.concat([bbox, classes], -1)


# print(pack)


def getbbox(input):
    return input[:4]

def getclass(input):
    return input[4:]

def perbatch(input):
    bbox = tf.map_fn(getbbox, input)
    clss = tf.map_fn(getclass, input)
    tf.print(bbox.shape)
    tf.print(clss.shape)

    bboxes = input[:, :4]

    tf.print(bboxes)

    return 0


tf.map_fn(perbatch, pack)
