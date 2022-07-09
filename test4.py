import tensorflow as tf

@tf.function
def cdist(x, y):
    # Calculate distance for a single row of x.
    # per_x_dist = lambda i : tf.reduce_sum(tf.abs(x[:, i:(i+1),:] - y), axis=2)
    per_x_dist = lambda i : tf.reduce_sum(tf.abs(x[:, i:(i+1),:] - y), axis=2)
    # Compute and stack distances for all rows of x.
    dist = tf.map_fn(fn=per_x_dist, elems=tf.range(tf.shape(x)[1], dtype=tf.int64), fn_output_signature=x.dtype)
    # Re-arrange stacks of distances.
    return tf.transpose(dist, perm=[1, 0, 2])


A = tf.random.uniform((3, 7))
B = tf.random.uniform((5, 7))

print(A)
print(B)
print(tf.squeeze(cdist(
        tf.expand_dims(A, 0),
        tf.expand_dims(B, 0)
)))
