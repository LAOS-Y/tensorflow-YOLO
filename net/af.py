import tensorflow as tf

def leaky_relu(x, alpha=0.2, name="leaky_relu"):
    with tf.name_scope(name, "leaky_relu") as name_scope:
        x = tf.convert_to_tensor(x, name="features")
        return tf.maximum(x, alpha * x, name=name_scope)