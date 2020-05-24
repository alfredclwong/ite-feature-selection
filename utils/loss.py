import tensorflow as tf


def mmd2(R, T, sigma=1, tensor=True):
    def k(x, y):
        return tf.exp(-(tf.reduce_sum(tf.square(x), axis=-1, keepdims=True)
                      + tf.transpose(tf.reduce_sum(tf.square(y), axis=-1, keepdims=True))
                      - 2 * x @ tf.transpose(y)) / tf.square(tf.cast(sigma, 'float32')) / 2.0)
    if not tensor:
        R, T = map(tf.constant, [R, T])
    ic = tf.where(T == 0)[:, 0]
    it = tf.where(T == 1)[:, 0]
    Rc = tf.gather(R, ic)
    Rt = tf.gather(R, it)
    Kcc = k(Rc, Rc)
    Ktt = k(Rt, Rt)
    Kct = k(Rc, Rt)
    Ktc = k(Rt, Rc)
    m = tf.cast(tf.shape(Rc)[0], 'float')
    n = tf.cast(tf.shape(Rt)[0], 'float')
    mmd2 = (tf.reduce_sum(Kcc)-m) / (m*(m-1)) + (tf.reduce_sum(Ktt)-n) / (n*(n-1))
    mmd2 -= (tf.reduce_sum(Kct) + tf.reduce_sum(Ktc)) / (m*n)
    return mmd2 if tensor else mmd2.numpy()


def factual(TYt, Yt_pred, t):
    T = TYt[:, 0]
    Yt = TYt[:, 1]
    it = tf.where(T == t)[:, 0]
    return tf.reduce_mean(tf.square(tf.gather(Yt, it) - tf.gather(Yt_pred, it)))
