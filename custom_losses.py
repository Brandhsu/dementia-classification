import tensorflow as tf

# --- Losses
def cosine_similarity(vects):
    a, b = vects
    return 1 - tf.keras.layers.Dot(axes=1, normalize=True)([a, b])


def euclidean_distance(vects):
    a, b = vects
    return tf.norm(a - b, ord='euclidean')


def norm_euclidean_distance(vects):
    a, b = vects
    return tf.norm(tf.nn.l2_normalize(a, 0) - tf.nn.l2_normalize(b, 0), ord='euclidean')


def ContrastiveLoss(margin=1, gamma=2):
    def ctr_loss(y_true, y_pred):
        square_pred = tf.math.square(y_pred) ** gamma
        margin_square = tf.math.square(tf.math.maximum((margin - y_pred), 0)) ** gamma
        return tf.math.reduce_mean(
            y_true * square_pred + (1 - y_true) * margin_square
        )

    return ctr_loss


def LogContrastiveLoss(margin=1, ep=1e-9, gamma=2):
    def ctr_loss(y_true, y_pred):
        clip = lambda x: tf.clip_by_value(x, clip_value_min=ep, clip_value_max=(margin - ep))
        log = lambda x: tf.math.log(x)
        
        clip_pred = clip(y_pred)
        clip_marg = clip(margin - y_pred)
        
        log_pred = log(clip_pred)
        log_marg = log(clip_marg)
        
        foc_pred = log_pred * clip_marg ** gamma
        foc_marg = log_marg * clip_pred ** gamma
        
        return -tf.math.reduce_mean((y_true * foc_marg + (1 - y_true) * foc_pred))

    return ctr_loss