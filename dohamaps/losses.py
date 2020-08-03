import  tensorflow  as tf

from    tensorflow.keras.losses import Loss, binary_crossentropy

def adversarial(y_true, y_pred):
    """
        @param y_true: list of tensors of shape (1, n) containing scale
                       labels.
        @param y_pred: list of tensors of shape (1, n) containing scale
                       predictions.
        @return: the adversarial loss (scalar).
    """
    assert(len(y_true) == len(y_pred));
    losses = [];
    for i in range(len(y_pred)):
        losses.append(binary_crossentropy(y_true[i], y_pred[i]));

    return tf.reduce_mean(tf.stack(losses));

def gdl(y_true, y_pred, alpha = 1):
    """
        @param y_true: list of tensors containing the ground truth images
                       at each scale.
        @param y_pred: list of tensors containing the generated images at
                       each scale.
        @return: gradient descent loss (scalar).
    """
    assert(len(y_true) == len(y_pred));
    losses = [];
    for i in range(len(y_pred)):
        y_true_dy, y_true_dx = tf.image.image_gradients(y_true[i]);
        y_pred_dy, y_pred_dx = tf.image.image_gradients(y_pred[i]);

        gdiff_y = tf.abs(tf.abs(y_true_dy) - tf.abs(y_pred_dy));
        gdiff_x = tf.abs(tf.abs(y_true_dx) - tf.abs(y_pred_dx));

        pow_x = tf.pow(gdiff_x, alpha);
        pow_y = tf.pow(gdiff_y, alpha);

        losses.append(tf.reduce_mean(pow_x + pow_y));

    return losses;

def lp(y_true, y_pred, l_num = 2):
    """
        @param y_true: list of tensors containing the ground truth images
                       at each scale.
        @param y_pred: list of tensors containing the generated images at
                       each scale.
        @return: l1 or l2 loss.
    """
    assert(len(y_true) == len(y_pred));
    losses = [];
    for i in range(len(y_true)):
        losses.append(tf.reduce_sum(tf.abs(y_pred[i] - y_true[i]) ** l_num));

    return tf.reduce_mean(tf.stack(losses));

def combined(y_true, y_pred, labels, l_adv = 0.05, l_lp = 1, l_gdl = 1, l_num = 2):
    assert(len(y_true) == len(y_pred) and len(y_pred) == len(labels));
    batch_size = tf.shape(y_pred[0])[0];

    loss = l_lp * lp(y_true, y_pred, l_num);
    loss += l_gdl * gdl(y_true, y_pred);
    one = tf.ones([ batch_size, 1 ]);
    ones = [ one for i in range(len(y_true)) ];
    loss += l_adv * adversarial(ones, labels);

    return loss;

class Adversarial(Loss):
    def __init__(self):
        super(Adversarial, self).__init__();

    def __call__(self, y_true, y_pred):
        return adversarial(y_true, y_pred);

class GDL(Loss):
    def __init__(self, alpha = 1):
        super(GDL, self).__init__();
        self.alpha = alpha;

    def __call__(self, y_true, y_pred):
        return gdl(y_true, y_pred, self.alpha);

class LP(Loss):
    def __init__(self, l_num = 2):
        super(LP, self).__init__();
        self.l_num = l_num;

    def __call__(self, y_true, y_pred):
        return lp(y_true, y_pred, self.l_num);

class Combined(Loss):
    def __init__(self, l_adv = 0.05, l_lp = 1, l_gdl = 1, l_num = 2):
        super(Combined, self).__init__();
        self.l_adv = l_adv;
        self.l_lp = l_lp;
        self.l_gdl = l_gdl;
        self.l_num = l_num;

    def __call__(self, y_true, y_pred, labels):
        return combined(y_true, y_pred, labels, self.l_adv, self.l_lp,
                        self.l_gdl, self.l_num);
