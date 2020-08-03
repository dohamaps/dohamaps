import  tensorflow  as tf
import  abc

from    abc     import ABCMeta, ABC, abstractmethod

def psnr(y_true, y_pred):
    """
        @param y_true: tensor of shape (batch_size, height, width, channels)
        @param y_pred: tensor of shape (batch_size, height, width, channels)
        @return: the peak signal to noise ratio as a scalar
    """
    return tf.reduce_mean(tf.image.psnr(y_true, y_pred, max_val = 2.0));

def sharpdiff(y_true, y_pred):
    """
        @param y_true: tensor of shape (batch_size, height, width, channels)
        @param y_pred: tensor of shape (batch_size, height, width, channels)
        @return: the sharpness difference as a scalar
    """
    def log10(tensor):
        numerator = tf.math.log(tensor);
        denominator = tf.math.log(tf.constant(10, dtype = numerator.dtype));
        return numerator / denominator;

    shape = tf.shape(y_pred);
    num_pixels = tf.cast(shape[1] * shape[2] * shape[3], tf.float32);

    y_true_dy, y_true_dx = tf.image.image_gradients(y_true);
    y_pred_dy, y_pred_dx = tf.image.image_gradients(y_pred);

    pred_grad_sum = y_pred_dx + y_pred_dy;
    true_grad_sum = y_true_dx + y_true_dy;

    grad_diff = tf.abs(true_grad_sum - pred_grad_sum);
    grad_diff_red = tf.reduce_sum(grad_diff, [1, 2, 3]);

    batch_errors = 10 * log10(1 / ((1 / num_pixels) * grad_diff_red));
    return tf.reduce_mean(batch_errors);

def ssim(y_true, y_pred):
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val = 2.0))

class Metric(metaclass = ABCMeta):
    def __init__(self):
        super(Metric, self).__init__();

    def __call__(self, y_true, y_pred):
        return self.call(y_true, y_pred);

    @abstractmethod
    def call(self, y_true, y_pred):
        pass;

class PSNR(Metric):
    def __init__(self, name = "psnr"):
        super(PSNR, self).__init__();
        self.name = name;

    def call(self, y_true, y_pred):
        return psnr(y_true, y_pred);

class SharpDiff(Metric):
    def __init__(self, name = "sharpdiff"):
        super(SharpDiff, self).__init__();
        self.name = name;

    def call(self, y_true, y_pred):
        return sharpdiff(y_true, y_pred);

class SSIM(Metric):
    def __init__(self, name = "ssim"):
        super(SSIM, self).__init__();
        self.name = name;

    def call(self, y_true, y_pred):
        return ssim(y_true, y_pred);
