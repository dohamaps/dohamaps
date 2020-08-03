import  os
import  numpy       as np
import  tensorflow  as tf
import  itertools

from    .       import util, img, const
from    glob    import glob

class Dataset(object):
    def __init__(self, cache_path, hist_len, pred_len, mode, dim):
        self.cache_path = os.path.join(util.get_dir(cache_path), "cache");
        self.clip_len = hist_len + pred_len;
        self.hist_len = hist_len;
        self.pred_len = pred_len;
        self.mode = mode;
        (self.height, self.width) = dim;
        self.dataset = None;

    def cache(self):
        self.dataset = self.dataset.cache(self.cache_path);

    def glob_files(self, path, pattern):
        self.dataset = tf.data.Dataset.list_files(os.path.join(path, pattern));

    def batch(self, batch_size):
        self.dataset = self.dataset.batch(batch_size, drop_remainder = True);

    def load_tensors(self):
        def path_to_tensor(path):
            clip = img.load_npz(path.numpy().decode("utf-8"));
            tensor = img.npz_to_tensor(clip);
            return tensor;
        def map_fn(path):
            [ tensor, ] = tf.py_function(path_to_tensor, [ path ], [ tf.float32 ]);
            tensor.set_shape((self.height, self.width, 3 * self.clip_len));
            history = tensor[ :, :, : 3 * self.hist_len ];
            ground_truth = tensor[ :, :, 3 * self.hist_len : ];
            return (history, ground_truth);
        def pred_map_fn(path):
            [ tensor, ] = tf.py_function(path_to_tensor, [ path ], [ tf.float32 ]);
            tensor.set_shape((self.height, self.width, 3 * self.clip_len));
            history = tensor[ :, :, 3 * self.pred_len : ];
            ground_truth = tensor[ :, :, : ];
            return (history, ground_truth);

        if self.mode != const.mode.pred:
            self.dataset = self.dataset.map(map_fn);
        else:
            self.dataset = self.dataset.map(pred_map_fn);

    def prefetch(self):
        self.dataset = self.dataset.prefetch(1);

    def tf_dataset(self):
        return self.dataset;
