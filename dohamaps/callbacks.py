import  tensorflow  as tf
import  os

from    .                           import img
from    tensorflow.keras.callbacks  import Callback

class ImageSave(Callback):
    def __init__(self, data_save_path, clips_path, **kwargs):
        super(ImageSave, self).__init__(**kwargs);
        self.data_save_path = data_save_path;
        self.clips_path = clips_path;
        self.train_save_path = os.path.join(self.data_save_path, "train");
        self.test_save_path = os.path.join(self.data_save_path, "test");
        self.pred_save_png_path = os.path.join(self.data_save_path, "pred");
        self.pred_save_npz_path = os.path.join(self.clips_path, "pred");

    def save_data_train(self, ext, scales, name):
        save_path = os.path.join(self.train_save_path, ext);
        def save_fn(path, name, tensor):
            img.save_png(path.numpy().decode("utf-8"),
                          name.numpy().decode("utf-8"),
                          img.tensor_to_clip(tensor));
        def save_single_batch(path, name, batch):
            # batch.shape = (batch_size, height, width, 3 * clip_len)
            clips = tf.unstack(batch);
            for i, clip in enumerate(clips):
                batch_path = os.path.join(path, "batch_" + str(i));
                [] = tf.py_function(save_fn, [ batch_path, name, clip ], []);

        for i, scale in enumerate(scales):
            save_single_batch(save_path, name + "_scale_" + str(i), scale);

    def save_data_test(self, ext, scales, name):
        save_path = os.path.join(self.test_save_path, ext);
        def save_fn(path, name, tensor):
            img.save_png(path.numpy().decode("utf-8"),
                          name.numpy().decode("utf-8"),
                          img.tensor_to_clip(tensor));
        def save_single_batch(path, name, batch):
            # batch.shape = (batch_size, height, width, 3 * clip_len)
            clips = tf.unstack(batch);
            for i, clip in enumerate(clips):
                batch_path = os.path.join(path, "batch_" + str(i));
                [] = tf.py_function(save_fn, [ batch_path, name, clip ], []);

        for i, scale in enumerate(scales):
            save_single_batch(save_path, name + "_scale_" + str(i), scale);

    def save_data_pred(self, tensor, name):
        def save_png_fn(path, name, tensor):
            img.save_png(path.numpy().decode("utf-8"),
                         name.numpy().decode("utf-8"),
                         img.tensor_to_clip(tensor));
        def save_npz_fn(path, name, tensor):
            path_str = path.numpy().decode("utf-8");
            name_str = name.numpy().decode("utf-8");
            prev_path = os.path.join(path_str, name_str + ".npz");
            # prev.shape = (height, width, 3 * clip_len)
            # tensor.shape = (height, width, 3 * pred_len)
            prev = img.npz_to_tensor(img.load_npz(prev_path));
            (height, width, _) = tensor.shape;
            size = tf.constant([ height, width ]);
            prev = tf.image.resize(prev, size);
            tensor = tf.concat([ prev, tensor ], axis = -1);
            tensor = tensor[:, :, 3 * self.model.pred_len : ];
            img.save_npz(path_str, name_str, img.tensor_to_clip(tensor));

        [] = tf.py_function(save_png_fn, [ self.pred_save_png_path, name, tensor ], []);
        [] = tf.py_function(save_npz_fn, [ self.pred_save_npz_path, name, tensor ], []);

    def on_epoch_end(self, epoch, logs = None):
        self.save_data_train(str(epoch), self.model.h_scales, "history");
        self.save_data_train(str(epoch), self.model.gt_scales, "ground_truth");
        self.save_data_train(str(epoch), self.model.rg_scales, "gen");

    def on_test_end(self, logs = None):
        self.save_data_test("test", self.model.h_scales, "history");
        self.save_data_test("test", self.model.gt_scales, "ground_truth");
        self.save_data_test("test", self.model.rg_scales, "gen");

    def on_predict_end(self, logs = None):
        self.save_data_pred(self.model.pred, "pred");
