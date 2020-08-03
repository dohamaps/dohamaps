import  tensorflow  as tf
import  getopt
import  sys
import  os
import  time


from    .       import img, layers, util, data, losses
from    .       import metrics, callbacks, const, models
from    .const  import info, mode


def train(epochs, batch_size, load_weights):
    util.__sysprint__("beginning training...");
    time.sleep(1);
    util.__sysprint__("loading training data...");
    dataset = data.Dataset(info.cache_path, hist_len = info.hist_len,
                           pred_len = info.pred_len, mode = mode.train,
                           dim = info.train_dim);
    dataset.glob_files(info.clips_path, "*.npz");
    dataset.load_tensors();
    dataset.batch(batch_size = batch_size);
    dataset.cache();
    dataset.prefetch();
    util.__sysprint__("loaded training data");

    checkpt_base = util.get_dir(os.path.join(info.save_path, "checkpt"));
    checkpt_path = os.path.join(checkpt_base, "checkpt");

    checkpoint = tf.keras.callbacks.ModelCheckpoint \
    (
        filepath = checkpt_path,
        save_weights_only = True
    );

    image_save = callbacks.ImageSave(info.save_path, info.clips_path);

    model = models.GAN(num_scales = info.num_scales, hist_len = info.hist_len,
                       pred_len = info.pred_len, dim = info.train_dim);

    util.__sysprint__("compiling model...");
    model.compile \
    (
        disc_optim = tf.keras.optimizers.Adam(learning_rate = info.disc_lrate),
        gen_optim = tf.keras.optimizers.Adam(learning_rate = info.gen_lrate),
        disc_loss = losses.Adversarial(),
        gen_loss = losses.Combined(),
        metrics = [ metrics.PSNR(), metrics.SharpDiff(), metrics.SSIM() ],
        run_eagerly = True
    );
    util.__sysprint__("compiled model");
    time.sleep(1);

    if load_weights:
        util.__sysprint__("loading weights...");
        model.load_weights(checkpt_path);
        util.__sysprint__("loaded weights");
        time.sleep(1);

    util.__sysprint__("training...");
    model.fit \
    (
        dataset.tf_dataset(),
        epochs = epochs,
        callbacks = [ checkpoint, image_save ]
    );
    util.__sysprint__("done");
    time.sleep(1);

def test(steps, batch_size):
    util.__sysprint__("beginning testing...");
    time.sleep(1);
    util.__sysprint__("loading testing data...");
    dataset = data.Dataset(info.cache_path, hist_len = info.hist_len,
                           pred_len = info.pred_len, mode = mode.test,
                           dim = info.train_dim);
    dataset.glob_files(util.get_dir(info.clips_path), "*.npz");
    dataset.load_tensors();
    dataset.batch(batch_size = batch_size);
    dataset.cache();
    dataset.prefetch();
    util.__sysprint__("loaded testing data");

    checkpt_base = util.get_dir(os.path.join(info.save_path, "checkpt"));
    checkpt_path = os.path.join(checkpt_base, "checkpt");

    checkpoint = tf.keras.callbacks.ModelCheckpoint \
    (
        filepath = checkpt_path,
        save_weights_only = True
    );

    image_save = callbacks.ImageSave(info.save_path, info.clips_path);

    model = models.GAN(num_scales = info.num_scales, hist_len = info.hist_len,
                       pred_len = info.pred_len, dim = info.train_dim);

    util.__sysprint__("compiling model...");
    model.compile \
    (
        disc_optim = tf.keras.optimizers.Adam(learning_rate = info.disc_lrate),
        gen_optim = tf.keras.optimizers.Adam(learning_rate = info.gen_lrate),
        disc_loss = losses.Adversarial(),
        gen_loss = losses.Combined(),
        metrics = [ metrics.PSNR(), metrics.SharpDiff(), metrics.SSIM() ],
        run_eagerly = True
    );
    util.__sysprint__("compiled model");
    time.sleep(1);

    util.__sysprint__("loading weights...");
    model.load_weights(checkpt_path);
    util.__sysprint__("loaded weights");
    time.sleep(1);

    util.__sysprint__("testing...");
    model.evaluate \
    (
        dataset.tf_dataset(),
        steps = steps,
        callbacks = [ checkpoint, image_save ]
    );
    util.__sysprint__("done");
    time.sleep(1);

def predict(checkpt_path = None):
    util.__sysprint__("beginning prediction...");
    time.sleep(1);
    util.__sysprint__("loading prediction data...");
    dataset = data.Dataset(info.cache_path, hist_len = info.hist_len,
                           pred_len = info.pred_len, mode = mode.pred,
                           dim = info.pred_dim);
    pred_clip_path = os.path.join(info.clips_path, "pred")
    dataset.glob_files(util.get_dir(pred_clip_path), "*.npz");
    dataset.load_tensors();
    dataset.batch(batch_size = 1);
    util.__sysprint__("loaded prediction data");

    if checkpt_path is None:
        checkpt_base = util.get_dir(os.path.join(info.save_path, "checkpt"));
        checkpt_path = os.path.join(checkpt_base, "checkpt");
    else:
        checkpt_path = os.path.join(checkpt_path, "checkpt");
    image_save = callbacks.ImageSave(info.save_path, info.clips_path);

    model = models.GAN(num_scales = info.num_scales, hist_len = info.hist_len,
                       pred_len = info.pred_len, dim = info.pred_dim);

    util.__sysprint__("compiling model...");
    model.compile \
    (
        disc_optim = tf.keras.optimizers.Adam(learning_rate = info.disc_lrate),
        gen_optim = tf.keras.optimizers.Adam(learning_rate = info.gen_lrate),
        disc_loss = losses.Adversarial(),
        gen_loss = losses.Combined(),
        metrics = [ metrics.PSNR(), metrics.SharpDiff(), metrics.SSIM() ],
        run_eagerly = True
    );
    util.__sysprint__("compiled model");
    time.sleep(1);

    util.__sysprint__("loading weights...");
    model.load_weights(checkpt_path);
    util.__sysprint__("loaded weights");
    time.sleep(1);

    util.__sysprint__("predicting...");
    model.predict \
    (
        dataset.tf_dataset(),
        callbacks = [ image_save ]
    );
    util.__sysprint__("done");
    time.sleep(1);
