import  tensorflow  as tf
import  os

from    .                   import layers, util, img
from    tensorflow.keras    import Model
from    glob                import glob
from    .util                import __sysprint__

class Discriminator(Model):
    def __init__(self, **kwargs):
        super(Discriminator, self).__init__(**kwargs);

        disc0_layers = \
        [
            layers.conv(filters = 64, kernel_size = 3, padding = "valid"),
            layers.maxpool(),
            layers.flatten(),
            layers.dense(units = 512),
            layers.dense(units = 256),
            layers.dense(units = 1, activation = "sigmoid"),
            layers.clipvalue()
        ];

        disc1_layers = \
        [
            layers.conv(filters = 64, kernel_size = 3, padding = "valid"),
            layers.conv(filters = 128, kernel_size = 3, padding = "valid"),
            layers.conv(filters = 128, kernel_size = 3, padding = "valid"),
            layers.maxpool(),
            layers.flatten(),
            layers.dense(units = 1024),
            layers.dense(units = 512),
            layers.dense(units = 1, activation = "sigmoid"),
            layers.clipvalue()
        ];

        disc2_layers = \
        [
            layers.conv(filters = 128, kernel_size = 5, padding = "valid"),
            layers.conv(filters = 256, kernel_size = 5, padding = "valid"),
            layers.conv(filters = 256, kernel_size = 5, padding = "valid"),
            layers.maxpool(),
            layers.flatten(),
            layers.dense(units = 1024),
            layers.dense(units = 512),
            layers.dense(units = 1, activation = "sigmoid"),
            layers.clipvalue()
        ];

        disc3_layers = \
        [
            layers.conv(filters = 128, kernel_size = 7, padding = "valid"),
            layers.conv(filters = 256, kernel_size = 7, padding = "valid"),
            layers.conv(filters = 512, kernel_size = 5, padding = "valid"),
            layers.conv(filters = 128, kernel_size = 5, padding = "valid"),
            layers.maxpool(),
            layers.flatten(),
            layers.dense(units = 1024),
            layers.dense(units = 512),
            layers.dense(units = 1, activation = "sigmoid"),
            layers.clipvalue()
        ];

        self.blocks = \
        [
            layers.Block(layer_list = disc0_layers),
            layers.Block(layer_list = disc1_layers),
            layers.Block(layer_list = disc2_layers),
            layers.Block(layer_list = disc3_layers)
        ];

        self.num_scales = len(self.blocks);

    def call(self, inputs):
        """
            @param inputs: list of tensors of shape
                           (batch_size, height, width, channels)
            @return: list of tensors containing each scale prediction of
                     shape (batch_size, 1)
        """
        assert(len(inputs) == self.num_scales);
        preds = [];
        for (i, block) in enumerate(self.blocks):
            preds.append(block(inputs[i]));

        return preds;




class Generator(Model):
    def __init__(self, pred_len, **kwargs):
        super(Generator, self).__init__(**kwargs);

        self.pred_len = pred_len;

        gen0_layers = \
        [
            layers.conv(filters = 128, kernel_size = 3, padding = "same"),
            layers.conv(filters = 256, kernel_size = 3, padding = "same"),
            layers.conv(filters = 128, kernel_size = 3, padding = "same"),
            layers.conv(filters = 3 * self.pred_len, kernel_size = 3,
                        padding = "same", activation = "tanh"),
        ];

        gen1_layers = \
        [
            layers.conv(filters = 128, kernel_size = 5, padding = "same"),
            layers.conv(filters = 256, kernel_size = 3, padding = "same"),
            layers.conv(filters = 128, kernel_size = 3, padding = "same"),
            layers.conv(filters = 3 * self.pred_len, kernel_size = 5,
                        padding = "same", activation = "tanh")
        ];

        gen2_layers = \
        [
            layers.conv(filters = 128, kernel_size = 5, padding = "same"),
            layers.conv(filters = 256, kernel_size = 3, padding = "same"),
            layers.conv(filters = 512, kernel_size = 3, padding = "same"),
            layers.conv(filters = 256, kernel_size = 3, padding = "same"),
            layers.conv(filters = 128, kernel_size = 3, padding = "same"),
            layers.conv(filters = 3 * self.pred_len, kernel_size = 5,
                        padding = "same", activation = "tanh")
        ];

        gen3_layers = \
        [
            layers.conv(filters = 128, kernel_size = 7, padding = "same"),
            layers.conv(filters = 256, kernel_size = 5, padding = "same"),
            layers.conv(filters = 512, kernel_size = 5, padding = "same"),
            layers.conv(filters = 256, kernel_size = 5, padding = "same"),
            layers.conv(filters = 128, kernel_size = 5, padding = "same"),
            layers.conv(filters = 3 * self.pred_len, kernel_size = 7,
                        padding = "same", activation = "tanh")
        ];

        self.blocks = \
        [
            layers.Block(layer_list = gen0_layers),
            layers.Block(layer_list = gen1_layers),
            layers.Block(layer_list = gen2_layers),
            layers.Block(layer_list = gen3_layers)
        ];

        self.upsample = layers.upsample();
        self.num_scales = len(self.blocks);

    def call(self, inputs):
        """
            @param inputs: list of tensors of shape
                           (batch_size, height, width, channels)
            @return: list of tensors containing each scale prediction of
                     shape (batch_size, 1)
        """
        assert(len(inputs) == self.num_scales);
        preds = [];
        for (i, block) in enumerate(self.blocks):
            if i > 0:
                scale_pred = self.upsample(scale_pred);
                scale_pred = tf.concat([ inputs[i], scale_pred ], axis = -1);
            else:
                scale_pred = inputs[i];

            scale_pred = block(scale_pred);
            preds.append(scale_pred);
        return preds;


class GAN(Model):
    def __init__(self, num_scales, hist_len, pred_len, dim, **kwargs):
        super(GAN, self).__init__(**kwargs);
        (self.height, self.width) = dim;
        self.hist_len = hist_len;
        self.pred_len = pred_len;
        self.clip_len = self.hist_len + self.pred_len;
        self.num_scales = num_scales;
        self.discriminator = Discriminator();
        self.generator = Generator(self.pred_len);

        self.h_scales = [];
        self.gt_scales = [];
        self.rg_scales = [];
        self.pred = None;

    def compile(self, disc_optim, gen_optim, disc_loss, gen_loss, metrics, **kwargs):
        super(GAN, self).compile(**kwargs);
        self.disc_optim = disc_optim;
        self.gen_optim = gen_optim;
        self.disc_loss = disc_loss;
        self.gen_loss = gen_loss;
        self.__metrics = metrics;

    def process_batch(self, data):
        history_batch = tf.stack(data[0]);
        ground_truth_batch = tf.stack(data[1]);

        h_scales = [];
        gt_scales = [];

        for i in range(self.num_scales):
            h_scales.append(img.resize_tensor(history_batch,
                                              (self.height, self.width), i));
            gt_scales.append(img.resize_tensor(ground_truth_batch,
                                               (self.height, self.width), i));

        return (h_scales, gt_scales);

    def save_data_scaled(self, h_scales, gt_scales, rg_scales):
        self.h_scales = h_scales;
        self.gt_scales = gt_scales;
        self.rg_scales = rg_scales;

    def save_data_single(self, pred):
        self.pred = pred;

    def train_step(self, data):
        """
            @param data: a batch of tensors pairs (history, ground_truth)
                         of shapes: history (height, width, hist_len) and
                         ground_truth (height, width, pred_len)
        """
        ###     init     ###

        performance_dict = {  };

        (h_scales, gt_scales) = self.process_batch(data);

        ###     discriminator     ###

        fg_scales = self.generator(h_scales); # fake gens for discriminator
        d_input = [];
        for i in range(self.num_scales):
            fake_sequence = tf.concat([ h_scales[i], fg_scales[i] ], -1);
            real_sequence = tf.concat([ h_scales[i], gt_scales[i] ], -1);
            d_input.append(tf.concat([ fake_sequence, real_sequence ], 0));

        fake_shape = (fake_sequence.shape[0], 1);
        real_shape = (real_sequence.shape[0], 1);
        d_label = tf.concat([ tf.zeros(fake_shape), tf.ones(real_shape) ], 0);

        # random noise
        d_label += 0.05 * tf.random.uniform(tf.shape(d_label));

        d_labels = [ d_label for i in range(self.num_scales) ];

        with tf.GradientTape() as tape:
            d_preds = self.discriminator(d_input);
            d_loss = self.disc_loss(d_labels, d_preds);
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights);
        self.disc_optim.apply_gradients(zip(grads, self.discriminator.trainable_weights));

        ###     generator     ###

        with tf.GradientTape() as tape:
            rg_scales = self.generator(h_scales); # real gens
            for metric in self.__metrics:
                performance_dict.update({ metric.name : metric(gt_scales[-1], rg_scales[-1]) });
            d_input = [];
            for i in range(self.num_scales):
                d_input.append(tf.concat([ h_scales[i], rg_scales[i] ], -1));
            g_preds = self.discriminator(d_input);
            g_loss = self.gen_loss(gt_scales, rg_scales, g_preds);

        grads = tape.gradient(g_loss, self.generator.trainable_weights);
        self.gen_optim.apply_gradients(zip(grads, self.generator.trainable_weights));

        performance_dict.update({ "disc_loss" : d_loss, "gen_loss" : g_loss });

        self.save_data_scaled(h_scales, gt_scales, rg_scales);

        return performance_dict;

    def test_step(self, data):
        """
            @param data: a batch of tensors pairs (history, ground_truth)
                         of shapes: history (height, width, hist_len) and
                         ground_truth (height, width, pred_len)
        """
        ###     init     ###

        performance_dict = {  };

        (h_scales, gt_scales) = self.process_batch(data);

        ###     generator     ###

        rg_scales = self.generator(h_scales); # real gens
        for metric in self.__metrics:
            performance_dict.update({ metric.name : metric(gt_scales[-1], rg_scales[-1]) });

        self.save_data_scaled(h_scales, gt_scales, rg_scales);

        return performance_dict;

    def predict_step(self, data):
        """
            @param data: a batch of tensors pairs (history, ground_truth)
                         of shapes: history (height, width, hist_len) and
                         ground_truth (height, width, pred_len)
        """
        ###     init     ###

        (h_scales, _) = self.process_batch(data);

        rg_scales = self.generator(h_scales);

        self.save_data_single(rg_scales[-1][0]);

        return rg_scales;
