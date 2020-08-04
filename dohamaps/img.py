import  os

import  numpy       as np
import  tensorflow  as tf
import  PIL
import  time

from    .                                       import const, util
from    glob                                    import glob
from    tensorflow.keras.preprocessing.image    import load_img, img_to_array
from    tensorflow.keras.preprocessing.image    import save_img, array_to_img
from    .const                                  import info


def normalize(image):
    """
        convert image from int8 [0, 255] to float32 [-1, 1]
        @param maps: a numpy array of the image to be converted
        @return: the normalized image
    """
    image = image.astype(np.float32);
    image /= (255 / 2);
    image -= 1;

    return image;

def denormalize(image):
    """
        performs the inverse operation of normalize
        @param image: a numpy array of the image to be converted.
        @return: the denormalized image
    """
    new_img = image + 1;
    new_img *= (255 / 2);
    new_img = new_img.astype(np.uint8);
    return new_img;

def filter(image):
    sharpness = PIL.ImageEnhance.Sharpness(image);
    __image = sharpness.enhance(1.2);
    contrast = PIL.ImageEnhance.Contrast(__image);
    __image = contrast.enhance(1.55);
    brightness = PIL.ImageEnhance.Brightness(__image);
    __image = brightness.enhance(0.8);
    return __image;


def create_clip(path, length, start_i = None):
    """
        creates a clip from a folder of standalone large images.

        @param path: path to where the frames in the map sequence are stored.
        @param length: length of the clip in frames.

        @return: a list of PIL images
    """
    clip = [];

    img_paths_total = sorted(glob(os.path.join(path, "*")));
    if start_i is None:
        start_i = np.random.choice(len(img_paths_total) - (length - 1));
    img_paths = img_paths_total[start_i : start_i + length];
    for img_path in img_paths:
        image = PIL.Image.open(img_path);
        clip.append(filter(image));

    return clip;

def crop_clip(clip, height, width):
    """
        Crops clip randomly to desired height and width.

        @param clip: list of PIL images of equal sizes.
        @param height: height of the final cropped clip.
        @param width: width of the final cropped clip.

        @return: list of cropped PIL images.
    """
    cropped_clip = [];

    (img_width, img_height) = clip[0].size;
    cx = np.random.choice(img_width - width + 1);
    cy = np.random.choice(img_height - height + 1);
    for image in clip:
        cropped_clip.append(image.crop((cx, cy, cx + width, cy + height)));

    return cropped_clip;

def save_png(path, name, clip):
    """
        saves a clip as an animated PNG
        @param path: path to the folder where clip should be stored
        @param clip: list of PIL images
    """
    path = util.get_dir(path);
    clip[0].save(os.path.join(path, name) + ".png", "png", save_all = True,
                 append_images = clip[1:], loop = 0);

def save_npz(path, name, clip):
    path = util.get_dir(path);
    arrays = [];
    for frame in clip:
        arrays.append(normalize(img_to_array(frame)));
    __array = np.concatenate(arrays, axis = 2);
    np.savez_compressed(os.path.join(path, name + ".npz"), __array);

def load_png(path, expected_len = None):
    """
        DEPRECATED
        loads an animated PNG as a PIL image
    """
    if expected_len != None:
        clip = PIL.Image.open(path);
        if clip.n_frames != expected_len:
            print("neq");
        # while clip.n_frames != expected_len:
        #     clip = load_img(path);
    else:
        clip = PIL.Image.open(path);
        clip = clip.load();
    assert(clip.is_animated and not clip.default_image);

    return clip;

def load_npz(path):
    return np.load(path)["arr_0"];

def npz_to_tensor(npz):
    return tf.convert_to_tensor(npz, dtype = tf.float32);

def tensor_to_clip(tensor):
    """
        converts a tensor of shape (height, width, 3 * clip_len)
        to a list of PIL images
    """
    num_splits = tensor.shape[2] // 3;
    frame_tensors = tf.split(tensor, num_or_size_splits = num_splits, axis = 2);
    clip = [];
    for ftensor in frame_tensors:
        new_ftensor = ftensor + 1;
        new_ftensor *= (255 / 2);
        clip.append(array_to_img(new_ftensor.numpy()));
    return clip;

def resize_clip(clip, scale_factor):
    (width, height) = clip.size;
    new_width = width * scale_factor;
    new_height = height * scale_factor;
    return clip.resize((new_width, new_height), resample = PIL.Image.BILINEAR);

def resize_tensor(clip, dim, scale_num):
    (height, width) = dim;

    new_height = int(int(height * info.scale_min) * info.scale_up ** scale_num);
    new_width = int(int(width * info.scale_min) * info.scale_up ** scale_num);
    size = tf.constant([ new_height, new_width ]);
    return tf.image.resize(clip,  size);

def process(num_clips, load_path, save_path):
    util.__sysprint__("processing clips...");
    (height, width) = info.train_dim;
    assert(os.path.exists(load_path));

    util.__sysprint__("processing prediction clip...");

    pred_path = os.path.join(save_path, "pred");
    start_i = len(glob(os.path.join(load_path, "*"))) - info.clip_len;
    pred_clip = create_clip(load_path, info.clip_len, start_i = start_i);
    save_npz(pred_path, "pred", pred_clip);

    util.__sysprint__("processed prediction clip");
    time.sleep(1);
    if num_clips > 0:
        util.__sysprint__("processing training/testing clips...");

    n_existing = len(glob(os.path.join(save_path, "*.npz")));
    for i in range(n_existing, n_existing + num_clips):
        clip = crop_clip(create_clip(load_path, info.clip_len), height, width);
        save_npz(save_path, str(i), clip);
        if (i + 1) % 10 == 0:
            util.__sysprint__("processed ", (i + 1), " clips...");
    util.__sysprint__("done");
    time.sleep(1);
