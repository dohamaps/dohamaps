import  tensorflow  as tf
import  numpy       as np
import  os
import  shutil
import  json

from    ast         import literal_eval     as parse

def get_dir(directory):
    """
        creates the given directory if it does not exist
        @param directory: the path to the directory
        @return: the path to the directory
    """
    if (not os.path.exists(directory)):
        original_umask = os.umask(0);
        try:
            os.makedirs(directory, 0o777);
        finally:
            os.umask(original_umask)
    return directory;

from    .           import const
from    glob        import glob

def __sysprint__(*args):
    """
        Prints dohamaps system messages distinctly.
        @param *args: Strings to be printed.
    """
    if (const.__log__ == const.log_mode.none):
        pass;
    if (const.__log__ == const.log_mode.bash):
        print("\033[36m", "  <dohamaps> ", "\033[m", *args);
    if (const.__log__ == const.log_mode.html):
        print(*args);

def eager_str():
    return "on" if tf.executing_eagerly() else "off";

def gpu_str():
    if tf.config.experimental.list_physical_devices("GPU"):
        return "available";
    else:
        return "not available";

def clear_dir(directory):
    """
        removes all files in the given directory
        @param directory: the path to the directory
    """
    for file in os.listdir(directory):
        path = os.path.join(directory, file);
        try:
            if os.path.isfile(path):
                os.unlink(path);
            elif os.path.isdir(path):
                shutil.rmtree(path);
        except Exception as exn:
            __sysprint__(exn);

def info_json(path):
    with open(path) as file:
        data = json.load(file);
    for (key, value) in data.items():
        setattr(const.info, key, parse(value));
    setattr(const.info, "scale_min",
            1.0 / (2.0 ** (const.info.num_scales - 1)));
    setattr(const.info, "clip_len", const.info.hist_len + const.info.pred_len);
