
import  os

from    .   import util

def __constattr__(arg):
    is_special = lambda name: (name.startswith("__") and name.endswith("__"));
    __cc = { n: getattr(arg, n) for n in vars(arg) if not is_special(n) };
    def unbind(value):
        return lambda self: value;
    __pc = { name: property(unbind(value)) for (name, value) in __cc.items() };
    receptor = type(arg.__name__, (object,), __pc);
    return receptor();

@__constattr__
class info(object):
    hist_len = 8;
    pred_len = 16;
    clip_len = hist_len + pred_len;
    num_scales = 4;
    disc_lrate = 5e-5;
    gen_lrate = 1e-5;
    train_dim = (64, 64);
    pred_dim = (415, 350);
    scale_min = 1.0 / (2.0 ** (num_scales - 1));
    scale_up = 2.0;
    path = util.get_dir(os.path.expanduser("~/.dohamaps"));
    clips_path = util.get_dir(os.path.join(path, "clips"));
    save_path = util.get_dir(os.path.join(path, "save"));
    cache_path = util.get_dir(os.path.join(path, "cache"));

@__constattr__
class mode(object):
    undefined = 0;
    process = 1;
    train = 2;
    test = 3;
    pred = 4;

@__constattr__
class log_mode(object):
    none = 0;
    bash = 1;
    html = 2;

def log(value):
    global __log__;
    __log__ = value;
