
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

class info(object):
    """
        parameters initialized to `None` should be added in
        using util.info_json()
    """
    hist_len = None;
    pred_len = None;
    clip_len = None;
    num_scales = None;
    disc_lrate = None;
    gen_lrate = None;
    train_dim = None;
    pred_dim = None;
    scale_min = None;
    scale_up = None;
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
