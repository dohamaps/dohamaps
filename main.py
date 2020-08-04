import  os
import  getopt
import  sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3';

sys.path.append("dohamaps");

import  tensorflow      as tf
import  PIL             as pillow

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL);

from    dohamaps        import util, img, run, const
from    dohamaps.const  import mode, info

def usage():
    util.__sysprint__("options:");
    util.__sysprint__("--help         : prints usage");
    util.__sysprint__("--process      = processes n images");
    util.__sysprint__("--data-dir     = directory where unprocessed data is located");
    util.__sysprint__("--train        = trains for n epochs");
    util.__sysprint__("--batch-size   = sets batch size to n");
    util.__sysprint__("--load-weights : loads weights in training");
    util.__sysprint__("--test         = test for n steps");
    util.__sysprint__("--predict      : runs a prediction");
    util.__sysprint__("--weights-dir  = load weights from a different directory");
    util.__sysprint__("--clear-cache  : clears cache directory");
    util.__sysprint__("--clear-save   : clears save directory");
    util.__sysprint__("--clear-clips  : clears clips directory");
    util.__sysprint__("--clear-all    : clears all saved directories");
    util.__sysprint__("--no-gpu       : disables use of GPU");
    util.__sysprint__("--log-html     : generates logs in HTML mode for electron");

def main():
    try:
        optlist = \
        [
            "help",
            "process=",
            "data-dir=",
            "train=",
            "batch-size=",
            "load-weights",
            "test=",
            "predict",
            "weights-dir=",
            "clear-cache",
            "clear-save",
            "clear-clips",
            "clear-all",
            "no-gpu",
            "log-html"
        ];
        opts, _ = getopt.getopt(sys.argv[1:], "", optlist);
    except getopt.GetoptError:
        usage();
        sys.exit(2);

    run_mode = mode.undefined;
    clear_cache = False;
    clear_save = False;
    clear_clips = False;
    verbose = False;

    data_dir = "data";

    num_clips = None;
    epochs = None;
    batch_size = 8;
    load_weights = False;

    steps = None;

    weights_dir = None;

    const.log(const.log_mode.bash);

    for opt, arg in opts:
        if opt in ("--help"):
            usage();
            sys.exit(2);
        if opt in ("--process"):
            if run_mode != mode.undefined:
                print("please select only one of --process, --train, --test, or --predict");
                usage();
                sys.exit(2);
            run_mode = mode.process;
            num_clips = int(arg);
        if opt in ("--data-dir"):
            data_dir = arg;
        if opt in ("--train"):
            if run_mode != mode.undefined:
                print("please select only one of --process, --train, --test, or --predict");
                usage();
                sys.exit(2);
            run_mode = mode.train;
            epochs = int(arg);
        if opt in ("--batch-size"):
            batch_size = int(arg);
        if opt in ("--load-weights"):
            load_weights = True;
        if opt in ("--test"):
            if run_mode != mode.undefined:
                print("please select only one of --process, --train, --test, or --predict");
                usage();
                sys.exit(2);
            run_mode = mode.test;
            steps = int(arg);
        if opt in ("--predict"):
            if run_mode != mode.undefined:
                print("please select only one of --process, --train, --test, or --predict");
                usage();
                sys.exit(2);
            run_mode = mode.pred;
        if opt in ("--weights-dir"):
            weights_dir = arg;
        if opt in ("--clear-cache"):
            clear_cache = True;
        if opt in ("--clear-save"):
            clear_save = True;
        if opt in ("--clear-clips"):
            clear_clips = True;
        if opt in ("--clear-all"):
            clear_cache = True;
            clear_save = True;
            clear_clips = True;
        if opt in ("--no-gpu"):
            tf.config.set_visible_devices([  ], "GPU");
        if opt in ("--log-html"):
            const.log(const.log_mode.html);

    util.__sysprint__("tensorflow version: ", tf.__version__);

    if clear_cache:
        util.clear_dir(info.cache_path);
    if clear_clips:
        util.clear_dir(info.clips_path);
    if clear_save:
        util.clear_dir(info.save_path);

    if run_mode == mode.process:
        img.process(num_clips, data_dir, info.clips_path);
    elif run_mode == mode.train:
        run.train(epochs, batch_size, load_weights);
    elif run_mode == mode.test:
        run.test(steps, batch_size);
    elif run_mode == mode.pred:
        run.predict(weights_dir);

if __name__ == "__main__":
    main();
