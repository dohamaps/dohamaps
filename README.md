# dohamaps
[![](https://img.shields.io/badge/pypi-v1.1.4-brightgreen)](https://pypi.org/project/dohamaps/)

![](https://i.imgur.com/YtSWbyg.png)

`dohamaps` is a program for generative adversarial prediction of urban growth in Doha.

## Instructions
### Setup
To use this from the command line, first clone the repository, either from GitHub Desktop or the command line

    git clone https://github.com/dohamaps/dohamaps.git
Then install `python3.7` from [here](https://www.python.org/downloads/release/python-370/), and then create a virtual environment in the `dohamaps` directory

    python3 -m venv --copies venv
Activate the virtual environment using the instructions [here](https://docs.python.org/3/library/venv.html) and install dependencies using `pip`

    pip3 install --upgrade pip
    pip3 install tensorflow==2.2.0 pillow==7.2.0
### From the command line
Make sure the virtual environment is still activated and run

    python3 main.py <options...>
To see the full list of options, run

    python3 main.py --help
Most commonly, to process, train, test, and predict, you will want to run something like,

    python3 main.py --process=<number of clips>
    python3 main.py --train=<epochs> --batch-size=<size>
    python3 main.py --test=<steps>
    python3 main.py --predict
To run a pre-trained model prediction, you will want to run,

    python3 main.py --predict --load-weights=<.../path/to/weights>
All outputs (cache, clips, predictions, and saved models) are saved in `~/.dohamaps/`. This is a hidden folder, so navigate to it with Terminal or using `cmd` + `shift` + `.` on MacOS.
### As a GUI app
Install `node` from [here](https://nodejs.org/en/download/), and then in the `dohamaps` folder, run

    npm install
    npm start
and use the app. That's it!
