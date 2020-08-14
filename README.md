# dohamaps
[![](https://img.shields.io/badge/pypi-v1.1.2-brightgreen)](https://pypi.org/project/dohamaps/)

AI are supposed to steal your jobs. If your job is to do this: 

![](https://i.imgur.com/71VPyOK.jpg)

then `dohamaps` will steal your job. Technically it can be used for any future video frame predictions, but it's called `dohamaps` so it's for Doha maps. If you use it for anything else your computer will explode. Just kidding. Or maybe not. I haven't tried.

It includes an associated [`python3` package](https://pypi.org/project/dohamaps/) that can be downloaded on its own and used from the command line. It is also included as a GUI [`electron`](https://electronjs.org) app that can run training, testing, and prediction. How do you run it? 
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
### As a GUI app
Install `node` from [here](https://nodejs.org/en/download/), and then in the `dohamaps` folder, run

    npm install
    npm start
and use the app. That's it! Unless it crashes your computer, which it probably will.
