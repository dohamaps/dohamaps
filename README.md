# dohamaps
## Dependencies
 * Python 3.7.0
 * TensorFlow 2.2.0
 * PIL (Pillow) 7.2.0
## Instructions
### Cloning this repository
Install GitHub Desktop (easier) or use the command line (harder but more mysterious-looking and makes everyone think you're a CIA agent (like me (whoops the CIA is going to fire me for revealing that))) and clone the repository to a folder of your choice. Using GitHub Desktop, just select the repositories tab and click "add". From the command line, navigate to a folder of your choice and run

    git clone https://github.com/aegooby/dohamaps/
See? Way more edgy and wild.
### Installing Python3

If you are on Mac/Linux, install Homebrew and use it to install Python by running

    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install.sh)"
    brew install python3
If you are on Windows, install [Chocolatey](https://chocolatey.org/install) and use it to install Python

    choco install python
### Installing `pip`
Install the `pip` package manager by running

    python3 -m pip install -U pip
    pip install --upgrade pip
### Creating a virtual environment
Navigate to the repository folder named `dohamaps` and run

    python3 -m venv venv
To activate the virtual environment on Mac/Linux, navigate to the `dohamaps` folder and run

    source venv/bin/activate
On Windows, run

    venv\Scripts\activate.bat
    
**Buddy, always activate the virtual environment before installing any packages using `pip` or before attempting to run the program. If you don't do this I will personally show up at your house, clog your toilet, and remove 3 of your limbs.**
### Installing packages
`pip` will let you install the remaining dependencies by running the following from the command line

    pip3 install tensorflow==2.2.0
    pip3 install Pillow==7.2.0
## Running the program
Activate the virtual environment if you didn't (if you didn't activate it then shame on you because I already told you to earlier and you know what happens if you don't).
### Image preprocessing
If the images haven't already been preprocessed into smaller size clips (which is needed to prevent memory overflow), then preprocess them. If you don't, your computer might crash which isn't too bad for me but I'm guessing it would suck for you. If you're interested in it not crashing then preprocess the images by navigating to the `dohamaps` folder and running

    python3 main.py --process=<number of clips>
This only needs to be done once if it works.
### Training the network
Now for the real cool stuff where the computers take everyone's jobs (especially yours): from the same folder (`dohamaps`) run 

    python3 main.py --train=<number of epochs> --batch-size=<batch size>
This takes a long time so doing it overnight is not a bad idea. If it's interrupted in the middle or your computer crashes, go to `dohamaps` and delete the folder named `save` and start it over again.

Ok that's it you can go back to whatever you were doing now.
