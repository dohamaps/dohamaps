import  setuptools
import  os

with open(os.path.join("dohamaps", "README.md"), "r") as readme:
    long_description = readme.read();

setuptools.setup \
(
    name = "dohamaps",
    version = "1.1.1",
    author = "dohamaps",
    author_email = "dohamaps20@gmail.com",
    description = "Generative adversarial urban growth prediction of Doha.",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "https://github.com/dohamaps/dohamaps",
    packages = setuptools.find_packages(),
    classifiers = \
    [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
