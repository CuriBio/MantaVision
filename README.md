# Mantarray Computer Vision Tool #
Features include:

- optical tracking in vidoes for tissue contraction analysis (e.g. well post magnet tracking)

- morphological measurements of tissue images

- Ca2+ signal extraction and analysis of floresence images

</br></br>
## Binaries ##
https://github.com/CuriBio/MantaVision/releases/latest

</br></br>
## Source ##

#### Installing Required Packages ###
MS Windows:

```> pip install -r \path\to\src\requirements.txt```

macOS, Linux:

```> pip install -r /path/to/src/requirements.txt```

</br></br>
#### Launching the GUI ###
MS Windows:

```> python \path\to\src\mantavision.py```

macOS, Linux:

```> python /path/to/src/mantavision.py```

</br></br>
### Issues  ###
1) In the GitHub actions yml file, the python interpreter has been set to a specific version because the GUI (built with Gooey) relies on wxWindows which has a broken build for python 3.10+.

2) In the GitHub actions yml file, pyinstaller has been set to a specific version because newer version trip the windows anti-virus.

3) In the requirements.txt file, python-opencv has been set to a specific version because there is a bug with later opencv versions on windows (that only seems to affect building an exe with pyinstaller) which results in launch failure due to a recursive import problem.
