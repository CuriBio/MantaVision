## Computer Vision tool for Mantarray ##
Features include:
- optical tracking in vidoes for tissue contraction analysis (e.g. well post magnet tracking)
- morphological measurements of tissue images
- Ca2+ signal extraction and analysis of floresence images

</br></br>
## Installing the required packages ##
MS Windows:

> pip install -r \path\to\src\requirements.txt

macOS, Linux:

> pip install -r /path/to/src/requirements.txt

</br></br>
## Running Mantavision  ##
MS Windows:

> python \path\to\src\mantavision_gui.py

macOS, Linux:

> python /path/to/src/mantavision_gui.py

</br></br>
## Issues  ##
1) In the requirements file the version of python-opencv has been set to a specific version. 
This is because there is a known bug with later opencv versions that results in a recursive import
that only seems to affect building an exe with pyinstaller.
2) In the GitHub actions yml file, the version of python has been set to a specific version.
This is because the GUI relies on wxWindows which is has a broken build for python 3.10+
