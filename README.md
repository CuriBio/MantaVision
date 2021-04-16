# README #

Scripts for performing optical tracking of well magnets in Mantarray.

## Using the notebook interface on windows ##

One way to run the notebook interface is to download the anaconda package

https://www.anaconda.com/products/individual

Once that has been installed, from the windows menu open spyder (anaconda).
From the main (top) menu in spyder select view -> window layouts -> spyder default.
Then in the sub window at the top right you will see some tabs at the botoom of that sub window,
one of those tabs is 'Files', select this tab and navigate to the curi_tracker directory.
This sets the console sub window below it to the directory you just navigated to.
In the Console (1/A) sub window in the bottom right, type following commands to install some required packages:


> pip install --user spyder-notebook

> pip install --user -r ./requirements.txt


Close spyder.

You can now run the notebook from either spyder or jupyter.

To run with jupyter (recommended):
from the windows menu open Jupyter (anaconda),
navigate to the curi_tracker directory and select the curi_tracker.ipynb file.
You can now run this notebook by pressing Shift and Enter.

To run with spyder: 
(Note: you still need to have closed spyder from the previous step):
Re-open spyder, from the windows menu open spyder (anaconda).
You will now see that in the main window there are two tabs, Editor and Notebook.
Select the Notebook tab. In the top right hand corner of the notebook sub window 
you will so 3 horizontal stripes, click this and select open. Navigate to the 
curi_tracker.ipynb file and select it.
You can now run the notebook by pressing Shift and Enter.

## Using the command line interface ##

### Install Required Packages ###
> pip install -r ./requirements.txt

### run template tracking with all args ###
> track_template.py -h for all required and optional args

### run template tracking with from a json config file ###
> track_template.py --json_config /path/to/config.json
