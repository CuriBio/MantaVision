# README #

Scripts for performing optical tracking of well magnets in Mantarray.

## Using the notebook interface ##
One way to run the notebook interface is to download the spyder application from

https://www.spyder-ide.org/

Once that has been installed, open spyder and from the file system sub window at the top right,
select the file tab at the bottom and navigate to the curi_tracker directory. This sets the 
console sub window below it to the directory you just navigated to, so can now type the 
following commands in the console sub window to install some required packages:

> pip install spyder-notebook <br/>
> pip install --user -r ./requirements.txt

Now restart spyder.
You will now see that in the main window there are two tabs, Editor and Notebook.
Open the curi_tracker.ipynb file from within spyder (file -> open), and then
select the Notebook tab. You can now run the notebook by pressing Shift and Enter.

## Using the command line interface ##

### Install Required Packages ###
> pip install --user -r ./requirements.txt

### run template tracking with all args ###
> track_template.py -h for all required and optional args

### run template tracking with from a json config file ###
> track_template.py --json_config /path/to/config.json
