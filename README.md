# README #

Scripts for performing optical tracking of well magnets in Mantarray.

### Install Required Packages ###
> pip install --user -r ./requirements.txt

### To convert a video to a sequence of jpg's for use in tracking software such as DICe ###
> python /path/to/video2jpgs.py /path/to/input_video /path/to/output/sequence/dir/ [-enhance_contrast]

or to see usage and options info:
> python video2jpgs.py 

### To perform template matching on a single image or a directory of images for use in tracking software such as DICe ###
> python /path/to/match_template.py /path/to/input /path/to/template_image

Note: Input can be either a single image or a directory of images. The image with the best match will be specified.

or to see usage and options info:
> python match_template.py