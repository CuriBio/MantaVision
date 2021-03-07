# README #

Scripts for performing optical tracking of well magnets in Mantarray.

### Install Required Packages ###
> pip install --user -r ./requirements.txt

### How to run template tracking ###
> python /path_to/track_template.py /path_to/video.mp4 /path_to/template.jpg --output_json_path /path_to_write/tracking_results.json --output_video_path /path_to_write/tracking_results.mp4 -template_as_guide --seconds_per_period 5 --path_to_excel_template /path_to/optical_tracking_results_template.xlsx --path_to_excel_results /path_to_write/tracking_results.xlsx

### Run template tracking with the test data in this repo ###
From the top level dir, do the following:

Create a directory called test_output i.e. 
> mkdir ./test_output

Then run the tracker
> python ./track_template.py ./test_data/videos/2021.01.27_EHT_Plate_A006.mp4 ./test_data/image_templates/magnet_tip_template_rotated-15degrees.jpg --output_json_path ./test_output/tracking_results.json --output_video_path ./test_output/tracking_results.mp4 -template_as_guide --seconds_per_period 5 --path_to_excel_template ./test_data/excel_templates/optical_tracking_results_template.xlsx --path_to_excel_results ./test_output/tracking_results.xlsx