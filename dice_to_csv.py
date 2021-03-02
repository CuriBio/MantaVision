
import argparse
import os
import sys
import csv


def csv_to_list_of_dicts(csv_path: str) -> {}:
    '''
    '''
    csv_file = open(csv_path)
    csv_reader = csv.DictReader(csv_file, delimiter=',')
    csv_as_list_of_dicts = []
    for row in csv_reader:
        csv_as_list_of_dicts.append(row)
    return csv_as_list_of_dicts


def csv_to_list_of_lists(csv_path: str) -> {}:
    '''
    '''
    csv_file = open(csv_path)
    csv_reader = csv.reader(csv_file, delimiter=',')
    csv_as_list_of_lists = []
    for row in csv_reader:
        csv_as_list_of_lists.append(row)
    return csv_as_list_of_lists


def reformat_dice_results(
    dice_results_csv_path: str,
    results_template_csv_path: str,
    reformatted_dice_results_csv_path: str,
    frames_per_second: float = 1.0,
    pixel_distance_in_microns: float = 1.0
):
    dice_results = csv_to_list_of_dicts(dice_results_csv_path)
    csv_template = csv_to_list_of_lists(results_template_csv_path)

    reformatted_results_file = open(reformatted_dice_results_csv_path, mode='w')
    reformatted_results_writer = csv.writer(reformatted_results_file)
    template_header_row = csv_template[0]
    reformatted_results_writer.writerow(template_header_row)
    csv_template = csv_template[1:]  # now ignore the first template header row

    num_rows_to_write = len(dice_results)
    len_of_template = len(csv_template)
    for row_num in range(num_rows_to_write):
        dice_results_row = dice_results[row_num]
        frame_number = float(dice_results_row['FRAME'])
        elapsed_time = frame_number/float(frames_per_second)
        post_displacement_pixels = float(dice_results_row['DISPLACEMENT_Y'])
        post_displacement_microns = post_displacement_pixels*float(pixel_distance_in_microns) 

        if row_num < len_of_template:
            csv_template_row = csv_template[row_num]
        else:
            csv_template_row = ["", ""]
        csv_template_row[0] = elapsed_time
        csv_template_row[1] = post_displacement_microns        
        reformatted_results_writer.writerow(csv_template_row)


if __name__ == '__main__':

    # parse the input args
    parser = argparse.ArgumentParser(
        description='import results from DICe (csv) and return them as a dict and/or dump them as a reformatted csv',
    )
    parser.add_argument(
        'dice_results_csv_path',
        default=None,
        help='path to the DICe results csv.',
    )
    parser.add_argument(
        'results_template_csv_path',
        default=None,
        help='path to a template csv for writing the results from DICE.',
    )
    parser.add_argument(
        'reformatted_dice_results_csv_path',
        default=None,
        help='path to a write the reformatted results from DICE.',
    )    
    parser.add_argument(
        '--frames_per_second',
        default=1.0,
        help='conversion factor for frame number to elapsed time in seconds.',
    )    
    parser.add_argument(
        '--pixel_distance_in_microns',
        default=1.0,
        help='conversion factor for displacment in pixels to microns.',
    )
    args = parser.parse_args()

    reformat_dice_results(
        args.dice_results_csv_path,
        args.results_template_csv_path,
        args.reformatted_dice_results_csv_path,
        args.frames_per_second,
        args.pixel_distance_in_microns
    )
