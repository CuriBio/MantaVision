
from ca2_analysis import analyzeCa2Data
from optical_tracking import runTrackTemplate
from morphology import computeMorphologyMetrics

from gooey import Gooey, GooeyParser
from json import dump as writeJSON, load as readJSON
from pathlib import Path 
from typing import Dict
from time import time


def runCa2Analysis(args: Dict):
    """ Runs analyzeCa2Data function with the arguments provided by the Gooey UI. """
    analyzeCa2Data(
        path_to_data=args.ca2_analysis_path_to_data,
        expected_frequency_hz=args.ca2_analysis_expected_frequency_hz,
        low_signal_to_noise=args.ca2_analysis_low_signal_to_noise,
        save_result_plots=args.ca2_analysis_save_result_plots,
        analysis_method=args.ca2_analysis_method,
        dynamic_roi_template_path=args.ca2_analysis_dynamic_roi_template_path,
        reference_roi_template_path=args.ca2_analysis_reference_roi_template_path,
        select_background_once=args.ca2_analysis_select_background_once
    )


def runTracking(args: Dict):
    """ Runs TrackTemplate function with the arguments provided by the Gooey UI. """
    if args.tracking_output_frames == 'Yes':
        output_frames = True
    else:
        output_frames = False
    if args.tracking_vertical_contraction_direction == 'down':
        tracking_vertical_contraction_direction = -1
    elif args.tracking_vertical_contraction_direction == 'up':
        tracking_vertical_contraction_direction = 1
    else:
        tracking_vertical_contraction_direction = 0
    if args.tracking_horizontal_contraction_direction == 'left':
        tracking_horizontal_contraction_direction = -1
    elif args.tracking_horizontal_contraction_direction == 'right':
        tracking_horizontal_contraction_direction = 1
    else:
        tracking_horizontal_contraction_direction = 0
    contraction_vector = (tracking_horizontal_contraction_direction, tracking_vertical_contraction_direction)

    runTrackTemplate(
        {
            'input_video_path': args.tracking_video_dir,
            'contraction_vector': contraction_vector,
            'output_frames': output_frames,
            'template_guide_image_path': args.tracking_template_path,
            'output_video_path': args.tracking_output_path,
            'guide_match_search_seconds': args.tracking_guide_match_search_seconds,
            'max_translation_per_frame': args.tracking_max_translation_per_frame,
            'max_rotation_per_frame': args.tracking_max_rotation_per_frame,
            'output_conversion_factor': args.tracking_output_conversion_factor,
            'microns_per_pixel': args.tracking_microns_per_pixel,
            'sub_pixel_search_increment': args.tracking_sub_pixel_search_increment,
            'sub_pixel_refinement_radius': args.tracking_sub_pixel_refinement_radius
        }
    )    


def runMorphology(args: Dict):
    """ Runs Morphology function with the arguments provided by the Gooey UI. """
    computeMorphologyMetrics(
        search_image_path=args.morphology_search_image_path,
        left_template_image_path=args.morphology_left_template_image_path,
        right_template_image_path=args.morphology_right_template_image_path,
        sub_pixel_search_increment=args.morphology_sub_pixel_search_increment,
        sub_pixel_refinement_radius=args.morphology_sub_pixel_refinement_radius,
        template_refinement_radius=args.morphology_template_refinement_radius,
        edge_finding_smoothing_radius=args.morphology_edge_finding_smoothing_radius,
        microns_per_pixel=args.morphology_microns_per_pixel,
        write_result_images=True,
        display_result_images=False
    )


def previousFieldValues(prev_field_values_file_path: str) -> Dict:
    """ load all the field values form the previous run """
    previous_field_values = readJSON(open(prev_field_values_file_path))
    # add any missing field values to an existing field value file
    # which can happen if a new version of this app adds some new fields
    # that won't yet be in the previous saved field values file
    default_field_values = defaultFieldValues()
    for default_field_key, default_field_value in default_field_values.items():
        if default_field_key not in previous_field_values:
            previous_field_values[default_field_key] = default_field_value
    return previous_field_values


def saveCurrentFieldValues(
    args,
    previous_field_values: Dict,
    current_field_values_file_path: str
):
    """ write the current ui values to prev_run_values_file_path """
    field_values = previous_field_values
    if args.actions == 'Tracking':
        field_values['tracking_video_dir'] = args.tracking_video_dir
        field_values['tracking_horizontal_contraction_direction'] = args.tracking_horizontal_contraction_direction
        field_values['tracking_vertical_contraction_direction'] = args.tracking_vertical_contraction_direction
        field_values['tracking_template_path'] = args.tracking_template_path
        field_values['tracking_output_path'] = args.tracking_output_path
        field_values['tracking_output_frames'] = args.tracking_output_frames
        field_values['tracking_guide_match_search_seconds'] = args.tracking_guide_match_search_seconds
        field_values['tracking_max_translation_per_frame'] = args.tracking_max_translation_per_frame
        field_values['tracking_max_rotation_per_frame'] = args.tracking_max_rotation_per_frame
        field_values['tracking_output_conversion_factor'] = args.tracking_output_conversion_factor
        field_values['tracking_microns_per_pixel'] = args.tracking_microns_per_pixel
        field_values['tracking_sub_pixel_search_increment'] = args.tracking_sub_pixel_search_increment
        field_values['tracking_sub_pixel_refinement_radius'] = args.tracking_sub_pixel_refinement_radius
    elif args.actions == 'Morphology':
        field_values['morphology_search_image_path'] = args.morphology_search_image_path
        field_values['morphology_left_template_image_path'] = args.morphology_left_template_image_path
        field_values['morphology_right_template_image_path'] = args.morphology_right_template_image_path
        field_values['morphology_template_refinement_radius'] = args.morphology_template_refinement_radius
        field_values['morphology_edge_finding_smoothing_radius'] = args.morphology_edge_finding_smoothing_radius
        field_values['morphology_microns_per_pixel'] = args.morphology_microns_per_pixel
        field_values['morphology_sub_pixel_search_increment'] = args.morphology_sub_pixel_search_increment
        field_values['morphology_sub_pixel_refinement_radius'] = args.morphology_sub_pixel_refinement_radius
    elif args.actions == 'Ca2+_Analysis':
        field_values['ca2_analysis_path_to_data'] = args.ca2_analysis_path_to_data
        field_values['ca2_analysis_expected_frequency_hz'] = args.ca2_analysis_expected_frequency_hz
        field_values['ca2_analysis_save_result_plots'] = args.ca2_analysis_save_result_plots
        field_values['ca2_analysis_method'] = args.ca2_analysis_method
        field_values['ca2_analysis_low_signal_to_noise'] = args.ca2_analysis_low_signal_to_noise
        field_values['ca2_analysis_dynamic_roi_template_path'] = args.ca2_analysis_dynamic_roi_template_path
        field_values['ca2_analysis_reference_roi_template_path'] = args.ca2_analysis_reference_roi_template_path
        field_values['ca2_analysis_select_background_once'] = args.ca2_analysis_select_background_once
    with open(current_field_values_file_path, 'w') as outfile:
        writeJSON(field_values, outfile, indent=4)


def defaultFieldValues() -> Dict:
    return {
        'tracking_video_dir': '',
        'tracking_horizontal_contraction_direction': 'right',
        'tracking_vertical_contraction_direction': 'none',
        'tracking_template_path': None,
        'tracking_output_path': None,
        'tracking_output_frames': False,
        'tracking_guide_match_search_seconds': 5.0,
        'tracking_max_translation_per_frame': 50,
        'tracking_max_rotation_per_frame': 2.0,
        'tracking_output_conversion_factor': 1.0,
        'tracking_microns_per_pixel': 1.0,
        'tracking_sub_pixel_search_increment': None,
        'tracking_sub_pixel_refinement_radius': None,
        'morphology_search_image_path': '',
        'morphology_left_template_image_path': '',
        'morphology_right_template_image_path': '',
        'morphology_template_refinement_radius': 40,
        'morphology_edge_finding_smoothing_radius': 10,
        'morphology_microns_per_pixel': 1.0,
        'morphology_sub_pixel_search_increment': None,
        'morphology_sub_pixel_refinement_radius': None,
        'ca2_analysis_path_to_data': None,
        'ca2_analysis_expected_frequency_hz': 1.0,
        'ca2_analysis_save_result_plots': False,
        'ca2_analysis_method': None,
        'ca2_analysis_low_signal_to_noise': False,
        'ca2_analysis_dynamic_roi_template_path': None,
        'ca2_analysis_reference_roi_template_path': None,
        'ca2_analysis_select_background_once': False
    }


def ensureDefaultFieldValuesExist(prev_run_values_file_path: str):
    if Path(prev_run_values_file_path).is_file():
        return
    with open(prev_run_values_file_path, 'w') as outfile:
        writeJSON(defaultFieldValues(), outfile, indent=4)


GUI_WIDTH = 1200
GUI_HEIGHT = 950


@Gooey(
    program_name="CuriBio MantaVision Toolkit",
    default_size=(GUI_WIDTH, GUI_HEIGHT),
    optional_cols=3,
    # disable_progress_bar_animation=True
)
def main():
    """ Mantavision (MV) GUI description/layout """

    program_description = 'Perform computer vision tasks on videos and images'
    parser = GooeyParser(description=program_description)
    subs = parser.add_subparsers(help='Actions', dest='actions')

    mv_config_file_path = ".mv_initial_values.json"
    ensureDefaultFieldValuesExist(mv_config_file_path)

    initial_values = previousFieldValues(mv_config_file_path)
    
    ###############
    # Tracking UI #
    ###############
    track_template_parser = subs.add_parser(
        'Tracking',
        help='Track a user defined or template ROI through one or more videos'
    )
    track_template_parser.add_argument(
        'tracking_video_dir',
        metavar='Input Video Path',
        help='path to a directory with videos',
        widget='DirChooser',
        type=str,
        gooey_options={'full_width': True, 'initial_value': initial_values['tracking_video_dir']}
    )
    track_template_parser.add_argument(
        'tracking_horizontal_contraction_direction',
        metavar='Horizontal Contraction Direction',        
        help='Horizontal direction component of contraction',
        choices=['left', 'right', 'none'],
        default=initial_values['tracking_horizontal_contraction_direction']
    )
    track_template_parser.add_argument(
        'tracking_vertical_contraction_direction',
        metavar='Vertical Contraction Direction',
        help='Vertical direction component of contraction',
        choices=['up', 'down', 'none'],
        default=initial_values['tracking_vertical_contraction_direction']
    )
    track_template_parser.add_argument(
        '--tracking_template_path',
        metavar='Template Path',        
        help='path to an image that is a template ROI to track\n[leave blank to draw a ROI]',
        widget='FileChooser',
        type=str,
        default=None,
        gooey_options={'initial_value': initial_values['tracking_template_path']}
    )
    track_template_parser.add_argument(
        '--tracking_output_path', 
        metavar='Output Path',
        help='directory to store output of tracked video/s\n[leave blank to use input directory]',
        widget='DirChooser',
        type=str,
        default=None,
        gooey_options={'initial_value': initial_values['tracking_output_path']}
    )
    track_template_parser.add_argument(
        '--tracking_output_frames',
        metavar='Output Frames',
        help=' Output individual tracking results frames',
        action='store_true',
        gooey_options={'initial_value': initial_values['tracking_output_frames']}
    )
    # TODO: use default_values for the value of any help text with "e.g. {value}"
    default_values = defaultFieldValues()
    default_guid_match_search_seconds = default_values['tracking_guide_match_search_seconds']
    guide_match_search_seconds_help = f"search this many seconds of the video for "
    guide_match_search_seconds_help += f"a ROI that best matches the template image "
    guide_match_search_seconds_help += f"(e.g. {default_guid_match_search_seconds})"
    guide_match_search_seconds_help += f"\n[leave blank to use the template]"
    track_template_parser.add_argument(
        '--tracking_guide_match_search_seconds',
        metavar='Guide Search Time',
        help=guide_match_search_seconds_help,
        type=float,
        default=None,
        gooey_options={'initial_value': initial_values['tracking_guide_match_search_seconds']}
    )
    track_template_parser.add_argument(
        '--tracking_max_translation_per_frame',
        metavar='Max Translation',
        help='search +/- this amount of translation in any direction from the last frames results (e.g. 50)\n[leave blank to search the entire frame]',
        type=float,
        default=None,
        gooey_options={'initial_value': initial_values['tracking_max_translation_per_frame']}
    )
    track_template_parser.add_argument(
        '--tracking_max_rotation_per_frame',
        metavar='Max Rotation',
        help='search +/- this amount of rotation in either direction from the last frames results (e.g. 2.0)\n[leave blank to ignore rotation]',
        type=float,
        default=None,
        gooey_options={'initial_value': initial_values['tracking_max_rotation_per_frame']}
    )
    track_template_parser.add_argument(
        '--tracking_output_conversion_factor',
        metavar='Conversion Factor',        
        help='multiply all distance calulations by this value (e.g. 1.0 is no change)\n[leave blank for no change]',
        type=float,
        default=None,
        gooey_options={'initial_value': initial_values['tracking_output_conversion_factor']}
    )
    track_template_parser.add_argument(
        '--tracking_microns_per_pixel',
        metavar='Microns per Pixel',
        help='report displacement values in microns using this value to convert from pixels\n[leave blank for 1:1]',
        type=float,
        default=None,
        gooey_options={'initial_value': initial_values['tracking_microns_per_pixel']}
    )
    track_template_parser.add_argument(
        '--tracking_spacer_1',
        gooey_options={'visible': False}
    )    
    track_template_parser.add_argument(
        '--tracking_sub_pixel_search_increment',
        metavar='Subpixel Search Increment',
        help='search will be sub pixel accurate to within +/- this fraction of a pixel (e.g. 0.5)\n[leave blank for no sub pixel accuracy]',
        type=float,
        default=None,
        gooey_options={'initial_value': initial_values['tracking_sub_pixel_search_increment']}
    )
    track_template_parser.add_argument(
        '--tracking_sub_pixel_refinement_radius',
        metavar='Subpixel Refinement Radius',
        help='sub pixel search will be limited to +/- this amount in each dimension (e.g. 2.0)\n[leave blank for no sub pixel accuracy]',
        type=float,
        default=None,
        gooey_options={'initial_value': initial_values['tracking_sub_pixel_refinement_radius']}        
    )
    #################
    # Morphology UI #
    #################
    morphology_parser = subs.add_parser(
        'Morphology',
        help='Compute Morphology Metrics in Images'
    )
    morphology_parser.add_argument(
        'morphology_search_image_path',
        metavar='Input Dir Path',
        help='path to a directory with images to analyze',
        widget='DirChooser',
        type=str,
        gooey_options={'full_width': True, 'initial_value': initial_values['morphology_search_image_path']}
    )
    morphology_parser.add_argument(
        '--morphology_left_template_image_path',
        metavar='Left Template Path',
        help='path to a template image of the left post',
        widget='FileChooser',
        type=str,
        default=None,
        gooey_options={'full_width': True, 'initial_value': initial_values['morphology_left_template_image_path']}
    )
    morphology_parser.add_argument(
        '--morphology_right_template_image_path',
        metavar='Right Template Path',
        help='path to a template image of the right post',
        widget='FileChooser',
        type=str,
        default=None,
        gooey_options={'full_width': True, 'initial_value': initial_values['morphology_right_template_image_path']}
    )    
    morphology_parser.add_argument(
        '--morphology_template_refinement_radius',
        metavar='Template Edge Refinement Radius',
        help='search +/- this value at the inner edge of each template for a better match (e.g. 40)\n[leave blank for no refinement]',
        type=int,
        default=None,
        gooey_options={'initial_value': initial_values['morphology_template_refinement_radius']}
    )
    morphology_parser.add_argument(
        '--morphology_edge_finding_smoothing_radius',
        metavar='Tissue Edge Smoothing Window',
        help='smooth tissue edges with a windowed rolling average of this many pixels (e.g. 10)\n[leave blank for no smoothing]',
        type=int,
        default=None,
        gooey_options={'initial_value': initial_values['morphology_edge_finding_smoothing_radius']}
    )        
    morphology_parser.add_argument(
        '--morphology_microns_per_pixel',
        metavar='Microns per Pixel',
        help='report displacement values in microns using this value to convert from pixels\n[leave blank for 1:1]',
        type=float,
        default=None,
        gooey_options={'initial_value': initial_values['morphology_microns_per_pixel']}
    )
    morphology_parser.add_argument(
        '--morphology_sub_pixel_search_increment',
        metavar='Subpixel Search Increment',
        help='search will be sub pixel accurate to within +/- this fraction of a pixel (e.g. 0.5)\n[leave blank for no sub pixel accuracy]',
        type=float,
        default=None,
        gooey_options={'initial_value': initial_values['morphology_sub_pixel_search_increment']}
    )
    morphology_parser.add_argument(
        '--morphology_sub_pixel_refinement_radius',
        metavar='Subpixel Refinement Radius',
        help='sub pixel search will be limited to +/- this amount in each dimension (e.g. 2.0)\n[leave blank for no sub pixel accuracy]',
        type=float,
        default=None,
        gooey_options={'initial_value': initial_values['morphology_sub_pixel_refinement_radius']}
    )
    ####################
    # Ca2+ Analysis UI #
    ####################
    ca2_analysis_parser = subs.add_parser(
        'Ca2+_Analysis',
        help='Analyze Ca2+ Videos'
    )
    ca2_analysis_parser.add_argument(
        'ca2_analysis_path_to_data',
        metavar='Input Dir Path',
        help='path to a directory with videos to analyze',
        widget='DirChooser',
        type=str,
        gooey_options={'full_width': True, 'initial_value': initial_values['ca2_analysis_path_to_data']}
    )
    ca2_analysis_parser.add_argument(
        'ca2_analysis_expected_frequency_hz',
        metavar='Expected Frequency Hz',
        help='Signal can be +/- 0.5Hz from this value',
        type=float,
        gooey_options={'initial_value': initial_values['ca2_analysis_expected_frequency_hz']}
    )
    ca2_analysis_parser.add_argument(
        'ca2_analysis_method',
        metavar='Method',
        help='Method to Extract Signal',
        choices=[
            'Auto_Determine_Tissue_Morphology',
            'Auto_Adjusted_Tissue_ROI',
            'Fixed_Tissue_ROI'
        ],
        default=initial_values['ca2_analysis_method']
    )
    ca2_analysis_parser.add_argument(
        '--ca2_analysis_save_result_plots',
        metavar='Plots',
        help=' Generate plots of detected signal\n with peaks and troughs marked',
        action='store_true',
        gooey_options={'initial_value': initial_values['ca2_analysis_save_result_plots']}
    )
    ca2_analysis_parser.add_argument(
        '--ca2_analysis_select_background_once',
        metavar='Draw Background ROI Once',
        help=' Draw background ROI once per batch,\n from the first video processed',
        action='store_true',
        gooey_options={'initial_value': initial_values['ca2_analysis_select_background_once']}
    )
    ca2_analysis_parser.add_argument(
        '--ca2_analysis_low_signal_to_noise',
        metavar='Low Signal to Noise Adjustment',
        help=' Auto detection of tissue ROI will compensate for low signal to noise (VERY SLOW)',
        action='store_true',
        gooey_options={'initial_value': initial_values['ca2_analysis_low_signal_to_noise']}
    )
    ca2_analysis_parser.add_argument(
        '--ca2_analysis_dynamic_roi_template_path',
        metavar='Dynamic ROI Template Path',
        help='path to a template image of the dynamic roi to track',
        widget='FileChooser',
        type=str,
        default=None,
        gooey_options={'full_width': True, 'initial_value': initial_values['ca2_analysis_dynamic_roi_template_path']}
    )
    ca2_analysis_parser.add_argument(
        '--ca2_analysis_reference_roi_template_path',
        metavar='Reference ROI Template Path',
        help='path to a template image of the reference roi',
        widget='FileChooser',
        type=str,
        default=None,
        gooey_options={'full_width': True, 'initial_value': initial_values['ca2_analysis_reference_roi_template_path']}
    )

    args = parser.parse_args()
    saveCurrentFieldValues(args, initial_values, mv_config_file_path)
    start_time = time()
    if args.actions == 'Tracking':
        runTracking(args)
    elif args.actions == 'Morphology':
        runMorphology(args)
    elif args.actions == 'Ca2+_Analysis':
        runCa2Analysis(args)
    else:
        raise RuntimeError('Invalid Action Chosen')
    runtime = round(time() - start_time, 2)
    print(f'\n{args.actions} Runtime: {runtime}s')


if __name__ == '__main__':
    main()
