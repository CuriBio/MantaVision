
from morphology import computeMorphologyMetrics
from mantavision import runTrackTemplate
from gooey import Gooey, GooeyParser
from json import dump as writeJSON, load as readJSON
from pathlib import Path 
from typing import Dict


def runTracking(args: Dict):
    ''' Runs TrackTemplate function with the arguments provided by the Gooey UI.'''
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
    ''' Runs Morphology function with the arguments provided by the Gooey UI.'''
    computeMorphologyMetrics(
        search_image_path=args.morphology_search_image_path,
        left_template_image_path=args.morphology_left_template_image_path,
        right_template_image_path=args.morphology_right_template_image_path,
        left_sub_template_image_path=None,
        right_sub_template_image_path=None,
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
    return readJSON(open(prev_field_values_file_path))


def saveCurrentFieldValues(
    args,
    previous_field_values: Dict,
    current_field_values_file_path: str
):
    """ write the current ui values to prev_run_values_file_path """
    
    current_field_values = previous_field_values
    if args.actions == 'Track':
        current_field_values['tracking_video_dir'] = args.tracking_video_dir
        current_field_values['tracking_horizontal_contraction_direction'] = args.tracking_horizontal_contraction_direction
        current_field_values['tracking_vertical_contraction_direction'] = args.tracking_vertical_contraction_direction
        current_field_values['tracking_template_path'] = args.tracking_template_path
        current_field_values['tracking_output_path'] = args.tracking_output_path
        current_field_values['tracking_output_frames'] = args.tracking_output_frames
        current_field_values['tracking_guide_match_search_seconds'] = args.tracking_guide_match_search_seconds
        current_field_values['tracking_max_translation_per_frame'] = args.tracking_max_translation_per_frame
        current_field_values['tracking_max_rotation_per_frame'] = args.tracking_max_rotation_per_frame
        current_field_values['tracking_output_conversion_factor'] = args.tracking_output_conversion_factor
        current_field_values['tracking_microns_per_pixel'] = args.tracking_microns_per_pixel
        current_field_values['tracking_sub_pixel_search_increment'] = args.tracking_sub_pixel_search_increment
        current_field_values['tracking_sub_pixel_refinement_radius'] = args.tracking_sub_pixel_refinement_radius
    elif args.actions == 'Morphology':
        current_field_values['morphology_search_image_path'] = args.morphology_search_image_path
        current_field_values['morphology_left_template_image_path'] = args.morphology_left_template_image_path
        current_field_values['morphology_right_template_image_path'] = args.morphology_right_template_image_path
        current_field_values['morphology_template_refinement_radius'] = args.morphology_template_refinement_radius
        current_field_values['morphology_edge_finding_smoothing_radius'] = args.morphology_edge_finding_smoothing_radius
        current_field_values['morphology_microns_per_pixel'] = args.morphology_microns_per_pixel
        current_field_values['morphology_sub_pixel_search_increment'] = args.morphology_sub_pixel_search_increment
        current_field_values['morphology_sub_pixel_refinement_radius'] = args.morphology_sub_pixel_refinement_radius
    with open(current_field_values_file_path, 'w') as outfile:
        writeJSON(current_field_values, outfile, indent=4)


def ensureDefaultFieldValuesExist(prev_run_values_file_path: str):
    if Path(prev_run_values_file_path).is_file():
        return
    default_field_values = {
        'tracking_video_dir': '',
        'tracking_horizontal_contraction_direction': 'right',
        'tracking_vertical_contraction_direction': 'none',
        'tracking_template_path': None,
        'tracking_output_path': None,
        'tracking_output_frames': False,
        'tracking_guide_match_search_seconds': 5.0,
        'tracking_max_translation_per_frame': 100,
        'tracking_max_rotation_per_frame': 3.0,
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
        'morphology_sub_pixel_refinement_radius': None
    }
    with open(prev_run_values_file_path, 'w') as outfile:
        writeJSON(default_field_values, outfile, indent=4)


@Gooey(
    program_name="Curibio Mantavision Toolkit",
    # fullscreen=True,
    default_size=(1200, 900),
    # navigation='TABBED',
    optional_cols=3,
)
def main():
    """ Mantavision (MV) GUI description/layout """

    program_description = 'Track objects in videos/images'
    parser = GooeyParser(description=program_description)
    subs = parser.add_subparsers(help='Actions', dest='actions')

    mv_config_file_path = ".mv_initial_values.json"
    ensureDefaultFieldValuesExist(mv_config_file_path)

    initial_values = previousFieldValues(mv_config_file_path)
    
    ###############
    # Tracking UI #
    ###############
    track_template_parser = subs.add_parser(
        'Track',
        help='Track a user defined or template ROI through one or more videos'
    )
    track_template_parser.add_argument(
        'tracking_video_dir',
        metavar='Input Video Path',
        help='path to a directory with the input videos',
        widget='DirChooser',
        type=str,
        gooey_options={'full_width': True, 'initial_value': initial_values['tracking_video_dir']}  # ''
    )
    track_template_parser.add_argument(
        'tracking_horizontal_contraction_direction',
        metavar='Horizontal Contraction Direction',        
        help='Horizontal direction component of contration',
        choices=['left', 'right', 'none'],
        default=initial_values['tracking_horizontal_contraction_direction']  # 'right'
    )
    track_template_parser.add_argument(
        'tracking_vertical_contraction_direction',
        metavar='Vertical Contraction Direction',
        help='Vertical direction component of contration',
        choices=['up', 'down', 'none'],
        default=initial_values['tracking_vertical_contraction_direction']  # 'none'
    )
    track_template_parser.add_argument(
        '--tracking_template_path',
        metavar='Template Path',        
        help='path to an image with to be used as a template to track [Leave blank to draw a ROI to track]',
        widget='FileChooser',
        type=str,        
        default=None,
        gooey_options={'initial_value': initial_values['tracking_template_path']} # None
    )
    track_template_parser.add_argument(
        '--tracking_output_path', 
        metavar='Output Path',
        help='dir path to store output of trcked video/s [Leave blank to use a sub dir of the input dir]',
        widget='DirChooser',
        type=str,
        default=None,
        gooey_options={'initial_value': initial_values['tracking_output_path']} # None
    )
    track_template_parser.add_argument(
        '--tracking_output_frames',
        metavar='Output Frames',
        help=' Ouput individual tracking results frames',
        action='store_true',
        default=initial_values['tracking_output_frames']  # False
    )
    track_template_parser.add_argument(
        '--tracking_guide_match_search_seconds',
        metavar='Guide Search Time',
        help='seconds of video to search for a match to the template guide ROI [Leave blank to use the template]',
        type=float,
        default=None,
        gooey_options={'initial_value': initial_values['tracking_guide_match_search_seconds']}  # 5.0
    )
    track_template_parser.add_argument(
        '--tracking_max_translation_per_frame',
        metavar='Max Translation', 
        help='limits search to this amount of translation in any direction from the last frames results [Leave blank to search the entire frame]',
        type=float,
        default=None,
        gooey_options={'initial_value': initial_values['tracking_max_translation_per_frame']}  # 100
    )
    track_template_parser.add_argument(
        '--tracking_max_rotation_per_frame',
        metavar='Max Rotation',        
        help='limits search to this amount of rotation in either direction from the last frames results [Leave blank to ignore rotation]',
        type=float,
        default=None,
        gooey_options={'initial_value': initial_values['tracking_max_rotation_per_frame']}  # 3.0
    )
    track_template_parser.add_argument(
        '--tracking_output_conversion_factor',
        metavar='Conversion Factor',        
        help='apply this multiplier to all distance calulations',
        type=float,
        default=None,
        gooey_options={'initial_value': initial_values['tracking_output_conversion_factor']}  # 1.0
    )
    track_template_parser.add_argument(
        '--tracking_microns_per_pixel',
        metavar='Microns per Pixel',        
        help='conversion from pixels to microns for distances in results',
        type=float,
        default=None,
        gooey_options={'initial_value': initial_values['tracking_microns_per_pixel']}  # 1.0
    )
    track_template_parser.add_argument(
        '--tracking_spacer_1',
        gooey_options={'visible': False}
    )    
    track_template_parser.add_argument(
        '--tracking_sub_pixel_search_increment',
        metavar='Subpixel Search Increment',        
        help='search will be sub pixel accurate to within +/- this value',
        type=float,
        default=None,
        gooey_options={'initial_value': initial_values['tracking_sub_pixel_search_increment']}  # None
    )
    track_template_parser.add_argument(
        '--tracking_sub_pixel_refinement_radius',
        metavar='Subpixel Refinement Radius',        
        help='sub pixel search will be limited to +/- this amount in each dimension',
        type=float,
        default=None,
        gooey_options={'initial_value': initial_values['tracking_sub_pixel_refinement_radius']}  # None        
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
        help='path to a directory with the input images to analyze',
        widget='DirChooser',
        type=str,
        gooey_options={'full_width': True, 'initial_value': initial_values['morphology_search_image_path']}  # ''
    )
    morphology_parser.add_argument(
        'morphology_left_template_image_path',
        metavar='Left Template Path',
        help='path to a template image of the left post',
        widget='FileChooser',
        type=str,
        gooey_options={'full_width': True, 'initial_value': initial_values['morphology_left_template_image_path']}  # ''
    )
    morphology_parser.add_argument(
        'morphology_right_template_image_path',
        metavar='Right Template Path',
        help='path to a template image of the right post',
        widget='FileChooser',
        type=str,
        gooey_options={'full_width': True, 'initial_value': initial_values['morphology_right_template_image_path']}  # ''
    )    
    morphology_parser.add_argument(
        '--morphology_template_refinement_radius',
        metavar='Template Edge Refinement Radius',
        help='search +/- this value at the inner edge of each template for a better edge match',
        type=int,
        default=None,
        gooey_options={'initial_value': initial_values['morphology_template_refinement_radius']}  # 40
    )
    morphology_parser.add_argument(
        '--morphology_edge_finding_smoothing_radius',
        metavar='Template Edge Smoothing Radius',
        help='the top and bottom edges will be average-smoothed by this amount',
        type=int,
        default=None,
        gooey_options={'initial_value': initial_values['morphology_edge_finding_smoothing_radius']}  # 10
    )        
    morphology_parser.add_argument(
        '--morphology_microns_per_pixel',
        metavar='Microns per Pixel',        
        help='conversion from pixels to microns for distances in results',
        type=float,
        default=None,
        gooey_options={'initial_value': initial_values['morphology_microns_per_pixel']}  # 1.0
    )
    morphology_parser.add_argument(
        '--morphology_sub_pixel_search_increment',
        metavar='Subpixel Search Increment',        
        help='search will be sub pixel accurate to within +/- this value',
        type=float,
        default=None,
        gooey_options={'initial_value': initial_values['morphology_sub_pixel_search_increment']}  # None
    )
    morphology_parser.add_argument(
        '--morphology_sub_pixel_refinement_radius',
        metavar='Subpixel Refinement Radius',        
        help='sub pixel search will be limited to +/- this amount in each dimension',
        type=float,
        default=None,
        gooey_options={'initial_value': initial_values['morphology_sub_pixel_refinement_radius']}  # None
    )
    
    args = parser.parse_args()
    saveCurrentFieldValues(args, initial_values, mv_config_file_path)
    if args.actions == 'Track':
        runTracking(args)
    elif args.actions == 'Morphology':
        runMorphology(args)
    else:
        raise RuntimeError('Invalid Action Chosen')

if __name__ == '__main__':
    main()
