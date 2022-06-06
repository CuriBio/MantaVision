"""
Example program to demonstrate Gooey's presentation of subparsers
"""

from typing import Dict
from gooey import Gooey, GooeyParser
from mantavision import runTrackTemplate
from morphology import computeMorphologyMetrics


def runTracking(args):
    ''' Runs TrackTemplate function with the arguments provided by the Gooey UI.'''
    if args.output_frames == 'Yes':
        output_frames = True
    else:
        output_frames = False
    if args.vertical_contraction_direction == 'down':
        vertical_contraction_vector = -1
    elif args.vertical_contraction_direction == 'up':
        vertical_contraction_vector = 1
    else:
        vertical_contraction_vector = 0
    if args.horizontal_contraction_direction == 'left':
        horizontal_contraction_direction = -1
    elif args.horizontal_contraction_direction == 'right':
        horizontal_contraction_direction = 1
    else:
        horizontal_contraction_direction = 0
    contraction_vector = (horizontal_contraction_direction, vertical_contraction_vector)
    runTrackTemplate(
        {
            'input_video_path': args.video,
            'contraction_vector': contraction_vector,
            'output_frames': output_frames,
            'template_guide_image_path': args.template,
            'output_video_path': args.output,
            'guide_match_search_seconds': args.guide_match_search_seconds,
            'max_translation_per_frame': args.max_translation_per_frame,
            'max_rotation_per_frame': args.max_rotation_per_frame,
            'output_conversion_factor': args.output_conversion_factor,
            'microns_per_pixel': args.microns_per_pixel,
            'sub_pixel_search_increment': args.sub_pixel_search_increment,
            'sub_pixel_refinement_radius': args.sub_pixel_refinement_radius
        }
    )    

def runMorphology(args):
    ''' Runs Morphology function with the arguments provided by the Gooey UI.'''    
    computeMorphologyMetrics(
        {
            # search_image_path=test_image_suite_dir,
            # left_template_image_path=left_template_image_path,
            # right_template_image_path=right_template_image_path,
            # template_refinement_radius=40,
            # edge_finding_smoothing_radius=10,
            # microns_per_pixel=1.0,    
            # display_result_images=True,
            # write_result_images=True,            
        }
    )

@Gooey(
    program_name="Curibio Mantavision Toolkit",
    fullscreen=True,
    # navigation='TABBED',
    optional_cols=2
)
def main():
    program_description = 'Track objects in videos/images'
    parser = GooeyParser(description=program_description)
    subs = parser.add_subparsers(help='Actions', dest='actions')
    track_template_parser = subs.add_parser(
        'Track',
        help='Track a user defined or template ROI through one or more videos'
    )
    track_template_parser.add_argument(
        'video',
        metavar='Input Video Path',
        help='path to a directory with the input videos',
        widget='DirChooser',
        type=str,
        gooey_options={'full_width': True}
    )
    track_template_parser.add_argument(
        'horizontal_contraction_direction',
        metavar='Horizontal Contraction Direction',        
        help='Horizontal direction component of contration',
        choices=['left', 'right', 'none'],
        default='right'
    )
    track_template_parser.add_argument(
        'vertical_contraction_direction',
        metavar='Vertical Contraction Direction',
        help='Vertical direction component of contration',
        choices=['up', 'down', 'none'],
        default='none'
    )    
    # track_template_parser.add_argument(
    #     'output_frames',
    #     help=' Ouput individual tracking results frames',
    #     choices=['Yes', 'No'],
    #     default='No'
    # )
    track_template_parser.add_argument(
        '--output_frames',
        metavar='Output Frames',
        help=' Ouput individual tracking results frames',
        action='store_true',
    )    
    track_template_parser.add_argument(
        '--template',
        metavar='Template Path',        
        help='path to an image with to be used as a template to track [Leave blank to draw a ROI.]',
        widget='FileChooser',
        type=str,        
        default=''
    )    
    track_template_parser.add_argument(
        '--output', 
        metavar='Output Path',
        help='dir path to store output of trcked video/s [Leave blank to leave results in sub dir of input.]',
        widget='DirChooser',
        type=str,
        default=None
    )
    track_template_parser.add_argument(
        '--guide_match_search_seconds',
        metavar='Guide Search Time',
        help='seconds of video to search for a match to the template guide ROI [Leave blank to use the template]',
        type=float,
        default=5.0
    )
    track_template_parser.add_argument(
        '--max_translation_per_frame',
        metavar='Max Translation', 
        help='limits search to this amount of translation in any direction. [Leave blank to search the entire frame]',
        type=float,
        default=100
    )
    track_template_parser.add_argument(
        '--max_rotation_per_frame',
        metavar='Max Rotation',        
        help='limits search to this amount of rotation in either direction. [Leave blank to not search with rotation]',
        type=float,
        default=3.0
    )
    track_template_parser.add_argument(
        '--output_conversion_factor',
        metavar='Conversion Factor',        
        help='will apply this multiplier to all distance calulations',
        type=float,
        default=1.0
    )
    track_template_parser.add_argument(
        '--microns_per_pixel',
        metavar='Microns per Pixel',        
        help='conversion from pixels to microns for distances in results',
        type=float,
        default=1.0
    )
    track_template_parser.add_argument(
        '--sub_pixel_search_increment',
        metavar='Subpixel Search Increment',        
        help='search will be sub pixel accurate to within +/- this value',
        type=float,
        default=None
    )
    track_template_parser.add_argument(
        '--sub_pixel_refinement_radius',
        metavar='Subpixel Refinement Radius',        
        help='sub pixel search will be limited to +/- this amount in each dimension',
        type=float,
        default=None
    )    

    # ########################################################
    morphology_parser = subs.add_parser(
        'Morphology',
        help='Description for Morphology action choice'
    )
    morphology_parser.add_argument(
        'non_optional_argument',
        help='description for non optional argument',
        type=int
    )
    morphology_parser.add_argument(
        '--option_1',
        help='description for option 1',
        type=str
    )
    args = parser.parse_args()

    if args.actions == 'Track':
        runTracking(args)
    elif args.actions == 'Morphology':
        runMorphology(args)
    else:
        raise RuntimeError('Invalid Action Chosen')

if __name__ == '__main__':
    main()
