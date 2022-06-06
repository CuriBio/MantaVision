"""
Example program to demonstrate Gooey's presentation of subparsers
"""

from typing import Dict
from gooey import Gooey, GooeyParser
from mantavision import runTrackTemplate


def runConfigForTracking(args) -> Dict:
    ''' converts the input args from gooey into a dictionary required by the trackTemplate function.'''
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
    return {
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

def runConfigForMorphology(args) -> Dict:
    return {}

@Gooey(optional_cols=2, program_name="Curibio Mantavision Toolkit")
def main():
    settings_msg = 'Track objects in videos/images'
    parser = GooeyParser(description=settings_msg)
    parser.add_argument(
        '--verbose',
        help='be verbose',
        dest='verbose',
        action='store_true',
        default=False
    )

    subs = parser.add_subparsers(help='Actions', dest='actions')
    track_template_parser = subs.add_parser(
        'Track',
        help='Track a user defined or template ROI through one or more videos'
    )
    track_template_parser.add_argument(
        'video',
        help='path to a directory with the input videos',
        widget='DirChooser',        
        type=str
    )
    track_template_parser.add_argument(
        'output_frames',
        help=' Ouput individual tracking results frames',
        choices=['Yes', 'No'],
        default='No'
    )
    track_template_parser.add_argument(
        'horizontal_contraction_direction',
        help='Horizontal direction component of contration',
        choices=['left', 'right', 'none'],
        default='right'
    )
    track_template_parser.add_argument(
        'vertical_contraction_direction',
        help='Vertical direction component of contration',
        choices=['up', 'down', 'none'],
        default='none'
    )    
    track_template_parser.add_argument(
        '--template',
        help='path to an image with to be used as a template to track [Leave blank to draw a ROI.]',
        widget='FileChooser',
        type=str,        
        default=''
    )    
    track_template_parser.add_argument(
        '--output', 
        help='dir path to store output of trcked video/s [Leave blank to leave results in sub dir of input.]',
        widget='DirChooser',
        type=str,
        default=None
    )
    track_template_parser.add_argument(
        '--guide_match_search_seconds',
        help='seconds of video to search for a match to the template guide ROI [Leave blank to use the template]',
        type=float,
        default=5.0
    )
    track_template_parser.add_argument(
        '--max_translation_per_frame',
        help='limits search to this amount of translation in any direction. [Leave blank to search the entire frame]',
        type=float,
        default=100
    )
    track_template_parser.add_argument(
        '--max_rotation_per_frame',
        help='limits search to this amount of rotation in either direction. [Leave blank to not search with rotation]',
        type=float,
        default=3.0
    )
    track_template_parser.add_argument(
        '--output_conversion_factor',
        help='will apply this multiplier to all distance calulations',
        type=float,
        default=1.0
    )
    track_template_parser.add_argument(
        '--microns_per_pixel',
        help='conversion from pixels to microns for distances in results',
        type=float,
        default=1.0
    )
    track_template_parser.add_argument(
        '--sub_pixel_search_increment',
        help='search will be sub pixel accurate to within +/- this value',
        type=float,
        default=None
    )
    track_template_parser.add_argument(
        '--sub_pixel_refinement_radius',
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
        run_config = runConfigForTracking(args)
    elif args.actions == 'Morphology':
        run_config = runConfigForMorphology(args)
    else:
        raise RuntimeError('Invalid Action Chosen')

    runTrackTemplate(run_config)

if __name__ == '__main__':
    main()
