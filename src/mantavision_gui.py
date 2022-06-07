
from morphology import computeMorphologyMetrics
from mantavision import runTrackTemplate
from gooey import Gooey, GooeyParser
from typing import Dict


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
            'microns_per_pixel': args.tracking_microns_per_pixel,
            'sub_pixel_search_increment': args.tracking_sub_pixel_search_increment,
            'sub_pixel_refinement_radius': args.tracking_sub_pixel_refinement_radius
        }
    )    

def runMorphology(args):
    ''' Runs Morphology function with the arguments provided by the Gooey UI.'''    
    computeMorphologyMetrics(
        search_image_path=args.search_image_path,
        left_template_image_path=args.left_template_image_path,
        right_template_image_path=args.right_template_image_path,
        left_sub_template_image_path=None,
        right_sub_template_image_path=None,
        sub_pixel_search_increment=args.morphology_sub_pixel_search_increment,
        sub_pixel_refinement_radius=args.morphology_sub_pixel_refinement_radius,
        template_refinement_radius=args.template_refinement_radius,
        edge_finding_smoothing_radius=args.edge_finding_smoothing_radius,
        microns_per_pixel=args.morphology_microns_per_pixel,
        write_result_images=args.write_result_images,
        display_result_images=args.display_result_images
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
        '--tracking_microns_per_pixel',
        metavar='Microns per Pixel',        
        help='conversion from pixels to microns for distances in results',
        type=float,
        default=1.0
    )
    track_template_parser.add_argument(
        '--tracking_sub_pixel_search_increment',
        metavar='Subpixel Search Increment',        
        help='search will be sub pixel accurate to within +/- this value',
        type=float,
        default=None
    )
    track_template_parser.add_argument(
        '--tracking_sub_pixel_refinement_radius',
        metavar='Subpixel Refinement Radius',        
        help='sub pixel search will be limited to +/- this amount in each dimension',
        type=float,
        default=None
    )    

    # ########################################################
    morphology_parser = subs.add_parser(
        'Morphology',
        help='Compute Morphology Metrics in Images'
    )
    morphology_parser.add_argument(
        'search_image_path',
        metavar='Input Dir Path',
        help='path to a directory with the input images to analyze',
        widget='DirChooser',
        type=str,
        gooey_options={'full_width': True}
    )
    morphology_parser.add_argument(
        'left_template_image_path',
        metavar='Left Template Path',
        help='path to a template image of the left post',
        widget='FileChooser',
        type=str,
        gooey_options={'full_width': True}
    )
    morphology_parser.add_argument(
        'right_template_image_path',
        metavar='Right Template Path',
        help='path to a template image of the right post',
        widget='FileChooser',
        type=str,
        gooey_options={'full_width': True}
    )    
    morphology_parser.add_argument(
        '--template_refinement_radius',
        metavar='Template Edge Refinement Radius',
        help='search +/- this value at the inner edge of each template for a better edge match',
        type=int,
        default=40
    )
    morphology_parser.add_argument(
        '--edge_finding_smoothing_radius',
        metavar='Template Edge Smoothing Radius',
        help='the top and bottom edges will be average-smoothed by this amount',
        type=int,
        default=10
    )        
    morphology_parser.add_argument(
        '--morphology_microns_per_pixel',
        metavar='Microns per Pixel',        
        help='conversion from pixels to microns for distances in results',
        type=float,
        default=1.0
    )
    morphology_parser.add_argument(
        '--morphology_sub_pixel_search_increment',
        metavar='Subpixel Search Increment',        
        help='search will be sub pixel accurate to within +/- this value',
        type=float,
        default=None
    )
    morphology_parser.add_argument(
        '--morphology_sub_pixel_refinement_radius',
        metavar='Subpixel Refinement Radius',        
        help='sub pixel search will be limited to +/- this amount in each dimension',
        type=float,
        default=None
    )
    morphology_parser.add_argument(
        '--write_result_images',
        metavar='Write Result Images',
        help=' ouput individual results images with morphological markers drawn',
        action='store_true',
    )
    morphology_parser.add_argument(
        '--display_result_images',
        metavar='Display Result Images',
        help=' show individual results images with morphological markers drawn',
        action='store_true',
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
