import argparse
from imaging_features.utils_lung_segmentation import *

def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--images_root_dir', type=str, required=True, \
                            help='Path to the directory with the images to segment.')
    parser.add_argument('--results_root_dir', type=str, required=True, \
                            help='Path to the directory where the segmentations will be stored.')
    args = parser.parse_args()

    lung_segmentation_obj = LungSegmentationClass()
    lung_segmentation_obj.obtain_lung_segmentation(args.images_root_dir, args.results_root_dir)

main()
