import ntpath
import os
import datetime
import argparse
from PIL import Image


def cut_image(image_path, destination, crop_width, crop_height, left_margin, upper_margin, divisor, num_tiles_x,
              num_tiles_y, num_files):
    image = Image.open(image_path)
    subdirectory = destination + f"/{ntpath.basename(image_path)[0]}"
    if not os.path.exists(subdirectory):
        os.makedirs(subdirectory)

    for y in range(num_tiles_y):
        for x in range(num_tiles_x):
            left = left_margin + x * crop_width + x * divisor
            upper = upper_margin + y * crop_height + y * divisor
            right = left + crop_width
            lower = upper + crop_height

            # Break conditions
            if y * num_tiles_x + x == num_files or lower > image.height:
                print(f"Image division for {ntpath.basename(image_path)[0]} completed successfully.")
                return

            # Crop the image
            cropped_image = image.crop((left, upper, right, lower))

            # Save the cropped image
            cropped_image.save(
                subdirectory + f"/{ntpath.basename(image_path)[0]}_{y * num_tiles_x + x + 1}"
                               f"_{hash(image_path + f'/{ntpath.basename(image_path)[0]}')}.png")

    print(f"Image division for {ntpath.basename(image_path)[0]} completed successfully.")


def cut_image_catalog(location, destination, crop_width, crop_height, left_margin, upper_margin, divisor, num_tiles_x,
                      num_tiles_y, num_files=-1):
    if not os.path.exists(destination):
        os.makedirs(destination)
    for address, dirs, files in os.walk(location):
        for file in files:
            cut_image(os.path.join(location, file), destination, crop_width, crop_height, left_margin, upper_margin,
                      divisor, num_tiles_x, num_tiles_y, num_files)


def main():

    parser = argparse.ArgumentParser(
        description='Divide previously fitted scans of letter sheets collected as part of '
                    'collecting datasets into a rectangular grid.')
    parser.add_argument('--source', type=str, required=True,
                        help='Dataset source directory.')
    parser.add_argument('--destination', type=str, required=True,
                        help='Processed dataset destination directory.')
    parser.add_argument('--left_margin', type=int, default=36,
                        help='Left margin.')
    parser.add_argument('--upper_margin', type=int, default=48,
                        help='Upper margin.')
    parser.add_argument('--crop_width', type=int, default=196,
                        help='Crop width.')
    parser.add_argument('--crop_height', type=int, default=196,
                        help='Crop height.')
    parser.add_argument('--divisor', type=int, default=5,
                        help='Size of vertical and horizontal separator(gap) applied when cutting each sample.')
    parser.add_argument('--num_tiles_x', type=int, default=12,
                        help='Amount of tiles in x direction on scanned page.')
    parser.add_argument('--num_tiles_y', type=int, default=17,
                        help='Amount of tiles in y direction on scanned page.')
    parser.add_argument('--num_files', type=int, default=124,
                        help='Amount of samples to be extracted from single scan file.')

    args = parser.parse_args()
    source = args.source
    destination = args.destination
    left_margin = args.left_margin
    upper_margin = args
    crop_width = args.crop_width
    crop_height = args
    divisor = args.divisor
    num_tiles_x = args.num_tiles_x
    num_tiles_y = args.num_tiles_y
    num_files = args.num_files

    cut_image_catalog(source, destination, crop_width, crop_height,
                      left_margin, upper_margin, divisor, num_tiles_x, num_tiles_y, num_files)


if __name__ == '__main__':
    main()
