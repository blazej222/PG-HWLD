import ntpath
import os
from PIL import Image


# TOP MARGIN: 188
# LEFT MARGIN: 64

# W: 187
# H: 187


def cut_image(image_path, destination, crop_width, crop_height, left_margin, upper_margin, divisor, num_tiles_x,
              num_tiles_y):
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

            # Crop the image
            cropped_image = image.crop((left, upper, right, lower))

            # Save the cropped image
            cropped_image.save(subdirectory + f"/{ntpath.basename(image_path)[0]}_{y * num_tiles_x + x + 1}.png")

    print(f"Image division for {ntpath.basename(image_path)[0]} completed successfully.")


def cut_image_catalog(location, destination, crop_width, crop_height, left_margin, upper_margin, divisor, num_tiles_x,
                      num_tiles_y):
    if not os.path.exists(destination):
        os.makedirs(destination)
    for address, dirs, files in os.walk(location):
        for file in files:
            cut_image(os.path.join(location, file), destination, crop_width, crop_height, left_margin, upper_margin,
                      divisor, num_tiles_x, num_tiles_y)


def main():
    # FIXME: Correct paths
    cut_image_catalog("..\\letters-marker-cropped", "..\\dataset-black-marker", crop_width=196, crop_height=196,
                      left_margin=36, upper_margin=48, divisor=5, num_tiles_x=12, num_tiles_y=17)


if __name__ == '__main__':
    main()
