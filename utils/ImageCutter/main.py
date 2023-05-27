import ntpath
import os
from PIL import Image


# TOP MARGIN: 188
# LEFT MARGIN: 64

# W: 187
# H: 187


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
            cropped_image.save(subdirectory + f"/{ntpath.basename(image_path)[0]}_{y * num_tiles_x + x + 1}_{hash(image_path)}.png")

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
    source_image_arrays_path_mp1 = "../../resources/datasets/archives/scans-multi-person-1"
    source_image_arrays_path_mp2 = "../../resources/datasets/archives/scans-multi-person-2"
    source_image_arrays_path_sp = "../../resources/datasets/archives/scans-single-person"
    destination_images_path_mp = "../../resources/datasets/unpacked/dataset-multi-person"
    destination_images_path_sp = "../../resources/datasets/unpacked/dataset-single-person"

    cut_image_catalog(source_image_arrays_path_mp1, destination_images_path_mp, crop_width=196, crop_height=196,
                      left_margin=36, upper_margin=48, divisor=5, num_tiles_x=12, num_tiles_y=17, num_files=124)

    cut_image_catalog(source_image_arrays_path_mp2, destination_images_path_mp, crop_width=196, crop_height=196,
                      left_margin=36, upper_margin=48, divisor=5, num_tiles_x=12, num_tiles_y=17)

    # cut_image_catalog(source_image_arrays_path_sp, destination_images_path_sp, crop_width=196, crop_height=196, left_margin=36, upper_margin=48, divisor=5, num_tiles_x=12, num_tiles_y=17)


if __name__ == '__main__':
    main()
