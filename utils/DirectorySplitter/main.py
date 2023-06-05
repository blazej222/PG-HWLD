import os
import random
import shutil


def split_directory(data_dir, train_dir, val_dir, split_ratio):
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    class_labels = os.listdir(data_dir)

    for label in class_labels:
        train_label_dir = os.path.join(train_dir, label)
        val_label_dir = os.path.join(val_dir, label)
        os.makedirs(train_label_dir, exist_ok=True)
        os.makedirs(val_label_dir, exist_ok=True)

        label_dir = os.path.join(data_dir, label)
        image_files = os.listdir(label_dir)
        random.shuffle(image_files)

        split_index = int(split_ratio * len(image_files))
        train_files = image_files[:split_index]
        val_files = image_files[split_index:]

        for file_name in train_files:
            src = os.path.join(label_dir, file_name)
            dst = os.path.join(train_label_dir, file_name)
            shutil.copyfile(src, dst)

        for file_name in val_files:
            src = os.path.join(label_dir, file_name)
            dst = os.path.join(val_label_dir, file_name)
            shutil.copyfile(src, dst)

        print('Label:', label)
        print('Training samples:', len(train_files))
        print('Validation samples:', len(val_files))
        print('---')


def main():
    split_directory(data_dir="../../resources/datasets/transformed/ultimate_dataset_3000",
                    train_dir="../../resources/datasets/divided/ultimate_dataset_3000/train-images",
                    val_dir="../../resources/datasets/divided/ultimate_dataset_3000/test-images",
                    split_ratio=0.8)


if __name__ == '__main__':
    main()
