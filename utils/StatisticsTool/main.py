import argparse
import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import time


def process_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Image at path {image_path} could not be read.")

    non_zero_pixels = np.count_nonzero(image)
    mean_weight = np.mean(image)

    return non_zero_pixels, mean_weight


def process_directory(root_dir, output_csv_path, verbose):
    results = []

    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                image_path = os.path.join(subdir, file)
                try:
                    non_zero_pixels, mean_weight = process_image(image_path)
                    results.append((image_path, non_zero_pixels, mean_weight))
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")
        if verbose:
            print(f"Processed directory {os.path.split(subdir)[1]}")

    df = pd.DataFrame(results, columns=['filename', 'non_zero_pixels', 'mean_weight'])
    df.to_csv(os.path.join(str(output_csv_path), root_dir.split(os.sep)[-1]) + ".csv", index=False)

    return df


def calculate_means(df):
    df['subdir'] = df['filename'].apply(lambda x: str(x).split(os.sep)[-2])

    # Mean statistics for each subdirectory
    subdir_means = df[['non_zero_pixels', 'mean_weight', 'subdir']].groupby('subdir').mean()

    # Mean statistics for the entire dataset
    total_means = df[['non_zero_pixels', 'mean_weight']].mean()

    print("\nMean statistics for each class:")
    print(subdir_means)

    print("\nMean statistics for the entire dataset:")
    print(total_means)

    return subdir_means, total_means


# Function to merge subdir_means DataFrames
def merge_subdir_means(dataframes):
    merged_df = dataframes[0].copy()
    merged_df.rename(columns={'non_zero_pixels': 'non_zero_pixels_1', 'mean_weight': 'mean_weight_1'}, inplace=True)

    for i, df in enumerate(dataframes[1:], start=2):
        df_subset = df.copy()
        df_subset.rename(columns={'non_zero_pixels': f'non_zero_pixels_{i}', 'mean_weight': f'mean_weight_{i}'},
                         inplace=True)
        merged_df = pd.merge(merged_df, df_subset, left_index=True, right_index=True, how='outer')

    return merged_df


# Function to plot the subdirectory means
def plot_subdir_means(merged_df, value_name, dataset_names):
    subdirs = merged_df.index
    x = range(len(subdirs))
    bar_width = 0.2

    fig, ax = plt.subplots(figsize=(12, 8))

    for i in range(1, len(merged_df.columns) // 2 + 1):
        ax.bar([pos + (i - 1) * bar_width for pos in x],
               merged_df[f'{value_name}_{i}'],
               width=bar_width,
               label=f'{dataset_names[i - 1]} dataset')

    ax.set_xlabel('Class')
    ax.set_ylabel(value_name)
    ax.set_title(f'{value_name} per class')
    ax.set_xticks([pos + (len(merged_df.columns) // 2 - 1) * bar_width / 2 for pos in x])
    ax.set_xticklabels(subdirs)
    ax.legend()

    plt.tight_layout()
    plt.show()


# Function to plot the total means
def plot_total_means(total_means_list, dataset_names):
    metrics = ['non_zero_pixels', 'mean_weight']
    total_means_df = pd.DataFrame(total_means_list)

    fig, ax = plt.subplots(figsize=(12, 8))
    bar_width = 0.35  # Width of the bars
    x = range(len(metrics))  # Positions for the metrics

    for i, dataset_name in enumerate(dataset_names):
        bar_values = total_means_df.loc[i]
        ax.bar([p + i * bar_width for p in x], bar_values, width=bar_width, label=dataset_name)
        for j in range(len(metrics)):
            plt.text(j + i * bar_width, bar_values[j] + 2, str(round(bar_values[j], 5)))

    ax.set_xlabel('Metric')
    ax.set_ylabel('Value')
    ax.set_title('Total Means')
    ax.set_xticks([p + (len(dataset_names) - 1) * bar_width / 2 for p in x])
    ax.set_xticklabels(metrics)
    ax.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Apply the EMNIST image processing steps to each sample.')
    parser.add_argument('--sources', nargs='+', required=True,
                        help='Source directories of datasets.')
    parser.add_argument('--output', type=str, required=True,
                        help='CSV statistics files output directory.')
    parser.add_argument('--verbose', action='store_true',
                        help='Print debug info.')
    args = parser.parse_args()

    sources = args.sources
    output_csv_path = args.output
    verbose = args.verbose

    start = time()

    dataset_names = []
    df = []
    results = []
    i = 0
    for source in sources:
        dataset_names.append(source.split(os.path.sep)[-1])
        df.append(process_directory(source, output_csv_path, verbose))
        results.append(calculate_means(df[i]))
        i = i + 1

    # Extract subdir_means DataFrames and total_means Series
    subdir_means_dataframes = [t[0] for t in results]
    total_means_series = [t[1] for t in results]

    # Merge the subdir_means DataFrames
    merged_subdir_means_df = merge_subdir_means(subdir_means_dataframes)

    # Plot subdirectory means
    plot_subdir_means(merged_subdir_means_df, 'non_zero_pixels', dataset_names)
    plot_subdir_means(merged_subdir_means_df, 'mean_weight', dataset_names)

    # Plot total means
    plot_total_means(total_means_series, dataset_names)

    end = time()
    print(f"Finished in {end - start}")
