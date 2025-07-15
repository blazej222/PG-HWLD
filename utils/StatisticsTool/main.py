import argparse
import os
import cv2
import pandas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from time import time


def process_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Image at path {image_path} could not be read.")

    non_zero_pixels = np.count_nonzero(image)
    mean_weight = np.mean(image)
    histogram = np.histogram(image, 256, (0, 256))

    return non_zero_pixels, mean_weight, histogram, image_path


def process_directory(root_dir, output_csv_path, verbose):
    results = []
    weight_distributions = []
    weight_results = []
    file_count = 0

    pool = mp.Pool()
    for subdir, _, files in os.walk(root_dir):
        data = pool.map(process_image, [os.path.join(subdir, file) for file in files if
                                        file.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])

        for row in data:
            try:
                results.append((row[3], row[0], row[1]))
                weight_distributions.append(row[2][0])
                weight_results.append(list((row[3],)) + list(row[2][0]))
                file_count += 1
            except Exception as e:
                print(f"Error processing {row[3]}: {e}")

        if verbose:
            print(f"Processed directory: {os.path.split(subdir)[1]}")

    cumulative_histogram = np.divide(np.sum(weight_distributions, axis=0), file_count)

    df = pd.DataFrame(results, columns=['filename', 'non_zero_pixels', 'mean_weight'])
    df2 = pd.DataFrame(weight_results, columns=['filename'] + [x for x in range(0, 256)])

    df.to_csv(os.path.join(str(output_csv_path), root_dir.split(os.sep)[-1]) + ".csv", index=False)
    df2.to_csv(os.path.join(str(output_csv_path), root_dir.split(os.sep)[-1]) + "_weights.csv", index=False)

    return df, df2, cumulative_histogram


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
def plot_subdir_means(merged_df, value_name, dataset_names, output_path):
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
    plt.savefig(f'{output_path}/{value_name}.png')
    # plt.show()


# Function to plot the total means
def plot_total_means(total_means_list, dataset_names, output_path):
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
    plt.savefig(f'{output_path}/total_means.png')
    # plt.show()


def plot_weight_distribution(pixel_weights, dataset_names, output_path):
    for i in range(len(dataset_names)):
        fig, ax = plt.subplots(figsize=(12, 8))

        ax.set_xlabel('Weight value')
        ax.set_ylabel('Average number of pixels')
        ax.set_title(f'Pixel weight distribution - {dataset_names[i]}')
        ax.set_xticks(np.linspace(0, 255, 18))
        ax.set_yscale('log')
        plt.bar(range(256), pixel_weights[i], width=0.8)
        plt.savefig(f'{output_path}/pixel_weights_{dataset_names[i]}.png')

    bar_width = 0.8 / len(dataset_names)
    x = np.arange(256)

    fig, ax = plt.subplots(figsize=(24, 8))

    for i in range(len(dataset_names)):
        ax.bar(x + (i - (len(dataset_names)-1)/2) * bar_width, pixel_weights[i], width=bar_width, label=dataset_names[i])

    ax.set_xlabel('Weight value')
    ax.set_ylabel('Average number of pixels')
    ax.set_title('Comparative pixel weight distribution')
    ax.set_xticks(np.linspace(0, 255, 18))
    ax.set_yscale('log')
    ax.legend()

    plt.tight_layout()
    plt.savefig(f'{output_path}/comparative_pixel_weights.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare given datasets in terms of relevant statistics.')
    parser.add_argument('--sources', nargs='+', required=True,
                        help='Source directories of compared datasets.')
    parser.add_argument('--output', type=str, required=True,
                        help='CSV statistics and image files output directory.')
    parser.add_argument('--verbose', action='store_true',
                        help='Print debug info.')
    args = parser.parse_args()

    sources = args.sources
    output_csv_path = args.output
    verbose = args.verbose

    start = time()

    dataset_names = []
    df = []
    df2 = []
    histograms = []
    results = []
    i = 0
    for source in sources:
        dataset_names.append(source.split(os.path.sep)[-1])
        dataframes = process_directory(source, output_csv_path, verbose)
        df.append(dataframes[0])
        df2.append(dataframes[1])
        histograms.append(dataframes[2])
        results.append(calculate_means(df[i]))
        i = i + 1

    # Extract subdir_means DataFrames and total_means Series
    subdir_means_dataframes = [t[0] for t in results]
    total_means_series = [t[1] for t in results]

    # Merge the subdir_means DataFrames
    merged_subdir_means_df = merge_subdir_means(subdir_means_dataframes)

    # Plot subdirectory means
    plot_subdir_means(merged_subdir_means_df, 'non_zero_pixels', dataset_names, output_csv_path)
    plot_subdir_means(merged_subdir_means_df, 'mean_weight', dataset_names, output_csv_path)

    # Plot total means
    plot_total_means(total_means_series, dataset_names, output_csv_path)

    # Plot cumulative pixel weight distribution
    plot_weight_distribution(histograms, dataset_names, output_csv_path)

    end = time()
    print(f"Finished in {end - start}")
