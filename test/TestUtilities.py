import os
import sys

import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(__file__))
from src.ForwardProb import add_noise_to_cp_values, load_and_show_plots


def property_counts(input_file):
    with open(input_file, 'r') as file:
        data = file.readlines()
    properties_count = {}
    current_experiment = None
    for line in data:
        line = line.strip()
        if line.startswith('[Experiment'):
            current_experiment = line
        elif line.startswith('model:') or line.startswith('optimizer:') \
                or line.startswith('initializer:') or line.startswith('activation:') \
                or line.startswith('batch_size:') or line.startswith('learning_rate:') \
                or line.startswith('loss:'):
            if current_experiment:
                property_name, property_value = line.split(':')
                property_name = property_name.strip()
                property_value = property_value.strip()
                key = f'{property_name}: {property_value}'
                properties_count[key] = properties_count.get(key, 0) + 1
    sorted_properties_count = dict(sorted(properties_count.items(), key=lambda x: x[0]))
    plt.figure(figsize=(10, 6))
    plt.barh(list(sorted_properties_count.keys()), list(sorted_properties_count.values()), color='skyblue')
    plt.xlabel('Count')
    plt.ylabel('Properties')
    plt.title('Property Counts (Sorted by Property Name)')
    plt.tight_layout()
    # plt.savefig('property_counts_sorted.png')  # Save the plot as a PNG file
    plt.show()


def test_combine_images(directory):
    for subdir in os.listdir(directory):
        subdir_path = os.path.join(directory, subdir)
        if os.path.isdir(subdir_path):
            images = []
            for file in os.listdir(subdir_path):
                file_path = os.path.join(subdir_path, file)
                if (
                        # file.startswith('prediction_cp_') or
                        # file.startswith('prediction_profile_') or

                        file.startswith('optim_cp_') or
                        file.startswith('optim_profile_') or

                        file == 'training_history.png'
                ) and file.endswith('.png'):
                    images.append(file_path)
            if images:
                combined_image = Image.new('RGB', (1920, 1440))  # Larger size: 4 times bigger
                for i, img_path in enumerate(images):
                    img = Image.open(img_path)
                    img = img.resize((640, 480))  # Resize to original dimensions
                    combined_image.paste(img, (i % 3 * 640, i // 3 * 480))  # Adjust position based on grid
                output_dir = os.path.dirname(images[0])
                combined_image.save(os.path.join(output_dir, 'combined_image.png'))


def test_extract_experiment_info(input_file, experiment_numbers, output_file):
    with open(input_file, 'r') as file:
        data = file.readlines()

    output_data = []
    current_experiment = None
    for line in data:
        line = line.strip()
        if line.startswith('[Experiment') and any(f'{exp_num}]' in line for exp_num in experiment_numbers):
            current_experiment = line
        elif line == '' and current_experiment:
            output_data.append(current_experiment)
            current_experiment = None
        elif current_experiment is not None:
            output_data.append(line)

    with open(output_file, 'w') as file:
        file.write('\n'.join(output_data))

    property_counts(output_file)


def test_load_and_show_plots():
    load_and_show_plots(directory='../out/out_test/summary')


def test_add_noise_to_cp_values():
    add_noise_to_cp_values(['../data/data_test_0', '../data/data_test_1'], noise_percentage=0.5)


if __name__ == "__main__":
    # test_combine_images('../out/out_test')
    # test_extract_experiment_info('../out/out_test/summary/param_list.txt', [1, 2, 3],
    #                              '../out/out_test/summary/param_list_ext.txt')
    # test_load_and_show_plots()
    test_add_noise_to_cp_values()
