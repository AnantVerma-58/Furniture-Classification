import os

output_folder = os.environ['CONDA_PREFIX'] + "/project/updated_furnture_data"
input_folder = os.environ['CONDA_PREFIX'] + "/project/furniture_data"
augment_folder = os.environ['CONDA_PREFIX'] + "/project/augmented"
final_data_folder = os.environ['CONDA_PREFIX']+"/project/final"

all_model_names = ['resnet', 'vgg', 'densenet']
# model_names = ['densenet']
batch_size = 32
lr = 0.01
num_classes = 5
num_epochs = 30
save_path = os.environ['CONDA_PREFIX'] + "/project/streamlit"

model_name_for_prediction = 'densenet'

image_paths = {
    1: os.environ['CONDA_PREFIX'] + "/project/samples/1_aug1.jpeg",
    2: os.environ['CONDA_PREFIX'] + "/project/samples/1_aug2.jpeg",
    3: os.environ['CONDA_PREFIX'] + "/project/samples/4_aug2.jpeg",
    4: os.environ['CONDA_PREFIX'] + "/project/samples/6_aug2.jpeg",
    5: os.environ['CONDA_PREFIX'] + "/project/samples/9_aug2.jpeg",
}

class_names = ['bed','chair','sofa','storage','table']