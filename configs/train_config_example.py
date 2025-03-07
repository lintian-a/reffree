from reffree.datasets import MRIHemiDataset, synth_config


in_shape = (128, 256, 128)
slice_shape = (96, 96)
split_folder = "" # The folder contains the file split_train.pkl, split_debug.pkl, split_test.pkl
data_folder = "" # The folder contains the MRI data
center_of_mass_file = "" # The file contains the center of mass of the brain

# Use the default config for synthetic data.
# You can also adjust the config here.
synth_config = synth_config

def get_train_dataset():
    return MRIHemiDataset(
        data_folder,
        split_folder=split_folder,
        cm_file=center_of_mass_file,
        use_split=True,
        phase='train',
        shape=in_shape)

def get_debug_dataset():
    return MRIHemiDataset(
        data_folder,
        split_folder=split_folder,
        cm_file=center_of_mass_file,
        phase='debug',
        shape=in_shape)