import numpy as np
import pandas as pd
from src.barycenter_utils import set_seed

def load_csv(path):
    return pd.read_csv(path, header=None).values

def get_reflecto_loaders(source_paths_list, num_samples_src=2000):
    set_seed(2)
    source_datasets = []

    for i, source in enumerate(source_paths_list):
        data = load_csv(source['data'])
        labels = load_csv(source['labels']).ravel()

        indices = np.random.choice(data.shape[0], num_samples_src, replace=False)
        data_sampled = data[indices]
        labels_sampled = labels[indices]

        source_datasets.append((data_sampled, labels_sampled))

    return source_datasets


