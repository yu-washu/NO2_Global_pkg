import numpy as np
import math
import toml
import os
from scipy.spatial.distance import cdist

Obs_version = 'cf_v6_filtered'

# Training data directories
Resampled_Training_BLISCO_data_outdir = '/fsx/yany1/1.project/NO2_DL_global/Training_Evaluation_Estimation/'
Figure_outdir = '/fsx/yany1/1.project/NO2_DL_global/Training_Evaluation_Estimation/'

def neighbors_haversine_indices(train_lat, train_lon, test_lat, test_lon, k):
    try:
        from sklearn.neighbors import NearestNeighbors
        train_rad = np.c_[np.radians(train_lat), np.radians(train_lon)]
        test_rad  = np.c_[np.radians(test_lat),  np.radians(test_lon)]
        k = int(min(k, len(train_rad)))
        if k <= 0:
            return np.empty((len(test_rad), 0), dtype=np.int64)
        nn = NearestNeighbors(n_neighbors=k, algorithm='ball_tree', metric='haversine')
        nn.fit(train_rad)
        _, idx = nn.kneighbors(test_rad, n_neighbors=k, return_distance=True)
        return idx.astype(np.int64, copy=False)
    except Exception:
        # Fallback uses the pure NumPy implementation defined above
        return batch_topk_indices(test_lat, test_lon, train_lat, train_lon, k)

def Get_typeName(bias, normalize_bias, normalize_species, absolute_species, log_species, species):
    if bias == True:
        typeName = '{}-bias'.format(species)
    elif normalize_bias:
        typeName = 'Normalized-{}-bias'.format(species)
    elif normalize_species == True:
        typeName = 'Normaized-{}'.format(species)
    elif absolute_species == True:
        typeName = 'Absolute-{}'.format(species)
    elif log_species == True:
        typeName = 'Log-{}'.format(species)
    return  typeName