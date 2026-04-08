import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import toml
import os

from Training_pkg.iostream import load_TrainingVariables, load_geophysical_biases_data, load_geophysical_species_data, load_monthly_obs_data, Learning_Object_Datasets
from Training_pkg.utils import *
from Training_pkg.data_func import normalize_Func, get_trainingdata_within_start_end_YEAR
from Training_pkg.Net_Construction import *
from Training_pkg.Statistic_Func import linear_regression

from Evaluation_pkg.utils import *
from Evaluation_pkg.data_func import Get_valid_index_for_temporal_periods,Get_month_based_Index,Get_month_based_XY_indices,GetXIndex,GetYIndex,Get_XY_indices, Get_XY_arraies, Get_final_output, ForcedSlopeUnity_Func, CalculateAnnualR2, CalculateMonthR2, calculate_Statistics_results
from Evaluation_pkg.iostream import *


from lassonet import LassoNetRegressor, LassoNetRegressorCV
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from functools import partial
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(current_dir, 'config.toml')
cfg = toml.load(config_path)
cfg = toml.load('/my-projects2/1.project/NO2_DL_global_2019/NO2_global_pkg/config.toml')
# *------------------------------------------------------------------------------*#
##   Initialize the array, variables and constants.
# *------------------------------------------------------------------------------*#
### Get training data, label data, initial observation data and geophysical species
width, height, sitesnumber,start_YYYY, TrainingDatasets = load_TrainingVariables(nametags=channel_names)
SPECIES_OBS, lat, lon = load_monthly_obs_data(species=species)
geophysical_species, geolat, geolon = load_geophysical_species_data(species=species)
true_input, mean, std = Learning_Object_Datasets(bias=bias,Normalized_bias=normalize_bias,Normlized_Speices=normalize_species,Absolute_Species=absolute_species,Log_PM25=log_species,species=species)
Initial_Normalized_TrainingData, input_mean, input_std = normalize_Func(inputarray=TrainingDatasets,observation_data=SPECIES_OBS)
population_data = load_coMonitor_Population()
MONTH = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
total_channel_names, main_stream_channel_names, side_channel_names = Get_channel_names(channels_to_exclude=[])
nchannel   = len(total_channel_names)
seed       = 20190130
typeName   = Get_typeName(bias=bias, normalize_bias=normalize_bias,normalize_species=normalize_species, absolute_species=absolute_species, log_species=log_species, species=species)
site_index = np.array(range(sitesnumber))
lassonet_plot_dir = '/my-projects2/1.project/NO2_DL_global_2019/Training_Evaluation_Estimation/lassonet_plot/'
os.makedirs(lassonet_plot_dir, exist_ok=True)

imodel_year = 0    
    
Normalized_TrainingData = get_trainingdata_within_start_end_YEAR(initial_array=Initial_Normalized_TrainingData, training_start_YYYY=beginyears[imodel_year],training_end_YYYY=endyears[imodel_year],start_YYYY=start_YYYY,sitesnumber=sitesnumber)

print('shape of Normalized_TrainingData: ', Normalized_TrainingData.shape)
valid_sites_index, temp_index_of_initial_array = Get_valid_index_for_temporal_periods(SPECIES_OBS=SPECIES_OBS,beginyear=beginyears[imodel_year],endyear=endyears[imodel_year],month_range=list(range(0, 12)),sitesnumber=sitesnumber)
imodel_siteindex = site_index[valid_sites_index]

train_index = imodel_siteindex
test_index  = imodel_siteindex
X_train, X_test, y_train, y_test = train_test_split(Normalized_TrainingData, true_input)

train_mask = np.where(~np.isnan(y_train))[0]
test_mask  = np.where(~np.isnan(y_test))[0]

X_train_masked = X_train[train_mask,:,3,3]
y_train_masked = y_train[train_mask]
X_test_masked = X_test[test_mask,:,3,3]
y_test_masked = y_test[test_mask]

CV_mode_default = LassoNetRegressorCV()
CV_model_customized = LassoNetRegressorCV(hidden_dims=(64, 128, 256, 512), batch_size=batchsize,lambda_start=1e-3, optim=partial(torch.optim.Adam, lr=lr0, betas=(Adam_beta0, Adam_beta1), eps=Adam_eps))
model_default = LassoNetRegressor()
model_customized = LassoNetRegressor(hidden_dims=(64, 128, 256, 512), batch_size=batchsize,lambda_start=1e-3, optim=partial(torch.optim.Adam, lr=lr0, betas=(Adam_beta0, Adam_beta1), eps=Adam_eps))

model_name_list = ['LassoNet_customized_layers2', 'LassoNet_default', 'LassoNetCV_customized_layers2', 'LassoNetCV_default']
for model, model_name in zip([model_customized, model_default, CV_model_customized, CV_mode_default], model_name_list): 
    fig = plt.figure(figsize=(12, 12))

    n_selected = []
    mse = []
    r2 = []
    lambda_ = []

    path = model.path(X_train_masked, y_train_masked, return_state_dicts=True)

    for i, save in enumerate(path):
        if i == 0:  # Skip the first path
            continue
        model.load(save.state_dict)
        y_pred = model.predict(X_test_masked)
        n_selected.append(save.selected.sum().cpu().numpy().item())
        mse.append(mean_squared_error(y_test_masked, y_pred))
        r2.append(np.round(linear_regression(y_test_masked, y_pred.flatten()), 4))
        lambda_.append(save.lambda_)
        
    print('processing model:', model_name, '\nnumber of selected features: ', n_selected, '\nmse: ', mse, '\nr2: ', r2, '\nlambda: ', lambda_)
    

    # Find the best lambda (minimum MSE)
    best_mse_idx = np.argmin(mse)
    best_mse_lambda = lambda_[best_mse_idx]
    best_mse_mse = mse[best_mse_idx]
    best_mse_r2 = r2[best_mse_idx]
    best_mse_n_selected = n_selected[best_mse_idx]
    # Find the best lambda (maximum R2)
    best_r2_idx = np.argmax(r2)
    best_r2_lambda = lambda_[best_r2_idx]
    best_r2_mse = mse[best_r2_idx]
    best_r2_r2 = r2[best_r2_idx]
    best_r2_n_selected = n_selected[best_r2_idx]

    # Main axis for MSE
    ax1 = plt.subplot(312)
    color_lambda = 'tab:red'
    ax1.set_xlabel("log(λ)")
    ax1.set_ylabel("MSE", color=color_lambda)
    ax1.plot(lambda_, mse, ".-", color=color_lambda, label=f"MSE (best: {best_mse_mse:.4f})")
    ax1.scatter([best_mse_lambda], [best_mse_mse], color='red', s=100, marker='*', 
                zorder=10, label=f"Best λ {best_mse_lambda:.6f} for minimum MSE {best_mse_mse:.4f}, {best_mse_n_selected} features")

    ax1.tick_params(axis='y', labelcolor=color_lambda)
    ax1.set_xscale("log")
    ax1.grid(True)
    
    # Get unique n_selected values and their corresponding lambda values
    unique_n_selected = []
    unique_lambda = []
    for i, n in enumerate(n_selected):
        if n not in unique_n_selected:
            unique_n_selected.append(n)
            unique_lambda.append(lambda_[i])

    # Top x-axis for number of selected features 
    ax2 = ax1.twiny()
    color_n_selected = 'tab:blue'
    ax2.xaxis.set_ticks_position('top')
    ax2.xaxis.set_label_position('top')
    ax2.set_xlabel("number of selected features")
    
    # Make the top axis share the same positions as the bottom axis
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xscale("log")
    ax2.set_xticks(unique_lambda)
    ax2.set_xticklabels(unique_n_selected)

    # Right y-axis for R²
    ax3 = ax1.twinx()
    color_r2 = 'tab:green'
    ax3.set_ylabel("R²", color=color_r2)
    ax3.plot(lambda_, r2, ".-", color=color_r2, label=f"R² (best: {best_r2_r2:.4f})")
    ax3.scatter([best_r2_lambda], [best_r2_r2], color='green', s=100, marker='*', 
                zorder=10, label=f"Best λ {best_r2_lambda:.6f} for maximum R² {best_r2_r2:.4f}, {best_r2_n_selected} features")
    ax3.tick_params(axis='y', labelcolor=color_r2)
    # Set specific y-axis ticks for R²
    ax3.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    
    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines3, labels3 = ax3.get_legend_handles_labels()
    ax1.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc='upper center', 
            bbox_to_anchor=(0.5, -0.15), ncol=3, fancybox=True, shadow=True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(lassonet_plot_dir, f'model_selection_{model_name}.png'))
    plt.clf()

##############
# Find the best model (maximum R2) and plot the feature importance
###############   
    best_path = path[np.argmax(r2)]
    model.load(best_path)
    importances_values = model.feature_importances_.numpy()
    best_order = np.argsort(importances_values)[::-1]

    ordered_importances = importances_values[best_order] 
    ordered_feature_names_plot = [total_channel_names[i] for i in best_order]

    ordered_colors = np.array(["g"] * len(importances_values))[best_order] 

    fig = plt.figure(figsize=(len(importances_values)*0.5, 8))  

    plt.bar(
        np.arange(len(importances_values)),
        ordered_importances, 
        color=ordered_colors,
    )
    plt.xticks(np.arange(len(importances_values)), ordered_feature_names_plot, rotation=90)
    plt.xlabel('Features name')
    plt.ylabel("Feature importance")
    plt.tight_layout()
    plt.savefig(os.path.join(lassonet_plot_dir, f'model_selection_{model_name}_feature_importance.png'))
    plt.clf()