# $var is the name of a variable.

Step 0. Collect all training datasets from BLISCO tests using derive_resampled_trainingdata_BLISCO_data.py to derive $resampled_RawObs_testing_site_input_data and $resampled_RawObs_training_site_input_data.

Step 1. Get the Mahalanobis distance to rRMSE relationship.
Code in data_func and calculate_the_mahalanobis_distance_binned_rRMSE is used to get the mahalanobis distance based on the BLISCO CV results.
After run calculate_the_mahalanobis_distance_binned_rRMSE.py to get the local reference (default 20 sites nearby) and the mahalanobis distances from the test stes to training sites nearby, then run mahalabobis_distance_uncertainty_test.ipynb to decide the relationship from the mahalanobis distances and rRMSE.

Step 2. Use derive_map_mahalanobis_uncertainty.py to get the mahalanobis distance of the map, and corresponding uncertainties.