#!/usr/bin/env python
import pandas as pd
import numpy as np

obs_file = '/my-projects2/1.project/NO2_ground_complied/global/no2_monthly_observations.csv'
df = pd.read_csv(obs_file)

#(24.45 / 46)
factors = [0.45, 0.35, 0.25]
# factors_name = ['45', '35', '25']
# for i, factor in enumerate(factors):

df_modified = df.copy()
mask = (df_modified['country'] == 'China') | (df_modified['country'] == 'Japan')

# Apply the division to the filtered rows
df_modified.loc[mask, 'no2_ppb'] = df_modified.loc[mask, 'no2_ppb'] / (24.45 / 46)

outfile = f'/my-projects2/1.project/NO2_ground_complied/global/no2_monthly_observations_asia_ugm3.csv'
df_modified.to_csv(outfile, index=False)