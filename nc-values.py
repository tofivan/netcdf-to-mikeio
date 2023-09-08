# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 16:30:38 2023

@author: cyuan
"""

import netCDF4 as nc

input_file = "D:/MIKEIO/pom-input/SRF.wpac10exp010.2023-02-28_to_2023-03-14.nc"  # 更改為您的檔案路徑
nc_dataset = nc.Dataset(input_file)

for var_name, var in nc_dataset.variables.items():
    if len(var.dimensions) == 3:
        print(f"{var_name} has dimensions {var.dimensions} and shape {var.shape}")

nc_dataset.close()
