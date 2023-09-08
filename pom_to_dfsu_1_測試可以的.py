# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 22:10:10 2023

@author: liouyufu
"""

import xarray as xr
import mikeio
from mikeio import Dfsu
import pandas as pd
import os
import datetime


print("開始時間:", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

input_folder = "E:/pom-input1"
output_folder = "E:/pom-output1/2D_dfsu"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

template_filename = 'D:/MIKEIO/Ocean.dfsu'
dfs = Dfsu(template_filename)
print(dfs)

for filename in os.listdir(input_folder):
    if filename.endswith(".nc"):
        print(f"Processing file: {filename}")

        pom_ds = xr.open_dataset(os.path.join(input_folder, filename))
        print(pom_ds.keys())

        water_u = pom_ds["u"].values[:,0,:,:]
        water_v = pom_ds["v"].values[:,0,:,:]
        water_temp = pom_ds["t"].values[:,0,:,:]
        salinity = pom_ds["s"].values[:,0,:,:]
        time = pd.to_datetime(pom_ds["time"])

       

        # Removed the interpolation code

        data = [
            mikeio.DataArray(water_u, time=time, item=mikeio.ItemInfo("U", mikeio.EUMType.u_velocity_component, mikeio.EUMUnit.meter_per_sec), geometry=dfs),
            mikeio.DataArray(water_v, time=time, item=mikeio.ItemInfo("V", mikeio.EUMType.v_velocity_component, mikeio.EUMUnit.meter_per_sec), geometry=dfs),
            mikeio.DataArray(water_temp, time=time, item=mikeio.ItemInfo("Temperature", mikeio.EUMType.Temperature, mikeio.EUMUnit.degree_Kelvin), geometry=dfs),
            mikeio.DataArray(salinity, time=time, item=mikeio.ItemInfo("Salinity", mikeio.EUMType.Salinity, mikeio.EUMUnit.PSU), geometry=dfs),
        ]

        my_ds = mikeio.Dataset(data=data, time=time)

        output_filename = f"{output_folder}/{os.path.splitext(filename)[0]}_surface_tt.dfsu"
        dfs.write(output_filename, my_ds)

print("結束時間:", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))





