# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 16:31:32 2023

@author: cyuan
"""
import numpy as np
import xarray as xr
import mikeio
from mikeio import Dfsu, Dataset
import pandas as pd
import os
import datetime
from scipy.interpolate import griddata
from tqdm import tqdm

print("開始時間:", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

input_folder = "E:/pom-input1"
output_folder = "E:/pom-output1/2D_output"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

template_filename = 'D:/MIKEIO/Ocean.dfsu'
dfs = Dfsu(template_filename)
non_equidistant_grid_coords = dfs.node_coordinates

x_coordinates = np.linspace(99.0, 195.1, 962)
y_coordinates = np.linspace(1.5, 51.6, 502)
equidistant_grid_coords = np.array([[x, y] for y in y_coordinates for x in x_coordinates])

non_equidistant_grid_coords = dfs.node_coordinates[:, :2]

def interpolate_to_dfsu(data, time_index):
    reshaped_data = data[time_index, :, :].flatten()
    assert len(reshaped_data) == len(equidistant_grid_coords), f"Data shape {len(reshaped_data)} does not match grid coordinates shape {len(equidistant_grid_coords)}"
    return griddata(equidistant_grid_coords, reshaped_data, non_equidistant_grid_coords, method='linear')

file_count = len([filename for filename in os.listdir(input_folder) if filename.endswith(".nc")])
processed_files = 0

for filename in tqdm(os.listdir(input_folder), desc="進度"):
    if filename.endswith(".nc"):
        print(f"處理檔案: {filename}")

        pom_ds = xr.open_dataset(os.path.join(input_folder, filename))
        print(pom_ds.keys())

        water_u = pom_ds["u"].values[:, 0, :, :]
        water_v = pom_ds["v"].values[:, 0, :, :]
        water_temp = pom_ds["t"].values[:, 0, :, :]
        salinity = pom_ds["s"].values[:, 0, :, :]
        time = pd.to_datetime(pom_ds["time"].values)

        data = []
        for time_index in range(time.shape[0]):
            water_u_interp = interpolate_to_dfsu(water_u, time_index)
            water_v_interp = interpolate_to_dfsu(water_v, time_index)
            water_temp_interp = interpolate_to_dfsu(water_temp, time_index)
            salinity_interp = interpolate_to_dfsu(salinity, time_index)

            data.append([
                mikeio.DataArray(water_u_interp, time=time[time_index], item=mikeio.ItemInfo("U", mikeio.EUMType.u_velocity_component, mikeio.EUMUnit.meter_per_sec), geometry=dfs),
                mikeio.DataArray(water_v_interp, time=time[time_index], item=mikeio.ItemInfo("V", mikeio.EUMType.v_velocity_component, mikeio.EUMUnit.meter_per_sec), geometry=dfs),
                mikeio.DataArray(water_temp_interp, time=time[time_index], item=mikeio.ItemInfo("Temperature", mikeio.EUMType.Temperature, mikeio.EUMUnit.degree_Kelvin), geometry=dfs),
                mikeio.DataArray(salinity_interp, time=time[time_index], item=mikeio.ItemInfo("Salinity", mikeio.EUMType.Salinity, mikeio.EUMUnit.PSU), geometry=dfs),
            ])

        my_ds = mikeio.Dataset(data=data, time=time)
        output_filename = f"{output_folder}/{os.path.splitext(filename)[0]}_surface1.dfsu"
        dfs.write(output_filename, my_ds)

        processed_files += 1
        print(f"已完成：{processed_files}/{file_count}，完成比例：{100.0 * processed_files / file_count:.2f}%")

print("結束時間:", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))




