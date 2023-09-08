# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 20:52:44 2023

@author: liouyufu
"""

import numpy as np
import xarray as xr
import mikeio
from mikeio import Dfsu, Dataset
from scipy.interpolate import griddata
import pandas as pd
import os
import datetime

print("開始時間:", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

input_folder = "E:/pom-input1"
output_folder = "E:/pom-output1/2D_dfsu"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Read the template file
template_filename = 'D:/MIKEIO/Ocean.dfsu'
dfs = Dfsu(template_filename)

# Define a function to interpolate data to the dfsu mesh
def interpolate_to_mesh(data, lon, lat, dfsu_mesh):
    mesh_coords = np.array(dfsu_mesh.node_coordinates)
    mesh_x = mesh_coords[:,0]
    mesh_y = mesh_coords[:,1]
    grid_x, grid_y = np.meshgrid(lon, lat)
    interpolated_data = np.zeros((data.shape[0], len(dfsu_mesh.node_coordinates)))
    for t in range(data.shape[0]):
        interpolated_data[t, :] = griddata((grid_x.ravel(), grid_y.ravel()), data[t, :, :].ravel(), (mesh_x, mesh_y), method='linear')
    return interpolated_data

# pom的x座標
x_coordinates = np.linspace(99.0, 195.1, 962)
# pom的y座標
y_coordinates = np.linspace(1.5, 51.6, 502)

# Loop over all nc files in the directory
for filename in os.listdir(input_folder):
    if filename.endswith(".nc"):  
        print(f"正在處理文件: {filename}")

        # Read the nc file
        pom_ds = xr.open_dataset(os.path.join(input_folder, filename))
        print(pom_ds.keys())

        # Read new variables and only select the top layer
        water_u = pom_ds["u"].values[:,0,:,:]  # Only select the top layer
        water_v = pom_ds["v"].values[:,0,:,:]  # Only select the top layer
        water_temp = pom_ds["t"].values[:,0,:,:]  # Only select the top layer
        salinity = pom_ds["s"].values[:,0,:,:]  # Only select the top layer
        time = pd.to_datetime(pom_ds["time"].values)

        # Interpolate data to the dfsu mesh
        water_u = interpolate_to_mesh(water_u, x_coordinates, y_coordinates, dfs)
        water_v = interpolate_to_mesh(water_v, x_coordinates, y_coordinates, dfs)
        water_temp = interpolate_to_mesh(water_temp, x_coordinates, y_coordinates, dfs)
        salinity = interpolate_to_mesh(salinity, x_coordinates, y_coordinates, dfs)

        # Debugging code
        print("Debugging information:")
        print(f"water_u shape: {water_u.shape}, type: {type(water_u)}")
        print(f"water_v shape: {water_v.shape}, type: {type(water_v)}")
        print(f"water_temp shape: {water_temp.shape}, type: {type(water_temp)}")
        print(f"salinity shape: {salinity.shape}, type: {type(salinity)}")
        print(f"time shape: {time.shape}, type: {type(time)}")

       # Create DataArray objects
        data = [
            mikeio.DataArray(water_u, time=time, item=mikeio.ItemInfo("U", mikeio.EUMType.u_velocity_component, mikeio.EUMUnit.meter_per_sec), geometry=dfs),
            mikeio.DataArray(water_v, time=time, item=mikeio.ItemInfo("V", mikeio.EUMType.v_velocity_component, mikeio.EUMUnit.meter_per_sec), geometry=dfs),
            mikeio.DataArray(water_temp, time=time, item=mikeio.ItemInfo("Temperature", mikeio.EUMType.Temperature, mikeio.EUMUnit.degree_Kelvin), geometry=dfs),
            mikeio.DataArray(salinity, time=time, item=mikeio.ItemInfo("Salinity", mikeio.EUMType.Salinity, mikeio.EUMUnit.PSU), geometry=dfs),
        ]

        # Create the Dataset object and Save the data to a new Dfsu file
        try:
            my_ds = mikeio.Dataset(data=data, time=time)
            output_filename = f"{output_folder}/{os.path.splitext(filename)[0]}_linear_surface.dfsu"
            dfs.write(output_filename, my_ds)
        except AttributeError as e:
            if 'Length' in str(e):
                with open("error_log.txt", "a") as log_file:
                    log_file.write(f"Error in file {filename}: {str(e)}\n")
                pass
            else:
                raise

print("結束時間:", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))




