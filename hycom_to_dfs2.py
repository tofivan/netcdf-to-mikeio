# -*- coding: utf-8 -*-
"""
Created on Wed May  3 16:24:00 2023

@author: liouyufu
"""

import numpy as np
import xarray as xr
from scipy.interpolate import griddata
import mikeio
import pandas as pd
import os

hycom_ds = xr.open_dataset("D:/MIKEIO/pom-input/20211231_21.nc")
print(hycom_ds.keys())

# Extract variables and dimensions
longitude = hycom_ds["lon"].values
latitude = hycom_ds["lat"].values
time = pd.to_datetime(hycom_ds["time"].values)

water_u = hycom_ds["water_u"].values
water_v = hycom_ds["water_v"].values
surf_el = hycom_ds["surf_el"].values
water_temp = hycom_ds["water_temp"].values
salinity = hycom_ds["salinity"].values

# Create target equidistant grid
target_lon = np.linspace(longitude.min(), longitude.max(), len(longitude))
target_lat = np.linspace(latitude.min(), latitude.max(), len(latitude))

# Function to interpolate data
def interpolate_data(data, source_lon, source_lat, target_lon, target_lat):
    data_flattened = data.flatten()
    source_grid = np.meshgrid(source_lon, source_lat)
    target_grid = np.meshgrid(target_lon, target_lat)
    source_points = np.array([source_grid[0].flatten(), source_grid[1].flatten()]).T
    return griddata(source_points, data_flattened, tuple(target_grid), method="linear")

# Initialize arrays for interpolated data
u_interp = np.zeros((len(time), len(target_lat), len(target_lon)))
v_interp = np.zeros((len(time), len(target_lat), len(target_lon)))
el_interp = np.zeros((len(time), len(target_lat), len(target_lon)))
temp_interp = np.zeros((len(time), len(target_lat), len(target_lon)))
sal_interp = np.zeros((len(time), len(target_lat), len(target_lon)))

# Interpolate each variable to the target grid
for t_index in range(len(time)):
    u_interp[t_index] = interpolate_data(water_u[t_index, 0], longitude, latitude, target_lon, target_lat)
    v_interp[t_index] = interpolate_data(water_v[t_index, 0], longitude, latitude, target_lon, target_lat)
    temp_interp[t_index] = interpolate_data(water_temp[t_index, 0], longitude, latitude, target_lon, target_lat)
    sal_interp[t_index] = interpolate_data(salinity[t_index, 0], longitude, latitude, target_lon, target_lat)
    el_interp[t_index] = interpolate_data(surf_el[t_index], longitude, latitude, target_lon, target_lat)

output_folder = "D:/MIKEIO/pom-input/2D_output"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

g = mikeio.Grid2D(x=target_lon, y=target_lat, projection="LONG/LAT")

das = [
    mikeio.DataArray(u_interp, time=time, geometry=g,
                     item=mikeio.ItemInfo("U", mikeio.EUMType.u_velocity_component, mikeio.EUMUnit.meter_per_sec)),
    mikeio.DataArray(v_interp, time=time, geometry=g,
                     item=mikeio.ItemInfo("V", mikeio.EUMType.v_velocity_component, mikeio.EUMUnit.meter_per_sec)),
    mikeio.DataArray(el_interp, time=time, geometry=g,
                     item=mikeio.ItemInfo("Surface Elevation", mikeio.EUMType.Water_Level, mikeio.EUMUnit.meter)),
    mikeio.DataArray(temp_interp, time=time, geometry=g,
                     item=mikeio.ItemInfo("Temperature", mikeio.EUMType.Temperature, mikeio.EUMUnit.degree_Celsius)),
    mikeio.DataArray(sal_interp, time=time, geometry=g,
                     item=mikeio.ItemInfo("Salinity", mikeio.EUMType.Salinity, mikeio.EUMUnit.PSU)),
]

# Create MIKE Dataset
my_ds = mikeio.Dataset(das)

# Save to DFS file
output_filename = f"{output_folder}/20211231_21_2D.dfs2"
my_ds.to_dfs(output_filename)








