# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 09:34:47 2023

@author: cyuan
"""
import xarray as xr
import mikeio
from mikeio import Dfsu
import pandas as pd
import os
import datetime
import numpy as np
from scipy.interpolate import griddata
from scipy.signal import savgol_filter

# 使用Savitzky-Golay濾波器平滑數據
def smooth_data(data, window_length, polyorder):
    return savgol_filter(data, window_length, polyorder, axis=0)

def griddata_interpolate_data(original_time, data, new_time):
    original_time_timestamp = original_time.astype(np.int64) // 10**9
    new_time_timestamp = new_time.astype(np.int64) // 10**9
    n_time, n_x, n_y = data.shape
    flattened_data = data.reshape(n_time, n_x*n_y)
    interpolated_flattened_data = np.empty((len(new_time_timestamp), n_x*n_y))
    for i in range(n_x*n_y):
        interp_values = griddata(original_time_timestamp, flattened_data[:, i], new_time_timestamp, method='cubic')
        interpolated_flattened_data[:, i] = interp_values
    interpolated_data = interpolated_flattened_data.reshape(-1, n_x, n_y)
    return interpolated_data

def interpolate_zeros(array):
    non_zero_indices = np.where(array != 0)[0]
    zero_indices = np.where(array == 0)[0]
    if len(non_zero_indices) == 0 or len(zero_indices) == 0:
        return array
    interp_values = np.interp(zero_indices, non_zero_indices, array[non_zero_indices])
    array[zero_indices] = interp_values
    return array

def interpolate_3d_zeros(matrix):
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            matrix[i, j, :] = interpolate_zeros(matrix[i, j, :])
    return matrix

def grid_to_mesh_interpolation(original_lon, original_lat, original_data, target_lon, target_lat):
    points = np.array([original_lon.flatten(), original_lat.flatten()]).T
    target_points = np.array([target_lon, target_lat]).T
    interpolated_data = griddata(points, original_data.flatten(), target_points, method='cubic')
    return interpolated_data

def post_process_zeros(data):
    """後處理數據，尋找和替換0值"""
    zero_positions = np.where(data == 0)
    for pos in zip(*zero_positions):
        # 假設你的數據是3D的，我們只處理空間的部分
        surrounding = [
            data[pos[0], max(pos[1]-1, 0), pos[2]],
            data[pos[0], min(pos[1]+1, data.shape[1]-1), pos[2]],
            data[pos[0], pos[1], max(pos[2]-1, 0)],
            data[pos[0], pos[1], min(pos[2]+1, data.shape[2]-1)]
        ]
        average_value = np.mean([val for val in surrounding if val != 0])  # 計算非0的周圍數值的平均
        data[pos] = average_value
    return data

print("開始時間:", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

input_folder = "E:/pom-input1"
output_folder = "E:/pom-output1/2D_dfsu"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

template_filename = 'D:/MIKEIO/Ocean.dfsu'
dfs = Dfsu(template_filename)
target_lon, target_lat = dfs.node_coordinates[:,0], dfs.node_coordinates[:,1]

x_coordinates = np.linspace(99.0, 195.1, 962)
y_coordinates = np.linspace(1.5, 51.6, 502)
original_lon, original_lat = np.meshgrid(x_coordinates, y_coordinates)

window_length = 3  # 平滑窗口大小
polyorder = 1      # 多項式階數


for filename in os.listdir(input_folder):
    if filename.endswith(".nc"):
        print(f"Processing file: {filename}")

        pom_ds = xr.open_dataset(os.path.join(input_folder, filename))
        print(pom_ds.keys())

        water_u = pom_ds["u"].values[:, 0, :, :]
        water_v = pom_ds["v"].values[:, 0, :, :]
        water_temp = pom_ds["t"].values[:, 0, :, :]
        salinity = pom_ds["s"].values[:, 0, :, :]
        elb = pom_ds["elb"].values[:, :, :]
        time = pd.to_datetime(pom_ds["time"])
        
      # 使用 atol=1e-08 來將 0 和 -0 的值設為 NaN
      #  zero_mask = np.isclose(water_u, 0, atol=1e-07) | (water_u == 0)
      #  water_u[zero_mask] = np.nan
        
      #  zero_mask = np.isclose(water_v, 0, atol=1e-07) | (water_v == 0)
      #  water_v[zero_mask] = np.nan
        
     
        new_time = pd.date_range(start=time[0], end=time[-1], periods=31)
        new_time = pd.to_datetime(new_time)

def process_data(original_lon, original_lat, data, target_lon, target_lat, time, new_time):
          
    interpolated_data = griddata_interpolate_data(time, data, new_time)
    interpolated_data = interpolate_3d_zeros(interpolated_data)
    
    return interpolated_data

def save_to_dfsu(data, new_time, item_info, dfs):
    return mikeio.DataArray(data, time=new_time, item=item_info, geometry=dfs)

interpolated_u = process_data(original_lon, original_lat, water_u, target_lon, target_lat, time, new_time)
interpolated_v = process_data(original_lon, original_lat, water_v, target_lon, target_lat, time, new_time)
interpolated_temp = process_data(original_lon, original_lat, water_temp, target_lon, target_lat, time, new_time)
interpolated_salinity = process_data(original_lon, original_lat, salinity, target_lon, target_lat, time, new_time)
interpolated_elb = process_data(original_lon, original_lat, elb, target_lon, target_lat, time, new_time)

# 在這裡加入後處理步驟
interpolated_u = post_process_zeros(interpolated_u)
interpolated_v = post_process_zeros(interpolated_v)
interpolated_temp = post_process_zeros(interpolated_temp)
interpolated_salinity = post_process_zeros(interpolated_salinity)
interpolated_elb = post_process_zeros(interpolated_elb)

data = [
    save_to_dfsu(interpolated_u, new_time, mikeio.ItemInfo("U", mikeio.EUMType.u_velocity_component, mikeio.EUMUnit.meter_per_sec), dfs),
    save_to_dfsu(interpolated_v, new_time, mikeio.ItemInfo("V", mikeio.EUMType.v_velocity_component, mikeio.EUMUnit.meter_per_sec), dfs),
    save_to_dfsu(interpolated_temp, new_time, mikeio.ItemInfo("Temperature", mikeio.EUMType.Temperature, mikeio.EUMUnit.degree_Kelvin), dfs),
    save_to_dfsu(interpolated_salinity, new_time, mikeio.ItemInfo("Salinity", mikeio.EUMType.Salinity, mikeio.EUMUnit.PSU), dfs),
    save_to_dfsu(interpolated_elb, new_time, mikeio.ItemInfo("Surface Elevation", mikeio.EUMType.Water_Level, mikeio.EUMUnit.meter), dfs),
]

my_ds = mikeio.Dataset(data=data, time=new_time)
output_filename = f"{output_folder}/{os.path.splitext(filename)[0]}_surface_tt_griddata_2_1.dfsu"
dfs.write(output_filename, my_ds)
print(f"Output: {output_filename}")
print("結束時間:", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
