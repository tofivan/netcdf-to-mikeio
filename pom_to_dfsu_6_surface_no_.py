# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 14:47:05 2023

@author: cyuan
"""

import xarray as xr
import mikeio
from mikeio import Dfsu
import pandas as pd
import os
import datetime
import numpy as np

def post_process_zeros(data):
    """後處理數據，尋找和替換0值"""
    zero_positions = np.where(data == 0)
    for pos in zip(*zero_positions):
        surrounding = [
            data[pos[0], max(pos[1]-1, 0), pos[2]],
            data[pos[0], min(pos[1]+1, data.shape[1]-1), pos[2]],
            data[pos[0], pos[1], max(pos[2]-1, 0)],
            data[pos[0], pos[1], min(pos[2]+1, data.shape[2]-1)]
        ]
        average_value = np.mean([val for val in surrounding if val != 0])
        data[pos] = average_value
    return data

def post_process_below_20(data):
    """後處理數據，尋找和替換20以下的值"""
    positions_below_20 = np.where(data < 20)
    for pos in zip(*positions_below_20):
        surrounding = [
            data[pos[0], max(pos[1]-1, 0), pos[2]],
            data[pos[0], min(pos[1]+1, data.shape[1]-1), pos[2]],
            data[pos[0], pos[1], max(pos[2]-1, 0)],
            data[pos[0], pos[1], min(pos[2]+1, data.shape[2]-1)]
        ]
        average_value = np.mean([val for val in surrounding if val >= 20])
        data[pos] = average_value
    return data

def save_to_dfsu(data, time, item_info, dfs):
    return mikeio.DataArray(data, time=time, item=item_info, geometry=dfs)

print("開始時間:", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

input_folder = "E:/pom-input1"
output_folder = "E:/pom-output1/2D_dfsu"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

template_filename = 'D:/MIKEIO/Ocean.dfsu'
dfs = Dfsu(template_filename)

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

        # Post-processing
        water_u = post_process_zeros(water_u)
        water_v = post_process_zeros(water_v)
        water_temp = post_process_below_20(water_temp)
        salinity = post_process_zeros(salinity)
        elb = post_process_zeros(elb)

        data = [
            save_to_dfsu(water_u, time, mikeio.ItemInfo("U", mikeio.EUMType.u_velocity_component, mikeio.EUMUnit.meter_per_sec), dfs),
            save_to_dfsu(water_v, time, mikeio.ItemInfo("V", mikeio.EUMType.v_velocity_component, mikeio.EUMUnit.meter_per_sec), dfs),
            save_to_dfsu(water_temp, time, mikeio.ItemInfo("Temperature", mikeio.EUMType.Temperature, mikeio.EUMUnit.degree_Kelvin), dfs),
            save_to_dfsu(salinity, time, mikeio.ItemInfo("Salinity", mikeio.EUMType.Salinity, mikeio.EUMUnit.PSU), dfs),
            save_to_dfsu(elb, time, mikeio.ItemInfo("Surface Elevation", mikeio.EUMType.Water_Level, mikeio.EUMUnit.meter), dfs),
        ]

        my_ds = mikeio.Dataset(data=data, time=time)
        output_filename = f"{output_folder}/{os.path.splitext(filename)[0]}_surface_tt_griddata_2_1_1.dfsu"
        dfs.write(output_filename, my_ds)
        print(f"Output: {output_filename}")

print("結束時間:", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
