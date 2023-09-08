# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 22:05:29 2023

@author: liouyufu
"""

import numpy as np
import xarray as xr
import mikeio
import pandas as pd
import os
import datetime

print("開始時間:", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

x_coordinates = np.linspace(99.0, 195.1, 962)  #  從99.0到195.1，總共963個數據點
y_coordinates = np.linspace(1.5, 51.6, 502)  # 從1.5到51.6，總共502個數據點

longitude = x_coordinates  # 將longitude變量替換為x座標
latitude = y_coordinates  # 將latitude變量替換為y座標

print(longitude)
print(latitude)

input_folder = "E:/pom-input1/"
output_folder = "E:/pom-output1/2D_output"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for filename in os.listdir(input_folder):
    if filename.endswith(".nc"):  
        print(f"Processing file: {filename}")

        pom_ds = xr.open_dataset(os.path.join(input_folder, filename))
        print(pom_ds.keys())

        all_time_values = pd.to_datetime(pom_ds["time"].values)
        
        # 在循環外部讀取所有數據
        u_values = pom_ds["u"].values[:,0,:,:]
        v_values = pom_ds["v"].values[:,0,:,:]
        t_values = pom_ds["t"].values[:,0,:,:]
        s_values = pom_ds["s"].values[:,0,:,:]
        
        # create MIKE IO's Grid2D object
        g = mikeio.Grid2D(x=longitude, y=latitude, projection="LONG/LAT")
        
        data_u = []
        data_v = []
        data_t = []
        data_s = []

        for i, time in enumerate(all_time_values):
            water_u = u_values[i,:,:]
            water_v = v_values[i,:,:]
            water_temp = t_values[i,:,:]
            salinity = s_values[i,:,:]

            # 添加到各自的列表中
            data_u.append(water_u)
            data_v.append(water_v)
            data_t.append(water_temp)
            data_s.append(salinity)

        # 轉換為 numpy array
        data_u = np.array(data_u)
        data_v = np.array(data_v)
        data_t = np.array(data_t)
        data_s = np.array(data_s)

        das = [
            mikeio.DataArray(data_u, time=all_time_values, geometry=g,
            item=mikeio.ItemInfo("U", mikeio.EUMType.u_velocity_component, mikeio.EUMUnit.meter_per_sec)),
            mikeio.DataArray(data_v, time=all_time_values, geometry=g,
            item=mikeio.ItemInfo("V", mikeio.EUMType.v_velocity_component, mikeio.EUMUnit.meter_per_sec)),
            mikeio.DataArray(data_t, time=all_time_values, geometry=g,
            item=mikeio.ItemInfo("Temperature", mikeio.EUMType.Temperature, mikeio.EUMUnit.degree_Kelvin)),
            mikeio.DataArray(data_s, time=all_time_values, geometry=g,
            item=mikeio.ItemInfo("Salinity", mikeio.EUMType.Salinity, mikeio.EUMUnit.PSU)),
        ]

        my_ds = mikeio.Dataset(das)

        # 針對每個輸入文件只輸出一個dfs2文件
        output_filename = f"{output_folder}/{os.path.splitext(filename)[0]}_2pom_alltime.dfs2"
        my_ds.to_dfs(output_filename)

print("結束時間:", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
