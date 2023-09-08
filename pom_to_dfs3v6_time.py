# -*- coding: utf-8 -*-
"""
Created on Fri May 26 22:31:26 2023

@author: liouyufu
"""

import numpy as np
import xarray as xr
import mikeio
import pandas as pd
import os
import datetime

print("開始時間:", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

# 提取變量和維度
#longitude = np.linspace(0, 961, 962)  
#latitude = np.linspace(0, 501, 502)

# pom的x座標
x_coordinates = np.linspace(99.0, 195.1, 962)  #  從99.0到195.1，總共962個數據點
# pom的y座標
y_coordinates = np.linspace(1.5, 51.6, 502)  # 從1.5到51.6，總共502個數據點
longitude = x_coordinates  # 將longitude變量替換為x座標
latitude = y_coordinates  # 將latitude變量替換為y座標
#depth = np.array(range(41))  # 根據你的新數據，深度從0到40
depth = np.linspace(1, 10260, 41)
print(longitude)
print(latitude)
print(depth)

input_folder = "E:/pom-input1"
output_folder = "E:/pom-output1/3D_output"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Loop over all nc files in the directory
for filename in os.listdir(input_folder):
    if filename.endswith(".nc"):  
        print(f"Processing file: {filename}")

        # 讀取 nc 檔案
        pom_ds = xr.open_dataset(os.path.join(input_folder, filename))
        print(pom_ds.keys())

        # 讀取新變量名
        water_u = pom_ds["u"].values
        water_v = pom_ds["v"].values
        water_w = pom_ds["w"].values
        water_temp = pom_ds["t"].values
        salinity = pom_ds["s"].values
        time = pd.to_datetime(pom_ds["time"].values)

        # 使用 POM 模型中的深度層創建 MIKE IO 的 Grid3D 物件
        g = mikeio.Grid3D(x=longitude, y=latitude, z=depth, projection="LONG/LAT")
       
       # 創建 MIKE IO 的 DataArray 對象
        das = [
            mikeio.DataArray(water_u, time=time, geometry=g,
            item=mikeio.ItemInfo("U", mikeio.EUMType.u_velocity_component, mikeio.EUMUnit.meter_per_sec)),
            mikeio.DataArray(water_v, time=time, geometry=g,
            item=mikeio.ItemInfo("V", mikeio.EUMType.v_velocity_component, mikeio.EUMUnit.meter_per_sec)),
            mikeio.DataArray(water_w, time=time, geometry=g,
            item=mikeio.ItemInfo("W", mikeio.EUMType.w_velocity_component, mikeio.EUMUnit.meter_per_sec)),
            mikeio.DataArray(water_temp, time=time, geometry=g,
            item=mikeio.ItemInfo("Temperature", mikeio.EUMType.Temperature, mikeio.EUMUnit.degree_Kelvin)),
            mikeio.DataArray(salinity, time=time, geometry=g,
            item=mikeio.ItemInfo("Salinity", mikeio.EUMType.Salinity, mikeio.EUMUnit.PSU)),
        ]

        # 創建 MIKE IO 的 Dataset 對象
        my_ds = mikeio.Dataset(das)

        # 將資料保存到 DFS3 檔案
        output_filename = f"{output_folder}/{os.path.splitext(filename)[0]}_1pomxy.dfs3"
        my_ds.to_dfs(output_filename)

print("結束時間:", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))