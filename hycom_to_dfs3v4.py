# -*- coding: utf-8 -*-
"""
Created on Tue May  9 14:41:34 2023

@author: liouyufu
"""
"""
這個程式碼基本上是從一個.NETCDF檔案讀取海洋數據，然後將這些數據插值到一個新的網格中，最後將處理後的數據存儲為MIKE IO格式的.DFS3檔案。以下是每個步驟的詳細解釋：

匯入所需的模組：包括處理數據和時間的常用模組，如numpy、pandas、datetime，以及處理特定數據格式的模組，如xarray、mikeio和scipy的griddata。

讀取.NETCDF檔案，並獲取數據的鍵值。

從.NETCDF檔案中提取變數和維度，這些包括經度、緯度、時間、深度、水流u分量、水流v分量、表面高程、水溫和鹽度。

創建目標網格，包括目標經度、緯度和深度。

定義內插函數。這個函數的目的是將原始數據插值到新的目標網格中。

創建初始的3D插值數據，並為每個時間和深度的變數內插數據到目標網格。

如果目標資料夾不存在，則創建它。

創建mikeio的Grid3D對象。

將表面高程數據擴展為3D數組。

創建mikeio的DataArray對象。

創建mikeio的Dataset對象。

將處理後的數據保存到.DFS3檔案中。
"""


# 匯入所需模組
import numpy as np
import xarray as xr
from scipy.interpolate import griddata
import mikeio
import pandas as pd
import os
import datetime


print("開始時間:", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

# 讀取 nc 檔案
hycom_ds = xr.open_dataset("D:/MIKEIO/pom-input/20211231_21.nc")
print(hycom_ds.keys())

# 提取變量和維度
longitude = hycom_ds["lon"].values
latitude = hycom_ds["lat"].values
time = pd.to_datetime(hycom_ds["time"].values)
original_depth = hycom_ds["depth"].values  
print(original_depth)
water_u = hycom_ds["water_u"].values
water_v = hycom_ds["water_v"].values
surf_el = hycom_ds["surf_el"].values
water_temp = hycom_ds["water_temp"].values
salinity = hycom_ds["salinity"].values

# 創建目標等距網格
target_lon = np.linspace(longitude.min(), longitude.max(), len(longitude))
target_lat = np.linspace(latitude.min(), latitude.max(), len(latitude))

# 原始深度值
#original_depth = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0,
#                           45.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 125.0, 150.0, 200.0, 250.0,
#                           300.0, 350.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0, 1250.0,
#                           1500.0, 2000.0, 2500.0, 3000.0, 4000.0, 5000.0])

# 深度索引
depth = np.array(range(len(original_depth))) #所有深度索引正值

#depth = np.array(range(len(original_depth)))
#depth = np.negative(depth) ##所有深度索引負值

# 使用原始深度值的最小值和最大值創建目標深度值
#depth = np.linspace(np.min(original_depth), np.max(original_depth), len(original_depth))

print(depth)

# 內插函數
def interpolate_data(data, source_lon, source_lat, target_lon, target_lat):
    data_flattened = data.flatten()
    source_grid = np.meshgrid(source_lon, source_lat)
    target_grid = np.meshgrid(target_lon, target_lat)
    source_points = np.array([source_grid[0].flatten(), source_grid[1].flatten()]).T
    return griddata(source_points, data_flattened, tuple(target_grid), method="linear")

# 初始內插值後的 3D 數值
u_interp = np.zeros((len(time), len(depth), len(target_lat), len(target_lon)))
v_interp = np.zeros((len(time), len(depth), len(target_lat), len(target_lon)))
temp_interp = np.zeros((len(time), len(depth), len(target_lat), len(target_lon)))
sal_interp = np.zeros((len(time), len(depth), len(target_lat), len(target_lon)))

# 對每個深度層的變量內插數值到目標網格
total_steps = len(time) * len(depth)
step = 0
for t_index in range(len(time)):
    for d_index in range(len(depth)):
        u_interp[t_index, d_index] = interpolate_data(water_u[t_index, d_index], longitude, latitude, target_lon, target_lat)
        v_interp[t_index, d_index] = interpolate_data(water_v[t_index, d_index], longitude, latitude, target_lon, target_lat)
        temp_interp[t_index, d_index] = interpolate_data(water_temp[t_index, d_index], longitude, latitude, target_lon, target_lat)
        sal_interp[t_index, d_index] = interpolate_data(salinity[t_index, d_index], longitude, latitude, target_lon, target_lat)
        # 顯示進度百分比
        step += 1
        progress = (step / total_steps) * 100
        print(f"進度: {progress:.2f}%")

output_folder = "D:/MIKEIO/pom-output/3D_output"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 使用 HYCOM 模型中的深度層創建 MIKE IO 的 Grid3D 物件
g = mikeio.Grid3D(x=target_lon, y=target_lat, z=depth, projection="LONG/LAT")

# 將 HYCOM 模型中的表面高程展開為 3D 數組
surf_el_3d = np.repeat(surf_el[:, np.newaxis, :, :], len(depth), axis=1)

# 創建 MIKE IO 的 DataArray 對象
das = [
    mikeio.DataArray(u_interp, time=time, geometry=g,
                      item=mikeio.ItemInfo("U", mikeio.EUMType.u_velocity_component, mikeio.EUMUnit.meter_per_sec)),
    mikeio.DataArray(v_interp, time=time, geometry=g,
                      item=mikeio.ItemInfo("V", mikeio.EUMType.v_velocity_component, mikeio.EUMUnit.meter_per_sec)),
    mikeio.DataArray(surf_el_3d, time=time, geometry=g,
                      item=mikeio.ItemInfo("Surface Elevation", mikeio.EUMType.Water_Level, mikeio.EUMUnit.meter)),
    mikeio.DataArray(temp_interp, time=time, geometry=g,
                      item=mikeio.ItemInfo("Temperature", mikeio.EUMType.Temperature, mikeio.EUMUnit.degree_Celsius)),
    mikeio.DataArray(sal_interp, time=time, geometry=g,
                      item=mikeio.ItemInfo("Salinity", mikeio.EUMType.Salinity, mikeio.EUMUnit.PSU)),
]

# 創建 MIKE IO 的 Dataset 對象
my_ds = mikeio.Dataset(das)

# 將資料保存到 DFS3 檔案
output_filename = f"{output_folder}/20211231_21_3D_v4_2.dfs3"
my_ds.to_dfs(output_filename)

print("結束時間:", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
