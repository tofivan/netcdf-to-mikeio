# -*- coding: utf-8 -*-
"""
Created on Tue May  9 17:26:34 2023

@author: cyuan
"""

# 匯入所需模組
import numpy as np
import xarray as xr
from scipy.interpolate import griddata
import mikeio
import pandas as pd
import os
import datetime
import dask.array as da
from dask import delayed
# 導入 Dask 所需的模組
import dask
from multiprocessing import cpu_count
import psutil

# 設定 worker 數量為系統核心數量的 90%
num_workers = int(cpu_count() * 0.9)

# 使用 psutil 獲取系統總內存
total_memory = psutil.virtual_memory().total

# 設定每個 worker 的內存限制為系統總內存的 90% 除以 worker 數量
memory_limit = int(total_memory * 0.9 / num_workers)

# 設定 Dask 使用本地多線程調度器
dask.config.set(scheduler="threads")

# 程式碼開始

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

depth = np.array(range(len(original_depth))) #所有深度索引正值

#depth = np.array(range(len(original_depth)))
#depth = np.negative(depth) ##所有深度索引負值
print(depth)

# 內插函數
def interpolate_data(data, source_lon, source_lat, target_lon, target_lat):
    data_flattened = data.flatten()
    source_grid = np.meshgrid(source_lon, source_lat)
    target_grid = np.meshgrid(target_lon, target_lat)
    source_points = np.array([source_grid[0].flatten(), source_grid[1].flatten()]).T
    return griddata(source_points, data_flattened, tuple(target_grid), method="linear")

# 使用 Dask 進行並行計算
def parallel_interpolation(data):
    delayed_results = []
    for t_index in range(len(time)):
        for d_index in range(len(depth)):
            delayed_result = delayed(interpolate_data)(data[t_index, d_index], longitude, latitude, target_lon, target_lat)
            delayed_results.append(da.from_delayed(delayed_result, shape=(len(target_lat), len(target_lon)), dtype=data.dtype))
    
    # 將所有延遲結果沿時間和深度維度堆疊在一起
    interp_data = da.stack(delayed_results, axis=0).reshape(len(time), len(depth), len(target_lat), len(target_lon))
    return interp_data.compute()


u_interp = parallel_interpolation(water_u)
v_interp = parallel_interpolation(water_v)
temp_interp = parallel_interpolation(water_temp)
sal_interp = parallel_interpolation(salinity)

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
output_filename = f"{output_folder}/20211231_21_3D_v5.dfs3"
my_ds.to_dfs(output_filename)

print("結束時間:", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
