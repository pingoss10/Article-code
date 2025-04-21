# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 07:53:46 2025

@author: yuping
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 20:18:54 2024

@author: 于平
"""

from netCDF4 import Dataset
import numpy as np


def selection():
    
    file = 'F:/heatwave/maxT/air.2m.gauss.1960.nc'
    nc = Dataset(file)
    lats = nc.variables['lat'][:]
    lons = nc.variables['lon'][:]
    
    node = []; lat = []; lon = []
    resolution = 2
    for i in range(0, len(lats), resolution):
        temp = lats[i]
        a = np.cos(temp*np.pi/180)
        if a==0:
            b = 1
        else:
            b = round(1/a)
        y = np.arange(0, len(lons), b)
        for j in y:
            lat.append(temp)
            lon.append(lons[j])
            n = i*len(lons)+j
            node.append(n)
    return node, lat, lon
nodes, lat_selected, lon_selected = selection()
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.interpolate import griddata
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import Normalize
import pandas as pd

lon = np.load("F:/heatwave/longitude.npy")
lat = np.load("F:/heatwave/latitude.npy")

heat= pd.read_csv('F:/heatwave/worldtongbu/42yearworld/40worldheat_allheatevents.csv',header=None)
heat=np.array(heat)
heat=heat[nodes]


cold= pd.read_csv('F:/heatwave/worldtongbu/42yearworld/cold/40worldcold_allheatevents.csv',header=None)
cold=np.array(cold)
cold=cold[nodes]





#fig1为全球热浪事件，fig2为全球寒流事件
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.interpolate import griddata
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# 加载散度值数据
value = np.load("F:/heatwave/6242/6242Crosscorrelationr_value.npy")
data = np.nan_to_num(value)

# 计算热浪散度
yuzhi = 0.321682113909693
xuhao1 = 6242
ayt11 = np.zeros((xuhao1, xuhao1))
for y in range(xuhao1):
    zhi = data[y, :]
    ayt22 = np.zeros(xuhao1)
    for h in range(xuhao1):
        if abs(zhi[h]) > yuzhi:
            ayt22[h] = 1
        else:
            ayt22[h] = 0
    ayt11[y, :] = ayt22
# value=np.load("F:/heatwave/worldtongbu/42yearworld/detrend_Crosscorrelationr_value.npy")
# data = np.nan_to_num(value)
# #热浪对寒流
# yuzhi=0.3160635107892089
# xuhao1=9024*2
# ayt11  = np.zeros((xuhao1,xuhao1))  
# for y in range(xuhao1):
#     print(y)
#     zhi=data[y,:]
    
#     ayt22=np.zeros(xuhao1) 
#     for h in range(xuhao1):
#         if abs(zhi[h])>yuzhi:
#             ayt22[h]=1
#             #ayt22[h]=zhi[h]
#         else:
#             ayt22[h]=0
            
#     ayt11[y,:]=ayt22
chu_heataa = np.sum(ayt11, axis=1)
ru_heataa = np.sum(ayt11, axis=0)
Divergence_heat = ru_heataa - chu_heataa

# 计算寒流散度
value = np.load("F:/heatwave/6242/6242_ctoh_Crosscorrelationr_value.npy")
data = np.nan_to_num(value)

yuzhi = 0.31603312550843543
ayt11 = np.zeros((xuhao1, xuhao1))
for y in range(xuhao1):
    zhi = data[y, :]
    ayt22 = np.zeros(xuhao1)
    for h in range(xuhao1):
        if abs(zhi[h]) > yuzhi:
            ayt22[h] = 1
        else:
            ayt22[h] = 0
    ayt11[y, :] = ayt22
# value=np.load("F:/heatwave/worldtongbu/42yearworld/ColdtoHeat_Crosscorrelationr_value.npy")
# data = np.nan_to_num(value)
# yuzhi=0.31594086174697744
# xuhao1=9024*2
# ayt11  = np.zeros((xuhao1,xuhao1))  
# for y in range(xuhao1):
#     print(y)
#     zhi=data[y,:]
    
#     ayt22=np.zeros(xuhao1) 
#     for h in range(xuhao1):
#         if abs(zhi[h])>yuzhi:
#             ayt22[h]=1
#             #ayt22[h]=zhi[h]
#         else:
#             ayt22[h]=0
            
#     ayt11[y,:]=ayt22
chu_coldaa = np.sum(ayt11, axis=1)
ru_coldaa = np.sum(ayt11, axis=0)
Divergence_cold = ru_coldaa - chu_coldaa


# lat_range = [-30, 25]
# lon_range = [80, 170]
# Divergence_heat_reshaped = Divergence_heat.reshape(94, 192)
# Divergence_cold_reshaped = Divergence_cold.reshape(94, 192)
# lon = np.load("F:/heatwave/longitude.npy")
# lat = np.load("F:/heatwave/latitude.npy")
# grid_lonn, grid_latt = np.meshgrid(
#     np.linspace(lon.min(), lon.max(), 192),
#     np.linspace(lat.max(), lat.min(), 47*2)
# )

# # 获取 Divergence_heat 和 Divergence_cold 的前 10% 和后 10% 阈值
# heat_threshold = np.percentile(Divergence_heat_reshaped, 10)  # 前 10%
# cold_threshold = np.percentile(Divergence_cold_reshaped, 90)  # 后 10%

# # 创建布尔掩码来标记符合条件的交叉点
# heat_mask = Divergence_heat_reshaped <= heat_threshold  # 前 10% 的热浪散度
# cold_mask = Divergence_cold_reshaped >= cold_threshold  # 后 10% 的寒流散度

# # 交叉点：同时满足 heat_mask 和 cold_mask 的点
# intersection_mask = heat_mask & cold_mask

# # 限制交叉点到感兴趣的经纬度范围内
# lat_range = [-30, 25]
# lon_range = [80, 170]
# lat_mask = (grid_latt >= lat_range[0]) & (grid_latt <= lat_range[1])
# lon_mask = (grid_lonn >= lon_range[0]) & (grid_lonn <= lon_range[1])

# # 最终筛选的交叉点
# final_mask = intersection_mask & lat_mask & lon_mask

# # 绘制全球地图，但只标记感兴趣范围内的交叉点
# fig = plt.figure(figsize=(12, 8))
# ax = plt.axes(projection=ccrs.Robinson(central_longitude=180))

# # 设置全球范围
# ax.set_global()
# # 添加海岸线和地理特征
# ax.coastlines()
# ax.add_feature(cfeature.LAND, edgecolor='black')
# ax.add_feature(cfeature.OCEAN, facecolor='white')

# # 添加地球背景
# ax.stock_img()

# # 在特定经纬度范围内标记交叉点，使用黑色叉号表示
# ax.scatter(grid_lonn[final_mask], grid_latt[final_mask], color='black', s=20, marker='x', transform=ccrs.PlateCarree())



#plt.savefig('F:/heatwave/实验三程序/xin实验图/s6_Selected Nodes1.pdf', format='pdf', bbox_inches='tight')











# 选择特定经纬度范围内的节点

lat_range = [-30, 25]
lon_range = [80, 170]


lower_threshold_heat = np.percentile(Divergence_heat, 10)
upper_threshold_cold = np.percentile(Divergence_cold, 90)

heat_lower_indices = np.where((Divergence_heat <= lower_threshold_heat) & 
                              (lat >= lat_range[0]) & (lat <= lat_range[1]) &
                              (lon >= lon_range[0]) & (lon <= lon_range[1]))[0]

cold_upper_indices = np.where((Divergence_cold >= upper_threshold_cold) & 
                              (lat >= lat_range[0]) & (lat <= lat_range[1]) &
                              (lon >= lon_range[0]) & (lon <= lon_range[1]))[0]
#索引
print("热浪散度值小于排名后10%的节点索引:", heat_lower_indices)
print("寒流散度值大于排名前10%的节点索引:", cold_upper_indices)

heat_lower_lon = lon[heat_lower_indices]
heat_lower_lat = lat[heat_lower_indices]
cold_upper_lon = lon[cold_upper_indices]
cold_upper_lat = lat[cold_upper_indices]




# 绘制地图并显示选择的节点
# import matplotlib.pyplot as plt
# import cartopy.crs as ccrs
# import cartopy.feature as cfeature

# # 创建 2x1 布局的子图
# fig, axs = plt.subplots(2, 1, figsize=(15, 12), subplot_kw={'projection': ccrs.Robinson(central_longitude=125)})

# # 第一个子图
# ax1 = axs[0]
# ax1.set_global()
# ax1.coastlines()
# ax1.add_feature(cfeature.LAND, edgecolor='black')
# ax1.add_feature(cfeature.OCEAN, facecolor='white')

# # 假设 heat_lower_lon, heat_lower_lat, cold_upper_lon, cold_upper_lat 是预定义的坐标
# ax1.scatter(heat_lower_lon, heat_lower_lat, color='blue', s=40, marker='x', transform=ccrs.PlateCarree())
# #ax1.scatter(cold_upper_lon, cold_upper_lat, color='red', s=30, marker='o', transform=ccrs.PlateCarree(), label='Cold upper 10%')

# ax1.set_title('(a) Selected Nodes(Network I)', fontsize=20,x=0.3)
# ax1.legend()

# # 第二个子图
# ax2 = axs[1]
# ax2.set_global()
# ax2.coastlines()
# ax2.add_feature(cfeature.LAND, edgecolor='black')
# ax2.add_feature(cfeature.OCEAN, facecolor='white')

# # 在第二个子图上绘制相同的散点图
# #ax2.scatter(heat_lower_lon, heat_lower_lat, color='blue', s=30, marker='x', transform=ccrs.PlateCarree(), label='Heat lower 10%')
# ax2.scatter(cold_upper_lon, cold_upper_lat, color='red', s=40, marker='x', transform=ccrs.PlateCarree())

# ax2.set_title('(b) Selected Nodes(Network II)', fontsize=20,x=0.3)
# #plt.savefig('F:/heatwave/实验三程序/xin实验图/s6_Selected Nodes.pdf', format='pdf', bbox_inches='tight')






#预测enso


#common_indices = np.intersect1d(heat_lower_indices, cold_upper_indices)

pacfic_heat=heat[heat_lower_indices,:]
pacfic_heat1=np.sum(pacfic_heat,axis=0)
avg=np.mean(pacfic_heat1)








#画colorbar
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# 假设 pacfic_heat1 和 avg 已经定义
# 计算 pacfic_heat2
pacfic_heat2 = pacfic_heat1 - avg

# 除以格点数
pacfic_heat3 = pacfic_heat2 / 209 

# 固定的纵轴数据
y_values = [0, 1, -1, -1, 0, 1, 1, -1, 0, 0, 1, 0, 0, 1, -1, 0, 1, -1, -1, -1, 0, 1, 0, 1, -1, 1, -1, -1, 1, -1, -1, 0, 0, 1, 1, -1, -1, 1, 0, -1]
strength = [-0.1, 2.2, -0.9, -1.1, -0.4, 1.2, 1.1, -1.8, -0.1, 0.4, 1.5, -0.1, 0.1, 1.1, -1, -0.5, 2.4, -1.6, -1.7, -0.7, -0.3, 1.1, 0.4, 0.7, -0.8, 0.9, -1.6, -0.7, 1.6, -1.6, -1, -0.2, -0.3, 0.7, 2.6, -0.6, -1, 0.8, 0.5, -1.2]
# 根据 y 值生成颜色列表
colors = ['blue' if y == -1 else 'red' if y == 1 else 'black' for y in y_values]

# 创建绘图
plt.figure(figsize=(12, 11))

# 填充不同区间
plt.axvspan(-2, -1, color='#EA8379', alpha=0.2)  # 填充区间 [-2, -1]
plt.axvspan(-1, 1, color='#B3953D', alpha=0.2)  # 填充区间 [-1, 1]
plt.axvspan(1, 4, color='#7DAEE0', alpha=0.2)   # 填充区间 [1, 4] 
# ax_bif.axvspan(-2.5, -1, color='#E4DBE8', alpha=0.5)  # 填充区间 [-3, -1]
# ax_bif.axvspan(-1, 1, color='#E6EEF6', alpha=0.4)  # 填充区间 [-1, 1]
# ax_bif.axvspan(1, 3, color='#AFB6D2', alpha=0.5)   # 填充区间 [1, 3]


# 计算点的大小，根据strength来映射
size = np.abs(strength) * 600  # 映射强度为点的大小，您可以调整100来控制点的大小范围

# 绘制散点图，使用渐变色
scatter = plt.scatter(pacfic_heat3, y_values, c=strength, cmap='PiYG_r', s=size, edgecolors='w', marker='o' ,vmin=-1.5, vmax=2.8)

# 设置x轴和y轴的标签
plt.xlabel('Western Pacific HWN anomaly', fontsize=25)
plt.ylabel('NDJ State 1981-2020', fontsize=25)
ax = plt.gca()  # Get current axis
ax.xaxis.set_label_position('top')

# 设置x轴的限制
plt.xlim(-2, 4)

# 设置y轴的刻度为 -1, 0, 1，并为每个刻度设置标签
plt.yticks([-1, 0, 1], ['La Niña (-1)', 'Normal (0)', 'El Niño (1)'], fontsize=15, fontweight='bold')

# 创建渐变色条（使用colorbar）
norm = Normalize(vmin=-1.5, vmax=2.8)
sm = ScalarMappable(cmap='PiYG_r', norm=norm)
sm.set_array([])  # 空数组来配置 colorbar
cbar = plt.colorbar(sm, label='Oceanic Niño Index (ONI)',orientation='horizontal', shrink=0.8, aspect=20, pad=0.06)

# 使用 colorbar 的 ax 属性来设置刻度标签的大小
cbar.ax.tick_params(labelsize=25)  # 设置 colorbar 刻度标签的字体大小

# 添加垂直参考线
plt.axvline(x=1, color='green', linestyle='--', linewidth=2)
plt.axvline(x=-1, color='green', linestyle='--', linewidth=2)

# 设置刻度标签的字体大小
plt.tick_params(axis='both', labelsize=20)

# 设置标题
plt.title('a', fontweight='bold', fontsize=60, x=-0.05)

# 保存图像
plt.savefig('F:/heatwave/6242/最终版Fig/图四/Figure4_a.svg', format='svg', bbox_inches='tight')

# 显示图像
plt.show()

#不画colorbar
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# 假设 pacfic_heat1 和 avg 已经定义
# 计算 pacfic_heat2
pacfic_heat2 = pacfic_heat1 - avg

# 除以格点数
pacfic_heat3 = pacfic_heat2 / 209

# 固定的纵轴数据
y_values = [0, 1, -1, -1, 0, 1, 1, -1, 0, 0, 1, 0, 0, 1, -1, 0, 1, -1, -1, -1, 0, 1, 0, 1, -1, 1, -1, -1, 1, -1, -1, 0, 0, 1, 1, -1, -1, 1, 0, -1]
strength = [-0.1, 2.2, -0.9, -1.1, -0.4, 1.2, 1.1, -1.8, -0.1, 0.4, 1.5, -0.1, 0.1, 1.1, -1, -0.5, 2.4, -1.6, -1.7, -0.7, -0.3, 1.1, 0.4, 0.7, -0.8, 0.9, -1.6, -0.7, 1.6, -1.6, -1, -0.2, -0.3, 0.7, 2.6, -0.6, -1, 0.8, 0.5, -1.2]

# 根据 y 值生成颜色列表
colors = ['blue' if y == -1 else 'red' if y == 1 else 'gray' for y in y_values]

# 创建绘图
plt.figure(figsize=(10, 8))

# 填充不同区间
plt.axvspan(-2, -1, color='#EA8379', alpha=0.2)  # 填充区间 [-2, -1]
plt.axvspan(-1, 1, color='#B3953D', alpha=0.2)  # 填充区间 [-1, 1]
plt.axvspan(1, 4, color='#7DAEE0', alpha=0.2)   # 填充区间 [1, 4] 

# 计算点的大小，根据strength来映射
size = np.abs(strength) * 300  # 映射强度为点的大小，您可以调整100来控制点的大小范围

# 绘制散点图，使用预定义颜色
scatter = plt.scatter(pacfic_heat3, y_values, c=colors, s=size, edgecolors='w', marker='o')

# 设置x轴和y轴的标签
plt.xlabel('Western Pacific HWN anomaly', fontsize=25)
plt.ylabel('NDJ State 1981-2020', fontsize=25)
ax = plt.gca()  # Get current axis
ax.xaxis.set_label_position('top')

# 设置x轴的限制
plt.xlim(-2, 4)

# 设置y轴的刻度为 -1, 0, 1，并为每个刻度设置标签
plt.yticks([-1, 0, 1], ['La Niña (-1)', 'Normal (0)', 'El Niño (1)'], fontsize=15, fontweight='bold')

# 创建渐变色条（使用colorbar）—此处不再需要colorbar，因为没有使用强度进行颜色映射
# 创建渐变色条（使用colorbar）
# norm = Normalize(vmin=-1.5, vmax=2.8)
# sm = ScalarMappable(cmap='PiYG_r', norm=norm)
# sm.set_array([])  # 空数组来配置 colorbar
# cbar = plt.colorbar(sm, label='Oceanic Niño Index (ONI)',orientation='horizontal', shrink=0.8, aspect=20, pad=0.06)

# # 使用 colorbar 的 ax 属性来设置刻度标签的大小
# cbar.ax.tick_params(labelsize=25)  # 设置 colorbar 刻度标签的字体大小

# 添加垂直参考线
plt.axvline(x=1, color='green', linestyle='--', linewidth=2)
plt.axvline(x=-1, color='green', linestyle='--', linewidth=2)

# 设置刻度标签的字体大小
plt.tick_params(axis='both', labelsize=20)

# 设置标题
plt.title('a', fontweight='bold', fontsize=60, x=-0.05)

# 保存图像
plt.savefig('F:/heatwave/6242/最终版Fig/图四/Figure4_a.svg', format='svg', bbox_inches='tight')

# 显示图像
plt.show()




import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams['font.family'] = 'Arial'

y_values = [0, 1, -1, -1, 0, 1, 1, -1, 0, 0, 1, 0, 0, 1, -1, 0, 1, -1, -1, -1, 0, 1, 0, 1, -1, 1, -1, -1, 1, -1, -1, 0, 0, 1, 1, -1, -1, 1, 0, -1]

# 颜色映射
colors = ['#377eb8' if y == -1 else '#e41a1c' if y == 1 else '#555555' for y in y_values]  # muted blue/red/gray

# 创建图
fig, ax = plt.subplots(figsize=(18, 6), dpi=300)

# 绘制散点图
sc = ax.scatter(pacfic_heat3, y_values, color=colors, edgecolor='black', s=250, linewidth=0.5)

# 坐标轴设置
ax.set_yticks([-1, 0, 1])
ax.set_yticklabels(['La Niña (-1)', 'Normal (0)', 'El Niño (1)'], fontsize=25, fontweight='bold')
ax.set_xlabel('Western Pacific HWN anomaly', fontsize=25, fontweight='bold')
ax.set_ylabel('NDJ ENSO State (1981–2020)', fontsize=25, fontweight='bold')
ax.axvline(x=1, color='#D3D3D3', linestyle=':', linewidth=5)
#ax.axvline(x=4, color='black', linestyle='--', linewidth=2)
ax.axvspan(1,2.8, color='#74B69F', alpha=0.2, zorder=0)
# 网格线
ax.grid(True, linestyle=':', linewidth=0.8, alpha=0.5, zorder=0)

# 坐标轴外观
ax.tick_params(axis='both', labelsize=25)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.2)
ax.spines['bottom'].set_linewidth(1.2)

# 图例
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='El Niño', markerfacecolor='#e41a1c', markeredgecolor='black', markersize=8),
    Line2D([0], [0], marker='o', color='w', label='Normal', markerfacecolor='#555555', markeredgecolor='black', markersize=8),
    Line2D([0], [0], marker='o', color='w', label='La Niña', markerfacecolor='#377eb8', markeredgecolor='black', markersize=8),
]
ax.legend(handles=legend_elements, fontsize=25, frameon=False)
#ax.set_title('a', x=-0.2, fontweight='bold',fontsize=42)
ax.set_title('ERA5', fontweight='bold',fontsize=25)
# 去掉上下边缘多余留白
plt.savefig('F:/heatwave/6242/最终版Fig/sup_ERA5test.pdf', format='pdf', bbox_inches='tight')
plt.tight_layout()
plt.show()












#ERA5预测enso

heat= np.loadtxt("G:/heatwave/heatwave30year/heatwave_40year.txt", delimiter=',')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.interpolate import griddata
import cartopy.crs as ccrs
import cartopy.feature as cfeature

a="G:/heatwave/ERA5/"

b=".nc"
yearnumber=2


Temperature_data=[]
#Pressure_data=[]
Tdata=[]
#SL_Pressure_data=[]
#for j in range(YEAR_NUMBER2):
for j in range(yearnumber):
#for j in range (40,73):
    print (j)
    data_lebel = 1981+j
    data=[]
    data = Dataset(a+str(data_lebel)+b,'r',format='NETCDF4')
#    data1 = Dataset(a1+'slp.'+str(data_lebel)+b,'r',format='NETCDF4')    
#    data2 = Dataset(a2+'air.2m.gauss.'+str(data_lebel)+b,'r',format='NETCDF4')
#    data3 = Dataset(a3+'pres.sfc.'+str(data_lebel)+b,'r',format='NETCDF4')  
#    Temperature_data.append(data.variables['t'][:]) #
#    Pressure_data.append(data1.variables['slp'][:]) #
#    TWOm_Temperature_data.append(data2.variables['air'][:,0,:,:]) #
    Tdata.append(data.variables['t2m'][:])
lat=data.variables['latitude'][:]
lon=data.variables['longitude'][:]

lon2d, lat2d = np.meshgrid(lon, lat)
lat = lat2d.flatten()
lon= lon2d.flatten()

# 加载散度值数据
value = np.load("G:/heatwave/heatwave30year/ERA5_network.npy")
data = np.nan_to_num(value)

# 计算热浪散度
yuzhi = 0.301682113909693
xuhao1 = 144*73
ayt11 = np.zeros((xuhao1, xuhao1))
for y in range(xuhao1):
    print(y)
    zhi = data[y, :]
    ayt22 = np.zeros(xuhao1)
    for h in range(xuhao1):
        if abs(zhi[h]) > yuzhi:
            ayt22[h] = 1
        else:
            ayt22[h] = 0
    ayt11[y, :] = ayt22

chu_heataa = np.sum(ayt11, axis=1)
ru_heataa = np.sum(ayt11, axis=0)
Divergence_heat = ru_heataa - chu_heataa


lat_range = [-30, 25]
lon_range = [80, 170]


lower_threshold_heat = np.percentile(Divergence_heat, 10)


heat_lower_indices = np.where((Divergence_heat <= lower_threshold_heat) & 
                              (lat >= lat_range[0]) & (lat <= lat_range[1]) &
                              (lon >= lon_range[0]) & (lon <= lon_range[1]))[0]



heat_lower_lon = lon[heat_lower_indices]
heat_lower_lat = lat[heat_lower_indices]



pacfic_heat=heat[heat_lower_indices,:]
pacfic_heat1=np.sum(pacfic_heat,axis=0)
avg=np.mean(pacfic_heat1)



hh=len(heat_lower_indices)


# 计算 pacfic_heat2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
# 假设 pacfic_heat1 和 avg 已经定义
# 计算 pacfic_heat2
pacfic_heat2 = pacfic_heat1 - avg

# 除以格点数
pacfic_heat3 = pacfic_heat2 / hh

# 固定的纵轴数据
y_values = [0, 1, -1, -1, 0, 1, 1, -1, 0, 0, 1, 0, 0, 1, -1, 0, 1, -1, -1, -1, 0, 1, 0, 1, -1, 1, -1, -1, 1, -1, -1, 0, 0, 1, 1, -1, -1, 1, 0, -1]

# 根据 y 值生成颜色列表
colors = ['blue' if y == -1 else 'red' if y == 1 else 'black' for y in y_values]

# 绘制散点图
plt.figure(figsize=(10, 6))
plt.scatter(pacfic_heat3, y_values, color=colors, marker='o', s=100)

# 设置y轴的刻度为 -1, 0, 1
plt.yticks([-1, 0, 1])

# 设置网格线样式
plt.grid(True, linestyle='--', linewidth=1, alpha=1)

# 设置x轴和y轴的标签
plt.xlabel('Pacific Heat Anomalies\n((Annual heatwave average - 40year average)/number of grid points)', fontsize=15)
plt.ylabel('NDJ State 1981-2020', fontsize=20)
legend_elements = [Line2D([0], [0], marker='o', color='w', label='El Niño Year (1)', 
                          markerfacecolor='red', markersize=10),
                   Line2D([0], [0], marker='o', color='w', label='Normal Year (0)', 
                          markerfacecolor='black', markersize=10),
                   Line2D([0], [0], marker='o', color='w', label='La Niña Year (-1)', 
                          markerfacecolor='blue', markersize=10)]

plt.legend(handles=legend_elements, loc='upper right', fontsize=15)


plt.title("ERA5", fontsize=30)



# 显示图形
plt.show()





############################################################NCEP 数据选点代码
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.interpolate import griddata
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# 加载散度值数据
value = np.load("F:/heatwave/6242/6242Crosscorrelationr_value.npy")
data = np.nan_to_num(value)

# 计算热浪散度
yuzhi = 0.321682113909693
xuhao1 = 6242
ayt11 = np.zeros((xuhao1, xuhao1))
for y in range(xuhao1):
    zhi = data[y, :]
    ayt22 = np.zeros(xuhao1)
    for h in range(xuhao1):
        if abs(zhi[h]) > yuzhi:
            ayt22[h] = 1
        else:
            ayt22[h] = 0
    ayt11[y, :] = ayt22

chu_heataa = np.sum(ayt11, axis=1)
ru_heataa = np.sum(ayt11, axis=0)
Divergence_heat = ru_heataa - chu_heataa

# 计算寒流散度
value = np.load("F:/heatwave/6242/6242_ctoh_Crosscorrelationr_value.npy")
data = np.nan_to_num(value)

yuzhi = 0.31603312550843543
ayt11 = np.zeros((xuhao1, xuhao1))
for y in range(xuhao1):
    zhi = data[y, :]
    ayt22 = np.zeros(xuhao1)
    for h in range(xuhao1):
        if abs(zhi[h]) > yuzhi:
            ayt22[h] = 1
        else:
            ayt22[h] = 0
    ayt11[y, :] = ayt22

chu_coldaa = np.sum(ayt11, axis=1)
ru_coldaa = np.sum(ayt11, axis=0)
Divergence_cold = ru_coldaa - chu_coldaa





lat_range = [-30, 25]
lon_range = [80, 170]
# Divergence_heat_reshaped = Divergence_heat.reshape(94, 192)
# Divergence_cold_reshaped = Divergence_cold.reshape(94, 192)
lon = np.load("F:/heatwave/longitude.npy")
lat = np.load("F:/heatwave/latitude.npy")

# 获取 Divergence_heat 和 Divergence_cold 的前 10% 和后 10% 阈值
heat_threshold = np.percentile(Divergence_heat, 10)  # 前 10%
cold_threshold = np.percentile(Divergence_cold, 90)  # 后 10%

heat_mask = Divergence_heat <= heat_threshold
cold_mask = Divergence_cold >= cold_threshold
intersection_mask = heat_mask & cold_mask

# 经纬度筛选范围
lon_range = [80, 170]
lat_range = [-30, 25]
region_mask = (lon >= lon_range[0]) & (lon <= lon_range[1]) & \
              (lat >= lat_range[0]) & (lat <= lat_range[1])

# 筛选区域内的点
heat_points_mask = heat_mask & region_mask
cold_points_mask = cold_mask & region_mask
intersection_points_mask = intersection_mask & region_mask

# 提取点坐标
heat_lons, heat_lats = lon[heat_points_mask], lat[heat_points_mask]
cold_lons, cold_lats = lon[cold_points_mask], lat[cold_points_mask]
inter_lons, inter_lats = lon[intersection_points_mask], lat[intersection_points_mask]


fig = plt.figure(figsize=(12, 8))
ax = plt.axes(projection=ccrs.Robinson(central_longitude=180))

# 设置全球范围
ax.set_global()
# 添加海岸线和地理特征
ax.coastlines()
ax.add_feature(cfeature.LAND, edgecolor='black')
ax.add_feature(cfeature.OCEAN, facecolor='white')

# 添加地球背景
ax.stock_img()

# 在特定经纬度范围内标记交叉点，使用黑色叉号表示
ax.scatter(inter_lons, inter_lats, color='black', s=30, marker='x', transform=ccrs.PlateCarree())
plt.savefig('F:/heatwave/6242/最终版Fig/Sup_NECP_location.pdf', format='pdf', bbox_inches='tight')
plt.show()






##################################################################ERA5数据
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.interpolate import griddata
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# 加载散度值数据
value = np.load("G:/heatwave/heatwave30year/ERA5_network.npy")
data = np.nan_to_num(value)
# absdata=abs(data)
# yuzhi=np.percentile(absdata,95)
# print(yuzhi)
# 计算热浪散度
yuzhi =0.30742838949472917
xuhao1 = 144*73
ayt11 = np.zeros((xuhao1, xuhao1))
for y in range(xuhao1):
    print(y)
    zhi = data[y, :]
    ayt22 = np.zeros(xuhao1)
    for h in range(xuhao1):
        if abs(zhi[h]) > yuzhi:
            ayt22[h] = 1
        else:
            ayt22[h] = 0
    ayt11[y, :] = ayt22

chu_heataa = np.sum(ayt11, axis=1)
ru_heataa = np.sum(ayt11, axis=0)
Divergence_heat = ru_heataa - chu_heataa


# 计算寒流散度
value = np.load("G:/heatwave/heatwave30year/ctoh_ERA5_network.npy")
data = np.nan_to_num(value)
# absdata=abs(data)
# yuzhi=np.percentile(absdata,95)
# print(yuzhi)
xuhao1 = 144*73
yuzhi = 0.30810826576216244
ayt11 = np.zeros((xuhao1, xuhao1))
for y in range(xuhao1):
    print(y)
    zhi = data[y, :]
    ayt22 = np.zeros(xuhao1)
    for h in range(xuhao1):
        if abs(zhi[h]) > yuzhi:
            ayt22[h] = 1
        else:
            ayt22[h] = 0
    ayt11[y, :] = ayt22

chu_coldaa = np.sum(ayt11, axis=1)
ru_coldaa = np.sum(ayt11, axis=0)
Divergence_cold = ru_coldaa - chu_coldaa





lat_range = [-30, 25]
lon_range = [80, 170]
# Divergence_heat_reshaped = Divergence_heat.reshape(94, 192)
# Divergence_cold_reshaped = Divergence_cold.reshape(94, 192)

a="G:/heatwave/ERA5/"

b=".nc"
yearnumber=2


Temperature_data=[]
#Pressure_data=[]
Tdata=[]
#SL_Pressure_data=[]
#for j in range(YEAR_NUMBER2):
for j in range(yearnumber):
#for j in range (40,73):
    print (j)
    data_lebel = 1981+j
    data=[]
    data = Dataset(a+str(data_lebel)+b,'r',format='NETCDF4')
#    data1 = Dataset(a1+'slp.'+str(data_lebel)+b,'r',format='NETCDF4')    
#    data2 = Dataset(a2+'air.2m.gauss.'+str(data_lebel)+b,'r',format='NETCDF4')
#    data3 = Dataset(a3+'pres.sfc.'+str(data_lebel)+b,'r',format='NETCDF4')  
#    Temperature_data.append(data.variables['t'][:]) #
#    Pressure_data.append(data1.variables['slp'][:]) #
#    TWOm_Temperature_data.append(data2.variables['air'][:,0,:,:]) #
    Tdata.append(data.variables['t2m'][:])
lat=data.variables['latitude'][:]
lon=data.variables['longitude'][:]
lon2d, lat2d = np.meshgrid(lon, lat)
lat = lat2d.flatten()
lon= lon2d.flatten()

# 获取 Divergence_heat 和 Divergence_cold 的前 10% 和后 10% 阈值
heat_threshold = np.percentile(Divergence_heat, 10)  # 前 10%
#cold_threshold = np.percentile(Divergence_cold, 90)  # 后 10%

heat_mask = Divergence_heat <= heat_threshold
#cold_mask = Divergence_cold >= cold_threshold
intersection_mask = heat_mask 

# 经纬度筛选范围
lon_range = [80, 170]
lat_range = [-30, 25]
region_mask = (lon >= lon_range[0]) & (lon <= lon_range[1]) & \
              (lat >= lat_range[0]) & (lat <= lat_range[1])

# 筛选区域内的点
# heat_points_mask = heat_mask & region_mask
# cold_points_mask = cold_mask & region_mask
intersection_points_mask = intersection_mask & region_mask

# 提取点坐标
# heat_lons, heat_lats = lon[heat_points_mask], lat[heat_points_mask]
# cold_lons, cold_lats = lon[cold_points_mask], lat[cold_points_mask]
inter_lons, inter_lats = lon[intersection_points_mask], lat[intersection_points_mask]


fig = plt.figure(figsize=(12, 8))
ax = plt.axes(projection=ccrs.Robinson(central_longitude=180))

# 设置全球范围
ax.set_global()
# 添加海岸线和地理特征
ax.coastlines()
ax.add_feature(cfeature.LAND, edgecolor='black')
ax.add_feature(cfeature.OCEAN, facecolor='white')

# 添加地球背景
ax.stock_img()
ax.set_title('a',fontweight='bold',fontsize=35, x=0.05,pad=20)
# 在特定经纬度范围内标记交叉点，使用黑色叉号表示
ax.scatter(inter_lons, inter_lats, color='black', s=30, marker='x', transform=ccrs.PlateCarree())
plt.savefig('F:/heatwave/6242/最终版Fig/Sup_ERA5_location.pdf', format='pdf', bbox_inches='tight')
plt.show()





# heatwave_events=np.array(allheatevents1)
tenper=Divergence_cold

llat=data.variables['latitude'][:]
llon=data.variables['longitude'][:]
tenper1 = np.reshape(tenper,(73,144))

fig = plt.figure(figsize=(10, 10))
ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))  # 设置中心经度为180
ax.set_global()

# 绘制温度数据
c = ax.pcolormesh(llon, llat, tenper1, transform=ccrs.PlateCarree(), cmap='coolwarm',vmin=-200, vmax=200)
plt.colorbar(c, ax=ax, orientation='horizontal', label='Temperature (°C)')

# 添加海岸线等特征
ax.coastlines()
ax.add_feature(cfeature.BORDERS, linestyle=':')

plt.title('Global Temperature Map Centered at 180 Longitude')
plt.show()
