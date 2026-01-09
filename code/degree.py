# -*- coding: utf-8 -*-
"""
Created on Sun Nov 30 14:11:33 2025

@author: yuping
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.interpolate import griddata


value=np.load("F:/heatwave/6242/6242Crosscorrelationr_value.npy")
dis = np.load("F:/heatwave/6242distance.npy")
data = np.nan_to_num(value)

absdata=abs(data)
yuzhi=np.percentile(absdata,95)
print(yuzhi)
#热浪对寒流
yuzhi=0.321682113909693
xuhao1=6242
ayt11  = np.zeros((xuhao1,xuhao1))  
for y in range(xuhao1):
    print(y)
    zhi=data[y,:]
    diss=dis[y,:]
    
    ayt22=np.zeros(xuhao1) 
    for h in range(xuhao1):
        if abs(zhi[h])>yuzhi:
            #ayt22[h]=diss[h]
            ayt22[h]=1
        else:
            ayt22[h]=0
            
    ayt11[y,:]=ayt22
    
#chudu    
# chu_heataa=np.sum(ayt11, axis=1) / np.count_nonzero(ayt11, axis=1)
# chu_heataa[np.isnan(chu_heataa)] = 0

# #rudu
# ru_heataa=np.sum(ayt11, axis=0) / np.count_nonzero(ayt11, axis=0)     
# ru_heataa[np.isnan(ru_heataa)] = 0 
chu_heataa=np.sum(ayt11, axis=1)
ru_heataa=np.sum(ayt11, axis=0)
Divergence_heat=ru_heataa-chu_heataa


value=np.load("F:/heatwave/6242/6242_ctoh_Crosscorrelationr_value.npy")
data = np.nan_to_num(value)

# absdata=abs(data)
# yuzhi=np.percentile(absdata,95)
# print(yuzhi)

yuzhi=0.31603312550843543
xuhao1=6242
ayt11  = np.zeros((xuhao1,xuhao1))  
for y in range(xuhao1):
    print(y)
    zhi=data[y,:]
    diss=dis[y,:]
    ayt22=np.zeros(xuhao1) 
    for h in range(xuhao1):
        if abs(zhi[h])>yuzhi:
            #ayt22[h]=diss[h]
            ayt22[h]=1
        else:
            ayt22[h]=0
            
    ayt11[y,:]=ayt22
    
#chudu    
             
chu_coldaa=np.sum(ayt11, axis=1)
ru_coldaa=np.sum(ayt11, axis=0)
Divergence_cold=ru_coldaa-chu_coldaa









from matplotlib.colors import ListedColormap

# Divergence_heat = chu_heataa/np.max(chu_heataa)
# Divergence_cold = ru_heataa/np.max(ru_heataa)

Divergence_heat = chu_heataa
Divergence_cold = ru_heataa

# 计算阈值
heat_threshold_high = np.percentile(Divergence_heat, 90)
heat_threshold_low = np.percentile(Divergence_heat, 10)
cold_threshold_high = np.percentile(Divergence_cold, 90)
cold_threshold_low = np.percentile(Divergence_cold, 10)

# 加载经纬度
lon = np.load("F:/heatwave/longitude.npy")
lat = np.load("F:/heatwave/latitude.npy")

# 创建网格
grid_lon = np.linspace(0, 360, 360)
grid_lat = np.linspace(-90, 90, 180)
lon_grid, lat_grid = np.meshgrid(grid_lon, grid_lat)

# 插值
deg_grid = griddata((lon, lat), Divergence_heat, (lon_grid, lat_grid), method='cubic')
deg_grid1 = griddata((lon, lat), Divergence_cold, (lon_grid, lat_grid), method='cubic')

# 创建图表：一行两列
fig = plt.figure(figsize=(20, 8), dpi=300, facecolor='white')
gs = gridspec.GridSpec(1, 2, wspace=0.2)  # 一行两列，调整间距

# 子图 1：Network I
ax5 = plt.subplot(gs[0, 0], projection=ccrs.Robinson(central_longitude=180))
ax5.coastlines(linewidth=1, color='black')
ax5.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='black')
ax5.add_feature(cfeature.OCEAN, facecolor='white')
base_cmap = plt.get_cmap('Reds')

# 比如只用后 60% 的颜色（0.4~1.0 区间可以自己调）
colors = base_cmap(np.linspace(0, 0.9, 256))

# 生成一个新的 colormap
new_cmap = ListedColormap(colors)
contourf_heat = ax5.pcolormesh(lon_grid, lat_grid, deg_grid, cmap=new_cmap,vmin=100,vmax=900, transform=ccrs.PlateCarree())
cbar1 = fig.colorbar(contourf_heat, ax=ax5, orientation='horizontal', shrink=0.9, extend='both', aspect=28, pad=0.04)
cbar1.ax.tick_params(labelsize=26, width=2, length=6)
cbar1.set_label('Network I out-degree', fontsize=24, labelpad=5)
cbar1.outline.set_linewidth(2)  # 颜色条边框
gl = ax5.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.7, color='black', alpha=0.5, linestyle='--')
gl.top_labels = True  # 经度标签放在上方
gl.left_labels = True  # 纬度标签在左侧
gl.right_labels = False  # 隐藏右侧经度标签
gl.bottom_labels = False  # 隐藏底部经度标签
gl.xlabel_style = {'size':25, 'color': 'black'}
gl.ylabel_style = {'size': 25, 'color': 'black'}
# try:
#     gl.xpadding = 15
#     gl.ypadding = 15
# except AttributeError:
#     pass
ax5.contourf(lon_grid, lat_grid, deg_grid, levels=[heat_threshold_high, deg_grid.max()],
              hatches=['\\\\'], alpha=0, transform=ccrs.PlateCarree())
# ax5.contourf(lon_grid, lat_grid, deg_grid, levels=[deg_grid.min(), heat_threshold_low],
#               hatches=['//'], alpha=0, transform=ccrs.PlateCarree())
ax5.set_title('a', fontsize=34, fontweight='bold', x=-0.01, pad=1)
#ax5.text(0.5, 1.05, 'Network I', ha='center', va='center', fontsize=40, transform=ax5.transAxes, fontweight='bold')

# 子图 2：Network II
ax6 = plt.subplot(gs[0, 1], projection=ccrs.Robinson(central_longitude=180))
ax6.coastlines(linewidth=1, color='black')
ax6.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='black')
ax6.add_feature(cfeature.OCEAN, facecolor='white')
base_cmap = plt.get_cmap('Purples')

# 比如只用后 60% 的颜色（0.4~1.0 区间可以自己调）
colors = base_cmap(np.linspace(0.2, 0.9, 256))

# 生成一个新的 colormap
new_cmap = ListedColormap(colors)
contourf_cold = ax6.pcolormesh(lon_grid, lat_grid, deg_grid1, cmap=new_cmap, vmin=100,vmax=1500, transform=ccrs.PlateCarree())
cbar2 = fig.colorbar(contourf_cold, ax=ax6, orientation='horizontal', shrink=0.9, extend='both', aspect=28, pad=0.04)
cbar2.ax.tick_params(labelsize=26, width=2, length=6)
cbar2.set_label('Network I in-degree', fontsize=24, labelpad=5)
cbar2.outline.set_linewidth(2)  # 颜色条边框
gl = ax6.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.7, color='black', alpha=0.5, linestyle='--')
gl.top_labels = True  # 经度标签放在上方
gl.left_labels = True  # 纬度标签在左侧
gl.right_labels = False  # 隐藏右侧经度标签
gl.bottom_labels = False  # 隐藏底部经度标签
gl.xlabel_style = {'size': 25, 'color': 'black'}
gl.ylabel_style = {'size': 25, 'color': 'black'}
ax6.contourf(lon_grid, lat_grid, deg_grid1, levels=[cold_threshold_high, deg_grid1.max()],
              hatches=['\\\\'], alpha=0, transform=ccrs.PlateCarree())
# ax6.contourf(lon_grid, lat_grid, deg_grid1, levels=[deg_grid1.min(), cold_threshold_low],
#               hatches=['//'], alpha=0, transform=ccrs.PlateCarree())
#ax6.text(0.5, 1.05, 'Network II', ha='center', va='center', fontsize=40, transform=ax6.transAxes, fontweight='bold')

# 美化布局
plt.tight_layout()
plt.subplots_adjust(wspace=0.15)
ax6.set_title('b', fontsize=34, fontweight='bold', x=-0.01, pad=0)
# 保存为高质量图片（可选）
plt.savefig('F:/heatwave_chaos/Fig3_1.pdf', dpi=300, bbox_inches='tight')
plt.show()








from matplotlib.colors import ListedColormap

# Divergence_heat = chu_coldaa/np.max(chu_coldaa)
# Divergence_cold = ru_coldaa/np.max(ru_coldaa)
Divergence_heat = chu_coldaa
Divergence_cold = ru_coldaa
# 计算阈值
heat_threshold_high = np.percentile(Divergence_heat, 88)
heat_threshold_low = np.percentile(Divergence_heat, 10)
cold_threshold_high = np.percentile(Divergence_cold, 88)
cold_threshold_low = np.percentile(Divergence_cold, 10)

# 加载经纬度
lon = np.load("F:/heatwave/longitude.npy")
lat = np.load("F:/heatwave/latitude.npy")

# 创建网格
grid_lon = np.linspace(0, 360, 360)
grid_lat = np.linspace(-90, 90, 180)
lon_grid, lat_grid = np.meshgrid(grid_lon, grid_lat)

# 插值
deg_grid = griddata((lon, lat), Divergence_heat, (lon_grid, lat_grid), method='cubic')
deg_grid1 = griddata((lon, lat), Divergence_cold, (lon_grid, lat_grid), method='cubic')

# 创建图表：一行两列
fig = plt.figure(figsize=(20, 8), dpi=300, facecolor='white')
gs = gridspec.GridSpec(1, 2, wspace=0.2)  # 一行两列，调整间距

# 子图 1：Network I
ax5 = plt.subplot(gs[0, 0], projection=ccrs.Robinson(central_longitude=180))
ax5.coastlines(linewidth=1, color='black')
ax5.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='black')
ax5.add_feature(cfeature.OCEAN, facecolor='white')
base_cmap = plt.get_cmap('Reds')

# 比如只用后 60% 的颜色（0.4~1.0 区间可以自己调）
colors = base_cmap(np.linspace(0, 0.9, 256))

# 生成一个新的 colormap
new_cmap = ListedColormap(colors)
contourf_heat = ax5.pcolormesh(lon_grid, lat_grid, deg_grid, cmap=new_cmap,vmin=100,vmax=1000, transform=ccrs.PlateCarree())
cbar1 = fig.colorbar(contourf_heat, ax=ax5, orientation='horizontal', shrink=0.9, extend='both', aspect=28, pad=0.04)
cbar1.ax.tick_params(labelsize=26, width=2, length=6)
cbar1.set_label('Network II out-degree', fontsize=24, labelpad=5)
cbar1.outline.set_linewidth(2)  # 颜色条边框
gl = ax5.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.7, color='black', alpha=0.5, linestyle='--')
gl.top_labels = True  # 经度标签放在上方
gl.left_labels = True  # 纬度标签在左侧
gl.right_labels = False  # 隐藏右侧经度标签
gl.bottom_labels = False  # 隐藏底部经度标签
gl.xlabel_style = {'size':25, 'color': 'black'}
gl.ylabel_style = {'size': 25, 'color': 'black'}
# try:
#     gl.xpadding = 15
#     gl.ypadding = 15
# except AttributeError:
#     pass
ax5.contourf(lon_grid, lat_grid, deg_grid, levels=[heat_threshold_high, deg_grid.max()],
              hatches=['\\\\'], alpha=0, transform=ccrs.PlateCarree())
# ax5.contourf(lon_grid, lat_grid, deg_grid, levels=[deg_grid.min(), heat_threshold_low],
#               hatches=['//'], alpha=0, transform=ccrs.PlateCarree())
ax5.set_title('c', fontsize=34, fontweight='bold', x=-0.01, pad=1)
#ax5.text(0.5, 1.05, 'Network I', ha='center', va='center', fontsize=40, transform=ax5.transAxes, fontweight='bold')

# 子图 2：Network II
ax6 = plt.subplot(gs[0, 1], projection=ccrs.Robinson(central_longitude=180))
ax6.coastlines(linewidth=1, color='black')
ax6.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='black')
ax6.add_feature(cfeature.OCEAN, facecolor='white')
base_cmap = plt.get_cmap('Purples')

# 比如只用后 60% 的颜色（0.4~1.0 区间可以自己调）
colors = base_cmap(np.linspace(0.2, 0.9, 256))

# 生成一个新的 colormap
new_cmap = ListedColormap(colors)
contourf_cold = ax6.pcolormesh(lon_grid, lat_grid, deg_grid1, cmap=new_cmap,vmin=100,vmax=1000,  transform=ccrs.PlateCarree())
cbar2 = fig.colorbar(contourf_cold, ax=ax6, orientation='horizontal', shrink=0.9, extend='both', aspect=28, pad=0.04)
cbar2.ax.tick_params(labelsize=26, width=2, length=6)
cbar2.set_label('Network II in-degree', fontsize=24, labelpad=5)
cbar2.outline.set_linewidth(2)  # 颜色条边框
gl = ax6.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.7, color='black', alpha=0.5, linestyle='--')
gl.top_labels = True  # 经度标签放在上方
gl.left_labels = True  # 纬度标签在左侧
gl.right_labels = False  # 隐藏右侧经度标签
gl.bottom_labels = False  # 隐藏底部经度标签
gl.xlabel_style = {'size': 25, 'color': 'black'}
gl.ylabel_style = {'size': 25, 'color': 'black'}
ax6.contourf(lon_grid, lat_grid, deg_grid1, levels=[cold_threshold_high, deg_grid1.max()],
              hatches=['\\\\'], alpha=0, transform=ccrs.PlateCarree())
# ax6.contourf(lon_grid, lat_grid, deg_grid1, levels=[deg_grid1.min(), cold_threshold_low],
#               hatches=['//'], alpha=0, transform=ccrs.PlateCarree())
#ax6.text(0.5, 1.05, 'Network II', ha='center', va='center', fontsize=40, transform=ax6.transAxes, fontweight='bold')

# 美化布局
plt.tight_layout()
plt.subplots_adjust(wspace=0.15)
ax6.set_title('d', fontsize=34, fontweight='bold', x=-0.01, pad=0)
# 保存为高质量图片（可选）
plt.savefig('F:/heatwave_chaos/Fig3_2.pdf', dpi=300, bbox_inches='tight')
plt.show()






