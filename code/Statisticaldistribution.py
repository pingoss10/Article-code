# -*- coding: utf-8 -*-
"""
Created on Sun Nov 30 22:39:19 2025

@author: yuping
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 17:08:03 2023

@author: 于平
"""


from pandas import DataFrame
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys 
import os
import netCDF4 as nc
import math
import datetime
import csv
from pandas import DataFrame
import pymannkendall as mk
test = pd.read_csv('F:/heatwave_chaos/审稿意见程序/allheat_events_yuzhi95.csv',header=None)
test=np.array(test)

#test1=np.array(test)[0,:]
totle=np.ones([9024*2,19])
for j in range(9024*2):
    print(j)
    test1=np.array(test)[j,:]
    sum=np.ones(19)
    for i in range(0,38,2):

        event=test1[i:i+5]
        sum[int(i/2)]=np.sum(event)
    totle[j,:]=sum


#计算斜率
MK_P=np.zeros(9024*2)
for h in range(9024*2):
    print(h)
    yi=mk.original_test(totle[h,:])[2]
    MK_P[h]=yi

heatPvalue=MK_P   

MK_P=np.zeros(9024*2)
for h in range(9024*2):
    print(h)
    yi=mk.original_test(totle[h,:])[7]
    MK_P[h]=yi

heatA=MK_P  


test = pd.read_csv('F:/heatwave_chaos/审稿意见程序/allcold_events_yuzhi95.csv',header=None)
test=np.array(test)

#test1=np.array(test)[0,:]
totle=np.ones([9024*2,19])
for j in range(9024*2):
    print(j)
    test1=np.array(test)[j,:]
    sum=np.ones(19)
    for i in range(0,38,2):

        event=test1[i:i+5]
        sum[int(i/2)]=np.sum(event)
    totle[j,:]=sum


#计算斜率
MK_P=np.zeros(9024*2)
for h in range(9024*2):
    print(h)
    yi=mk.original_test(totle[h,:])[2]
    MK_P[h]=yi

coldPvalue=MK_P   

MK_P=np.zeros(9024*2)
for h in range(9024*2):
    print(h)
    yi=mk.original_test(totle[h,:])[7]
    MK_P[h]=yi

coldA=MK_P  





#fig1为全球热浪事件，fig2为全球寒流事件
heat= pd.read_csv('F:/heatwave/worldtongbu/42yearworld/40worldheat_allheatevents.csv',header=None)
heat=np.array(heat)
heat=np.sum(heat,axis=1)
cold= pd.read_csv('F:/heatwave/worldtongbu/42yearworld/cold/40worldcold_allheatevents.csv',header=None)
cold=np.array(cold)
cold=np.sum(cold,axis=1)
fig1= np.reshape(heat,(47*2,192))
fig1 = np.ma.masked_invalid(fig1)
fig2 = np.reshape(cold,(47*2,192))
fig2 = np.ma.masked_invalid(fig2)

#fig4为heat_MK里面的A
fig3 = np.reshape(heatA,(47*2,192))
fig3 = np.ma.masked_invalid(fig3)
fig4 = np.reshape(coldA,(47*2,192))
fig4 = np.ma.masked_invalid(fig4)

heatPvalue1= np.reshape(heatPvalue,(47*2,192))
heatPvalue1 = np.ma.masked_invalid(heatPvalue1)
coldPvalue1= np.reshape(coldPvalue,(47*2,192))
coldPvalue1 = np.ma.masked_invalid(coldPvalue1)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

# 假设 coff1, coff2, coff3, coff4 是预先定义的不同的数据数组
# coff1, coff2, coff3, coff4 = ...

lon11 = np.zeros(shape=(192, 47*2))
lat11 = np.zeros(shape=(192, 47*2))

lon1 = np.linspace(0, 358.125, 192)
lat1 = np.linspace(88.542, -88.542, 47*2)

for i in range(47*2):
    lon11[:, i] = lon1
for i in range(192):
    lat11[i, :] = lat1

# # 创建子图
# fig, axs = plt.subplots(2, 2, figsize=(20, 10))

# # 自定义子图
# for i, ax in enumerate(axs.flatten()):
#     ax.set_adjustable('datalim')
#     # ax.set_global()

#     # Basemap 设置
#     mp = Basemap(ax=ax, resolution='c', projection='robin', lat_0=0, lon_0=-180)
#     coasts = mp.drawcoastlines(zorder=2, color='white', linewidth=0.3)
#     coasts_paths = coasts.get_paths()
#     ipolygons = list(range(83)) + [84]
#     mp.drawparallels(np.arange(-90.,91.,30.),labels=[True,False,False,True],linewidth=0.2)
#     mp.drawmeridians(np.arange(-180,180,60.),labels=[True,False,False,True],linewidth=0.2)
#     for ipoly in range(len(coasts_paths)):
#         r = coasts_paths[ipoly]
#         polygon_vertices = [(vertex[0], vertex[1]) for (vertex, code) in r.iter_segments(simplify=False)]
#         px = [polygon_vertices[j][0] for j in range(len(polygon_vertices))]
#         py = [polygon_vertices[j][1] for j in range(len(polygon_vertices))]

#         if ipoly in ipolygons:
#             mp.plot(px, py, linewidth=0.6, zorder=3, color='black')
#         else:
#             mp.plot(px, py, linewidth=0.6, zorder=4, color='grey')
        
#     # 使用不同的数据数组绘制数据
#     lon1111, lat1111 = mp(lon11.T, lat11.T)

#     if i == 0:
#         data_array = fig1
#         ticks = [0, 40,80,120]
#         pcm = mp.pcolor(lon1111, lat1111, data_array, cmap=plt.cm.coolwarm, vmax=120, vmin=0, snap=True)
#         cbar = mp.colorbar(pcm, ax=ax, ticks=ticks, pad=0.1, size=0.11, extend='both')
#         ax.set_title('(a)', fontsize=30, x=0.05)
#     elif i == 1:
#         data_array = fig2
#         ticks =[0, 40,80,120]
#         pcm = mp.pcolor(lon1111, lat1111, data_array, cmap=plt.cm.coolwarm, vmax=120, vmin=0, snap=True)
#         cbar = mp.colorbar(pcm, ax=ax, ticks=ticks, pad=0.1, size=0.11, extend='both')
#         ax.set_title('(b)', fontsize=30, x=0.05)
#     elif i == 2:
#         data_array = fig3*10
#         ticks = [-4, 4]
#         pcm = mp.pcolor(lon1111, lat1111, data_array, cmap=plt.cm.coolwarm, vmax=5, vmin=-5, snap=True)
#         cbar = mp.colorbar(pcm, ax=ax, ticks=ticks, pad=0.1, size=0.11, extend='both')
#         # pvalue=heatA1
#         # pvalue1 = np.reshape(pvalue,(94,192))
#         for i in range (94):
#             print(i)
#             for j in range(192):
#                 if heatPvalue1[i,j]<0.05:
#                     mp.scatter(lon1111[i,j],lat1111[i,i],s=1,marker='o',c='k',zorder=3, alpha=0.5)
                    
#         ax.set_title('(c)', fontsize=30, x=0.05)
#     else:
#         data_array = fig4*10
#         ticks = [-4, 4]
#         pcm = mp.pcolor(lon1111, lat1111, data_array, cmap=plt.cm.coolwarm, vmax=5, vmin=-5, snap=True)
#         cbar = mp.colorbar(pcm, ax=ax, ticks=ticks, pad=0.1, size=0.11, extend='both')
#         # pvalue=coldA1
#         # pvalue1 = np.reshape(pvalue,(94,192))
#         for i in range (94):
#             print(i)
#             for j in range(192):
#                 if coldPvalue1[i,j]<0.05:
#                     mp.scatter(lon1111[i,j],lat1111[i,i],s=1,marker='o',c='k',zorder=3, alpha=0.5)

#         ax.set_title('(d)', fontsize=30, x=0.05)
#     # 标题


# # 调整布局，使得 colorbar 不会重叠
# plt.tight_layout()

# # 显示图形
# plt.show()





import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['font.size'] = 15

# 画布比例适中（16:9 更协调）
fig, axs = plt.subplots(2, 2, figsize=(14, 8), dpi=300)
plt.subplots_adjust(wspace=-0.08, hspace=-0.5)

# ==== 数据配置 ====
data_list = [fig1, fig2, fig3 * 10, fig4 * 10]
titles = ['a', 'b', 'c', 'd']
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def truncate_colormap(cmap_name, minval=0.0, maxval=1.0, n=256):
    """截取 colormap 的一部分"""
    cmap = plt.get_cmap(cmap_name)
    new_colors = cmap(np.linspace(minval, maxval, n))
    return mcolors.LinearSegmentedColormap.from_list(
        f"trunc({cmap_name},{minval:.2f},{maxval:.2f})", new_colors)

# 例如：
cmaps = [
    truncate_colormap('Oranges', 0, 0.6),  # 取后半段（更深的橙红）
    truncate_colormap('Blues', 0, 0.6),
    truncate_colormap('RdGy_r', 0.1, 0.8),     # 只取红色部分（正异常）
    truncate_colormap('RdGy_r', 0.1, 0.8)      # 只取蓝色部分（负异常）
]
vmins = [0, 0, -4, -4]
vmaxs = [120, 90, 4, 4]
ticks_list = [[0, 40, 80, 120],
              [0, 30, 60, 90],
              [-4, -2,0,2, 4],
              [-4, -2,0,2, 4]]

colorbar_labels = [
    'Number of heatwaves',
    'Number of cold waves',
    'Changes per decade',
    'Changes per decade'
]

# ==== 子图循环 ====
for idx, ax in enumerate(axs.flatten()):
    ax.set_facecolor('#F7F7F7')

    mp = Basemap(ax=ax, projection='robin', lon_0=-180, resolution='c')
    mp.drawmapboundary(fill_color='white', linewidth=0.8)
    mp.drawcoastlines(linewidth=0.4, color='black')
    mp.drawparallels(np.arange(-60, 91, 30), linewidth=0.2, dashes=[1, 1], labels=[1, 0, 0, 0], fontsize=20)
    mp.drawmeridians(np.arange(-180, 181, 90), linewidth=0.2, dashes=[1, 1], labels=[0, 0, 0, 1], fontsize=20)

    lon_proj, lat_proj = mp(lon11.T, lat11.T)
    data = data_list[idx]

    pcm = mp.pcolormesh(lon_proj, lat_proj, data,
                        cmap=cmaps[idx],
                        vmin=vmins[idx],
                        vmax=vmaxs[idx],
                        shading='auto')

    # colorbar 美化：竖直、靠右、紧贴
    cbar = plt.colorbar(pcm, ax=ax, orientation='vertical',
                        pad=0.02, shrink=0.6,extend='both')
    cbar.ax.tick_params(labelsize=20, width=0.8)
    cbar.outline.set_linewidth(0.6)
    cbar.set_ticks(ticks_list[idx])
    cbar.set_label(colorbar_labels[idx], fontsize=18, labelpad=8)
    # 子图标题
    ax.text(-0.1, 1.12, titles[idx],
            transform=ax.transAxes,
            fontsize=30, fontweight='bold', ha='left', va='top')
    
    # 统计显著性标记（示例）
    if idx == 2:
        for i in range(94):
            print(i)
            for j in range(192):
                if heatPvalue1[i, j] < 0.05:
                    mp.scatter(lon1111[i, j], lat1111[i, j], s=0.5,
                                marker='o', c='k', zorder=5, alpha=0.3)
    if idx == 3:
        for i in range(94):
            print(i)
            for j in range(192):
                if coldPvalue1[i, j] < 0.05:
                    mp.scatter(lon1111[i, j], lat1111[i, j], s=0.5,
                                marker='o', c='k', zorder=5, alpha=0.3)

# ==== 版面整体微调 ====
plt.tight_layout(pad=1.0)
plt.savefig('F:/heatwave_chaos/Fig1.pdf', format='pdf', bbox_inches='tight')
plt.show()







import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import matplotlib.colors as mcolors

# =========================
# 0) 读取热浪/寒潮总数并reshape
# =========================
# heat_path = r'F:/heatwave_chaos/审稿意见程序/11allheat_events_yuzhi95.csv'
# cold_path = r'F:/heatwave_chaos/审稿意见程序/allcold_events_yuzhi5.csv'
heat_path = r'F:/heatwave_chaos/审稿意见程序/阈值2day_4day建网结果/allheat_events_yuzhi90.csv'
cold_path = r'F:/heatwave_chaos/审稿意见程序/阈值2day_4day建网结果/allcold_events_yuzhi10.csv'

heat = pd.read_csv(heat_path, header=None).values
cold = pd.read_csv(cold_path, header=None).values

# 你原来的逻辑：对每个格点沿时间/事件维度求和，得到每格点总事件数
heat_sum = np.nansum(heat, axis=1)   # (9024,)
cold_sum = np.nansum(cold, axis=1)   # (9024,)

# 网格：你原来是 (47*2, 192) = (94, 192) 共 18048?
# 但你这里 reshape 用的是 (47*2, 192) -> 94*192=18048
# 而 heat_sum 是 9024，说明你之前是 heat_sum = sum(axis=1) 后 reshape(94,192) 前，
# heat 的行数应为 18048 才对。
# 你给的程序里是：fig1=reshape(heat,(47*2,192))，表示 heat_sum 的长度应当是 18048。
# 因此这里做一个兼容：若长度是 9024，则尝试按 (47,192)；若是 18048，则 (94,192)。

def reshape_to_map(arr, nlon=192):
    L = arr.shape[0]
    if L % nlon != 0:
        raise ValueError(f"无法reshape：长度 {L} 不能被 nlon={nlon} 整除。请检查CSV维度。")
    nlat = L // nlon
    return np.reshape(arr, (nlat, nlon))

fig_heat = np.ma.masked_invalid(reshape_to_map(heat_sum, nlon=192))
fig_cold = np.ma.masked_invalid(reshape_to_map(cold_sum, nlon=192))

nlat, nlon = fig_heat.shape
assert fig_cold.shape == (nlat, nlon), "热浪与寒潮网格shape不一致，请检查输入CSV。"

# =========================
# 1) 构造经纬度网格（沿用你之前写法）
# =========================
lon1 = np.linspace(0, 358.125, nlon)
# 你之前 lat1 = np.linspace(88.542, -88.542, 47*2)
# 这里根据nlat自适应，保证与你reshape一致
lat1 = np.linspace(88.542, -88.542, nlat)

lon11 = np.zeros((nlon, nlat))
lat11 = np.zeros((nlon, nlat))
for i in range(nlat):
    lon11[:, i] = lon1
for j in range(nlon):
    lat11[j, :] = lat1

# =========================
# 2) colormap（稍微美化：截取更“Nature风”的深色段）
# =========================
def truncate_colormap(cmap_name, minval=0.0, maxval=1.0, n=256):
    cmap = plt.get_cmap(cmap_name)
    new_colors = cmap(np.linspace(minval, maxval, n))
    return mcolors.LinearSegmentedColormap.from_list(
        f"trunc({cmap_name},{minval:.2f},{maxval:.2f})", new_colors
    )

cmap_heat = truncate_colormap('RdYlBu_r', 0.10, 0.85)
cmap_cold = truncate_colormap('RdYlBu_r',   0.10, 0.85)

# =========================
# 3) 画图：只画数目（2×1）
# =========================
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['font.size'] = 16

fig, axs = plt.subplots(2, 1, figsize=(12, 10), dpi=300)
plt.subplots_adjust(hspace=0.12)

data_list = [fig_heat, fig_cold]
cmaps = [cmap_heat, cmap_cold]
titles = ['a Number of heatwaves', 'b Number of cold waves']

# 你原来给的范围：heat 0~120, cold 0~90（可按需要改）
vmins = [0, 0]
vmaxs = [70, 70]
ticks_list = [[0, 30, 70],
              [0, 30, 70]]
cbar_labels = ['Number of heatwaves', 'Number of cold waves']

for idx, ax in enumerate(axs):
    ax.set_facecolor('#F7F7F7')

    mp = Basemap(ax=ax, projection='robin', lon_0=-180, resolution='c')
    mp.drawmapboundary(fill_color='white', linewidth=0.8)
    mp.drawcoastlines(linewidth=0.4, color='black')

    # 网格线与标签（你原来设置较大字体，这里保留但稍微收敛）
    mp.drawparallels(np.arange(-60, 91, 30), linewidth=0.25, dashes=[1, 1],
                     labels=[1, 0, 0, 0], fontsize=14)
    mp.drawmeridians(np.arange(-180, 181, 90), linewidth=0.25, dashes=[1, 1],
                     labels=[0, 0, 0, 1], fontsize=14)

    lon_proj, lat_proj = mp(lon11.T, lat11.T)
    data = data_list[idx]

    pcm = mp.pcolormesh(lon_proj, lat_proj, data,
                        cmap=cmaps[idx],
                        vmin=vmins[idx],
                        vmax=vmaxs[idx],
                        shading='auto')

    cbar = plt.colorbar(pcm, ax=ax, orientation='vertical',
                        pad=0.02, shrink=0.72, extend='both')
    cbar.set_ticks(ticks_list[idx])
    cbar.set_label(cbar_labels[idx], fontsize=15, labelpad=10)
    cbar.ax.tick_params(labelsize=13, width=0.8)
    cbar.outline.set_linewidth(0.6)

    ax.set_title(titles[idx], fontsize=25, loc='left', pad=6)

plt.tight_layout(pad=0.8)
# out_pdf = r'F:/heatwave_chaos/Fig_counts_only.pdf'
# plt.savefig(out_pdf, format='pdf', bbox_inches='tight')
plt.show()

# print("Saved:", out_pdf)



import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.interpolate import griddata
#plt.rcParams['font.family'] = 'Arial'
#plt.rcParams['font.weight'] = 'bold'
# 加载数据
# Divergence_heat = np.load("F:/heatwave_chaos/审稿意见程序/阈值TN5_TX95建网结果/Net_I_div6242.npy")
# Divergence_cold = np.load("F:/heatwave_chaos/审稿意见程序/阈值TN5_TX95建网结果/Net_II_div6242.npy")
Divergence_heat = np.load("F:/heatwave_chaos/审稿意见程序/阈值2day_4day建网结果/Net_I_div6242.npy")
Divergence_cold = np.load("F:/heatwave_chaos/审稿意见程序/阈值2day_4day建网结果/Net_II_div6242.npy")
# Divergence_heat = -Divergence_heat
# Divergence_cold = -Divergence_cold

# 计算阈值
# heat_threshold_high = np.percentile(Divergence_heat, 90)
# heat_threshold_low = np.percentile(Divergence_heat, 10)
# cold_threshold_high = np.percentile(Divergence_cold, 90)
# cold_threshold_low = np.percentile(Divergence_cold, 10)

# 加载经纬度
lon = np.load("F:/heatwave/longitude.npy")
lat = np.load("F:/heatwave/latitude.npy")

# 创建网格
grid_lon = np.linspace(0, 360, 360)
grid_lat = np.linspace(-90, 90, 90)
lon_grid, lat_grid = np.meshgrid(grid_lon, grid_lat)

# 插值
deg_grid = griddata((lon, lat), Divergence_heat, (lon_grid, lat_grid), method='cubic')
deg_grid1 = griddata((lon, lat), Divergence_cold, (lon_grid, lat_grid), method='cubic')

# # 创建图表：一行两列
# fig = plt.figure(figsize=(20, 8), dpi=300, facecolor='white')
# gs = gridspec.GridSpec(1, 2, wspace=0.2)  # 一行两列，调整间距

# # 子图 1：Network I
# ax5 = plt.subplot(gs[0, 0], projection=ccrs.Robinson(central_longitude=180))
# ax5.coastlines(linewidth=1, color='black')
# ax5.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='black')
# ax5.add_feature(cfeature.OCEAN, facecolor='white')
# contourf_heat = ax5.pcolormesh(lon_grid, lat_grid, deg_grid, cmap='RdBu_r', vmin=-900, vmax=900, transform=ccrs.PlateCarree())
# cbar1 = fig.colorbar(contourf_heat, ax=ax5, orientation='horizontal', shrink=0.9, extend='both', aspect=28, pad=0.04)
# cbar1.ax.tick_params(labelsize=24, width=2, length=6)
# cbar1.set_label('Network I Divergence', fontsize=22, labelpad=5)
# cbar1.outline.set_linewidth(2)  # 颜色条边框
# gl = ax5.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.7, color='black', alpha=0.5, linestyle='--')
# gl.top_labels = True  # 经度标签放在上方
# gl.left_labels = True  # 纬度标签在左侧
# gl.right_labels = False  # 隐藏右侧经度标签
# gl.bottom_labels = False  # 隐藏底部经度标签
# gl.xlabel_style = {'size':23, 'color': 'black'}
# gl.ylabel_style = {'size': 23, 'color': 'black'}
# # try:
# #     gl.xpadding = 15
# #     gl.ypadding = 15
# # except AttributeError:
# #     pass
# # ax5.contourf(lon_grid, lat_grid, deg_grid, levels=[heat_threshold_high, deg_grid.max()],
# #               hatches=['\\'], alpha=0, transform=ccrs.PlateCarree())
# # ax5.contourf(lon_grid, lat_grid, deg_grid, levels=[deg_grid.min(), heat_threshold_low],
# #               hatches=['//'], alpha=0, transform=ccrs.PlateCarree())
# ax5.set_title('a', fontsize=32, fontweight='bold', x=-0.01, pad=1)
# #ax5.text(0.5, 1.05, 'Network I', ha='center', va='center', fontsize=40, transform=ax5.transAxes, fontweight='bold')

# # 子图 2：Network II
# ax6 = plt.subplot(gs[0, 1], projection=ccrs.Robinson(central_longitude=180))
# ax6.coastlines(linewidth=1, color='black')
# ax6.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='black')
# ax6.add_feature(cfeature.OCEAN, facecolor='white')
# contourf_cold = ax6.pcolormesh(lon_grid, lat_grid, deg_grid1, cmap='RdBu_r', vmin=-650, vmax=650, transform=ccrs.PlateCarree())
# cbar2 = fig.colorbar(contourf_cold, ax=ax6, orientation='horizontal', shrink=0.9, extend='both', aspect=28, pad=0.04)
# cbar2.ax.tick_params(labelsize=24, width=2, length=6)
# cbar2.set_label('Network II Divergence', fontsize=22, labelpad=5)
# cbar2.outline.set_linewidth(2)  # 颜色条边框
# gl = ax6.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.7, color='black', alpha=0.5, linestyle='--')
# gl.top_labels = True  # 经度标签放在上方
# gl.left_labels = True  # 纬度标签在左侧
# gl.right_labels = False  # 隐藏右侧经度标签
# gl.bottom_labels = False  # 隐藏底部经度标签
# gl.xlabel_style = {'size': 23, 'color': 'black'}
# gl.ylabel_style = {'size': 23, 'color': 'black'}
# # ax6.contourf(lon_grid, lat_grid, deg_grid1, levels=[cold_threshold_high, deg_grid1.max()],
# #               hatches=['\\'], alpha=0, transform=ccrs.PlateCarree())
# # ax6.contourf(lon_grid, lat_grid, deg_grid1, levels=[deg_grid1.min(), cold_threshold_low],
# #               hatches=['//'], alpha=0, transform=ccrs.PlateCarree())
# #ax6.text(0.5, 1.05, 'Network II', ha='center', va='center', fontsize=40, transform=ax6.transAxes, fontweight='bold')

# # 美化布局
# plt.tight_layout()
# plt.subplots_adjust(wspace=0.15)
# ax6.set_title('b', fontsize=32, fontweight='bold', x=-0.01, pad=0)
# # 保存为高质量图片（可选）
# #plt.savefig('F:/heatwave_chaos/Fig3.pdf', dpi=300, bbox_inches='tight')
# plt.show()




import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# =========================
# 关键热补丁：修复 GeoAxes 缺少 _autoscaleXon/_autoscaleYon
# 必须对每个 cartopy GeoAxes 调用一次
# =========================
def fix_cartopy_autoscale(ax):
    if not hasattr(ax, "_autoscaleXon"):
        ax._autoscaleXon = True
    if not hasattr(ax, "_autoscaleYon"):
        ax._autoscaleYon = True
    return ax

fig = plt.figure(figsize=(20, 8), dpi=300, facecolor='white')
gs = gridspec.GridSpec(1, 2, wspace=0.2)

# -------------------------
# 子图 1：Network I
# -------------------------
ax5 = plt.subplot(gs[0, 0], projection=ccrs.Robinson(central_longitude=180))
fix_cartopy_autoscale(ax5)  # ✅ 必须加在最前面

ax5.coastlines(linewidth=1, color='black')
ax5.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='black')
ax5.add_feature(cfeature.OCEAN, facecolor='white')

contourf_heat = ax5.pcolormesh(
    lon_grid, lat_grid, deg_grid,
    cmap='RdBu_r', vmin=-800, vmax=800,
    transform=ccrs.PlateCarree()
)

cbar1 = fig.colorbar(
    contourf_heat, ax=ax5,
    orientation='horizontal', shrink=0.9,
    extend='both', aspect=28, pad=0.04
)
cbar1.ax.tick_params(labelsize=24, width=2, length=6)
cbar1.set_label('Network I Divergence', fontsize=22, labelpad=5)
cbar1.outline.set_linewidth(2)

gl1 = ax5.gridlines(
    crs=ccrs.PlateCarree(), draw_labels=True,
    linewidth=0.7, color='black', alpha=0.5, linestyle='--'
)
gl1.top_labels = True
gl1.left_labels = True
gl1.right_labels = False
gl1.bottom_labels = False
gl1.xlabel_style = {'size': 23, 'color': 'black'}
gl1.ylabel_style = {'size': 23, 'color': 'black'}

ax5.set_title('a', fontsize=32, fontweight='bold', x=-0.01, pad=1)

# -------------------------
# 子图 2：Network II
# -------------------------
ax6 = plt.subplot(gs[0, 1], projection=ccrs.Robinson(central_longitude=180))
fix_cartopy_autoscale(ax6)  # ✅ 同样必须加

ax6.coastlines(linewidth=1, color='black')
ax6.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='black')
ax6.add_feature(cfeature.OCEAN, facecolor='white')

contourf_cold = ax6.pcolormesh(
    lon_grid, lat_grid, deg_grid1,
    cmap='RdBu_r', vmin=-650, vmax=650,
    transform=ccrs.PlateCarree()
)

cbar2 = fig.colorbar(
    contourf_cold, ax=ax6,
    orientation='horizontal', shrink=0.9,
    extend='both', aspect=28, pad=0.04
)
cbar2.ax.tick_params(labelsize=24, width=2, length=6)
cbar2.set_label('Network II Divergence', fontsize=22, labelpad=5)
cbar2.outline.set_linewidth(2)

gl2 = ax6.gridlines(
    crs=ccrs.PlateCarree(), draw_labels=True,
    linewidth=0.7, color='black', alpha=0.5, linestyle='--'
)
gl2.top_labels = True
gl2.left_labels = True
gl2.right_labels = False
gl2.bottom_labels = False
gl2.xlabel_style = {'size': 23, 'color': 'black'}
gl2.ylabel_style = {'size': 23, 'color': 'black'}

ax6.set_title('b', fontsize=32, fontweight='bold', x=-0.01, pad=0)

# -------------------------
# 布局（tight_layout 在某些版本 cartopy 仍可能触发 extent 相关问题）
# 建议：只用 subplots_adjust，不用 tight_layout
# -------------------------
plt.subplots_adjust(wspace=0.15, left=0.02, right=0.98, top=0.95, bottom=0.08)

plt.show()


