# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 12:03:36 2025

@author: yuping
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.interpolate import griddata
#plt.rcParams['font.family'] = 'Arial'
#plt.rcParams['font.weight'] = 'bold'
# 加载数据
Divergence_heat = np.load("F:/heatwave/6242/最终版Fig/data/Divergence_heat.npy")
Divergence_cold = np.load("F:/heatwave/6242/最终版Fig/data/Divergence_cold.npy")
Divergence_heat = -Divergence_heat
Divergence_cold = -Divergence_cold

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
contourf_heat = ax5.pcolormesh(lon_grid, lat_grid, deg_grid, cmap='RdBu_r', vmin=-900, vmax=900, transform=ccrs.PlateCarree())
cbar1 = fig.colorbar(contourf_heat, ax=ax5, orientation='horizontal', shrink=0.9, extend='both', aspect=28, pad=0.04)
cbar1.ax.tick_params(labelsize=24, width=2, length=6)
cbar1.set_label('Network I Divergence', fontsize=22, labelpad=5)
cbar1.outline.set_linewidth(2)  # 颜色条边框
gl = ax5.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.7, color='black', alpha=0.5, linestyle='--')
gl.top_labels = True  # 经度标签放在上方
gl.left_labels = True  # 纬度标签在左侧
gl.right_labels = False  # 隐藏右侧经度标签
gl.bottom_labels = False  # 隐藏底部经度标签
gl.xlabel_style = {'size':23, 'color': 'black'}
gl.ylabel_style = {'size': 23, 'color': 'black'}
# try:
#     gl.xpadding = 15
#     gl.ypadding = 15
# except AttributeError:
#     pass
ax5.contourf(lon_grid, lat_grid, deg_grid, levels=[heat_threshold_high, deg_grid.max()],
              hatches=['\\'], alpha=0, transform=ccrs.PlateCarree())
ax5.contourf(lon_grid, lat_grid, deg_grid, levels=[deg_grid.min(), heat_threshold_low],
              hatches=['//'], alpha=0, transform=ccrs.PlateCarree())
ax5.set_title('a', fontsize=32, fontweight='bold', x=-0.01, pad=1)
#ax5.text(0.5, 1.05, 'Network I', ha='center', va='center', fontsize=40, transform=ax5.transAxes, fontweight='bold')

# 子图 2：Network II
ax6 = plt.subplot(gs[0, 1], projection=ccrs.Robinson(central_longitude=180))
ax6.coastlines(linewidth=1, color='black')
ax6.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='black')
ax6.add_feature(cfeature.OCEAN, facecolor='white')
contourf_cold = ax6.pcolormesh(lon_grid, lat_grid, deg_grid1, cmap='RdBu_r', vmin=-650, vmax=650, transform=ccrs.PlateCarree())
cbar2 = fig.colorbar(contourf_cold, ax=ax6, orientation='horizontal', shrink=0.9, extend='both', aspect=28, pad=0.04)
cbar2.ax.tick_params(labelsize=24, width=2, length=6)
cbar2.set_label('Network II Divergence', fontsize=22, labelpad=5)
cbar2.outline.set_linewidth(2)  # 颜色条边框
gl = ax6.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.7, color='black', alpha=0.5, linestyle='--')
gl.top_labels = True  # 经度标签放在上方
gl.left_labels = True  # 纬度标签在左侧
gl.right_labels = False  # 隐藏右侧经度标签
gl.bottom_labels = False  # 隐藏底部经度标签
gl.xlabel_style = {'size': 23, 'color': 'black'}
gl.ylabel_style = {'size': 23, 'color': 'black'}
ax6.contourf(lon_grid, lat_grid, deg_grid1, levels=[cold_threshold_high, deg_grid1.max()],
              hatches=['\\'], alpha=0, transform=ccrs.PlateCarree())
ax6.contourf(lon_grid, lat_grid, deg_grid1, levels=[deg_grid1.min(), cold_threshold_low],
              hatches=['//'], alpha=0, transform=ccrs.PlateCarree())
#ax6.text(0.5, 1.05, 'Network II', ha='center', va='center', fontsize=40, transform=ax6.transAxes, fontweight='bold')

# 美化布局
plt.tight_layout()
plt.subplots_adjust(wspace=0.15)
ax6.set_title('b', fontsize=32, fontweight='bold', x=-0.01, pad=0)
# 保存为高质量图片（可选）
plt.savefig('F:/heatwave_chaos/Fig3.pdf', dpi=300, bbox_inches='tight')
plt.show()




import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.ticker import MaxNLocator
from netCDF4 import Dataset

# =========================
# 0) 读散度（18048）
# =========================
Divergence_heat = np.load(r"F:/heatwave_chaos/审稿意见程序/建网结果/Net_I_div18048.npy")
Divergence_cold = np.load(r"F:/heatwave_chaos/审稿意见程序/建网结果/Net_II_div18048.npy")

# =========================
# 1) 读 lat/lon（94,192）
#    你随便选一年文件拿 lat/lon 就行
# =========================
nc_path = r"F:/heatwave/minT/air.2m.gauss.1951.nc"
ds = Dataset(nc_path, "r")
lon = np.array(ds.variables["lon"][:])   # (192,)
lat = np.array(ds.variables["lat"][:])   # (94,)
ds.close()

nlat, nlon = lat.size, lon.size
assert nlat * nlon == 18048, f"lat*lon={nlat*nlon} 与 18048 不一致，请检查 lat/lon 维度"

# =========================
# 2) 1D -> 2D
#    默认按 (lat, lon) 展开顺序重塑
# =========================
divH = Divergence_heat.reshape(nlat, nlon)
divC = Divergence_cold.reshape(nlat, nlon)

# 若你画出来发现经度方向错位/翻转，打开下面两行试一下：
# divH = divH[:, ::-1]
# divC = divC[:, ::-1]

# 构造网格
lon2d, lat2d = np.meshgrid(lon, lat)

# =========================
# 3) 颜色范围：用分位数裁剪更稳健
#    也可改成对称色标（看你散度有无正负）
# =========================
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.ticker import MaxNLocator

allv = np.concatenate([divH[np.isfinite(divH)], divC[np.isfinite(divC)]])
vmin = np.nanpercentile(allv, 1)
vmax = np.nanpercentile(allv, 99)

plt.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 14,
    "axes.titlesize": 15,
    "axes.linewidth": 1.0
})

proj = ccrs.PlateCarree(central_longitude=180)

def setup_ax(ax, extent=None):
    ax._autoscaleXon = getattr(ax, "_autoscaleXon", True)
    ax._autoscaleYon = getattr(ax, "_autoscaleYon", True)

    if extent is None:
        extent = [0, 360, -90, 90]
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.LAND, facecolor="#F2F2F2", zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.7, zorder=3)

    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=0.35, color="gray", alpha=0.5, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 12}
    gl.ylabel_style = {"size": 12}
    return gl

# ---- 关键：GridSpec 预留色条行 ----
fig = plt.figure(figsize=(14, 6), dpi=300)
gs = gridspec.GridSpec(
    nrows=2, ncols=2,
    height_ratios=[1.0, 0.08],   # 第二行专给 colorbar
    hspace=0.15, wspace=0.1
)

ax1 = fig.add_subplot(gs[0, 0], projection=proj)
ax2 = fig.add_subplot(gs[0, 1], projection=proj)
cax = fig.add_subplot(gs[1, :])   # 横跨两列的 colorbar 专用轴

setup_ax(ax1, extent=[0, 360, -90, 90])
im1 = ax1.pcolormesh(lon2d, lat2d, divH,
                     transform=ccrs.PlateCarree(),
                     shading="auto",
                     vmin=vmin, vmax=vmax,
                     cmap="RdBu_r")
ax1.set_title("Network I divergence")

setup_ax(ax2, extent=[0, 360, -90, 90])
im2 = ax2.pcolormesh(lon2d, lat2d, divC,
                     transform=ccrs.PlateCarree(),
                     shading="auto",
                     vmin=vmin, vmax=vmax,
                     cmap="RdBu_r")
ax2.set_title("Network II divergence")

# ---- colorbar 画到专用 cax 上，就不会压图 ----
cbar = fig.colorbar(im2, cax=cax, orientation="horizontal", extend="both")
cbar.set_label("Divergence", fontsize=14)
cbar.locator = MaxNLocator(nbins=6)
cbar.update_ticks()
cbar.ax.tick_params(labelsize=12, length=3)
pos = cax.get_position()
# 让色条更靠下、更短、更薄
cax.set_position([0.18, pos.y0 + 0.12, 0.64, pos.height * 0.6])
# 不要 tight_layout；GridSpec 已经控住布局了
plt.show()




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

