# -*- coding: utf-8 -*-
"""
Created on Mon Dec  1 20:12:22 2025

@author: yuping
"""



import numpy as np
from scipy import stats
from scipy.linalg import svd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ===== 读取经纬度 =====

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import Normalize
import pandas as pd
from scipy.interpolate import griddata
from scipy.linalg import svd
from netCDF4 import Dataset

# 以下部分假设你已经加载了数据，并做了处理，省略了文件路径和数据加载

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

nodes, lat, lon = selection()

heat= pd.read_csv('F:/heatwave/worldtongbu/42yearworld/40worldheat_allheatevents.csv',header=None)
heat=np.array(heat)
heat=heat[nodes]

cold= pd.read_csv('F:/heatwave/worldtongbu/42yearworld/cold/40worldcold_allheatevents.csv',header=None)
cold=np.array(cold)
cold=cold[nodes]


avgtem_summer=np.load('F:/heatwave_chaos/summer_mean_1981_2020.npy')
avgtem_summer=avgtem_summer[nodes]
avgtem_winter= np.load('F:/heatwave_chaos/winter_mean_1981_2020.npy')
avgtem_winter=avgtem_winter[nodes]
lon = np.load("F:/heatwave/longitude.npy")
lat = np.load("F:/heatwave/latitude.npy")

heat1 = heat.astype(np.float32)
cold1 = cold.astype(np.float32)
tsum1 = avgtem_summer.astype(np.float32)
twin1 = avgtem_winter.astype(np.float32)

n_grid, n_year = heat1.shape
time = np.arange(n_year)

# ===== 去趋势函数 =====
def detrend_field(data):
    detrended = np.empty_like(data)
    for i in range(n_grid):
        y = data[i, :]
        if np.all(np.isnan(y)):
            detrended[i, :] = np.nan
        else:
            slope, intercept, _, _, _ = stats.linregress(time, y)
            detrended[i, :] = y - (slope * time + intercept)
    return detrended

heat = detrend_field(heat1)
cold = detrend_field(cold1)
tsum = detrend_field(tsum1)
twin = detrend_field(twin1)

# ==================================================
# 去掉 ENSO 区域：纬度 [-10,10]，经度 [190,230]
# ==================================================
#mask = ~((lat >= 30) & (lat <= 70) & (lon >= 2) & (lon <= 135))
mask = ~((lat >= -15) & (lat <= 15) & (lon >= 190) & (lon <= 280))
valid_indices = np.where(mask)[0]

# 对每个变量只保留非 ENSO 区域的格点
heat = heat[valid_indices, :]
cold = cold[valid_indices, :]
tsum = tsum[valid_indices, :]
twin = twin[valid_indices, :]
lon = lon[valid_indices]
lat = lat[valid_indices]

print(f"去除 ENSO 区域后保留格点数: {len(valid_indices)}")

# ==================================================
# 做 anomaly、归一化、拼接联合场
# ==================================================

def make_anomaly(x):
    x_ts = x.T
    return x_ts - x_ts.mean(axis=0, keepdims=True)

def normalize_field(x):
    std = x.std()
    return x / std if std != 0 else x

heat_n = normalize_field(make_anomaly(heat))
cold_n = normalize_field(make_anomaly(cold))
tsum_n = normalize_field(make_anomaly(tsum))
twin_n = normalize_field(make_anomaly(twin))

# 拼接联合场
X = np.concatenate([heat_n, cold_n, tsum_n, twin_n], axis=1)
ntime, nspace_total = X.shape
print("联合场维度:", X.shape)

# ==================================================
# EOF 分解
# ==================================================
def eof_analysis_fast(data, neofs=5):
    X = data - data.mean(axis=0, keepdims=True)
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    varfrac = (s**2) / np.sum(s**2)
    pcs  = U[:, :neofs] * s[:neofs]
    eofs = Vt[:neofs, :].T
    return eofs, pcs, varfrac[:neofs]

neofs = 5
eofs, pcs, varfrac = eof_analysis_fast(X, neofs=neofs)

print("前 {} 个模态方差贡献率:".format(neofs))
for i in range(neofs):
    print(f"Mode {i+1}: {varfrac[i]*100:.2f}%")

# ==================================================
# 拆分回各变量
# ==================================================
npoint = len(valid_indices)
mode = 0

eof_all = eofs[:, mode]
eof_heat = eof_all[0*npoint:1*npoint]
eof_cold = eof_all[1*npoint:2*npoint]
eof_tsum = eof_all[2*npoint:3*npoint]
eof_twin = eof_all[3*npoint:4*npoint]
pc1 = pcs[:, mode]

print("Mode 1 explains: {:.2f}% variance".format(varfrac[0]*100))


import pandas as pd



df = pd.read_csv("F:/heatwave_chaos/nina34.anom(time).csv", header=None, names=["Date", "ONI"])

# 2️⃣ 转换日期格式
df["Date"] = pd.to_datetime(df["Date"])

# 3️⃣ 提取年份和月份
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month

# 4️⃣ 筛选出 1981–2020 年的夏季（5–9月）
summer_df = df[(df["Year"] >= 1981) & (df["Year"] <= 2020) & (df["Month"].between(5, 9))]

# 5️⃣ 按年份计算夏季平均 ONI
summer_mean = summer_df.groupby("Year")["ONI"].mean().reset_index()
summer_mean=np.array(summer_mean)
summer_mean=summer_mean[:,1]
# 6️⃣ 输出结果
print(summer_mean)

# 7️⃣ （可选）保存结果
#np.save("F:/heatwave_chaos/nina34.summer5-9.npy",summer_mean)

def standardize(arr):
    """对数组做 Z-score 标准化，忽略 NaN"""
    mean = np.nanmean(arr)
    std = np.nanstd(arr)
    if std == 0 or np.isnan(std):
        std = 1.0
    return (arr - mean) / std

# 分别标准化
summer_mean1 = standardize(summer_mean)
pc1_std = standardize(pc1)

plt.figure(figsize=(8, 3.8), dpi=300)

# ===== 基础曲线 =====
# ==== PC1（主线）====
plt.plot(np.arange(nyear) + 1981, pc1_std, 
         color='#E63946', linewidth=3.2, marker='o', markersize=6,
         markerfacecolor='white', markeredgewidth=1.5,
         label='PC1', zorder=10)

# ==== 其他变量 ====
plt.plot(np.arange(nyear) + 1981, summer_mean1,
         color='#457B9D', linewidth=2.0, alpha=0.8, marker='o', markersize=5,
         markerfacecolor='white', markeredgewidth=1.0,
         label='ONI5-9')

# plt.plot(np.arange(nyear) + 1981, shuju2_std,
#          color='#A8DADC', linewidth=2.0, alpha=0.8, marker='o', markersize=5,
#          markerfacecolor='white', markeredgewidth=1.0,
#          label='CWN r=0.818')


# ===== 零线 =====
plt.axhline(0, color='gray', linewidth=1.2, linestyle='--', alpha=0.6, zorder=0)

# ===== 坐标轴样式 =====
plt.xlabel("Year", fontsize=16, fontweight='bold', labelpad=6)
plt.ylabel("PC1", fontsize=16, fontweight='bold', labelpad=6)

#plt.xlim(1981, 1980 + nyear)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# ===== 网格线 =====
plt.grid(True, linestyle=':', linewidth=0.8, alpha=0.4)

# ===== 去除右上边框 =====
for spine in ['right', 'top']:
    plt.gca().spines[spine].set_visible(False)
plt.gca().spines['left'].set_linewidth(1.2)
plt.gca().spines['bottom'].set_linewidth(1.2)

# ===== 增强整体配色感 =====
plt.gca().set_facecolor('#FAFAFA')  # 柔和背景
plt.legend(frameon=False, fontsize=13, loc='upper left')


def corr(x, y):
    """忽略 NaN 的 Pearson 相关系数"""
    mask = ~np.isnan(x) & ~np.isnan(y)
    if np.sum(mask) == 0:
        return np.nan
    return np.corrcoef(x[mask], y[mask])[0, 1]

r1 = corr(pc1_std,  summer_mean1)

print("=== 相关系数 (PC1 vs 各变量) ===")
print(f"PC1 vs HWN  : {r1:.3f}")
# ===== 调整边距并展示 =====
plt.tight_layout(pad=1)
plt.show()












# ######################和南极涛动指数

# # 查看前几行
# with open("F:/heatwave_chaos/SAMI (AAOI) 1948-present.ASCII", "r", encoding="utf-8", errors="ignore") as f:
#     for i in range(10):   # 打印前10行
#         print(f.readline().strip())

# import numpy as np
# import pandas as pd

# # 1️⃣ 读取文件（跳过前两行说明）
# file_path = "F:/heatwave_chaos/SAMI (AAOI) 1948-present.ASCII"

# df = pd.read_csv(file_path, 
#                  delim_whitespace=True, 
#                  skiprows=2, 
#                  na_values=-99.99)

# # 2️⃣ 补全列名
# df.columns = ['Year', 1,2,3,4,5,6,7,8,9,10,11,12,
#               'DJF','MAM','JJA','SON','ANN','DJFM']

# # 3️⃣ 只取 1981–2020 年段（含1981冬到2020冬）
# df = df[(df['Year'] >= 1980) & (df['Year'] <= 2021)]

# # 4️⃣ 计算冬季（NDJFM：11,12,1,2,3），冬季以起始年命名
# winter_means = []
# for year in range(1981, 2021):  # 冬季1981–2020
#     # 当前年 (year) 的11、12月
#     this_year = df[df['Year'] == year]
#     # 下一年 (year+1) 的1–3月
#     next_year = df[df['Year'] == year + 1]

#     if len(this_year) and len(next_year):
#         months = np.concatenate([
#             this_year[[11, 12]].values.flatten(),
#             next_year[[1, 2, 3]].values.flatten()
#         ])
#         winter_means.append([year, np.nanmean(months)])

# winter_df = pd.DataFrame(winter_means, columns=['Winter_Year', 'AAO_winter_mean'])
# winter_df=np.array(winter_df)
# winter_df=winter_df[:,1]




# import numpy as np
# import pandas as pd
# with open("F:/heatwave_chaos/SAMI (AAOI) 1948-present.ASCII", "r", encoding="utf-8", errors="ignore") as f:
#     for i in range(10):   # 打印前10行
#         print(f.readline().strip())

# # 1️⃣ 读取文件（跳过前两行说明）
# file_path = "F:/heatwave_chaos/SAMI (AAOI) 1948-present.ASCII"

# df = pd.read_csv(file_path, 
#                  delim_whitespace=True, 
#                  skiprows=2, 
#                  na_values=-99.99)

# # 2️⃣ 补全列名
# df.columns = ['Year', 1,2,3,4,5,6,7,8,9,10,11,12,
#               'DJF','MAM','JJA','SON','ANN','DJFM']

# # 3️⃣ 只取 1981–2020 年段（含1981冬到2020冬）
# summer_means = []
# for year in range(1981, 2021):
#     this_year = df[df['Year'] == year]
#     if len(this_year):
#         months = this_year[[5,6,7,8,9]].values.flatten()
#         months = pd.to_numeric(months, errors='coerce')
#         summer_means.append([year, np.nanmean(months)])

# # winter_df = pd.DataFrame(winter_means, columns=['Winter_Year', 'AAO_winter_mean'])
# winter_df=np.array(summer_means)
# summer_df=winter_df[:,1]



# def standardize(arr):
#     """对数组做 Z-score 标准化，忽略 NaN"""
#     mean = np.nanmean(arr)
#     std = np.nanstd(arr)
#     if std == 0 or np.isnan(std):
#         std = 1.0
#     return (arr - mean) / std

# # 分别标准化
# winter_df1 = standardize(summer_df)



# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.stats import pearsonr

# from scipy.stats import pearsonr


# # 计算皮尔逊相关系数
# r, p = pearsonr(pc1_std,  winter_df1 )
# print(f"相关系数 r = {r:.3f}, 显著性 p = {p:.3f}")

# # 绘制散点图
# plt.figure(figsize=(5.5, 5), dpi=300)
# plt.scatter(winter_df1, pc1_std, color="#EF767A", edgecolor="k", s=70, alpha=0.85)

# # 拟合回归线
# m, b = np.polyfit(winter_df1 , pc1_std, 1)
# plt.plot(winter_df1 , m * winter_df1  + b, color="black", linewidth=1.5, linestyle="--", label=f"r = {r:.3f}")

# # 坐标轴标签
# plt.xlabel("Summer AAO (1981–2020)", fontsize=14, fontweight='bold')
# plt.ylabel("PC1 (standardized)", fontsize=14, fontweight='bold')

# # 美化外观
# plt.grid(True, linestyle=":", alpha=0.5)
# plt.legend(frameon=False, fontsize=12)
# plt.gca().spines['top'].set_visible(False)
# plt.gca().spines['right'].set_visible(False)
# plt.gca().set_facecolor("#FAFAFA")

# plt.tight_layout()
# plt.show()




# from mpl_toolkits.basemap import Basemap
# import matplotlib.pyplot as plt
# import numpy as np
# from scipy.interpolate import griddata
# from matplotlib.ticker import MaxNLocator

# # 构建插值网格
# grid_lon = np.linspace(0, 360, 360)
# grid_lat = np.linspace(-90, 90, 180)
# grid_lon, grid_lat = np.meshgrid(grid_lon, grid_lat)

# # 插值 EOF 空间型
# eof_heat1 = griddata((lon, lat), eof_heat, (grid_lon, grid_lat), method='cubic')

# # ===== 屏蔽 Niño3.4 区域 (10S–10N, 170W–130W，即 190–230E) =====
# mask = ((grid_lat >= 30) & (grid_lat <= 70) &
#         (grid_lon >= 60) & (grid_lon <= 145))
# eof_heat1_masked = np.copy(eof_heat1)
# eof_heat1_masked[mask] = np.nan  # 屏蔽区域设为 NaN（显示为白色）

# # =============== 绘图部分 ===============
# plt.figure(figsize=(8, 8), dpi=300)
# m = Basemap(projection='cyl', resolution='l',
#             llcrnrlat=-88, urcrnrlat=90,
#             llcrnrlon=0, urcrnrlon=350)
# m.drawcoastlines(linewidth=0.5)
# m.fillcontinents(color='none')

# x, y = m(grid_lon, grid_lat)
# contour = m.contourf(x, y, eof_heat1_masked, 60, cmap='RdBu_r',
#                      vmin=-0.02, vmax=0.02, extend='both')

# # 纬线经线
# parallels = np.arange(-80, 81, 40)
# m.drawparallels(parallels, labels=[1, 0, 0, 0],
#                 fontsize=20, linewidth=0, dashes=[2, 2])
# meridians = np.arange(0, 361, 60)
# m.drawmeridians(meridians, labels=[0, 0, 0, 1],
#                 fontsize=20, linewidth=0, dashes=[2, 2])

# # ======= 添加显著区域（Top 5%） =======
# abs_vals = eof_heat1_masked
# threshold = np.nanpercentile(np.abs(abs_vals), 95)
# pos_mask = np.where(abs_vals >= threshold, 1, np.nan)

# m.contourf(x, y, pos_mask, levels=[0.5, 1.5],
#            colors='none', hatches=['////'], zorder=6)

# # ======= 颜色条 =======
# cbar = plt.colorbar(contour, orientation='horizontal',
#                     pad=0.05, shrink=1, aspect=35)
# cbar.ax.tick_params(labelsize=20)
# cbar.locator = MaxNLocator(nbins=5)
# cbar.update_ticks()

# # ======= 标题与说明 =======
# plt.title("(a)", fontsize=34, fontweight='bold', x=-0.05)
# plt.text(108, 92, r'$\mathbf{Z_I}$: HWN EOF1 (7.81%)',
#          ha='center', va='bottom', fontsize=23, fontweight='bold')

# plt.tight_layout()
# plt.show()



###############################################g图在这
import pandas as pd


df = pd.read_csv("F:/heatwave_chaos/nina34.anom(time).csv", header=None, names=["Date", "ONI"])

# 2️⃣ 转换日期格式
df["Date"] = pd.to_datetime(df["Date"])

# 3️⃣ 提取年份和月份
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month

# 4️⃣ 筛选出 1981–2020 年的夏季（5–9月）
summer_df = df[(df["Year"] >= 1981) & (df["Year"] <= 2020) & (df["Month"].between(5, 9))]

# 5️⃣ 按年份计算夏季平均 ONI
summer_mean = summer_df.groupby("Year")["ONI"].mean().reset_index()
summer_mean=np.array(summer_mean)
summer_mean=summer_mean[:,1]
# 6️⃣ 输出结果
print(summer_mean)

# 7️⃣ （可选）保存结果
np.save("F:/heatwave_chaos/nina34.summer5-9.npy",summer_mean)

def standardize(arr):
    """对数组做 Z-score 标准化，忽略 NaN"""
    mean = np.nanmean(arr)
    std = np.nanstd(arr)
    if std == 0 or np.isnan(std):
        std = 1.0
    return (arr - mean) / std

# 分别标准化
summer_mean1 = standardize(summer_mean)
pc1_std = standardize(pc1)

plt.figure(figsize=(8, 3.8), dpi=300)

# ===== 基础曲线 =====
# ==== PC1（主线）====
plt.plot(np.arange(nyear) + 1981, pc1_std, 
         color='#E63946', linewidth=3.2, marker='o', markersize=6,
         markerfacecolor='white', markeredgewidth=1.5,
         label='PC1', zorder=10)

# ==== 其他变量 ====
plt.plot(np.arange(nyear) + 1981, summer_mean1,
         color='#457B9D', linewidth=2.0, alpha=0.8, marker='o', markersize=5,
         markerfacecolor='white', markeredgewidth=1.0,
         label='ONI5-9')

# plt.plot(np.arange(nyear) + 1981, shuju2_std,
#          color='#A8DADC', linewidth=2.0, alpha=0.8, marker='o', markersize=5,
#          markerfacecolor='white', markeredgewidth=1.0,
#          label='CWN r=0.818')


# ===== 零线 =====
plt.axhline(0, color='gray', linewidth=1.2, linestyle='--', alpha=0.6, zorder=0)

# ===== 坐标轴样式 =====
plt.xlabel("Year", fontsize=16, fontweight='bold', labelpad=6)
plt.ylabel("PC1", fontsize=16, fontweight='bold', labelpad=6)

#plt.xlim(1981, 1980 + nyear)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# ===== 网格线 =====
plt.grid(True, linestyle=':', linewidth=0.8, alpha=0.4)

# ===== 去除右上边框 =====
for spine in ['right', 'top']:
    plt.gca().spines[spine].set_visible(False)
plt.gca().spines['left'].set_linewidth(1.2)
plt.gca().spines['bottom'].set_linewidth(1.2)

# ===== 增强整体配色感 =====
plt.gca().set_facecolor('#FAFAFA')  # 柔和背景
plt.legend(frameon=False, fontsize=13, loc='upper left')



# ===== 调整边距并展示 =====
plt.tight_layout(pad=1)
plt.show()



def corr(x, y):
    """忽略 NaN 的 Pearson 相关系数"""
    mask = ~np.isnan(x) & ~np.isnan(y)
    if np.sum(mask) == 0:
        return np.nan
    return np.corrcoef(x[mask], y[mask])[0, 1]

r1 = corr(pc1_std,  summer_mean1)

print("=== 相关系数 (PC1 vs 各变量) ===")
print(f"PC1 vs HWN  : {r1:.3f}")
# print(f"PC1 vs CWN  : {r2:.3f}")
# print(f"PC1 vs TSUM : {r3:.3f}")
# print(f"PC1 vs TWIN : {r4:.3f}")



import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# 计算皮尔逊相关系数
r, p = pearsonr(pc1_std, summer_mean1)
print(f"相关系数 r = {r:.3f}, 显著性 p = {p:.3f}")

# 拟合回归线
m, b = np.polyfit(summer_mean1, pc1_std, 1)
x_line = np.linspace(summer_mean1.min(), summer_mean1.max(), 100)
y_line = m * x_line + b

plt.figure(figsize=(5.8, 4.5), dpi=300)

# 散点
plt.scatter(
    summer_mean1, pc1_std,
    color="#92B4C8",
    edgecolor="k",
    s=120,
    alpha=0.85,
    zorder=3
)

# 回归线
plt.plot(
    x_line, y_line,
    color="black",
    linewidth=1.8,
    linestyle="--",
    label=f"r = {r:.2f}, p = {p:.3f}"
)

# 坐标轴标签
plt.xlabel("ONI index(Summer)", fontsize=18, fontweight='bold')
plt.ylabel("PC1(Remove ENSO Region)", fontsize=18, fontweight='bold')

# 网格线
plt.grid(True, linestyle=":", alpha=0.4, zorder=0)

# 坐标轴样式
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.2)
ax.spines['bottom'].set_linewidth(1.2)
#ax.set_facecolor("#F9F9F9")

# 刻度
ax.tick_params(axis='both', labelsize=15, width=1.1, length=5)
p_text = "p < 0.05" if p < 0.05 else f"p = {p:.3f}"
# 在图内标注 r、p（放左上角）
plt.text(
    0.88, 0.96,
    f"r = {r:.2f}\n{p_text}",
    transform=ax.transAxes,
    fontsize=18,
    va='top', ha='right',
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
)

# 图例（只显示回归线说明）
#plt.legend(frameon=False, fontsize=11, loc='lower right')
plt.title("g", fontsize=28, fontweight='bold',x=-0.05)
#plt.text(0, 2.32,  'Network I', ha='center', va='bottom', fontsize=20, fontweight='bold')
plt.tight_layout()
plt.savefig('F:/heatwave_chaos/EOF/g.pdf', format='pdf', bbox_inches='tight')
plt.show()








##########################和ao的关系-------再去掉亚欧大陆重新跑EOF
import numpy as np
import pandas as pd
# 读取数据（假设文件中用空格分隔）
data = np.loadtxt('F:/heatwave_chaos/ao.dat')

# 如果文件有特定分隔符（例如逗号、制表符等）
# data = np.loadtxt('yourfile.dat', delimiter=',')
print(data)
data1=data[:888,1]
# 定义年份和月份
start_year = 1950
years = np.repeat(np.arange(start_year, start_year + 74), 12)  # 重复每年的值 12 次
months = np.tile(np.arange(1, 13), 74)  # 每年有 12 个月，重复 74 次

assert len(years) == len(data1) and len(months) == len(data1), "年份和月份数组的长度必须与数据匹配"

winter_months = [11, 12, 1, 2, 3]
winter_data = []

for year in range(1982, 2022):
    # 处理 11 月和 12 月属于上一年的情况
    indices = np.where(((years == year - 1) & (months == 11)) |
                       ((years == year - 1) & (months == 12)) |
                       ((years == year) & (months == 1)) |
                       ((years == year) & (months == 2)) |
                       ((years == year) & (months == 3)))[0]
    winter_data.append(data1[indices])

winter_data = np.array(winter_data)
print(winter_data.shape)  # 应输出 (40, 5)


AO=np.sum(winter_data,axis=1)


# summer_data = []
# for year in range(1981, 2021):   # 和冬季年份保持一致
#     idx = np.where(
#         (years == year) & (months >= 5) & (months <= 9)
#     )[0]
#     summer_data.append(data1[idx])

# summer_data = np.array(summer_data)      # (40, 5)
# AO_summer_mean = summer_data.mean(axis=1)








def standardize(arr):
    """对数组做 Z-score 标准化，忽略 NaN"""
    mean = np.nanmean(arr)
    std = np.nanstd(arr)
    if std == 0 or np.isnan(std):
        std = 1.0
    return (arr - mean) / std

# 分别标准化
AO_winter_mean_std = standardize(AO)
pc1_std = standardize(pc1)

plt.figure(figsize=(8, 3), dpi=300)

# ===== 基础曲线 =====
# ==== PC1（主线）====
plt.plot(np.arange(nyear) + 1981, pc1_std, 
         color='#E63946', linewidth=3.2, marker='o', markersize=6,
         markerfacecolor='white', markeredgewidth=1.5,
         label='PC1', zorder=10)

# ==== 其他变量 ====
plt.plot(np.arange(nyear) + 1981, AO_winter_mean_std ,
         color='#457B9D', linewidth=2.0, alpha=0.8, marker='o', markersize=5,
         markerfacecolor='white', markeredgewidth=1.0,
         label='AO11-3')

# plt.plot(np.arange(nyear) + 1981, shuju2_std,
#          color='#A8DADC', linewidth=2.0, alpha=0.8, marker='o', markersize=5,
#          markerfacecolor='white', markeredgewidth=1.0,
#          label='CWN r=0.818')


# ===== 零线 =====
plt.axhline(0, color='gray', linewidth=1.2, linestyle='--', alpha=0.6, zorder=0)

# ===== 坐标轴样式 =====
plt.xlabel("Year", fontsize=16, fontweight='bold', labelpad=6)
plt.ylabel("PC1", fontsize=16, fontweight='bold', labelpad=6)

#plt.xlim(1981, 1980 + nyear)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# ===== 网格线 =====
plt.grid(True, linestyle=':', linewidth=0.8, alpha=0.4)

# ===== 去除右上边框 =====
for spine in ['right', 'top']:
    plt.gca().spines[spine].set_visible(False)
plt.gca().spines['left'].set_linewidth(1.2)
plt.gca().spines['bottom'].set_linewidth(1.2)

# ===== 增强整体配色感 =====
plt.gca().set_facecolor('#FAFAFA')  # 柔和背景
plt.legend(frameon=False, fontsize=13, loc='upper left')



# ===== 调整边距并展示 =====
plt.tight_layout(pad=1)
plt.show()



def corr(x, y):
    """忽略 NaN 的 Pearson 相关系数"""
    mask = ~np.isnan(x) & ~np.isnan(y)
    if np.sum(mask) == 0:
        return np.nan
    return np.corrcoef(x[mask], y[mask])[0, 1]

r1 = corr(pc1_std, AO_winter_mean_std)

print("=== 相关系数 (PC1 vs 各变量) ===")
print(f"PC1 vs HWN  : {r1:.3f}")



import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

from scipy.stats import pearsonr


# 计算皮尔逊相关系数
r, p = pearsonr(pc1_std,  AO_winter_mean_std)
print(f"相关系数 r = {r:.3f}, 显著性 p = {p:.3f}")

# 绘制散点图
plt.figure(figsize=(5.5, 5), dpi=300)
plt.scatter(AO_winter_mean_std, pc1_std, color="#EF767A", edgecolor="k", s=70, alpha=0.85)

# 拟合回归线
m, b = np.polyfit(AO_winter_mean_std, pc1_std, 1)
plt.plot( AO_winter_mean_std, m *  AO_winter_mean_std + b, color="black", linewidth=1.5, linestyle="--", label=f"r = {r:.3f}")

# 坐标轴标签
plt.xlabel("AO(Winter)", fontsize=14, fontweight='bold')
plt.ylabel("PC1", fontsize=14, fontweight='bold')

# 美化外观
plt.grid(True, linestyle=":", alpha=0.5)
plt.legend(frameon=False, fontsize=12)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().set_facecolor("#FAFAFA")

plt.tight_layout()
plt.show()



import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# 计算皮尔逊相关系数
r, p = pearsonr(pc1_std, AO_winter_mean_std)
print(f"相关系数 r = {r:.3f}, 显著性 p = {p:.3f}")

# 拟合回归线
m, b = np.polyfit(AO_winter_mean_std, pc1_std, 1)
x_line = np.linspace(AO_winter_mean_std.min(), AO_winter_mean_std.max(), 100)
y_line = m * x_line + b

plt.figure(figsize=(5.8, 4.5), dpi=300)

# 散点
plt.scatter(
    AO_winter_mean_std, pc1_std,
    color="#92B4C8",
    edgecolor="k",
    s=120,
    alpha=0.85,
    zorder=3
)

# 回归线
plt.plot(
    x_line, y_line,
    color="black",
    linewidth=1.8,
    linestyle="--",
    label=f"r = {r:.2f}, p = {p:.3f}"
)

# 坐标轴标签
plt.xlabel("AO(Winter)", fontsize=18, fontweight='bold')
plt.ylabel("PC1(Remove Eurasia region)", fontsize=18, fontweight='bold')

# 网格线
plt.grid(True, linestyle=":", alpha=0.4, zorder=0)

# 坐标轴样式
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.2)
ax.spines['bottom'].set_linewidth(1.2)
#ax.set_facecolor("#F9F9F9")

# 刻度
ax.tick_params(axis='both', labelsize=18, width=1.1, length=5)
p_text = "p < 0.05" if p < 0.05 else f"p = {p:.3f}"
# 在图内标注 r、p（放左上角）
plt.text(
    0.3, 0.28,
    f"r = {r:.2f}\n{p_text}",
    transform=ax.transAxes,
    fontsize=18,
    va='top', ha='right',
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
)

# 图例（只显示回归线说明）
#plt.legend(frameon=False, fontsize=11, loc='lower right')
plt.title("h", fontsize=28, fontweight='bold',x=-0.05)
#plt.text(0, 2.32,  'Network I', ha='center', va='bottom', fontsize=20, fontweight='bold')
plt.tight_layout()
plt.savefig('F:/heatwave_chaos/EOF/h.pdf', format='pdf', bbox_inches='tight')
plt.show()


















####################################################网络II


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.interpolate import griddata
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import Normalize
import pandas as pd

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

    

nodes, lat, lon = selection()

heat= pd.read_csv('F:/heatwave/worldtongbu/42yearworld/40worldheat_allheatevents.csv',header=None)
heat=np.array(heat)
heat=heat[nodes]
heat=heat[:,1:]


cold= pd.read_csv('F:/heatwave/worldtongbu/42yearworld/cold/40worldcold_allheatevents.csv',header=None)
cold=np.array(cold)
cold=cold[nodes]
cold=cold[:,:-1]

lon = np.load("F:/heatwave/longitude.npy")
lat = np.load("F:/heatwave/latitude.npy")



avgtem_summer=np.load('F:/heatwave_chaos/summer_mean_1981_2020.npy')
avgtem_summer=avgtem_summer[nodes]
avgtem_summer=avgtem_summer[:,1:]
avgtem_winter= np.load('F:/heatwave_chaos/winter_mean_1981_2020.npy')
avgtem_winter=avgtem_winter[nodes]
avgtem_winter=avgtem_winter[:,:-1]
lon = np.load("F:/heatwave/longitude.npy")
lat = np.load("F:/heatwave/latitude.npy")

heat1 = heat.astype(np.float32)
cold1 = cold.astype(np.float32)
tsum1 = avgtem_summer.astype(np.float32)
twin1 = avgtem_winter.astype(np.float32)

n_grid, n_year = heat1.shape
time = np.arange(n_year)

# ===== 去趋势函数 =====
def detrend_field(data):
    detrended = np.empty_like(data)
    for i in range(n_grid):
        y = data[i, :]
        if np.all(np.isnan(y)):
            detrended[i, :] = np.nan
        else:
            slope, intercept, _, _, _ = stats.linregress(time, y)
            detrended[i, :] = y - (slope * time + intercept)
    return detrended

heat = detrend_field(heat1)
cold = detrend_field(cold1)
tsum = detrend_field(tsum1)
twin = detrend_field(twin1)

# ==================================================
# 去掉 ENSO 区域：纬度 [-10,10]，经度 [190,230]
# ==================================================
mask = ~((lat >= 30) & (lat <= 70) & (lon >= 2) & (lon <= 135))
#mask = ~((lat >= -20) & (lat <=20) & (lon >= 180) & (lon <= 290))
valid_indices = np.where(mask)[0]

# 对每个变量只保留非 ENSO 区域的格点
heat = heat[valid_indices, :]
cold = cold[valid_indices, :]
tsum = tsum[valid_indices, :]
twin = twin[valid_indices, :]
lon = lon[valid_indices]
lat = lat[valid_indices]

print(f"去除 ENSO 区域后保留格点数: {len(valid_indices)}")

# ==================================================
# 做 anomaly、归一化、拼接联合场
# ==================================================

def make_anomaly(x):
    x_ts = x.T
    return x_ts - x_ts.mean(axis=0, keepdims=True)

def normalize_field(x):
    std = x.std()
    return x / std if std != 0 else x

heat_n = normalize_field(make_anomaly(heat))
cold_n = normalize_field(make_anomaly(cold))
tsum_n = normalize_field(make_anomaly(tsum))
twin_n = normalize_field(make_anomaly(twin))

# 拼接联合场
X = np.concatenate([cold_n, heat_n, twin_n, tsum_n], axis=1)
ntime, nspace_total = X.shape
print("联合场维度:", X.shape)

# ==================================================
# EOF 分解
# ==================================================
def eof_analysis_fast(data, neofs=5):
    X = data - data.mean(axis=0, keepdims=True)
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    varfrac = (s**2) / np.sum(s**2)
    pcs  = U[:, :neofs] * s[:neofs]
    eofs = Vt[:neofs, :].T
    return eofs, pcs, varfrac[:neofs]

neofs = 5
eofs, pcs, varfrac = eof_analysis_fast(X, neofs=neofs)

print("前 {} 个模态方差贡献率:".format(neofs))
for i in range(neofs):
    print(f"Mode {i+1}: {varfrac[i]*100:.2f}%")

# ==================================================
# 拆分回各变量
# ==================================================
npoint = len(valid_indices)
mode = 0

eof_all = eofs[:, mode]
eof_heat = eof_all[0*npoint:1*npoint]
eof_cold = eof_all[1*npoint:2*npoint]
eof_tsum = eof_all[2*npoint:3*npoint]
eof_twin = eof_all[3*npoint:4*npoint]
pc1 = pcs[:, mode]

print("Mode 1 explains: {:.2f}% variance".format(varfrac[0]*100))













from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
from matplotlib.ticker import MaxNLocator

# 构建插值网格
grid_lon = np.linspace(0, 360, 360)
grid_lat = np.linspace(-90, 90, 180)
grid_lon, grid_lat = np.meshgrid(grid_lon, grid_lat)

# 插值 EOF 空间型
eof_heat1 = griddata((lon, lat), eof_heat, (grid_lon, grid_lat), method='cubic')

# ===== 屏蔽 Niño3.4 区域 (10S–10N, 170W–130W，即 190–230E) =====
mask = ((grid_lat >= -15) & (grid_lat <= 15) &
        (grid_lon >= 190) & (grid_lon <= 280))
eof_heat1_masked = np.copy(eof_heat1)
eof_heat1_masked[mask] = np.nan  # 屏蔽区域设为 NaN（显示为白色）

# =============== 绘图部分 ===============
plt.figure(figsize=(8, 8), dpi=300)
m = Basemap(projection='cyl', resolution='l',
            llcrnrlat=-88, urcrnrlat=90,
            llcrnrlon=0, urcrnrlon=350)
m.drawcoastlines(linewidth=0.5)
m.fillcontinents(color='none')

x, y = m(grid_lon, grid_lat)
contour = m.contourf(x, y, eof_heat1_masked, 60, cmap='RdBu_r',
                     vmin=-0.02, vmax=0.02, extend='both')

# 纬线经线
parallels = np.arange(-80, 81, 40)
m.drawparallels(parallels, labels=[1, 0, 0, 0],
                fontsize=20, linewidth=0, dashes=[2, 2])
meridians = np.arange(0, 361, 60)
m.drawmeridians(meridians, labels=[0, 0, 0, 1],
                fontsize=20, linewidth=0, dashes=[2, 2])

# ======= 添加显著区域（Top 5%） =======
abs_vals = eof_heat1_masked
threshold = np.nanpercentile(np.abs(abs_vals), 95)
pos_mask = np.where(abs_vals >= threshold, 1, np.nan)

m.contourf(x, y, pos_mask, levels=[0.5, 1.5],
           colors='none', hatches=['////'], zorder=6)

# ======= 颜色条 =======
cbar = plt.colorbar(contour, orientation='horizontal',
                    pad=0.05, shrink=1, aspect=35)
cbar.ax.tick_params(labelsize=20)
cbar.locator = MaxNLocator(nbins=5)
cbar.update_ticks()

# ======= 标题与说明 =======
plt.title("(a)", fontsize=34, fontweight='bold', x=-0.05)
plt.text(108, 92, r'$\mathbf{Z_I}$: HWN EOF1 (7.81%)',
         ha='center', va='bottom', fontsize=23, fontweight='bold')

plt.tight_layout()
plt.show()




import pandas as pd



df = pd.read_csv("F:/heatwave_chaos/nina34.anom(time).csv", header=None, names=["Date", "ONI"])

# 2️⃣ 转换日期格式
df["Date"] = pd.to_datetime(df["Date"])

# 3️⃣ 提取年份和月份
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month

# 4️⃣ 筛选出 1981–2020 年的夏季（5–9月）
summer_df = df[(df["Year"] >= 1982) & (df["Year"] <= 2020) & (df["Month"].between(5, 9))]

# 5️⃣ 按年份计算夏季平均 ONI
summer_mean = summer_df.groupby("Year")["ONI"].mean().reset_index()
summer_mean=np.array(summer_mean)
summer_mean=summer_mean[:,1]
# 6️⃣ 输出结果
print(summer_mean)

# 7️⃣ （可选）保存结果
np.save("F:/heatwave_chaos/nina34.summer5-9.npy",summer_mean)

def standardize(arr):
    """对数组做 Z-score 标准化，忽略 NaN"""
    mean = np.nanmean(arr)
    std = np.nanstd(arr)
    if std == 0 or np.isnan(std):
        std = 1.0
    return (arr - mean) / std

# 分别标准化
summer_mean1 = standardize(summer_mean)
pc1_std = standardize(pc1)

plt.figure(figsize=(8, 3.8), dpi=300)

# ===== 基础曲线 =====
# ==== PC1（主线）====
plt.plot(np.arange(nyear) + 1981, pc1_std, 
         color='#E63946', linewidth=3.2, marker='o', markersize=6,
         markerfacecolor='white', markeredgewidth=1.5,
         label='PC1', zorder=10)

# ==== 其他变量 ====
plt.plot(np.arange(nyear) + 1981, summer_mean1,
         color='#457B9D', linewidth=2.0, alpha=0.8, marker='o', markersize=5,
         markerfacecolor='white', markeredgewidth=1.0,
         label='ONI5-9')

# plt.plot(np.arange(nyear) + 1981, shuju2_std,
#          color='#A8DADC', linewidth=2.0, alpha=0.8, marker='o', markersize=5,
#          markerfacecolor='white', markeredgewidth=1.0,
#          label='CWN r=0.818')


# ===== 零线 =====
plt.axhline(0, color='gray', linewidth=1.2, linestyle='--', alpha=0.6, zorder=0)

# ===== 坐标轴样式 =====
plt.xlabel("Year", fontsize=16, fontweight='bold', labelpad=6)
plt.ylabel("PC1", fontsize=16, fontweight='bold', labelpad=6)

#plt.xlim(1981, 1980 + nyear)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# ===== 网格线 =====
plt.grid(True, linestyle=':', linewidth=0.8, alpha=0.4)

# ===== 去除右上边框 =====
for spine in ['right', 'top']:
    plt.gca().spines[spine].set_visible(False)
plt.gca().spines['left'].set_linewidth(1.2)
plt.gca().spines['bottom'].set_linewidth(1.2)

# ===== 增强整体配色感 =====
plt.gca().set_facecolor('#FAFAFA')  # 柔和背景
plt.legend(frameon=False, fontsize=13, loc='upper left')



# ===== 调整边距并展示 =====
plt.tight_layout(pad=1)
plt.show()



def corr(x, y):
    """忽略 NaN 的 Pearson 相关系数"""
    mask = ~np.isnan(x) & ~np.isnan(y)
    if np.sum(mask) == 0:
        return np.nan
    return np.corrcoef(x[mask], y[mask])[0, 1]

r1 = corr(pc1_std,  summer_mean1)

print("=== 相关系数 (PC1 vs 各变量) ===")
print(f"PC1 vs HWN  : {r1:.3f}")
# print(f"PC1 vs CWN  : {r2:.3f}")
# print(f"PC1 vs TSUM : {r3:.3f}")
# print(f"PC1 vs TWIN : {r4:.3f}")



import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# 计算皮尔逊相关系数
r, p = pearsonr(pc1_std, summer_mean1)
print(f"相关系数 r = {r:.3f}, 显著性 p = {p:.3f}")

# 拟合回归线
m, b = np.polyfit(summer_mean1, pc1_std, 1)
x_line = np.linspace(summer_mean1.min(), summer_mean1.max(), 100)
y_line = m * x_line + b

plt.figure(figsize=(5.8, 4.5), dpi=300)

# 散点
plt.scatter(
    summer_mean1, pc1_std,
    color="#92B4C8",
    edgecolor="k",
    s=120,
    alpha=0.85,
    zorder=3
)

# 回归线
plt.plot(
    x_line, y_line,
    color="black",
    linewidth=1.8,
    linestyle="--",
    label=f"r = {r:.2f}, p = {p:.3f}"
)

# 坐标轴标签
plt.xlabel("ONI index(Summer)", fontsize=18, fontweight='bold')
plt.ylabel("PC1(Remove ENSO Region)", fontsize=18, fontweight='bold')

# 网格线
plt.grid(True, linestyle=":", alpha=0.4, zorder=0)

# 坐标轴样式
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.2)
ax.spines['bottom'].set_linewidth(1.2)
#ax.set_facecolor("#F9F9F9")

# 刻度
ax.tick_params(axis='both', labelsize=18, width=1.1, length=5)
p_text = "p < 0.05" if p < 0.05 else f"p = {p:.3f}"
# 在图内标注 r、p（放左上角）
plt.text(
    0.8, 0.9,
    f"r = {r:.2f}\n{p_text}",
    transform=ax.transAxes,
    fontsize=18,
    va='top', ha='right',
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
)

# 图例（只显示回归线说明）
#plt.legend(frameon=False, fontsize=11, loc='lower right')
plt.title("o", fontsize=28, fontweight='bold',x=-0.07)
#plt.text(0, 2.4,  'Network II', ha='center', va='bottom', fontsize=20, fontweight='bold')
plt.tight_layout()
plt.savefig('F:/heatwave_chaos/EOF/netII_o.pdf', format='pdf', bbox_inches='tight')
plt.show()














import numpy as np
import pandas as pd
# 读取数据（假设文件中用空格分隔）
data = np.loadtxt('F:/heatwave_chaos/ao.dat')

# 如果文件有特定分隔符（例如逗号、制表符等）
# data = np.loadtxt('yourfile.dat', delimiter=',')
print(data)
data1=data[:888,1]
# 定义年份和月份
start_year = 1950
years = np.repeat(np.arange(start_year, start_year + 74), 12)  # 重复每年的值 12 次
months = np.tile(np.arange(1, 13), 74)  # 每年有 12 个月，重复 74 次

assert len(years) == len(data1) and len(months) == len(data1), "年份和月份数组的长度必须与数据匹配"

winter_months = [11, 12, 1, 2, 3]
winter_data = []

for year in range(1982, 2021):
    # 处理 11 月和 12 月属于上一年的情况
    indices = np.where(((years == year - 1) & (months == 11)) |
                       ((years == year - 1) & (months == 12)) |
                       ((years == year) & (months == 1)) |
                       ((years == year) & (months == 2)) |
                       ((years == year) & (months == 3)))[0]
    winter_data.append(data1[indices])

winter_data = np.array(winter_data)
print(winter_data.shape)  # 应输出 (40, 5)


AO=np.sum(winter_data,axis=1)






def standardize(arr):
    """对数组做 Z-score 标准化，忽略 NaN"""
    mean = np.nanmean(arr)
    std = np.nanstd(arr)
    if std == 0 or np.isnan(std):
        std = 1.0
    return (arr - mean) / std

# 分别标准化
AO_winter_mean_std = standardize(AO)
pc1_std = standardize(pc1)








import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# 计算皮尔逊相关系数
r, p = pearsonr(pc1_std, AO_winter_mean_std)
print(f"相关系数 r = {r:.3f}, 显著性 p = {p:.3f}")

# 拟合回归线
m, b = np.polyfit(AO_winter_mean_std, pc1_std, 1)
x_line = np.linspace(AO_winter_mean_std.min(), AO_winter_mean_std.max(), 100)
y_line = m * x_line + b

plt.figure(figsize=(5.8,4.5), dpi=300)

# 散点
plt.scatter(
    AO_winter_mean_std, pc1_std,
    color="#92B4C8",
    edgecolor="k",
    s=120,
    alpha=0.85,
    zorder=3
)

# 回归线
plt.plot(
    x_line, y_line,
    color="black",
    linewidth=1.8,
    linestyle="--",
    label=f"r = {r:.2f}, p = {p:.3f}"
)

# 坐标轴标签
plt.xlabel("AO(Winter)", fontsize=18, fontweight='bold')
plt.ylabel("PC1(Remove Eurasia region)", fontsize=18, fontweight='bold')

# 网格线
plt.grid(True, linestyle=":", alpha=0.4, zorder=0)

# 坐标轴样式
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.2)
ax.spines['bottom'].set_linewidth(1.2)
#ax.set_facecolor("#F9F9F9")

# 刻度
ax.tick_params(axis='both', labelsize=18, width=1.1, length=5)
p_text = "p < 0.05" if p < 0.05 else f"p = {p:.3f}"
# 在图内标注 r、p（放左上角）
plt.text(
    0.28, 0.27,
    f"r = {r:.2f}\n{p_text}",
    transform=ax.transAxes,
    fontsize=18,
    va='top', ha='right',
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
)

# 图例（只显示回归线说明）
#plt.legend(frameon=False, fontsize=11, loc='lower right')
plt.title("p", fontsize=28, fontweight='bold',x=-0.07)
#plt.text(0, 2.18,  'Network II', ha='center', va='bottom', fontsize=20, fontweight='bold')
plt.tight_layout()
plt.savefig('F:/heatwave_chaos/EOF/netII_p.pdf', format='pdf', bbox_inches='tight')
plt.show()
