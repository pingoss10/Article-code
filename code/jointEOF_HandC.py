# -*- coding: utf-8 -*-
"""
Created on Mon Dec  1 13:59:54 2025

@author: yuping
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Dec  1 12:43:45 2025

@author: yuping
"""

################################处理温度数据
import numpy as np
from netCDF4 import Dataset
import cftime

# 1. 读取数据
ds = Dataset("F:/heatwave_chaos/air.2m.mon.mean.nc", "r")
air = ds.variables["air"][:]     # (time, lat, lon)
lat = ds.variables["lat"][:]
lon = ds.variables["lon"][:]
time = ds.variables["time"][:]
dates = cftime.num2date(time, units=ds.variables["time"].units, calendar="standard")

# 2. 选择 1981–2020 年的 5–9 月
mask = np.array([(1981 <= d.year <= 2020) and (5 <= d.month <= 9) for d in dates])
selected_air = air[mask, :, :]
selected_dates = np.array(dates)[mask]

# 3. 按年份计算每年夏季平均
years = np.arange(1981, 2021)
nyear = len(years)
nlat, nlon = len(lat), len(lon)
summer_each_year = np.zeros((nyear, nlat, nlon))

for i, y in enumerate(years):
    idx = [j for j, d in enumerate(selected_dates) if d.year == y]
    summer_each_year[i, :, :] = np.mean(selected_air[idx, :, :], axis=0)

# 4. 重塑为 (格点数, 年份数)
summer_each_year_flat = summer_each_year.reshape(nyear, -1).T  # shape: (nlat*nlon, 40)

print("结果维度:", summer_each_year_flat.shape)
# 保存结果
np.save("F:/heatwave_chaos/summer_mean_1981_2020.npy", summer_each_year_flat)



import numpy as np
from netCDF4 import Dataset
import cftime

# 1. 读取数据
ds = Dataset("F:/heatwave_chaos/air.2m.mon.mean.nc", "r")
air = ds.variables["air"][:]     # (time, lat, lon)
lat = ds.variables["lat"][:]
lon = ds.variables["lon"][:]
time = ds.variables["time"][:]
dates = cftime.num2date(time, units=ds.variables["time"].units, calendar="standard")

# 2. 仅保留 1981–2021 年（因 2020 冬季要用到 2021 年 3 月）
mask = np.array([(1981 <= d.year <= 2021) for d in dates])
air = air[mask, :, :]
dates = np.array(dates)[mask]

# 3. 定义年份范围（1981–2020 冬季）
years = np.arange(1981, 2021)
nyear = len(years)
nlat, nlon = len(lat), len(lon)

winter_each_year = np.full((nyear, nlat, nlon), np.nan)

# 4. 逐年取 11–3 月平均（跨年）
for i, y in enumerate(years):
    months = [(y, 11), (y, 12), (y+1, 1), (y+1, 2), (y+1, 3)]
    idx = [j for j, d in enumerate(dates) if (d.year, d.month) in months]
    if len(idx) > 0:
        winter_each_year[i, :, :] = np.mean(air[idx, :, :], axis=0)

# 5. 转为 (格点数, 年份数)
winter_flat = winter_each_year.reshape(nyear, -1).T  # (格点数, 40)
print("冬季平均温度形状:", winter_flat.shape)

# 6. 保存
np.save("F:/heatwave_chaos/winter_mean_1981_2020.npy", winter_flat)



###################################开始跑




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

#挑选出西太平洋和亚欧大陆地区的韩流和热浪
#equatorial_indices = np.where((lat >= -90) & (lat <= 85) & (lon >= 30) & (lon <= 180))[0]

import numpy as np
from scipy.linalg import svd









import numpy as np
from scipy.linalg import svd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature


heat1 = heat.astype(np.float32)          # 热浪
cold1 = cold.astype(np.float32)          # 寒流
tsum1 = avgtem_summer.astype(np.float32) # 夏季平均温度
twin1 = avgtem_winter.astype(np.float32) # 冬季平均温度

import numpy as np
from scipy import stats

# 假设每个变量形状为 (n_grid, n_year)
n_grid, n_year = heat.shape
time = np.arange(n_year)

def detrend_field(data):
    """对每个格点去除线性趋势"""
    detrended = np.empty_like(data)
    for i in range(n_grid):
        y = data[i, :]
        if np.all(np.isnan(y)):
            detrended[i, :] = np.nan
        else:
            # 线性回归去趋势
            slope, intercept, _, _, _ = stats.linregress(time, y)
            trend = slope * time + intercept
            detrended[i, :] = y - trend
    return detrended

# 对四个场分别去趋势
heat = detrend_field(heat1)
cold = detrend_field(cold1)
tsum = detrend_field(tsum1)
twin = detrend_field(twin1)


lon  = np.load("F:/heatwave/longitude.npy")
lat  = np.load("F:/heatwave/latitude.npy")

npoint, nyear = heat.shape
assert cold.shape == (npoint, nyear)
assert tsum.shape == (npoint, nyear)
assert twin.shape == (npoint, nyear)

print("格点数:", npoint, " 年数:", nyear)

# ==============================
# 1. 转成 (time, space) 并做距平
# ==============================

def make_anomaly(x):
    """
    x: (space, time) -> 返回 (time, space) 的 anomaly
    """
    x_ts = x.T                        # (time, space)
    x_anom = x_ts - x_ts.mean(axis=0, keepdims=True)
    return x_anom                     # (time, space)

heat_a = make_anomaly(heat)
cold_a = make_anomaly(cold)
tsum_a = make_anomaly(tsum)
twin_a = make_anomaly(twin)

# ==============================
# 2. 变量归一化（不做面积权重）
# ==============================

def normalize_field(x_anom):
    """
    x_anom: (time, space)
    用整体 RMS 做归一化
    """
    # rms = np.sqrt(np.mean(x_anom**2))
    # if rms == 0:
    #     rms = 1.0
    # return x_anom / rms

    std = x_anom.std()
    if std == 0:
        std = 1.0
    return x_anom / std

heat_n = normalize_field(heat_a)
cold_n = normalize_field(cold_a)
tsum_n = normalize_field(tsum_a)
twin_n = normalize_field(twin_a)

# ==============================
# 3. 拼成联合场，做快速 EOF (SVD)
# ==============================

# 在“空间维”上拼接四个变量
# X: (time, 4 * npoint)
X = np.concatenate([heat_n, cold_n, tsum_n, twin_n], axis=1)
ntime, nspace_total = X.shape
print("联合场维度: time =", ntime, ", space_total =", nspace_total)

def eof_analysis_fast(data, neofs=5):
    """
    data: (time, space_total)
    用 SVD 直接做 EOF，不显式构造协方差矩阵
    返回:
      eofs:    (space_total, neofs)
      pcs:     (time, neofs)
      varfrac: (neofs,)
    """
    X = data - data.mean(axis=0, keepdims=True)   # 再去一次时间均值，保险
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    varfrac = (s**2) / np.sum(s**2)
    pcs  = U[:, :neofs] * s[:neofs]              # (time, neofs)
    eofs = Vt[:neofs, :].T                       # (space_total, neofs)
    return eofs, pcs, varfrac[:neofs]

neofs = 5
eofs, pcs, varfrac = eof_analysis_fast(X, neofs=neofs)

print("前 {} 个模态方差贡献率:".format(neofs))
for i in range(neofs):
    print("Mode {}: {:.2f}%".format(i+1, varfrac[i]*100))

# ==============================
# 4. 拆分回四个变量的 EOF 空间型
# ==============================

mode = 0  # 取第1模态
eof_all = eofs[:, mode]   # (4 * npoint,)

eof_heat = eof_all[0*npoint : 1*npoint]   # 热浪 EOF1
eof_cold = eof_all[1*npoint : 2*npoint]   # 寒流 EOF1
eof_tsum = eof_all[2*npoint : 3*npoint]   # 夏季温度 EOF1
eof_twin = eof_all[3*npoint : 4*npoint]   # 冬季温度 EOF1

pc1 = pcs[:, mode]  # 第1模态 PC 时间序列

print("Mode 1 explains: {:.2f}% variance".format(varfrac[0]*100))
print("PC1 shape:", pc1.shape)

# ==============================
# 5. 简单画图（可按需删掉）
# ==============================

def plot_eof_scatter(lon, lat, field, title, vmin=None, vmax=None):
    proj = ccrs.PlateCarree(central_longitude=180)
    fig = plt.figure(figsize=(6, 3), dpi=200)
    ax = plt.axes(projection=proj)
    ax.set_global()
    ax.coastlines(linewidth=0.6)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3)

    sc = ax.scatter(lon, lat, c=field, s=8, cmap='RdBu_r',
                    vmin=vmin, vmax=vmax,
                    transform=ccrs.PlateCarree())
    plt.title(title, fontsize=12)
    cbar = plt.colorbar(sc, orientation='horizontal', pad=0.05, shrink=0.8)
    cbar.set_label("EOF1 loading", fontsize=10)
    plt.tight_layout()
    plt.show()

# 用统一色标方便对比四个变量
vmax = np.max(np.abs([eof_heat, eof_cold, eof_tsum, eof_twin]))
vmin = -vmax

plot_eof_scatter(lon, lat, eof_heat, "EOF1 - Heatwave", vmin, vmax)
plot_eof_scatter(lon, lat, eof_cold, "EOF1 - Cold events", vmin, vmax)
plot_eof_scatter(lon, lat, eof_tsum, "EOF1 - Summer T", vmin, vmax)
plot_eof_scatter(lon, lat, eof_twin, "EOF1 - Winter T", vmin, vmax)

# PC1 时间序列
import matplotlib.pyplot as plt
import numpy as np

shuju1=np.load('F:/heatwave_chaos/data/EOF/Net1_EOF1_heat.npy')
shuju2=np.load('F:/heatwave_chaos/data/EOF/Net1_EOF1_cold.npy')
shuju3=np.load('F:/heatwave_chaos/data/EOF/Net1_EOF1_TSUM.npy')
shuju4=np.load('F:/heatwave_chaos/data/EOF/Net1_EOF1_TWIN.npy')

def standardize(arr):
    """对数组做 Z-score 标准化，忽略 NaN"""
    mean = np.nanmean(arr)
    std = np.nanstd(arr)
    if std == 0 or np.isnan(std):
        std = 1.0
    return (arr - mean) / std

# 分别标准化
shuju1_std = standardize(shuju1)
shuju2_std = standardize(shuju2)
shuju3_std = standardize(shuju3)
shuju4_std = standardize(shuju4)
pc1_std = standardize(pc1)

plt.figure(figsize=(8, 3), dpi=300)
#plt.figure(figsize=(8, 3.8), dpi=300)
# ===== 基础曲线 =====
# ==== PC1（主线）====
plt.plot(np.arange(nyear) + 1981, pc1_std, 
         color='#E63946', linewidth=3.2, marker='o', markersize=6,
         markerfacecolor='white', markeredgewidth=1.5,
         label='PC1', zorder=10)

# ==== 其他变量 ====
# plt.plot(np.arange(nyear) + 1981, shuju1_std,
#          color='#457B9D', linewidth=2.0, alpha=0.8, marker='o', markersize=5,
#          markerfacecolor='white', markeredgewidth=1.0,
#          label='HWN r=0.70')

# plt.plot(np.arange(nyear) + 1981, shuju2_std,
#          color='#A8DADC', linewidth=2.0, alpha=0.8, marker='o', markersize=5,
#          markerfacecolor='white', markeredgewidth=1.0,
#          label='CWN r=0.85')

plt.plot(np.arange(nyear) + 1981, shuju3_std,
          color='#6D6875', linewidth=2.0, alpha=0.8, marker='o', markersize=5,
          markerfacecolor='white', markeredgewidth=1.0,
          label='TSUM r=0.60')

plt.plot(np.arange(nyear) + 1981, shuju4_std,
          color='#B0B0B0', linewidth=2.0, alpha=0.8, marker='o', markersize=5,
          markerfacecolor='white', markeredgewidth=1.0,
          label='TWIN r=0.57')

# ===== 零线 =====
plt.axhline(0, color='gray', linewidth=1.2, linestyle='--', alpha=0.6, zorder=0)

# ===== 坐标轴样式 =====
plt.xlabel("Year", fontsize=16, fontweight='bold', labelpad=6)
#plt.ylabel("PC1", fontsize=16, fontweight='bold', labelpad=6)

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
#plt.legend(frameon=False, fontsize=13, loc='upper left')
plt.legend(
    frameon=False,
    fontsize=13,
    loc='upper right',
    bbox_to_anchor=(0.55, 1.15),
    ncol=2,              # 每行 2 个图例
    columnspacing=0.6,   # 列间距（可调）
    handlelength=2.0,    # 图例线长（可调）
)



plt.title("f", fontsize=20, fontweight='bold',x=-0.05)
#plt.text(108, 92,  r'$\mathbf{Z_I}$: HWN EOF1(8.10%)', ha='center', va='bottom', fontsize=23, fontweight='bold')
# 使用 tight_layout 来自动调整子图参数，避免边缘空白
plt.tight_layout()
plt.savefig('F:/heatwave_chaos/data/EOF/f.pdf', format='pdf', bbox_inches='tight')
plt.show()


def corr(x, y):
    """忽略 NaN 的 Pearson 相关系数"""
    mask = ~np.isnan(x) & ~np.isnan(y)
    if np.sum(mask) == 0:
        return np.nan
    return np.corrcoef(x[mask], y[mask])[0, 1]

r1 = corr(pc1_std, shuju1_std)
r2 = corr(pc1_std, shuju2_std)
r3 = corr(pc1_std, shuju3_std)
r4 = corr(pc1_std, shuju4_std)

print("=== 相关系数 (PC1 vs 各变量) ===")
print(f"PC1 vs HWN  : {r1:.3f}")
print(f"PC1 vs CWN  : {r2:.3f}")
print(f"PC1 vs TSUM : {r3:.3f}")
print(f"PC1 vs TWIN : {r4:.3f}")




#############################################pc和enso的关系
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

plt.figure(figsize=(5.8, 5.2), dpi=300)

# 散点
plt.scatter(
    summer_mean1, pc1_std,
    color="#EF767A",
    edgecolor="k",
    s=70,
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
plt.xlabel("ONI(Summer)", fontsize=16, fontweight='bold')
plt.ylabel("PC1", fontsize=16, fontweight='bold')

# 网格线
plt.grid(True, linestyle=":", alpha=0.4, zorder=0)

# 坐标轴样式
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.2)
ax.spines['bottom'].set_linewidth(1.2)
ax.set_facecolor("#F9F9F9")

# 刻度
ax.tick_params(axis='both', labelsize=15, width=1.1, length=5)
p_text = "p < 0.05" if p < 0.05 else f"p = {p:.3f}"
# 在图内标注 r、p（放左上角）
plt.text(
    0.8, 0.9,
    f"r = {r:.2f}\n{p_text}",
    transform=ax.transAxes,
    fontsize=15,
    va='top', ha='right',
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
)

# 图例（只显示回归线说明）
#plt.legend(frameon=False, fontsize=11, loc='lower right')
plt.title("g", fontsize=25, fontweight='bold',x=-0.05)
plt.tight_layout()
plt.savefig('F:/heatwave_chaos/data/EOF/g.pdf', format='pdf', bbox_inches='tight')
plt.show()







###############################################冬季
import pandas as pd
import numpy as np

# 1️⃣ 读取数据
df = pd.read_csv("F:/heatwave_chaos/nina34.anom(time).csv", header=None, names=["Date", "ONI"])
df["Date"] = pd.to_datetime(df["Date"])
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month

# 2️⃣ 为每条记录指定“冬季归属年”（即 DJF 属于前一年）
# 例如：1981年11月、12月 → 1981 冬季；1982年1-3月 → 1981 冬季
df["WinterYear"] = np.where(df["Month"] >= 11, df["Year"], df["Year"] - 1)

# 3️⃣ 筛选 1981–2020 年冬季
winter_df = df[(df["WinterYear"] >= 1981) & (df["WinterYear"] <= 2020) &
               (df["Month"].isin([11, 12, 1, 2, 3]))]

# 4️⃣ 按“冬季归属年”求平均 ONI
winter_mean = winter_df.groupby("WinterYear")["ONI"].mean().reset_index()
winter_mean.rename(columns={"WinterYear": "Year"}, inplace=True)

# 5️⃣ 转为 numpy 数组（只取 ONI）
winter_mean = np.array(winter_mean)
winter_mean = winter_mean[:, 1]

# 6️⃣ 打印与保存
print(winter_mean)
np.save("F:/heatwave_chaos/nina34.winter11-3.npy", winter_mean)

def standardize(arr):
    """对数组做 Z-score 标准化，忽略 NaN"""
    mean = np.nanmean(arr)
    std = np.nanstd(arr)
    if std == 0 or np.isnan(std):
        std = 1.0
    return (arr - mean) / std

# 分别标准化
winter_mean1 = standardize(winter_mean)
pc1_std = standardize(pc1)

plt.figure(figsize=(8, 3.8), dpi=300)

# ===== 基础曲线 =====
# ==== PC1（主线）====
plt.plot(np.arange(nyear) + 1981, pc1_std, 
         color='#E63946', linewidth=3.2, marker='o', markersize=6,
         markerfacecolor='white', markeredgewidth=1.5,
         label='PC1', zorder=10)

# ==== 其他变量 ====
plt.plot(np.arange(nyear) + 1981, winter_mean1,
         color='#457B9D', linewidth=2.0, alpha=0.8, marker='o', markersize=5,
         markerfacecolor='white', markeredgewidth=1.0,
         label='ONI11-3')

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

r1 = corr(pc1_std,  winter_mean1)

print("=== 相关系数 (PC1 vs 各变量) ===")
print(f"PC1 vs HWN  : {r1:.3f}")



import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

from scipy.stats import pearsonr


# 计算皮尔逊相关系数
r, p = pearsonr(pc1_std, winter_mean1)
print(f"相关系数 r = {r:.3f}, 显著性 p = {p:.3f}")

# 绘制散点图
plt.figure(figsize=(5.5, 5), dpi=300)
plt.scatter(winter_mean1, pc1_std, color="#EF767A", edgecolor="k", s=70, alpha=0.85)

# 拟合回归线
m, b = np.polyfit(winter_mean1, pc1_std, 1)
plt.plot(winter_mean1, m * winter_mean1 + b, color="black", linewidth=1.5, linestyle="--", label=f"r = {r:.3f}")

# 坐标轴标签
plt.xlabel("Winter ONI (1981–2020)", fontsize=14, fontweight='bold')
plt.ylabel("PC1 (standardized)", fontsize=14, fontweight='bold')

# 美化外观
plt.grid(True, linestyle=":", alpha=0.5)
plt.legend(frameon=False, fontsize=12)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().set_facecolor("#FAFAFA")

plt.tight_layout()
plt.show()




##########################和ao的关系
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

plt.figure(figsize=(8, 3.8), dpi=300)

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






import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

from scipy.stats import pearsonr


# 计算皮尔逊相关系数
r, p = pearsonr(pc1_std,  AO_winter_mean_std)
print(f"相关系数 r = {r:.3f}, 显著性 p = {p:.3f}")

# 绘制散点图
plt.figure(figsize=(5.5, 5), dpi=300)
plt.scatter(AO_winter_mean_std , pc1_std, color="#EF767A", edgecolor="k", s=70, alpha=0.85)

# 拟合回归线
m, b = np.polyfit(AO_winter_mean_std, pc1_std, 1)
plt.plot(AO_winter_mean_std , m * AO_winter_mean_std  + b, color="black", linewidth=1.5, linestyle="--", label=f"r = {r:.3f}")

# 坐标轴标签
plt.xlabel("Winter AO (1981–2020)", fontsize=14, fontweight='bold')
plt.ylabel("PC1 (standardized)", fontsize=14, fontweight='bold')

# 美化外观
plt.grid(True, linestyle=":", alpha=0.5)
plt.legend(frameon=False, fontsize=12)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().set_facecolor("#FAFAFA")

plt.tight_layout()
plt.show()





#################################pc1，ao和oni

import matplotlib.pyplot as plt
import numpy as np

shuju1=AO
shuju2=np.load('F:/heatwave_chaos/nina34.summer5-9.npy')


def standardize(arr):
    """对数组做 Z-score 标准化，忽略 NaN"""
    mean = np.nanmean(arr)
    std = np.nanstd(arr)
    if std == 0 or np.isnan(std):
        std = 1.0
    return (arr - mean) / std

# 分别标准化
shuju1_std = standardize(shuju1)
shuju2_std = standardize(shuju2)
# shuju3_std = standardize(shuju3)
# shuju4_std = standardize(shuju4)
pc1_std = standardize(pc1)

plt.figure(figsize=(8, 3), dpi=300)
#plt.figure(figsize=(8, 3.8), dpi=300)
# ===== 基础曲线 =====
# ==== PC1（主线）====
plt.plot(np.arange(nyear) + 1981, pc1_std, 
         color='#E63946', linewidth=3.2, marker='o', markersize=6,
         markerfacecolor='white', markeredgewidth=1.5,
         label='PC1', zorder=10)

# ==== 其他变量 ====
plt.plot(np.arange(nyear) + 1981, shuju1_std,
          color='#457B9D', linewidth=2.0, alpha=0.8, marker='o', markersize=5,
          markerfacecolor='white', markeredgewidth=1.0,
          label='AO r=-0.381')

plt.plot(np.arange(nyear) + 1981, shuju2_std,
          color='#A8DADC', linewidth=2.0, alpha=0.8, marker='o', markersize=5,
          markerfacecolor='white', markeredgewidth=1.0,
          label='ONI r=-0.676')

# plt.plot(np.arange(nyear) + 1981, shuju3_std,
#           color='#6D6875', linewidth=2.0, alpha=0.8, marker='o', markersize=5,
#           markerfacecolor='white', markeredgewidth=1.0,
#           label='TSUM r=0.60')

# plt.plot(np.arange(nyear) + 1981, shuju4_std,
#           color='#B0B0B0', linewidth=2.0, alpha=0.8, marker='o', markersize=5,
#           markerfacecolor='white', markeredgewidth=1.0,
#           label='TWIN r=0.57')

# ===== 零线 =====
plt.axhline(0, color='gray', linewidth=1.2, linestyle='--', alpha=0.6, zorder=0)

# ===== 坐标轴样式 =====
plt.xlabel("Year", fontsize=16, fontweight='bold', labelpad=6)
#plt.ylabel("PC1", fontsize=16, fontweight='bold', labelpad=6)

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
#plt.legend(frameon=False, fontsize=13, loc='upper left')
plt.legend(
    frameon=False,
    fontsize=13,
    loc='upper right',
    bbox_to_anchor=(0.69, 1.15),
    ncol=3,              # 每行 2 个图例
    columnspacing=0.6,   # 列间距（可调）
    handlelength=2.0,    # 图例线长（可调）
)



plt.title("e", fontsize=20, fontweight='bold',x=-0.05)
#plt.text(108, 92,  r'$\mathbf{Z_I}$: HWN EOF1(8.10%)', ha='center', va='bottom', fontsize=23, fontweight='bold')
# 使用 tight_layout 来自动调整子图参数，避免边缘空白
plt.tight_layout()
plt.savefig('F:/heatwave_chaos/data/EOF/e.pdf', format='pdf', bbox_inches='tight')
plt.show()


def corr(x, y):
    """忽略 NaN 的 Pearson 相关系数"""
    mask = ~np.isnan(x) & ~np.isnan(y)
    if np.sum(mask) == 0:
        return np.nan
    return np.corrcoef(x[mask], y[mask])[0, 1]

r1 = corr(pc1_std, shuju1_std)
r2 = corr(pc1_std, shuju2_std)
# r3 = corr(pc1_std, shuju3_std)
# r4 = corr(pc1_std, shuju4_std)

print("=== 相关系数 (PC1 vs 各变量) ===")
print(f"PC1 vs HWN  : {r1:.3f}")
print(f"PC1 vs CWN  : {r2:.3f}")
# print(f"PC1 vs TSUM : {r3:.3f}")
# print(f"PC1 vs TWIN : {r4:.3f}")





###############################################################3



from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.ticker import MaxNLocator
from matplotlib.gridspec import GridSpec


grid_lon = np.linspace(0,360, 360)
grid_lat = np.linspace(-90, 90, 180)
grid_lon, grid_lat = np.meshgrid(grid_lon, grid_lat)

# 对数据进行插值
eof_heat1= griddata((lon, lat), eof_heat, (grid_lon, grid_lat), method='cubic')



# 创建一个新的图形
plt.figure(figsize=(8, 8), dpi=300)
m = Basemap(projection='cyl', resolution='l', llcrnrlat=-88, urcrnrlat=90, llcrnrlon=0, urcrnrlon=350)
m.drawcoastlines(linewidth=0.5)
m.fillcontinents(color='none')
x, y = m(grid_lon, grid_lat)
contour = m.contourf(x, y, eof_heat1, 60, cmap='RdBu_r', vmin=-0.02, vmax=0.02)

parallels = np.arange(-80, 81, 40)  # 自己调间隔
m.drawparallels(parallels, labels=[1, 0, 0, 0],  # [左,右,上,下]，这里只在左边标
                fontsize=20, linewidth=0, dashes=[2, 2])

abs_vals = eof_heat
threshold = np.nanpercentile(abs_vals, 90)  # top 10%
mask1 = np.where(abs_vals >= threshold)  # 1 表示 top10%，其他为 nan
heatmask=heat[mask1]
avgheatmask=np.mean(heatmask,axis=0)
np.save('F:/heatwave_chaos/data/EOF/Net1_EOF1_heat.npy',avgheatmask)


abs_vals = eof_heat1
threshold = np.nanpercentile(abs_vals, 90)  # top 10%
mask = np.where(abs_vals >= threshold, 1, np.nan)  # 1 表示 top10%，其他为 nan

pos_mask = np.where(eof_heat1 >= threshold, 1, np.nan)
#neg_mask = np.where(eof_heat1 <= -threshold, 1, np.nan)

# 正异常区域 → 红斜线
m.contourf(x, y, pos_mask, levels=[0.5, 1.5],
           colors='none', hatches=['////'], zorder=6)

# 负异常区域 → 蓝斜线
# m.contourf(x, y, neg_mask, levels=[0.5, 1.5],
#            colors='none', hatches=['\\\\\\\\'], zorder=6)



meridians = np.arange(0, 361, 60)
m.drawmeridians(meridians, labels=[0, 0, 0, 1],  # 只在下边标
                fontsize=20, linewidth=0, dashes=[2, 2])
# x11, y11 = m(longitudes11, latitudes11)

# m.scatter(x11, y11, color='#74F2EB', s=100, edgecolor='#74F2EB', label='Laton1 Points', alpha=1, marker='^')
# 添加颜色条，增加字体大小
cbar = plt.colorbar(contour, orientation='horizontal', pad=0.05, shrink=1, aspect=35)
cbar.ax.tick_params(labelsize=20)  # 增大颜色条刻度字体
cbar.locator = MaxNLocator(nbins=5)  # 设置颜色条刻度为5
cbar.update_ticks()

# 添加标题
plt.title("a", fontsize=34, fontweight='bold',x=-0.05)
plt.text(108, 92,  r'$\mathbf{Z_I}$: HWN EOF1(8.10%)', ha='center', va='bottom', fontsize=23, fontweight='bold')
# 使用 tight_layout 来自动调整子图参数，避免边缘空白
plt.tight_layout()
plt.savefig('F:/heatwave_chaos/data/EOF/a.pdf', format='pdf', bbox_inches='tight')
# 显示图像
plt.show()



#图二
grid_lon = np.linspace(0,360, 360)
grid_lat = np.linspace(-90, 90, 180)
grid_lon, grid_lat = np.meshgrid(grid_lon, grid_lat)

# 对数据进行插值
eof_cold1= griddata((lon, lat), eof_cold, (grid_lon, grid_lat), method='cubic')
plt.figure(figsize=(8, 8), dpi=300)
m = Basemap(projection='cyl', resolution='l', llcrnrlat=-88, urcrnrlat=90, llcrnrlon=0, urcrnrlon=350)
m.drawcoastlines(linewidth=0.5)

m.fillcontinents(color='none')


contour = m.contourf(x, y, eof_cold1, 60, cmap='RdBu_r', vmin=-0.02, vmax=0.02)

parallels = np.arange(-80, 81, 40)  # 自己调间隔
m.drawparallels(parallels, labels=[1, 0, 0, 0],  # [左,右,上,下]，这里只在左边标
                fontsize=20, linewidth=0, dashes=[2, 2])
abs_vals = eof_cold
threshold = np.nanpercentile(abs_vals, 90)  # top 10%
mask1 = np.where(abs_vals >= threshold)  # 1 表示 top10%，其他为 nan
heatmask=cold[mask1]
avgheatmask=np.mean(heatmask,axis=0)
np.save('F:/heatwave_chaos/data/EOF/Net1_EOF1_cold.npy',avgheatmask)


abs_vals = eof_cold1
threshold = np.nanpercentile(abs_vals, 90)  # top 10%
mask = np.where(abs_vals >= threshold, 1, np.nan)  # 1 表示 top10%，其他为 nan

pos_mask = np.where(eof_cold1 >= threshold, 1, np.nan)
#neg_mask = np.where(eof_cold1 <= -threshold, 1, np.nan)

# 正异常区域 → 红斜线
m.contourf(x, y, pos_mask, levels=[0.5, 1.5],
           colors='none', hatches=['////'], zorder=6)

# 负异常区域 → 蓝斜线
# m.contourf(x, y, neg_mask, levels=[0.5, 1.5],
#            colors='none', hatches=['\\\\\\\\'], zorder=6)
# 加：经线（下方标经度）
meridians = np.arange(0, 361, 60)
m.drawmeridians(meridians, labels=[0, 0, 0, 1],  # 只在下边标
                fontsize=20, linewidth=0, dashes=[2, 2])
# x11, y11 = m(longitudes11, latitudes11)

# m.scatter(x11, y11, color='#74F2EB', s=100, edgecolor='#74F2EB', label='Laton1 Points', alpha=1, marker='^')
# 添加颜色条，增加字体大小
cbar = plt.colorbar(contour, orientation='horizontal', pad=0.05, shrink=1, aspect=35)
cbar.ax.tick_params(labelsize=20)  # 增大颜色条刻度字体
cbar.locator = MaxNLocator(nbins=5)  # 设置颜色条刻度为5
cbar.update_ticks()

# 添加标题
plt.title("b", fontsize=34, fontweight='bold',x=-0.05)
plt.text(108, 92,  r'$\mathbf{Z_I}$: CWN EOF1(8.10%)', ha='center', va='bottom', fontsize=23, fontweight='bold')
# 使用 tight_layout 来自动调整子图参数，避免边缘空白
plt.tight_layout()
plt.savefig('F:/heatwave_chaos/data/EOF/b.pdf', format='pdf', bbox_inches='tight')
# 显示图像
plt.show()





#图三
grid_lon = np.linspace(0,360, 360)
grid_lat = np.linspace(-90, 90, 180)
grid_lon, grid_lat = np.meshgrid(grid_lon, grid_lat)

# 对数据进行插值
eof_tsum1= griddata((lon, lat), eof_tsum, (grid_lon, grid_lat), method='cubic')


plt.figure(figsize=(8, 8), dpi=300)
m = Basemap(projection='cyl', resolution='l', llcrnrlat=-88, urcrnrlat=90, llcrnrlon=0, urcrnrlon=350)
m.drawcoastlines(linewidth=0.5)

m.fillcontinents(color='none')


contour = m.contourf(x, y, eof_tsum1, 60, cmap='RdBu_r', vmin=-0.02, vmax=0.02)

parallels = np.arange(-80, 81, 40)  # 自己调间隔
m.drawparallels(parallels, labels=[1, 0, 0, 0],  # [左,右,上,下]，这里只在左边标
                fontsize=20, linewidth=0, dashes=[2, 2])
abs_vals = eof_tsum
threshold = np.nanpercentile(abs_vals, 90)  # top 10%
mask1 = np.where(abs_vals >= threshold)  # 1 表示 top10%，其他为 nan
heatmask=avgtem_summer[mask1]
avgheatmask=np.mean(heatmask,axis=0)
np.save('F:/heatwave_chaos/data/EOF/Net1_EOF1_TSUM.npy',avgheatmask)


abs_vals = eof_tsum1
threshold = np.nanpercentile(abs_vals, 90)  # top 10%
mask = np.where(abs_vals >= threshold, 1, np.nan)  # 1 表示 top10%，其他为 nan

pos_mask = np.where(eof_tsum1 >= threshold, 1, np.nan)
#neg_mask = np.where(eof_cold1 <= -threshold, 1, np.nan)

# 正异常区域 → 红斜线
m.contourf(x, y, pos_mask, levels=[0.5, 1.5],
           colors='none', hatches=['////'], zorder=6)
# 加：经线（下方标经度）
meridians = np.arange(0, 361, 60)
m.drawmeridians(meridians, labels=[0, 0, 0, 1],  # 只在下边标
                fontsize=20, linewidth=0, dashes=[2, 2])
# x11, y11 = m(longitudes11, latitudes11)

# m.scatter(x11, y11, color='#74F2EB', s=100, edgecolor='#74F2EB', label='Laton1 Points', alpha=1, marker='^')
# 添加颜色条，增加字体大小
cbar = plt.colorbar(contour, orientation='horizontal', pad=0.05, shrink=1, aspect=35)
cbar.ax.tick_params(labelsize=20)  # 增大颜色条刻度字体
cbar.locator = MaxNLocator(nbins=5)  # 设置颜色条刻度为5
cbar.update_ticks()

# 添加标题
plt.title("c", fontsize=34, fontweight='bold',x=-0.05)
plt.text(108, 92,  r'$\mathbf{Z_I}$: TSUM EOF1(8.10%)', ha='center', va='bottom', fontsize=23, fontweight='bold')
# 使用 tight_layout 来自动调整子图参数，避免边缘空白
plt.tight_layout()
plt.savefig('F:/heatwave_chaos/data/EOF/c.pdf', format='pdf', bbox_inches='tight')
# 显示图像
plt.show()







#图四
grid_lon = np.linspace(0,360, 360)
grid_lat = np.linspace(-90, 90, 180)
grid_lon, grid_lat = np.meshgrid(grid_lon, grid_lat)

# 对数据进行插值
eof_twin1= griddata((lon, lat),  eof_twin, (grid_lon, grid_lat), method='cubic')


plt.figure(figsize=(8, 8), dpi=300)
m = Basemap(projection='cyl', resolution='l', llcrnrlat=-88, urcrnrlat=90, llcrnrlon=0, urcrnrlon=350)
m.drawcoastlines(linewidth=0.5)

m.fillcontinents(color='none')


contour = m.contourf(x, y,  eof_twin1, 60, cmap='RdBu_r', vmin=-0.02, vmax=0.02)

parallels = np.arange(-80, 81, 40)  # 自己调间隔
m.drawparallels(parallels, labels=[1, 0, 0, 0],  # [左,右,上,下]，这里只在左边标
                fontsize=20, linewidth=0, dashes=[2, 2])
abs_vals = eof_twin
threshold = np.nanpercentile(abs_vals, 90)  # top 10%
mask1 = np.where(abs_vals >= threshold)  # 1 表示 top10%，其他为 nan
heatmask=avgtem_winter[mask1]
avgheatmask=np.mean(heatmask,axis=0)
np.save('F:/heatwave_chaos/data/EOF/Net1_EOF1_TWIN.npy',avgheatmask)


abs_vals = eof_twin1
threshold = np.nanpercentile(abs_vals, 90)  # top 10%
mask = np.where(abs_vals >= threshold, 1, np.nan)  # 1 表示 top10%，其他为 nan

pos_mask = np.where(eof_twin1 >= threshold, 1, np.nan)
#neg_mask = np.where(eof_cold1 <= -threshold, 1, np.nan)

# 正异常区域 → 红斜线
m.contourf(x, y, pos_mask, levels=[0.5, 1.5],
           colors='none', hatches=['////'], zorder=6)
# 加：经线（下方标经度）
meridians = np.arange(0, 361, 60)
m.drawmeridians(meridians, labels=[0, 0, 0, 1],  # 只在下边标
                fontsize=20, linewidth=0, dashes=[2, 2])
# x11, y11 = m(longitudes11, latitudes11)

# m.scatter(x11, y11, color='#74F2EB', s=100, edgecolor='#74F2EB', label='Laton1 Points', alpha=1, marker='^')
# 添加颜色条，增加字体大小
cbar = plt.colorbar(contour, orientation='horizontal', pad=0.05, shrink=1, aspect=35)
cbar.ax.tick_params(labelsize=20)  # 增大颜色条刻度字体
cbar.locator = MaxNLocator(nbins=5)  # 设置颜色条刻度为5
cbar.update_ticks()

# 添加标题
plt.title("d", fontsize=34, fontweight='bold',x=-0.05)
plt.text(108, 92,  r'$\mathbf{Z_I}$: TWIN EOF1(8.10%)', ha='center', va='bottom', fontsize=23, fontweight='bold')
# 使用 tight_layout 来自动调整子图参数，避免边缘空白
plt.tight_layout()
plt.savefig('F:/heatwave_chaos/data/EOF/d.pdf', format='pdf', bbox_inches='tight')
# 显示图像
plt.show()





############################################################网络I-EOF2
mode = 1  # 取第1模态
eof_all = eofs[:, mode]   # (4 * npoint,)
pc2 = pcs[:, mode] 
eof_heat = eof_all[0*npoint : 1*npoint]   # 热浪 EOF1
eof_cold = eof_all[1*npoint : 2*npoint]   # 寒流 EOF1
eof_tsum = eof_all[2*npoint : 3*npoint]   # 夏季温度 EOF1
eof_twin = eof_all[3*npoint : 4*npoint]   # 冬季温度 EOF1

eof_heat = eof_heat*(-1)
eof_cold = eof_cold*(-1)
eof_tsum = eof_tsum*(-1)
eof_twin = eof_twin*(-1)

pc22 = pc2*(-1)


print("Mode 1 explains: {:.2f}% variance".format(varfrac[1]*100))
print("PC1 shape:", pc1.shape)


import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(8, 3.8), dpi=300)

# ===== 基础曲线 =====
plt.plot(np.arange(nyear) + 1981, pc22, 
         color='#EF767A', linewidth=2.5, marker='o', markersize=5,
         markerfacecolor='white', markeredgewidth=1.5,
         label='PC2')

# ===== 零线 =====
plt.axhline(0, color='gray', linewidth=1.2, linestyle='--', alpha=0.6, zorder=0)

# ===== 坐标轴样式 =====
plt.xlabel("Year", fontsize=16, fontweight='bold', labelpad=6)
plt.ylabel("PC2", fontsize=16, fontweight='bold', labelpad=6)

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




from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.ticker import MaxNLocator
from matplotlib.gridspec import GridSpec


grid_lon = np.linspace(0,360, 360)
grid_lat = np.linspace(-90, 90, 180)
grid_lon, grid_lat = np.meshgrid(grid_lon, grid_lat)

# 对数据进行插值
eof_heat1= griddata((lon, lat), eof_heat, (grid_lon, grid_lat), method='cubic')



# 创建一个新的图形
plt.figure(figsize=(8, 8), dpi=300)
m = Basemap(projection='cyl', resolution='l', llcrnrlat=-88, urcrnrlat=90, llcrnrlon=0, urcrnrlon=350)
m.drawcoastlines(linewidth=0.5)
m.fillcontinents(color='none')
contour = m.contourf(x, y, eof_heat1, 60, cmap='RdBu_r', vmin=-0.02, vmax=0.02)

parallels = np.arange(-80, 81, 40)  # 自己调间隔
m.drawparallels(parallels, labels=[1, 0, 0, 0],  # [左,右,上,下]，这里只在左边标
                fontsize=20, linewidth=0, dashes=[2, 2])


meridians = np.arange(0, 361, 60)
m.drawmeridians(meridians, labels=[0, 0, 0, 1],  # 只在下边标
                fontsize=20, linewidth=0, dashes=[2, 2])
# x11, y11 = m(longitudes11, latitudes11)

# m.scatter(x11, y11, color='#74F2EB', s=100, edgecolor='#74F2EB', label='Laton1 Points', alpha=1, marker='^')
# 添加颜色条，增加字体大小
cbar = plt.colorbar(contour, orientation='horizontal', pad=0.05, shrink=1, aspect=35)
cbar.ax.tick_params(labelsize=20)  # 增大颜色条刻度字体
cbar.locator = MaxNLocator(nbins=5)  # 设置颜色条刻度为5
cbar.update_ticks()

# 添加标题
plt.title("e", fontsize=34, fontweight='bold',x=-0.05)
plt.text(108, 92,  r'$\mathbf{Z_I}$: HWN EOF2(6.51%)', ha='center', va='bottom', fontsize=23, fontweight='bold')
# 使用 tight_layout 来自动调整子图参数，避免边缘空白
plt.tight_layout()
#plt.savefig('F:/heatwave/6242/最终版Fig/EOF/1.pdf', format='pdf', bbox_inches='tight')
# 显示图像
plt.show()



#图二
grid_lon = np.linspace(0,360, 360)
grid_lat = np.linspace(-90, 90, 180)
grid_lon, grid_lat = np.meshgrid(grid_lon, grid_lat)

# 对数据进行插值
eof_cold1= griddata((lon, lat), eof_cold, (grid_lon, grid_lat), method='cubic')
plt.figure(figsize=(8, 8), dpi=300)
m = Basemap(projection='cyl', resolution='l', llcrnrlat=-88, urcrnrlat=90, llcrnrlon=0, urcrnrlon=350)
m.drawcoastlines(linewidth=0.5)

m.fillcontinents(color='none')


contour = m.contourf(x, y, eof_cold1, 60, cmap='RdBu_r', vmin=-0.02, vmax=0.02)

parallels = np.arange(-80, 81, 40)  # 自己调间隔
m.drawparallels(parallels, labels=[1, 0, 0, 0],  # [左,右,上,下]，这里只在左边标
                fontsize=20, linewidth=0, dashes=[2, 2])

# 加：经线（下方标经度）
meridians = np.arange(0, 361, 60)
m.drawmeridians(meridians, labels=[0, 0, 0, 1],  # 只在下边标
                fontsize=20, linewidth=0, dashes=[2, 2])
# x11, y11 = m(longitudes11, latitudes11)

# m.scatter(x11, y11, color='#74F2EB', s=100, edgecolor='#74F2EB', label='Laton1 Points', alpha=1, marker='^')
# 添加颜色条，增加字体大小
cbar = plt.colorbar(contour, orientation='horizontal', pad=0.05, shrink=1, aspect=35)
cbar.ax.tick_params(labelsize=20)  # 增大颜色条刻度字体
cbar.locator = MaxNLocator(nbins=5)  # 设置颜色条刻度为5
cbar.update_ticks()

# 添加标题
plt.title("f", fontsize=34, fontweight='bold',x=-0.05)
plt.text(108, 92,  r'$\mathbf{Z_I}$: CWN EOF2(6.51%)', ha='center', va='bottom', fontsize=23, fontweight='bold')
# 使用 tight_layout 来自动调整子图参数，避免边缘空白
plt.tight_layout()
#plt.savefig('F:/heatwave/6242/最终版Fig/EOF/1.pdf', format='pdf', bbox_inches='tight')
# 显示图像
plt.show()



#图三
grid_lon = np.linspace(0,360, 360)
grid_lat = np.linspace(-90, 90, 180)
grid_lon, grid_lat = np.meshgrid(grid_lon, grid_lat)

# 对数据进行插值
eof_tsum1= griddata((lon, lat), eof_tsum, (grid_lon, grid_lat), method='cubic')


plt.figure(figsize=(8, 8), dpi=300)
m = Basemap(projection='cyl', resolution='l', llcrnrlat=-88, urcrnrlat=90, llcrnrlon=0, urcrnrlon=350)
m.drawcoastlines(linewidth=0.5)

m.fillcontinents(color='none')


contour = m.contourf(x, y, eof_tsum1, 60, cmap='RdBu_r', vmin=-0.02, vmax=0.02)

parallels = np.arange(-80, 81, 40)  # 自己调间隔
m.drawparallels(parallels, labels=[1, 0, 0, 0],  # [左,右,上,下]，这里只在左边标
                fontsize=20, linewidth=0, dashes=[2, 2])

# 加：经线（下方标经度）
meridians = np.arange(0, 361, 60)
m.drawmeridians(meridians, labels=[0, 0, 0, 1],  # 只在下边标
                fontsize=20, linewidth=0, dashes=[2, 2])
# x11, y11 = m(longitudes11, latitudes11)

# m.scatter(x11, y11, color='#74F2EB', s=100, edgecolor='#74F2EB', label='Laton1 Points', alpha=1, marker='^')
# 添加颜色条，增加字体大小
cbar = plt.colorbar(contour, orientation='horizontal', pad=0.05, shrink=1, aspect=35)
cbar.ax.tick_params(labelsize=20)  # 增大颜色条刻度字体
cbar.locator = MaxNLocator(nbins=5)  # 设置颜色条刻度为5
cbar.update_ticks()

# 添加标题
plt.title("g", fontsize=34, fontweight='bold',x=-0.05)
plt.text(108, 92,  r'$\mathbf{Z_I}$: TSUM EOF2(6.51%)', ha='center', va='bottom', fontsize=23, fontweight='bold')
# 使用 tight_layout 来自动调整子图参数，避免边缘空白
plt.tight_layout()
#plt.savefig('F:/heatwave/6242/最终版Fig/EOF/1.pdf', format='pdf', bbox_inches='tight')
# 显示图像
plt.show()







#图四
grid_lon = np.linspace(0,360, 360)
grid_lat = np.linspace(-90, 90, 180)
grid_lon, grid_lat = np.meshgrid(grid_lon, grid_lat)

# 对数据进行插值
eof_twin1= griddata((lon, lat),  eof_twin, (grid_lon, grid_lat), method='cubic')


plt.figure(figsize=(8, 8), dpi=300)
m = Basemap(projection='cyl', resolution='l', llcrnrlat=-88, urcrnrlat=90, llcrnrlon=0, urcrnrlon=350)
m.drawcoastlines(linewidth=0.5)

m.fillcontinents(color='none')


contour = m.contourf(x, y,  eof_twin1, 60, cmap='RdBu_r', vmin=-0.02, vmax=0.02)

parallels = np.arange(-80, 81, 40)  # 自己调间隔
m.drawparallels(parallels, labels=[1, 0, 0, 0],  # [左,右,上,下]，这里只在左边标
                fontsize=20, linewidth=0, dashes=[2, 2])

# 加：经线（下方标经度）
meridians = np.arange(0, 361, 60)
m.drawmeridians(meridians, labels=[0, 0, 0, 1],  # 只在下边标
                fontsize=20, linewidth=0, dashes=[2, 2])
# x11, y11 = m(longitudes11, latitudes11)

# m.scatter(x11, y11, color='#74F2EB', s=100, edgecolor='#74F2EB', label='Laton1 Points', alpha=1, marker='^')
# 添加颜色条，增加字体大小
cbar = plt.colorbar(contour, orientation='horizontal', pad=0.05, shrink=1, aspect=35)
cbar.ax.tick_params(labelsize=20)  # 增大颜色条刻度字体
cbar.locator = MaxNLocator(nbins=5)  # 设置颜色条刻度为5
cbar.update_ticks()

# 添加标题
plt.title("h", fontsize=34, fontweight='bold',x=-0.05)
plt.text(108, 92,  r'$\mathbf{Z_I}$: TWIN EOF2(6.51%)', ha='center', va='bottom', fontsize=23, fontweight='bold')
# 使用 tight_layout 来自动调整子图参数，避免边缘空白
plt.tight_layout()
#plt.savefig('F:/heatwave/6242/最终版Fig/EOF/1.pdf', format='pdf', bbox_inches='tight')
# 显示图像
plt.show()







############################################################网络I-EOF3
mode = 2  # 取第1模态
eof_all = eofs[:, mode]   # (4 * npoint,)
pc3 = pcs[:, mode] 
eof_heat = eof_all[0*npoint : 1*npoint]   # 热浪 EOF1
eof_cold = eof_all[1*npoint : 2*npoint]   # 寒流 EOF1
eof_tsum = eof_all[2*npoint : 3*npoint]   # 夏季温度 EOF1
eof_twin = eof_all[3*npoint : 4*npoint]   # 冬季温度 EOF1

# eof_heat = eof_heat*(-1)
# eof_cold = eof_cold*(-1)
# eof_tsum = eof_tsum*(-1)
# eof_twin = eof_twin*(-1)

pc3 = pc3


print("Mode 1 explains: {:.2f}% variance".format(varfrac[2]*100))
print("PC1 shape:", pc1.shape)


import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(8, 3.8), dpi=300)

# ===== 基础曲线 =====
plt.plot(np.arange(nyear) + 1981, pc3, 
         color='#EF767A', linewidth=2.5, marker='o', markersize=5,
         markerfacecolor='white', markeredgewidth=1.5,
         label='PC3')

# ===== 零线 =====
plt.axhline(0, color='gray', linewidth=1.2, linestyle='--', alpha=0.6, zorder=0)

# ===== 坐标轴样式 =====
plt.xlabel("Year", fontsize=16, fontweight='bold', labelpad=6)
plt.ylabel("PC3", fontsize=16, fontweight='bold', labelpad=6)

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


from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.ticker import MaxNLocator
from matplotlib.gridspec import GridSpec


grid_lon = np.linspace(0,360, 360)
grid_lat = np.linspace(-90, 90, 180)
grid_lon, grid_lat = np.meshgrid(grid_lon, grid_lat)

# 对数据进行插值
eof_heat1= griddata((lon, lat), eof_heat, (grid_lon, grid_lat), method='cubic')



# 创建一个新的图形
plt.figure(figsize=(8, 8), dpi=300)
m = Basemap(projection='cyl', resolution='l', llcrnrlat=-88, urcrnrlat=90, llcrnrlon=0, urcrnrlon=350)
m.drawcoastlines(linewidth=0.5)
m.fillcontinents(color='none')
contour = m.contourf(x, y, eof_heat1, 60, cmap='RdBu_r', vmin=-0.015, vmax=0.015)

parallels = np.arange(-80, 81, 40)  # 自己调间隔
m.drawparallels(parallels, labels=[1, 0, 0, 0],  # [左,右,上,下]，这里只在左边标
                fontsize=20, linewidth=0, dashes=[2, 2])


meridians = np.arange(0, 361, 60)
m.drawmeridians(meridians, labels=[0, 0, 0, 1],  # 只在下边标
                fontsize=20, linewidth=0, dashes=[2, 2])
# x11, y11 = m(longitudes11, latitudes11)

# m.scatter(x11, y11, color='#74F2EB', s=100, edgecolor='#74F2EB', label='Laton1 Points', alpha=1, marker='^')
# 添加颜色条，增加字体大小
cbar = plt.colorbar(contour, orientation='horizontal', pad=0.05, shrink=1, aspect=35)
cbar.ax.tick_params(labelsize=20)  # 增大颜色条刻度字体
cbar.locator = MaxNLocator(nbins=5)  # 设置颜色条刻度为5
cbar.update_ticks()

# 添加标题
plt.title("i", fontsize=34, fontweight='bold',x=-0.05)
plt.text(108, 92,  r'$\mathbf{Z_I}$: HWN EOF3(5.33%)', ha='center', va='bottom', fontsize=23, fontweight='bold')
# 使用 tight_layout 来自动调整子图参数，避免边缘空白
plt.tight_layout()
#plt.savefig('F:/heatwave/6242/最终版Fig/EOF/1.pdf', format='pdf', bbox_inches='tight')
# 显示图像
plt.show()



#图二
grid_lon = np.linspace(0,360, 360)
grid_lat = np.linspace(-90, 90, 180)
grid_lon, grid_lat = np.meshgrid(grid_lon, grid_lat)

# 对数据进行插值
eof_cold1= griddata((lon, lat), eof_cold, (grid_lon, grid_lat), method='cubic')
plt.figure(figsize=(8, 8), dpi=300)
m = Basemap(projection='cyl', resolution='l', llcrnrlat=-88, urcrnrlat=90, llcrnrlon=0, urcrnrlon=350)
m.drawcoastlines(linewidth=0.5)

m.fillcontinents(color='none')


contour = m.contourf(x, y, eof_cold1, 60, cmap='RdBu_r', vmin=-0.02, vmax=0.02)

parallels = np.arange(-80, 81, 40)  # 自己调间隔
m.drawparallels(parallels, labels=[1, 0, 0, 0],  # [左,右,上,下]，这里只在左边标
                fontsize=20, linewidth=0, dashes=[2, 2])

# 加：经线（下方标经度）
meridians = np.arange(0, 361, 60)
m.drawmeridians(meridians, labels=[0, 0, 0, 1],  # 只在下边标
                fontsize=20, linewidth=0, dashes=[2, 2])
# x11, y11 = m(longitudes11, latitudes11)

# m.scatter(x11, y11, color='#74F2EB', s=100, edgecolor='#74F2EB', label='Laton1 Points', alpha=1, marker='^')
# 添加颜色条，增加字体大小
cbar = plt.colorbar(contour, orientation='horizontal', pad=0.05, shrink=1, aspect=35)
cbar.ax.tick_params(labelsize=20)  # 增大颜色条刻度字体
cbar.locator = MaxNLocator(nbins=5)  # 设置颜色条刻度为5
cbar.update_ticks()

# 添加标题
plt.title("j", fontsize=34, fontweight='bold',x=-0.05)
plt.text(108, 92,  r'$\mathbf{Z_I}$: CWN EOF3(5.33%)', ha='center', va='bottom', fontsize=23, fontweight='bold')
# 使用 tight_layout 来自动调整子图参数，避免边缘空白
plt.tight_layout()
#plt.savefig('F:/heatwave/6242/最终版Fig/EOF/1.pdf', format='pdf', bbox_inches='tight')
# 显示图像
plt.show()



#图三
grid_lon = np.linspace(0,360, 360)
grid_lat = np.linspace(-90, 90, 180)
grid_lon, grid_lat = np.meshgrid(grid_lon, grid_lat)

# 对数据进行插值
eof_tsum1= griddata((lon, lat), eof_tsum, (grid_lon, grid_lat), method='cubic')


plt.figure(figsize=(8, 8), dpi=300)
m = Basemap(projection='cyl', resolution='l', llcrnrlat=-88, urcrnrlat=90, llcrnrlon=0, urcrnrlon=350)
m.drawcoastlines(linewidth=0.5)

m.fillcontinents(color='none')


contour = m.contourf(x, y, eof_tsum1, 60, cmap='RdBu_r', vmin=-0.02, vmax=0.02)

parallels = np.arange(-80, 81, 40)  # 自己调间隔
m.drawparallels(parallels, labels=[1, 0, 0, 0],  # [左,右,上,下]，这里只在左边标
                fontsize=20, linewidth=0, dashes=[2, 2])

# 加：经线（下方标经度）
meridians = np.arange(0, 361, 60)
m.drawmeridians(meridians, labels=[0, 0, 0, 1],  # 只在下边标
                fontsize=20, linewidth=0, dashes=[2, 2])
# x11, y11 = m(longitudes11, latitudes11)

# m.scatter(x11, y11, color='#74F2EB', s=100, edgecolor='#74F2EB', label='Laton1 Points', alpha=1, marker='^')
# 添加颜色条，增加字体大小
cbar = plt.colorbar(contour, orientation='horizontal', pad=0.05, shrink=1, aspect=35)
cbar.ax.tick_params(labelsize=20)  # 增大颜色条刻度字体
cbar.locator = MaxNLocator(nbins=5)  # 设置颜色条刻度为5
cbar.update_ticks()

# 添加标题
plt.title("k", fontsize=34, fontweight='bold',x=-0.05)
plt.text(108, 92,  r'$\mathbf{Z_I}$: TSUM EOF3(5.33%)', ha='center', va='bottom', fontsize=23, fontweight='bold')
# 使用 tight_layout 来自动调整子图参数，避免边缘空白
plt.tight_layout()
#plt.savefig('F:/heatwave/6242/最终版Fig/EOF/1.pdf', format='pdf', bbox_inches='tight')
# 显示图像
plt.show()







#图四
grid_lon = np.linspace(0,360, 360)
grid_lat = np.linspace(-90, 90, 180)
grid_lon, grid_lat = np.meshgrid(grid_lon, grid_lat)

# 对数据进行插值
eof_twin1= griddata((lon, lat),  eof_twin, (grid_lon, grid_lat), method='cubic')


plt.figure(figsize=(8, 8), dpi=300)
m = Basemap(projection='cyl', resolution='l', llcrnrlat=-88, urcrnrlat=90, llcrnrlon=0, urcrnrlon=350)
m.drawcoastlines(linewidth=0.5)

m.fillcontinents(color='none')


contour = m.contourf(x, y,  eof_twin1, 60, cmap='RdBu_r', vmin=-0.02, vmax=0.02)

parallels = np.arange(-80, 81, 40)  # 自己调间隔
m.drawparallels(parallels, labels=[1, 0, 0, 0],  # [左,右,上,下]，这里只在左边标
                fontsize=20, linewidth=0, dashes=[2, 2])

# 加：经线（下方标经度）
meridians = np.arange(0, 361, 60)
m.drawmeridians(meridians, labels=[0, 0, 0, 1],  # 只在下边标
                fontsize=20, linewidth=0, dashes=[2, 2])
# x11, y11 = m(longitudes11, latitudes11)

# m.scatter(x11, y11, color='#74F2EB', s=100, edgecolor='#74F2EB', label='Laton1 Points', alpha=1, marker='^')
# 添加颜色条，增加字体大小
cbar = plt.colorbar(contour, orientation='horizontal', pad=0.05, shrink=1, aspect=35)
cbar.ax.tick_params(labelsize=20)  # 增大颜色条刻度字体
cbar.locator = MaxNLocator(nbins=5)  # 设置颜色条刻度为5
cbar.update_ticks()

# 添加标题
plt.title("l", fontsize=34, fontweight='bold',x=-0.05)
plt.text(108, 92,  r'$\mathbf{Z_I}$: TWIN EOF3(5.33%)', ha='center', va='bottom', fontsize=23, fontweight='bold')
# 使用 tight_layout 来自动调整子图参数，避免边缘空白
plt.tight_layout()
#plt.savefig('F:/heatwave/6242/最终版Fig/EOF/1.pdf', format='pdf', bbox_inches='tight')
# 显示图像
plt.show()




















#pc


# from matplotlib import colors
# from matplotlib.ticker import MaxNLocator
# from matplotlib.ticker import MultipleLocator
# # 子图4: 绘制热浪、寒流和 PC1 的时间序列
# fig4 = plt.figure(figsize=(12, 8), dpi=300)  # 子图4大小
# ax4 = fig4.add_subplot(111)  # 使用 gridspec 布局中的子图
# heatwave = np.load('F:/heatwave/6242/最终版Fig/data/第一种分解方式/average_heat_values.npy')
# coldwave = np.load('F:/heatwave/6242/最终版Fig/data/第一种分解方式/average_cold_values.npy')
# pc1 = np.load('F:/heatwave/6242/最终版Fig/data/第一种分解方式/pc1.npy')
# years = np.arange(1981, 2021)  # 假设年份从1981到2021

# # 数据标准化
# allheat_scaled = heatwave / np.max(heatwave)
# allcold_scaled = coldwave / np.max(coldwave)
# pc1_scaled = pc1 / np.max(pc1)

# # 绘制时间序列折线图
# # ax4.plot(years, allheat_scaled, marker='o', markersize=10,linestyle='-', color='#B5382C', alpha=0.9, label='HWN (r=0.74)', linewidth=2)
# # ax4.plot(years, allcold_scaled, marker='^',markersize=10, linestyle='-', color='#254689', alpha=0.9, label='CWN (r=0.8)', linewidth=2)
# ax4.plot(years, allheat_scaled, marker='o', markersize=10,linestyle='-', color='#C74647', alpha=1, label='HWN', linewidth=2)
# ax4.plot(years, allcold_scaled, marker='^',markersize=10, linestyle='-', color='#456990', alpha=1, label='CWN', linewidth=2)
# ax4.plot(years, pc1_scaled, marker='s', markersize=10,linestyle='-', color='#6f4e39', alpha=1, label='PC2', linewidth=2)
# ax4.tick_params(axis='x', labelsize=25)  # 调整横轴刻度字体大小
# ax4.tick_params(axis='y', labelsize=25) 
# ax4.set_facecolor(colors.to_rgba('#FFFFFF', alpha=0.2))  # 透明度为0.5
# plt.title("c", fontsize=46, fontweight='bold',x=-0.03)
# ax4.set_xlabel('Years', fontsize=28, fontweight='bold')
# ax4.set_ylabel('Scaled Values', fontsize=28, fontweight='bold')
# #ax4.grid(True, linestyle='--', alpha=0.5)
# ax4.legend(fontsize=20, markerscale=1.5, loc='upper left', borderpad=1.2)

# # ax4.xaxis.set_major_locator(MultipleLocator(10))  # 设置主刻度间隔
# # ax4.xaxis.set_minor_locator(MultipleLocator(5))  # 设置副刻度间隔
# # ax4.yaxis.set_major_locator(MultipleLocator(0.4))  # 设置主刻度间隔
# # ax4.yaxis.set_minor_locator(MultipleLocator(0.2))  # 设置副刻度间隔

# # # 设置刻度和标签
# # ax4.tick_params(axis='both', which='major',  length=9,labelsize=25)
# # ax4.tick_params(axis='both', which='minor', length=6, color='black')

# # # 隐藏副刻度标签
# # ax4.tick_params(axis='x', which='minor', labelbottom=False)
# # ax4.tick_params(axis='y', which='minor', labelleft=False)
# xmajorLocator = MultipleLocator(10) #将x主刻度标签设置为20的倍数
# xmajorFormatter = FormatStrFormatter('%d') #设置x轴标签文本的格式
# ymajorLocator = MultipleLocator(0.4) #将y轴主刻度标签设置为0.5的倍数
# #ymajorFormatter = FormatStrFormatter('%d') #设置y轴标签文本的格式
#     #设置主刻度标签的位置,标签文本的格式
# ax4.xaxis.set_major_locator(xmajorLocator)
# ax4.xaxis.set_major_formatter(xmajorFormatter)
# ax4.yaxis.set_major_locator(ymajorLocator)
# #ax3.yaxis.set_major_formatter(ymajorFormatter)
    
#     #修改次刻度
# # xminorLocator = MultipleLocator(4) #将x轴次刻度标签设置为5的倍数
# # yminorLocator = MultipleLocator(0.2) #将此y轴次刻度标签设置为0.1的倍数
#     #设置次刻度标签的位置,没有标签文本格式
# # ax4.xaxis.set_minor_locator(xminorLocator)
# # ax4.yaxis.set_minor_locator(yminorLocator) 
# ax4.tick_params(which='both', width=5)
# ax4.tick_params(which='major', length=10)  
# ax4.tick_params(which='minor',length=7)
# ax4.tick_params(axis='both', labelsize=25)
# # 去掉不需要的边框（半框效果）
# # ax4.spines['top'].set_color('none')
# # ax4.spines['right'].set_color('none')
# ax4.spines['left'].set_linewidth(2)
# ax4.spines['bottom'].set_linewidth(2)

# plt.savefig('F:/heatwave/6242/最终版Fig/EOF/4.pdf', format='pdf', bbox_inches='tight')
# fig4.show()









#网络II
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

#挑选出西太平洋和亚欧大陆地区的韩流和热浪
#equatorial_indices = np.where((lat >= -90) & (lat <= 85) & (lon >= 30) & (lon <= 180))[0]

import numpy as np
from scipy.linalg import svd









import numpy as np
from scipy.linalg import svd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature


heat = heat.astype(np.float32)          # 热浪
cold = cold.astype(np.float32)          # 寒流
tsum = avgtem_summer.astype(np.float32) # 夏季平均温度
twin = avgtem_winter.astype(np.float32) # 冬季平均温度

lon  = np.load("F:/heatwave/longitude.npy")
lat  = np.load("F:/heatwave/latitude.npy")

npoint, nyear = heat.shape
assert cold.shape == (npoint, nyear)
assert tsum.shape == (npoint, nyear)
assert twin.shape == (npoint, nyear)

print("格点数:", npoint, " 年数:", nyear)

# ==============================
# 1. 转成 (time, space) 并做距平
# ==============================

def make_anomaly(x):
    """
    x: (space, time) -> 返回 (time, space) 的 anomaly
    """
    x_ts = x.T                        # (time, space)
    x_anom = x_ts - x_ts.mean(axis=0, keepdims=True)
    return x_anom                     # (time, space)

heat_a = make_anomaly(heat)
cold_a = make_anomaly(cold)
tsum_a = make_anomaly(tsum)
twin_a = make_anomaly(twin)

# ==============================
# 2. 变量归一化（不做面积权重）
# ==============================

def normalize_field(x_anom):
    """
    x_anom: (time, space)
    用整体 RMS 做归一化
    """
    # rms = np.sqrt(np.mean(x_anom**2))
    # if rms == 0:
    #     rms = 1.0
    # return x_anom / rms

    std = x_anom.std()
    if std == 0:
        std = 1.0
    return x_anom / std

heat_n = normalize_field(heat_a)
cold_n = normalize_field(cold_a)
tsum_n = normalize_field(tsum_a)
twin_n = normalize_field(twin_a)

# ==============================
# 3. 拼成联合场，做快速 EOF (SVD)
# ==============================

# 在“空间维”上拼接四个变量
# X: (time, 4 * npoint)
X = np.concatenate([cold_n, heat_n, twin_n, tsum_n], axis=1)
ntime, nspace_total = X.shape
print("联合场维度: time =", ntime, ", space_total =", nspace_total)

def eof_analysis_fast(data, neofs=5):
    """
    data: (time, space_total)
    用 SVD 直接做 EOF，不显式构造协方差矩阵
    返回:
      eofs:    (space_total, neofs)
      pcs:     (time, neofs)
      varfrac: (neofs,)
    """
    X = data - data.mean(axis=0, keepdims=True)   # 再去一次时间均值，保险
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    varfrac = (s**2) / np.sum(s**2)
    pcs  = U[:, :neofs] * s[:neofs]              # (time, neofs)
    eofs = Vt[:neofs, :].T                       # (space_total, neofs)
    return eofs, pcs, varfrac[:neofs]

neofs = 5
eofs, pcs, varfrac = eof_analysis_fast(X, neofs=neofs)

print("前 {} 个模态方差贡献率:".format(neofs))
for i in range(neofs):
    print("Mode {}: {:.2f}%".format(i+1, varfrac[i]*100))

# ==============================
# 4. 拆分回四个变量的 EOF 空间型
# ==============================

mode = 0  # 取第1模态
eof_all = eofs[:, mode]   # (4 * npoint,)

eof_heat = eof_all[0*npoint : 1*npoint]   # 热浪 EOF1
eof_cold = eof_all[1*npoint : 2*npoint]   # 寒流 EOF1
eof_tsum = eof_all[2*npoint : 3*npoint]   # 夏季温度 EOF1
eof_twin = eof_all[3*npoint : 4*npoint]   # 冬季温度 EOF1

pc1 = pcs[:, mode]  # 第1模态 PC 时间序列

print("Mode 1 explains: {:.2f}% variance".format(varfrac[0]*100))
print("PC1 shape:", pc1.shape)

# ==============================
# 5. 简单画图（可按需删掉）
# ==============================

def plot_eof_scatter(lon, lat, field, title, vmin=None, vmax=None):
    proj = ccrs.PlateCarree(central_longitude=180)
    fig = plt.figure(figsize=(6, 3), dpi=200)
    ax = plt.axes(projection=proj)
    ax.set_global()
    ax.coastlines(linewidth=0.6)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3)

    sc = ax.scatter(lon, lat, c=field, s=8, cmap='RdBu_r',
                    vmin=vmin, vmax=vmax,
                    transform=ccrs.PlateCarree())
    plt.title(title, fontsize=12)
    cbar = plt.colorbar(sc, orientation='horizontal', pad=0.05, shrink=0.8)
    cbar.set_label("EOF1 loading", fontsize=10)
    plt.tight_layout()
    plt.show()

# 用统一色标方便对比四个变量
vmax = np.max(np.abs([eof_heat, eof_cold, eof_tsum, eof_twin]))
vmin = -vmax

plot_eof_scatter(lon, lat, eof_heat, "EOF1 - Heatwave", vmin, vmax)
plot_eof_scatter(lon, lat, eof_cold, "EOF1 - Cold events", vmin, vmax)
plot_eof_scatter(lon, lat, eof_tsum, "EOF1 - Summer T", vmin, vmax)
plot_eof_scatter(lon, lat, eof_twin, "EOF1 - Winter T", vmin, vmax)

# PC1 时间序列
import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(8, 3.8), dpi=300)

# ===== 基础曲线 =====
plt.plot(np.arange(nyear) + 1981, pc1, 
         color='#EF767A', linewidth=2.5, marker='o', markersize=5,
         markerfacecolor='white', markeredgewidth=1.5,
         label='PC1')

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







from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.ticker import MaxNLocator
from matplotlib.gridspec import GridSpec


grid_lon = np.linspace(0,360, 360)
grid_lat = np.linspace(-90, 90, 180)
grid_lon, grid_lat = np.meshgrid(grid_lon, grid_lat)

# 对数据进行插值
eof_heat1= griddata((lon, lat), eof_heat, (grid_lon, grid_lat), method='cubic')



# 创建一个新的图形
plt.figure(figsize=(8, 8), dpi=300)
m = Basemap(projection='cyl', resolution='l', llcrnrlat=-88, urcrnrlat=90, llcrnrlon=0, urcrnrlon=350)
m.drawcoastlines(linewidth=0.5)
m.fillcontinents(color='none')
contour = m.contourf(x, y, eof_heat1, 60, cmap='RdBu_r', vmin=-0.02, vmax=0.02)

parallels = np.arange(-80, 81, 40)  # 自己调间隔
m.drawparallels(parallels, labels=[1, 0, 0, 0],  # [左,右,上,下]，这里只在左边标
                fontsize=20, linewidth=0, dashes=[2, 2])


meridians = np.arange(0, 361, 60)
m.drawmeridians(meridians, labels=[0, 0, 0, 1],  # 只在下边标
                fontsize=20, linewidth=0, dashes=[2, 2])
# x11, y11 = m(longitudes11, latitudes11)

# m.scatter(x11, y11, color='#74F2EB', s=100, edgecolor='#74F2EB', label='Laton1 Points', alpha=1, marker='^')
# 添加颜色条，增加字体大小
cbar = plt.colorbar(contour, orientation='horizontal', pad=0.05, shrink=1, aspect=35)
cbar.ax.tick_params(labelsize=20)  # 增大颜色条刻度字体
cbar.locator = MaxNLocator(nbins=5)  # 设置颜色条刻度为5
cbar.update_ticks()

# 添加标题
plt.title("a", fontsize=34, fontweight='bold',x=-0.05)
plt.text(108, 92,  r'$\mathbf{Z_{II}}$: CWN EOF1(16.84%)', ha='center', va='bottom', fontsize=23, fontweight='bold')
# 使用 tight_layout 来自动调整子图参数，避免边缘空白
plt.tight_layout()
#plt.savefig('F:/heatwave/6242/最终版Fig/EOF/1.pdf', format='pdf', bbox_inches='tight')
# 显示图像
plt.show()



#图二
grid_lon = np.linspace(0,360, 360)
grid_lat = np.linspace(-90, 90, 180)
grid_lon, grid_lat = np.meshgrid(grid_lon, grid_lat)

# 对数据进行插值
eof_cold1= griddata((lon, lat), eof_cold, (grid_lon, grid_lat), method='cubic')
plt.figure(figsize=(8, 8), dpi=300)
m = Basemap(projection='cyl', resolution='l', llcrnrlat=-88, urcrnrlat=90, llcrnrlon=0, urcrnrlon=350)
m.drawcoastlines(linewidth=0.5)

m.fillcontinents(color='none')


contour = m.contourf(x, y, eof_cold1, 60, cmap='RdBu_r', vmin=-0.02, vmax=0.02)

parallels = np.arange(-80, 81, 40)  # 自己调间隔
m.drawparallels(parallels, labels=[1, 0, 0, 0],  # [左,右,上,下]，这里只在左边标
                fontsize=20, linewidth=0, dashes=[2, 2])

# 加：经线（下方标经度）
meridians = np.arange(0, 361, 60)
m.drawmeridians(meridians, labels=[0, 0, 0, 1],  # 只在下边标
                fontsize=20, linewidth=0, dashes=[2, 2])
# x11, y11 = m(longitudes11, latitudes11)

# m.scatter(x11, y11, color='#74F2EB', s=100, edgecolor='#74F2EB', label='Laton1 Points', alpha=1, marker='^')
# 添加颜色条，增加字体大小
cbar = plt.colorbar(contour, orientation='horizontal', pad=0.05, shrink=1, aspect=35)
cbar.ax.tick_params(labelsize=20)  # 增大颜色条刻度字体
cbar.locator = MaxNLocator(nbins=5)  # 设置颜色条刻度为5
cbar.update_ticks()

# 添加标题
plt.title("b", fontsize=34, fontweight='bold',x=-0.05)
plt.text(108, 92,  r'$\mathbf{Z_{II}}$: HWN EOF1(16.84%)', ha='center', va='bottom', fontsize=23, fontweight='bold')
# 使用 tight_layout 来自动调整子图参数，避免边缘空白
plt.tight_layout()
#plt.savefig('F:/heatwave/6242/最终版Fig/EOF/1.pdf', format='pdf', bbox_inches='tight')
# 显示图像
plt.show()





#图三
grid_lon = np.linspace(0,360, 360)
grid_lat = np.linspace(-90, 90, 180)
grid_lon, grid_lat = np.meshgrid(grid_lon, grid_lat)

# 对数据进行插值
eof_tsum1= griddata((lon, lat), eof_tsum, (grid_lon, grid_lat), method='cubic')


plt.figure(figsize=(8, 8), dpi=300)
m = Basemap(projection='cyl', resolution='l', llcrnrlat=-88, urcrnrlat=90, llcrnrlon=0, urcrnrlon=350)
m.drawcoastlines(linewidth=0.5)

m.fillcontinents(color='none')


contour = m.contourf(x, y, eof_tsum1, 60, cmap='RdBu_r', vmin=-0.03, vmax=0.03)

parallels = np.arange(-80, 81, 40)  # 自己调间隔
m.drawparallels(parallels, labels=[1, 0, 0, 0],  # [左,右,上,下]，这里只在左边标
                fontsize=20, linewidth=0, dashes=[2, 2])

# 加：经线（下方标经度）
meridians = np.arange(0, 361, 60)
m.drawmeridians(meridians, labels=[0, 0, 0, 1],  # 只在下边标
                fontsize=20, linewidth=0, dashes=[2, 2])
# x11, y11 = m(longitudes11, latitudes11)

# m.scatter(x11, y11, color='#74F2EB', s=100, edgecolor='#74F2EB', label='Laton1 Points', alpha=1, marker='^')
# 添加颜色条，增加字体大小
cbar = plt.colorbar(contour, orientation='horizontal', pad=0.05, shrink=1, aspect=35)
cbar.ax.tick_params(labelsize=20)  # 增大颜色条刻度字体
cbar.locator = MaxNLocator(nbins=5)  # 设置颜色条刻度为5
cbar.update_ticks()

# 添加标题
plt.title("c", fontsize=34, fontweight='bold',x=-0.05)
plt.text(108, 92,  r'$\mathbf{Z_{II}}$: TWIN EOF1(16.84%)', ha='center', va='bottom', fontsize=23, fontweight='bold')
# 使用 tight_layout 来自动调整子图参数，避免边缘空白
plt.tight_layout()
#plt.savefig('F:/heatwave/6242/最终版Fig/EOF/1.pdf', format='pdf', bbox_inches='tight')
# 显示图像
plt.show()







#图四
grid_lon = np.linspace(0,360, 360)
grid_lat = np.linspace(-90, 90, 180)
grid_lon, grid_lat = np.meshgrid(grid_lon, grid_lat)

# 对数据进行插值
eof_twin1= griddata((lon, lat),  eof_twin, (grid_lon, grid_lat), method='cubic')


plt.figure(figsize=(8, 8), dpi=300)
m = Basemap(projection='cyl', resolution='l', llcrnrlat=-88, urcrnrlat=90, llcrnrlon=0, urcrnrlon=350)
m.drawcoastlines(linewidth=0.5)

m.fillcontinents(color='none')


contour = m.contourf(x, y,  eof_twin1, 60, cmap='RdBu_r', vmin=-0.03, vmax=0.03)

parallels = np.arange(-80, 81, 40)  # 自己调间隔
m.drawparallels(parallels, labels=[1, 0, 0, 0],  # [左,右,上,下]，这里只在左边标
                fontsize=20, linewidth=0, dashes=[2, 2])

# 加：经线（下方标经度）
meridians = np.arange(0, 361, 60)
m.drawmeridians(meridians, labels=[0, 0, 0, 1],  # 只在下边标
                fontsize=20, linewidth=0, dashes=[2, 2])
# x11, y11 = m(longitudes11, latitudes11)

# m.scatter(x11, y11, color='#74F2EB', s=100, edgecolor='#74F2EB', label='Laton1 Points', alpha=1, marker='^')
# 添加颜色条，增加字体大小
cbar = plt.colorbar(contour, orientation='horizontal', pad=0.05, shrink=1, aspect=35)
cbar.ax.tick_params(labelsize=20)  # 增大颜色条刻度字体
cbar.locator = MaxNLocator(nbins=5)  # 设置颜色条刻度为5
cbar.update_ticks()

# 添加标题
plt.title("d", fontsize=34, fontweight='bold',x=-0.05)
plt.text(108, 92,  r'$\mathbf{Z_{II}}$: TSUM EOF1(16.84%)', ha='center', va='bottom', fontsize=23, fontweight='bold')
# 使用 tight_layout 来自动调整子图参数，避免边缘空白
plt.tight_layout()
#plt.savefig('F:/heatwave/6242/最终版Fig/EOF/1.pdf', format='pdf', bbox_inches='tight')
# 显示图像
plt.show()





############################################################网络I-EOF2
mode = 1  # 取第1模态
eof_all = eofs[:, mode]   # (4 * npoint,)
pc2 = pcs[:, mode] 
eof_heat = eof_all[0*npoint : 1*npoint]   # 热浪 EOF1
eof_cold = eof_all[1*npoint : 2*npoint]   # 寒流 EOF1
eof_tsum = eof_all[2*npoint : 3*npoint]   # 夏季温度 EOF1
eof_twin = eof_all[3*npoint : 4*npoint]   # 冬季温度 EOF1

eof_heat = eof_heat
eof_cold = eof_cold
eof_tsum = eof_tsum
eof_twin = eof_twin

pc22 = pc2


print("Mode 1 explains: {:.2f}% variance".format(varfrac[1]*100))
print("PC1 shape:", pc1.shape)


import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(8, 3.8), dpi=300)

# ===== 基础曲线 =====
plt.plot(np.arange(nyear) + 1981, pc22, 
         color='#EF767A', linewidth=2.5, marker='o', markersize=5,
         markerfacecolor='white', markeredgewidth=1.5,
         label='PC2')

# ===== 零线 =====
plt.axhline(0, color='gray', linewidth=1.2, linestyle='--', alpha=0.6, zorder=0)

# ===== 坐标轴样式 =====
plt.xlabel("Year", fontsize=16, fontweight='bold', labelpad=6)
plt.ylabel("PC2", fontsize=16, fontweight='bold', labelpad=6)

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




from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.ticker import MaxNLocator
from matplotlib.gridspec import GridSpec


grid_lon = np.linspace(0,360, 360)
grid_lat = np.linspace(-90, 90, 180)
grid_lon, grid_lat = np.meshgrid(grid_lon, grid_lat)

# 对数据进行插值
eof_heat1= griddata((lon, lat), eof_heat, (grid_lon, grid_lat), method='cubic')



# 创建一个新的图形
plt.figure(figsize=(8, 8), dpi=300)
m = Basemap(projection='cyl', resolution='l', llcrnrlat=-88, urcrnrlat=90, llcrnrlon=0, urcrnrlon=350)
m.drawcoastlines(linewidth=0.5)
m.fillcontinents(color='none')
contour = m.contourf(x, y, eof_heat1, 60, cmap='RdBu_r', vmin=-0.02, vmax=0.02)

parallels = np.arange(-80, 81, 40)  # 自己调间隔
m.drawparallels(parallels, labels=[1, 0, 0, 0],  # [左,右,上,下]，这里只在左边标
                fontsize=20, linewidth=0, dashes=[2, 2])


meridians = np.arange(0, 361, 60)
m.drawmeridians(meridians, labels=[0, 0, 0, 1],  # 只在下边标
                fontsize=20, linewidth=0, dashes=[2, 2])
# x11, y11 = m(longitudes11, latitudes11)

# m.scatter(x11, y11, color='#74F2EB', s=100, edgecolor='#74F2EB', label='Laton1 Points', alpha=1, marker='^')
# 添加颜色条，增加字体大小
cbar = plt.colorbar(contour, orientation='horizontal', pad=0.05, shrink=1, aspect=35)
cbar.ax.tick_params(labelsize=20)  # 增大颜色条刻度字体
cbar.locator = MaxNLocator(nbins=5)  # 设置颜色条刻度为5
cbar.update_ticks()

# 添加标题
plt.title("e", fontsize=34, fontweight='bold',x=-0.05)
plt.text(108, 92,  r'$\mathbf{Z_{II}}$: CWN EOF2(6.97%)', ha='center', va='bottom', fontsize=23, fontweight='bold')
# 使用 tight_layout 来自动调整子图参数，避免边缘空白
plt.tight_layout()
#plt.savefig('F:/heatwave/6242/最终版Fig/EOF/1.pdf', format='pdf', bbox_inches='tight')
# 显示图像
plt.show()



#图二
grid_lon = np.linspace(0,360, 360)
grid_lat = np.linspace(-90, 90, 180)
grid_lon, grid_lat = np.meshgrid(grid_lon, grid_lat)

# 对数据进行插值
eof_cold1= griddata((lon, lat), eof_cold, (grid_lon, grid_lat), method='cubic')
plt.figure(figsize=(8, 8), dpi=300)
m = Basemap(projection='cyl', resolution='l', llcrnrlat=-88, urcrnrlat=90, llcrnrlon=0, urcrnrlon=350)
m.drawcoastlines(linewidth=0.5)

m.fillcontinents(color='none')


contour = m.contourf(x, y, eof_cold1, 60, cmap='RdBu_r', vmin=-0.016, vmax=0.015)

parallels = np.arange(-80, 81, 40)  # 自己调间隔
m.drawparallels(parallels, labels=[1, 0, 0, 0],  # [左,右,上,下]，这里只在左边标
                fontsize=20, linewidth=0, dashes=[2, 2])

# 加：经线（下方标经度）
meridians = np.arange(0, 361, 60)
m.drawmeridians(meridians, labels=[0, 0, 0, 1],  # 只在下边标
                fontsize=20, linewidth=0, dashes=[2, 2])
# x11, y11 = m(longitudes11, latitudes11)

# m.scatter(x11, y11, color='#74F2EB', s=100, edgecolor='#74F2EB', label='Laton1 Points', alpha=1, marker='^')
# 添加颜色条，增加字体大小
cbar = plt.colorbar(contour, orientation='horizontal', pad=0.05, shrink=1, aspect=35)
cbar.ax.tick_params(labelsize=20)  # 增大颜色条刻度字体
cbar.locator = MaxNLocator(nbins=5)  # 设置颜色条刻度为5
cbar.update_ticks()

# 添加标题
plt.title("f", fontsize=34, fontweight='bold',x=-0.05)
plt.text(108, 92,  r'$\mathbf{Z_{II}}$: HWN EOF2(6.97%)', ha='center', va='bottom', fontsize=23, fontweight='bold')
# 使用 tight_layout 来自动调整子图参数，避免边缘空白
plt.tight_layout()
#plt.savefig('F:/heatwave/6242/最终版Fig/EOF/1.pdf', format='pdf', bbox_inches='tight')
# 显示图像
plt.show()



#图三
grid_lon = np.linspace(0,360, 360)
grid_lat = np.linspace(-90, 90, 180)
grid_lon, grid_lat = np.meshgrid(grid_lon, grid_lat)

# 对数据进行插值
eof_tsum1= griddata((lon, lat), eof_tsum, (grid_lon, grid_lat), method='cubic')


plt.figure(figsize=(8, 8), dpi=300)
m = Basemap(projection='cyl', resolution='l', llcrnrlat=-88, urcrnrlat=90, llcrnrlon=0, urcrnrlon=350)
m.drawcoastlines(linewidth=0.5)

m.fillcontinents(color='none')


contour = m.contourf(x, y, eof_tsum1, 60, cmap='RdBu_r', vmin=-0.02, vmax=0.02)

parallels = np.arange(-80, 81, 40)  # 自己调间隔
m.drawparallels(parallels, labels=[1, 0, 0, 0],  # [左,右,上,下]，这里只在左边标
                fontsize=20, linewidth=0, dashes=[2, 2])

# 加：经线（下方标经度）
meridians = np.arange(0, 361, 60)
m.drawmeridians(meridians, labels=[0, 0, 0, 1],  # 只在下边标
                fontsize=20, linewidth=0, dashes=[2, 2])
# x11, y11 = m(longitudes11, latitudes11)

# m.scatter(x11, y11, color='#74F2EB', s=100, edgecolor='#74F2EB', label='Laton1 Points', alpha=1, marker='^')
# 添加颜色条，增加字体大小
cbar = plt.colorbar(contour, orientation='horizontal', pad=0.05, shrink=1, aspect=35)
cbar.ax.tick_params(labelsize=20)  # 增大颜色条刻度字体
cbar.locator = MaxNLocator(nbins=5)  # 设置颜色条刻度为5
cbar.update_ticks()

# 添加标题
plt.title("g", fontsize=34, fontweight='bold',x=-0.05)
plt.text(108, 92,  r'$\mathbf{Z_{II}}$: TWIN EOF2(6.97%)', ha='center', va='bottom', fontsize=23, fontweight='bold')
# 使用 tight_layout 来自动调整子图参数，避免边缘空白
plt.tight_layout()
#plt.savefig('F:/heatwave/6242/最终版Fig/EOF/1.pdf', format='pdf', bbox_inches='tight')
# 显示图像
plt.show()







#图四
grid_lon = np.linspace(0,360, 360)
grid_lat = np.linspace(-90, 90, 180)
grid_lon, grid_lat = np.meshgrid(grid_lon, grid_lat)

# 对数据进行插值
eof_twin1= griddata((lon, lat),  eof_twin, (grid_lon, grid_lat), method='cubic')


plt.figure(figsize=(8, 8), dpi=300)
m = Basemap(projection='cyl', resolution='l', llcrnrlat=-88, urcrnrlat=90, llcrnrlon=0, urcrnrlon=350)
m.drawcoastlines(linewidth=0.5)

m.fillcontinents(color='none')


contour = m.contourf(x, y,  eof_twin1, 60, cmap='RdBu_r', vmin=-0.02, vmax=0.02)

parallels = np.arange(-80, 81, 40)  # 自己调间隔
m.drawparallels(parallels, labels=[1, 0, 0, 0],  # [左,右,上,下]，这里只在左边标
                fontsize=20, linewidth=0, dashes=[2, 2])

# 加：经线（下方标经度）
meridians = np.arange(0, 361, 60)
m.drawmeridians(meridians, labels=[0, 0, 0, 1],  # 只在下边标
                fontsize=20, linewidth=0, dashes=[2, 2])
# x11, y11 = m(longitudes11, latitudes11)

# m.scatter(x11, y11, color='#74F2EB', s=100, edgecolor='#74F2EB', label='Laton1 Points', alpha=1, marker='^')
# 添加颜色条，增加字体大小
cbar = plt.colorbar(contour, orientation='horizontal', pad=0.05, shrink=1, aspect=35)
cbar.ax.tick_params(labelsize=20)  # 增大颜色条刻度字体
cbar.locator = MaxNLocator(nbins=5)  # 设置颜色条刻度为5
cbar.update_ticks()

# 添加标题
plt.title("h", fontsize=34, fontweight='bold',x=-0.05)
plt.text(108, 92,  r'$\mathbf{Z_{II}}$: TSUM EOF2(6.97%)', ha='center', va='bottom', fontsize=23, fontweight='bold')
# 使用 tight_layout 来自动调整子图参数，避免边缘空白
plt.tight_layout()
#plt.savefig('F:/heatwave/6242/最终版Fig/EOF/1.pdf', format='pdf', bbox_inches='tight')
# 显示图像
plt.show()







############################################################网络I-EOF3
mode = 2  # 取第1模态
eof_all = eofs[:, mode]   # (4 * npoint,)
pc3 = pcs[:, mode] 
eof_heat = eof_all[0*npoint : 1*npoint]   # 热浪 EOF1
eof_cold = eof_all[1*npoint : 2*npoint]   # 寒流 EOF1
eof_tsum = eof_all[2*npoint : 3*npoint]   # 夏季温度 EOF1
eof_twin = eof_all[3*npoint : 4*npoint]   # 冬季温度 EOF1

# eof_heat = eof_heat*(-1)
# eof_cold = eof_cold*(-1)
# eof_tsum = eof_tsum*(-1)
# eof_twin = eof_twin*(-1)

pc3 = pc3


print("Mode 1 explains: {:.2f}% variance".format(varfrac[2]*100))
print("PC1 shape:", pc1.shape)


import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(8, 3.8), dpi=300)

# ===== 基础曲线 =====
plt.plot(np.arange(nyear) + 1981, pc3, 
         color='#EF767A', linewidth=2.5, marker='o', markersize=5,
         markerfacecolor='white', markeredgewidth=1.5,
         label='PC3')

# ===== 零线 =====
plt.axhline(0, color='gray', linewidth=1.2, linestyle='--', alpha=0.6, zorder=0)

# ===== 坐标轴样式 =====
plt.xlabel("Year", fontsize=16, fontweight='bold', labelpad=6)
plt.ylabel("PC3", fontsize=16, fontweight='bold', labelpad=6)

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


from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.ticker import MaxNLocator
from matplotlib.gridspec import GridSpec


grid_lon = np.linspace(0,360, 360)
grid_lat = np.linspace(-90, 90, 180)
grid_lon, grid_lat = np.meshgrid(grid_lon, grid_lat)

# 对数据进行插值
eof_heat1= griddata((lon, lat), eof_heat, (grid_lon, grid_lat), method='cubic')



# 创建一个新的图形
plt.figure(figsize=(8, 8), dpi=300)
m = Basemap(projection='cyl', resolution='l', llcrnrlat=-88, urcrnrlat=90, llcrnrlon=0, urcrnrlon=350)
m.drawcoastlines(linewidth=0.5)
m.fillcontinents(color='none')
contour = m.contourf(x, y, eof_heat1, 60, cmap='RdBu_r', vmin=-0.015, vmax=0.015)

parallels = np.arange(-80, 81, 40)  # 自己调间隔
m.drawparallels(parallels, labels=[1, 0, 0, 0],  # [左,右,上,下]，这里只在左边标
                fontsize=20, linewidth=0, dashes=[2, 2])


meridians = np.arange(0, 361, 60)
m.drawmeridians(meridians, labels=[0, 0, 0, 1],  # 只在下边标
                fontsize=20, linewidth=0, dashes=[2, 2])
# x11, y11 = m(longitudes11, latitudes11)

# m.scatter(x11, y11, color='#74F2EB', s=100, edgecolor='#74F2EB', label='Laton1 Points', alpha=1, marker='^')
# 添加颜色条，增加字体大小
cbar = plt.colorbar(contour, orientation='horizontal', pad=0.05, shrink=1, aspect=35)
cbar.ax.tick_params(labelsize=20)  # 增大颜色条刻度字体
cbar.locator = MaxNLocator(nbins=5)  # 设置颜色条刻度为5
cbar.update_ticks()

# 添加标题
plt.title("i", fontsize=34, fontweight='bold',x=-0.05)
plt.text(108, 92,  r'$\mathbf{Z_{II}}$: CWN EOF3(5.11%)', ha='center', va='bottom', fontsize=23, fontweight='bold')
# 使用 tight_layout 来自动调整子图参数，避免边缘空白
plt.tight_layout()
#plt.savefig('F:/heatwave/6242/最终版Fig/EOF/1.pdf', format='pdf', bbox_inches='tight')
# 显示图像
plt.show()



#图二
grid_lon = np.linspace(0,360, 360)
grid_lat = np.linspace(-90, 90, 180)
grid_lon, grid_lat = np.meshgrid(grid_lon, grid_lat)

# 对数据进行插值
eof_cold1= griddata((lon, lat), eof_cold, (grid_lon, grid_lat), method='cubic')
plt.figure(figsize=(8, 8), dpi=300)
m = Basemap(projection='cyl', resolution='l', llcrnrlat=-88, urcrnrlat=90, llcrnrlon=0, urcrnrlon=350)
m.drawcoastlines(linewidth=0.5)

m.fillcontinents(color='none')


contour = m.contourf(x, y, eof_cold1, 60, cmap='RdBu_r', vmin=-0.015, vmax=0.015)

parallels = np.arange(-80, 81, 40)  # 自己调间隔
m.drawparallels(parallels, labels=[1, 0, 0, 0],  # [左,右,上,下]，这里只在左边标
                fontsize=20, linewidth=0, dashes=[2, 2])

# 加：经线（下方标经度）
meridians = np.arange(0, 361, 60)
m.drawmeridians(meridians, labels=[0, 0, 0, 1],  # 只在下边标
                fontsize=20, linewidth=0, dashes=[2, 2])
# x11, y11 = m(longitudes11, latitudes11)

# m.scatter(x11, y11, color='#74F2EB', s=100, edgecolor='#74F2EB', label='Laton1 Points', alpha=1, marker='^')
# 添加颜色条，增加字体大小
cbar = plt.colorbar(contour, orientation='horizontal', pad=0.05, shrink=1, aspect=35)
cbar.ax.tick_params(labelsize=20)  # 增大颜色条刻度字体
cbar.locator = MaxNLocator(nbins=5)  # 设置颜色条刻度为5
cbar.update_ticks()

# 添加标题
plt.title("j", fontsize=34, fontweight='bold',x=-0.05)
plt.text(108, 92,  r'$\mathbf{Z_{II}}$: HWN EOF3(5.11%)', ha='center', va='bottom', fontsize=23, fontweight='bold')
# 使用 tight_layout 来自动调整子图参数，避免边缘空白
plt.tight_layout()
#plt.savefig('F:/heatwave/6242/最终版Fig/EOF/1.pdf', format='pdf', bbox_inches='tight')
# 显示图像
plt.show()



#图三
grid_lon = np.linspace(0,360, 360)
grid_lat = np.linspace(-90, 90, 180)
grid_lon, grid_lat = np.meshgrid(grid_lon, grid_lat)

# 对数据进行插值
eof_tsum1= griddata((lon, lat), eof_tsum, (grid_lon, grid_lat), method='cubic')


plt.figure(figsize=(8, 8), dpi=300)
m = Basemap(projection='cyl', resolution='l', llcrnrlat=-88, urcrnrlat=90, llcrnrlon=0, urcrnrlon=350)
m.drawcoastlines(linewidth=0.5)

m.fillcontinents(color='none')


contour = m.contourf(x, y, eof_tsum1, 60, cmap='RdBu_r', vmin=-0.02, vmax=0.02)

parallels = np.arange(-80, 81, 40)  # 自己调间隔
m.drawparallels(parallels, labels=[1, 0, 0, 0],  # [左,右,上,下]，这里只在左边标
                fontsize=20, linewidth=0, dashes=[2, 2])

# 加：经线（下方标经度）
meridians = np.arange(0, 361, 60)
m.drawmeridians(meridians, labels=[0, 0, 0, 1],  # 只在下边标
                fontsize=20, linewidth=0, dashes=[2, 2])
# x11, y11 = m(longitudes11, latitudes11)

# m.scatter(x11, y11, color='#74F2EB', s=100, edgecolor='#74F2EB', label='Laton1 Points', alpha=1, marker='^')
# 添加颜色条，增加字体大小
cbar = plt.colorbar(contour, orientation='horizontal', pad=0.05, shrink=1, aspect=35)
cbar.ax.tick_params(labelsize=20)  # 增大颜色条刻度字体
cbar.locator = MaxNLocator(nbins=5)  # 设置颜色条刻度为5
cbar.update_ticks()

# 添加标题
plt.title("k", fontsize=34, fontweight='bold',x=-0.05)
plt.text(108, 92,  r'$\mathbf{Z_{II}}$: TWIN EOF3(5.11%)', ha='center', va='bottom', fontsize=23, fontweight='bold')
# 使用 tight_layout 来自动调整子图参数，避免边缘空白
plt.tight_layout()
#plt.savefig('F:/heatwave/6242/最终版Fig/EOF/1.pdf', format='pdf', bbox_inches='tight')
# 显示图像
plt.show()







#图四
grid_lon = np.linspace(0,360, 360)
grid_lat = np.linspace(-90, 90, 180)
grid_lon, grid_lat = np.meshgrid(grid_lon, grid_lat)

# 对数据进行插值
eof_twin1= griddata((lon, lat),  eof_twin, (grid_lon, grid_lat), method='cubic')


plt.figure(figsize=(8, 8), dpi=300)
m = Basemap(projection='cyl', resolution='l', llcrnrlat=-88, urcrnrlat=90, llcrnrlon=0, urcrnrlon=350)
m.drawcoastlines(linewidth=0.5)

m.fillcontinents(color='none')


contour = m.contourf(x, y,  eof_twin1, 60, cmap='RdBu_r', vmin=-0.015, vmax=0.015)

parallels = np.arange(-80, 81, 40)  # 自己调间隔
m.drawparallels(parallels, labels=[1, 0, 0, 0],  # [左,右,上,下]，这里只在左边标
                fontsize=20, linewidth=0, dashes=[2, 2])

# 加：经线（下方标经度）
meridians = np.arange(0, 361, 60)
m.drawmeridians(meridians, labels=[0, 0, 0, 1],  # 只在下边标
                fontsize=20, linewidth=0, dashes=[2, 2])
# x11, y11 = m(longitudes11, latitudes11)

# m.scatter(x11, y11, color='#74F2EB', s=100, edgecolor='#74F2EB', label='Laton1 Points', alpha=1, marker='^')
# 添加颜色条，增加字体大小
cbar = plt.colorbar(contour, orientation='horizontal', pad=0.05, shrink=1, aspect=35)
cbar.ax.tick_params(labelsize=20)  # 增大颜色条刻度字体
cbar.locator = MaxNLocator(nbins=5)  # 设置颜色条刻度为5
cbar.update_ticks()

# 添加标题
plt.title("l", fontsize=34, fontweight='bold',x=-0.05)
plt.text(108, 92,  r'$\mathbf{Z_{II}}$: TSUM EOF3(5.11%)', ha='center', va='bottom', fontsize=23, fontweight='bold')
# 使用 tight_layout 来自动调整子图参数，避免边缘空白
plt.tight_layout()
#plt.savefig('F:/heatwave/6242/最终版Fig/EOF/1.pdf', format='pdf', bbox_inches='tight')
# 显示图像
plt.show()

