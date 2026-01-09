# -*- coding: utf-8 -*-
"""
Created on Sun Nov 30 20:00:21 2025

@author: yuping
"""






#value=np.load("F:/heatwave/worldtongbu/42yearworld/detrend_Crosscorrelationr_value.npy")
value=np.load("F:/heatwave_chaos/审稿意见程序/阈值2day_4day建网结果/htoc_network_yuzhi90.npy")

data = np.nan_to_num(value)
# data = data[np.ix_(nodes, nodes)]
# absdata=abs(data)
# yuzhi=np.percentile(absdata,95)
# print(yuzhi)
#热浪对寒流
yuzhi=0.3194306181922703
xuhao1=6242
Hayt11  = np.zeros((xuhao1,xuhao1))  
for y in range(xuhao1):
    print(y)
    zhi=data[y,:]
    
    ayt22=np.zeros(xuhao1) 
    for h in range(xuhao1):
        if abs(zhi[h])>yuzhi:
            ayt22[h]=1
            #ayt22[h]=zhi[h]
        else:
            ayt22[h]=0
            
    Hayt11[y,:]=ayt22
    
#chudu    
chu_heataa=np.sum(Hayt11,axis=1)
#rudu
ru_heataa=np.sum(Hayt11,axis=0)             
 
degree1=chu_heataa-ru_heataa

np.save("F:/heatwave_chaos/审稿意见程序/阈值2day_4day建网结果/Net_I_div6242_yuzhi90.npy",degree1)

# d1=plt.hist(chu_heataa, bins=20,density=True,alpha=1)
# realchudy=d1[0]
# realchudx=d1[1][:-1]+(d1[1][1:]-d1[1][:-1])/2
# plt.plot(realchudx,realchudy,'-',marker = 's')




# d2=plt.hist(ru_heataa, bins=20,density=True,alpha=1)
# realrudy2=d2[0]
# realrudx2=d2[1][:-1]+(d2[1][1:]-d2[1][:-1])/2
# plt.plot(realrudx2,realrudy2,'-',marker = 's')


d=plt.hist(degree1, bins=20,density=True,alpha=1)
realdivergencedy=d[0]
realdivergencedx=d[1][:-1]+(d[1][1:]-d[1][:-1])/2
plt.plot(realdivergencedx,realdivergencedy,'-',marker = 's')




# phasevalue=np.load("F:/heatwave/worldtongbu/42yearworld/相位/ramdom_rvalue1.npy")
# phasedata = np.nan_to_num(phasevalue)
# phasedata = phasedata[np.ix_(nodes, nodes)]
# # phasedata1=abs(phasedata)
# # yuzhi=np.percentile(phasedata1,98)
# # print(yuzhi)
# #热浪对寒流
# yuzhi=0.37224363377378117
# xuhao1=6242
# phaseHayt11  = np.zeros((xuhao1,xuhao1))  
# for y in range(xuhao1):
#     print(y)
#     zhi=phasedata[y,:]
    
#     ayt22=np.zeros(xuhao1) 
#     for h in range(xuhao1):
#         if abs(zhi[h])>yuzhi:
#             ayt22[h]=1
#             #ayt22[h]=zhi[h]
#         else:
#             ayt22[h]=0
            
#     phaseHayt11[y,:]=ayt22
    
# #chudu    
# phasechu_heataa=np.sum(phaseHayt11,axis=1)
# #rudu
# phaseru_heataa=np.sum(phaseHayt11,axis=0)             
 

# phasedivergence=phaseru_heataa-phasechu_heataa


# # d3=plt.hist(phasechu_heataa, bins=30,density=True,alpha=1)
# # phasechudy1=d3[0]
# # phasechudx1=d3[1][:-1]+(d3[1][1:]-d3[1][:-1])/2
# # plt.plot(phasechudx1,phasechudy1,'-',marker = 's')
# # #plt.title('shuffle out-degree')



# # d4=plt.hist(phaseru_heataa, bins=20,density=True,alpha=1)
# # phaserudy2=d4[0]
# # phaserudx2=d4[1][:-1]+(d4[1][1:]-d4[1][:-1])/2
# # plt.plot(phaserudx2,phaserudy2,'-',marker = 's')
# #plt.title('shuffle in-degree')

# d4=plt.hist(phasedivergence, bins=20,density=True,alpha=1)
# phasedivergencedy=d4[0]
# phasedivergencedx=d4[1][:-1]+(d4[1][1:]-d4[1][:-1])/2
# plt.plot(phasedivergencedx,phasedivergencedy,'-',marker = 's')


ramvalue=np.load("F:/heatwave/worldtongbu/42yearworld/随机打乱/shuffle_rvalue7.npy")
ramdata = np.nan_to_num(ramvalue)
#ramdata = ramdata[np.ix_(nodes, nodes)]
# ramdata1=abs(ramdata)
# yuzhi=np.percentile(ramdata1,98)
# print(yuzhi)
#热浪对寒流
yuzhi=0.37924712031545604
xuhao1=6242
ramHayt11  = np.zeros((xuhao1,xuhao1))  
for y in range(xuhao1):
    print(y)
    zhi=ramdata[y,:]
    
    ayt22=np.zeros(xuhao1) 
    for h in range(xuhao1):
        if abs(zhi[h])>yuzhi:
            ayt22[h]=1
            #ayt22[h]=zhi[h]
        else:
            ayt22[h]=0
            
    ramHayt11[y,:]=ayt22
    
#chudu    
ramchu_heataa=np.sum(ramHayt11,axis=1)
#rudu
ramru_heataa=np.sum(ramHayt11,axis=0)   

ramdegree=ramru_heataa+ramchu_heataa


# d5=plt.hist(ramchu_heataa, bins=30,density=True,alpha=1)
# ramchudy1=d5[0]
# ramchudx1=d5[1][:-1]+(d5[1][1:]-d5[1][:-1])/2
# plt.plot(ramchudx1,ramchudy1,'-',marker = 's')
# #plt.title('shuffle out-degree')



# d6=plt.hist(ramru_heataa, bins=20,density=True,alpha=1)
# ramrudy2=d6[0]
# ramrudx2=d6[1][:-1]+(d6[1][1:]-d6[1][:-1])/2
# plt.plot(ramrudx2,ramrudy2,'-',marker = 's')
#plt.title('shuffle in-degree')

d6=plt.hist(ramdegree, bins=20,density=True,alpha=1)
ramdivergencedy=d6[0]
ramdivergencedx=d6[1][:-1]+(d6[1][1:]-d6[1][:-1])/2
plt.plot(ramdivergencedx,ramdivergencedy,'-',marker = 's')



plt.plot(realdivergencedx,realdivergencedy,'-',marker = 's',label='original data')
plt.plot(ramdivergencedx,ramdivergencedy,'-',marker = 's',label='phase random data')
#plt.plot(phasedivergencedx,phasedivergencedy,'-',marker = 's',label='ramdom data')
plt.legend(prop={'size': 10})
#plt.xlabel('Network I in-degree')
plt.ylabel('PDF')
#plt.title('Average R Value by Distance')
plt.grid(True)









import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np



import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#Hayt11/Cayt11带权重，Hayt111/Cayt111不带权重
# value=np.load("F:/heatwave/worldtongbu/42yearworld/detrend_Crosscorrelationr_value.npy")
# data = np.nan_to_num(value)
# absdata=abs(data)
# yuzhi=np.percentile(absdata,95)
# print(yuzhi)

#value=np.load("F:/heatwave/worldtongbu/42yearworld/ColdtoHeat_Crosscorrelationr_value.npy")
value=np.load("F:/heatwave_chaos/审稿意见程序/阈值2day_4day建网结果/ctoh_network_yuzhi10.npy")

data = np.nan_to_num(value)
#data = data[np.ix_(nodes, nodes)]
# absdata=abs(data)
# yuzhi=np.percentile(absdata,95)
# print(yuzhi)
# #热浪对寒流
yuzhi=0.3190930664932894
xuhao1=6242
Hayt11  = np.zeros((xuhao1,xuhao1))  
for y in range(xuhao1):
    print(y)
    zhi=data[y,:]
    
    ayt22=np.zeros(xuhao1) 
    for h in range(xuhao1):
        if abs(zhi[h])>yuzhi:
            ayt22[h]=1
            #ayt22[h]=zhi[h]
        else:
            ayt22[h]=0
            
    Hayt11[y,:]=ayt22
    
#chudu    
chu_coldaa=np.sum(Hayt11,axis=1)
#rudu
ru_coldaa=np.sum(Hayt11,axis=0)             
 
colddivergence=chu_coldaa-ru_coldaa

np.save("F:/heatwave_chaos/审稿意见程序/阈值2day_4day建网结果/Net_II_div6242_yuzhi10.npy",colddivergence)

# d1=plt.hist(chu_coldaa, bins=20,density=True,alpha=1)
# realchudy=d1[0]
# realchudx=d1[1][:-1]+(d1[1][1:]-d1[1][:-1])/2
# plt.plot(realchudx,realchudy,'-',marker = 's')



# d2=plt.hist(ru_coldaa, bins=20,density=True,alpha=1)
# realrudy2=d2[0]
# realrudx2=d2[1][:-1]+(d2[1][1:]-d2[1][:-1])/2
# plt.plot(realrudx2,realrudy2,'-',marker = 's')



d2=plt.hist(colddivergence, bins=20,density=True,alpha=1)
colddivergencedy=d2[0]
colddivergencedx=d2[1][:-1]+(d2[1][1:]-d2[1][:-1])/2
plt.plot(colddivergencedx,colddivergencedy,'-',marker = 's')




# phasevalue=np.load("F:/heatwave/worldtongbu/42yearworld/相位/ctoh_ramdom_rvalue1.npy")
# phasedata = np.nan_to_num(phasevalue)
# phasedata = phasedata[np.ix_(nodes, nodes)]
# # phasedata1=abs(phasedata)
# # yuzhi=np.percentile(phasedata1,98)
# # print(yuzhi)
# #热浪对寒流
# yuzhi=0.37589853385264194
# xuhao1=6242
# phaseHayt11  = np.zeros((xuhao1,xuhao1))  
# for y in range(xuhao1):
#     print(y)
#     zhi=phasedata[y,:]
    
#     ayt22=np.zeros(xuhao1) 
#     for h in range(xuhao1):
#         if abs(zhi[h])>yuzhi:
#             ayt22[h]=1
#             #ayt22[h]=zhi[h]
#         else:
#             ayt22[h]=0
            
#     phaseHayt11[y,:]=ayt22
    
# #chudu    
# phasechu_coldaa=np.sum(phaseHayt11,axis=1)
# #rudu
# phaseru_coldaa=np.sum(phaseHayt11,axis=0)             
 
# phasecolddivergence=phaseru_coldaa-phasechu_coldaa



# d3=plt.hist(phasechu_coldaa, bins=30,density=True,alpha=1)
# phasechudy1=d3[0]
# phasechudx1=d3[1][:-1]+(d3[1][1:]-d3[1][:-1])/2
# plt.plot(phasechudx1,phasechudy1,'-',marker = 's')
# #plt.title('shuffle out-degree')



# d4=plt.hist(phaseru_coldaa, bins=20,density=True,alpha=1)
# phaserudy2=d4[0]
# phaserudx2=d4[1][:-1]+(d4[1][1:]-d4[1][:-1])/2
# plt.plot(phaserudx2,phaserudy2,'-',marker = 's')
#plt.title('shuffle in-degree')

d4=plt.hist(phasecolddivergence, bins=20,density=True,alpha=1)
phasecolddivergencedy=d4[0]
phasecolddivergencedx=d4[1][:-1]+(d4[1][1:]-d4[1][:-1])/2
plt.plot(phasecolddivergencedx,phasecolddivergencedy,'-',marker = 's')
#plt.title('shuffle in-degree')


ramvalue=np.load("F:/heatwave/worldtongbu/42yearworld/随机打乱/ColdtoHeat_shuffle_rvalue.npy")
ramdata = np.nan_to_num(ramvalue)
ramdata = ramdata[np.ix_(nodes, nodes)]
# ramdata1=abs(ramdata)
# yuzhi=np.percentile(ramdata1,98)
# print(yuzhi)
#热浪对寒流
yuzhi=0.38256521004767496
xuhao1=6242
ramHayt11  = np.zeros((xuhao1,xuhao1))  
for y in range(xuhao1):
    print(y)
    zhi=ramdata[y,:]
    
    ayt22=np.zeros(xuhao1) 
    for h in range(xuhao1):
        if abs(zhi[h])>yuzhi:
            ayt22[h]=1
            #ayt22[h]=zhi[h]
        else:
            ayt22[h]=0
            
    ramHayt11[y,:]=ayt22
    
#chudu    
ramchu_coldaa=np.sum(ramHayt11,axis=1)
#rudu
ramru_coldaa=np.sum(ramHayt11,axis=0)   

ramcolddivergence=ramru_coldaa+ramchu_coldaa

# d5=plt.hist(ramchu_coldaa, bins=30,density=True,alpha=1)
# ramchudy1=d5[0]
# ramchudx1=d5[1][:-1]+(d5[1][1:]-d5[1][:-1])/2
# plt.plot(ramchudx1,ramchudy1,'-',marker = 's')
# #plt.title('shuffle out-degree')



# d6=plt.hist(ramru_coldaa, bins=20,density=True,alpha=1)
# ramrudy2=d6[0]
# ramrudx2=d6[1][:-1]+(d6[1][1:]-d6[1][:-1])/2
# plt.plot(ramrudx2,ramrudy2,'-',marker = 's')
#plt.title('shuffle in-degree')

d6=plt.hist(ramcolddivergence, bins=20,density=True,alpha=1)
ramcolddivergencedy=d6[0]
ramcolddivergencedx=d6[1][:-1]+(d6[1][1:]-d6[1][:-1])/2
plt.plot(ramcolddivergencedx,ramcolddivergencedy,'-',marker = 's')
#plt.title('shuffle in-degree')



plt.plot(colddivergencedx,colddivergencedy,'-',marker = 's',label='original data')
plt.plot(ramcolddivergencedx,ramcolddivergencedy,'-',marker = 's',label='phase random data')
#plt.plot(phasecolddivergencedx,phasecolddivergencedy,'-',marker = 's',label='ramdom data')
plt.legend(prop={'size': 10})
plt.xlabel('Network II out-degree')
plt.ylabel('PDF')
#plt.title('Average R Value by Distance')
plt.grid(True)












############################################################画图
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import FuncFormatter
from scipy.interpolate import interp1d  # 新增插值库
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import FormatStrFormatter

plt.rcParams['font.family'] = 'sans-serif'  # 无衬线字体
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica'] + plt.rcParams['font.sans-serif'] 
plt.rcParams['axes.linewidth'] = 1.5  # 坐标轴线宽
plt.rcParams['legend.frameon'] = False  # 图例无边框
plt.rcParams['xtick.major.width'] = 1.5  # X轴主刻度线宽
plt.rcParams['ytick.major.width'] = 1.5  # Y轴主刻度线宽
plt.rcParams['xtick.minor.width'] = 1.  # X轴次刻度线宽
plt.rcParams['ytick.minor.width'] = 1.  # Y轴次刻度线宽
plt.rcParams['xtick.labelsize'] = 35  # X轴刻度标签字体大小
plt.rcParams['ytick.labelsize'] = 35  # Y轴刻度标签字体大小
plt.rcParams['font.size'] = 35  # 全局字体大小
plt.rcParams['legend.fontsize'] = 35  # 图例字体大小
#plt.rcParams['mathtext.fontset'] = 'stix'  # 设置数学字体
#np.save('F:/heatwave_chaos/data/realdivergencedx.npy', realdivergencedx)
realdivergencedx = np.load('F:/heatwave_chaos/data/realdivergencedx.npy')

#np.save('F:/heatwave_chaos/data/realdivergencedy.npy', realdivergencedy)
realdivergencedy = np.load('F:/heatwave_chaos/data/realdivergencedy.npy')

#np.save('F:/heatwave_chaos/data/ramdivergencedx.npy', ramdivergencedx)
ramdivergencedx= np.load('F:/heatwave_chaos/data/ramdivergencedx.npy')

#np.save('F:/heatwave_chaos/data/ramdivergencedy.npy',ramdivergencedy)
ramdivergencedy= np.load('F:/heatwave_chaos/data/ramdivergencedy.npy')


#np.save('F:/heatwave_chaos/data/colddivergencedx.npy', colddivergencedx)
colddivergencedx = np.load('F:/heatwave_chaos/data/colddivergencedx.npy')

#np.save('F:/heatwave_chaos/data/colddivergencedy.npy', colddivergencedy)
colddivergencedy = np.load('F:/heatwave_chaos/data/colddivergencedy.npy')


#.save('F:/heatwave_chaos/data/ramcolddivergencedx.npy', ramcolddivergencedx)
ramcolddivergencedx = np.load('F:/heatwave_chaos/data/ramcolddivergencedx.npy')

#np.save('F:/heatwave_chaos/data/ramcolddivergencedy.npy', ramcolddivergencedy)
ramcolddivergencedy = np.load('F:/heatwave_chaos/data/ramcolddivergencedy.npy')


def setup_axis(ax, x_major, x_minor, y_major, y_minor, xlabel=None, ylabel=None):
    """统一设置坐标轴格式"""
    ax.xaxis.set_major_locator(MultipleLocator(x_major))
    ax.xaxis.set_minor_locator(MultipleLocator(x_minor))
    ax.yaxis.set_major_locator(MultipleLocator(y_major))
    ax.yaxis.set_minor_locator(MultipleLocator(y_minor))
    
    ax.tick_params(which='both', direction='out', width=1.5)
    ax.tick_params(which='major', length=5)
    ax.tick_params(which='minor', length=3)
    
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    
    def y_fmt(x, pos):
        return f"{x*1000:g}"
    ax.yaxis.set_major_formatter(FuncFormatter(y_fmt))
    
    ax.text(-0., 1.1, r'$\times10^{-3}$', 
            transform=ax.transAxes,
            va='top', 
            ha='left',
            fontsize=30)
    
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)

# 数据平滑处理函数
def smooth_data(x, y, num=200):
    """三次样条插值平滑"""
    # 按x值排序
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]
    
    # 生成插值函数
    f = interp1d(x_sorted, y_sorted, kind='cubic', fill_value='extrapolate')
    
    # 创建密集采样点
    x_new = np.linspace(x.min(), x.max(), num)
    y_new = f(x_new)
    
    return x_new, y_new

def add_hist_background(ax, bin_centers, density, color='#DFB6BC', alpha=0.5):
    # 计算分箱边界
    bin_widths = np.diff(bin_centers)
    bin_edges = np.concatenate([
        [bin_centers[0] - bin_widths[0]/2],
        bin_centers[:-1] + bin_widths/2,
        [bin_centers[-1] + bin_widths[-1]/2]
    ])
    
    # 添加边框设置
    ax.bar(bin_centers, density, 
           width=np.diff(bin_edges), 
           align='center', 
           color=color, 
           alpha=alpha,
           edgecolor='k',  # 边框颜色
           linewidth=1,        # 边框线宽
           linestyle='-',        # 边框线型
           zorder=1)

# 创建图形
fig = plt.figure(figsize=(12, 15), dpi=450)
gs = gridspec.GridSpec(2, 1)

# ========== 处理Network I数据 ==========
# 原始数据
raw_x1 = realdivergencedx
raw_y1 = realdivergencedy
raw_x = ramdivergencedx
raw_y = ramdivergencedy

# 应用平滑（新增部分）
smooth_x1, smooth_y1 = smooth_data(raw_x1, raw_y1, num=100)
smooth_x, smooth_y = smooth_data(raw_x, raw_y, num=30)

# Network I 子图
ax1 = fig.add_subplot(gs[0])

setup_axis(ax1, 
          x_major=400, 
          x_minor=200,
          y_major=0.002,
          y_minor=0.001,
          ylabel='PDF')
ax1.set_ylabel('PDF', fontsize=40)


ax1.plot(smooth_x1, smooth_y1, color='#C9352B', lw=4, label='Real', zorder=5, marker='o', markersize=2)
ax1.plot(smooth_x, smooth_y, color='#999A9E', lw=2.5, label='Shuffle')  # 使用平滑数据
#add_hist_background(ax1, realdivergencedx, realdivergencedy)
ax1.fill_between(smooth_x, smooth_y, color='#999A9E', alpha=0.2)  # 同步更新填充
ax1.fill_between(realdivergencedx, realdivergencedy, color='#DFB6BC', alpha=0.4)  # 同步更新填充
ax1.legend(frameon=False, fontsize=14)
#ax1.set_xlim(-1000, 1000)
#ax1.set_ylim(0, 0.009)
xmajorLocator = MultipleLocator(400) #将x主刻度标签设置为20的倍数
xmajorFormatter = FormatStrFormatter('%d') #设置x轴标签文本的格式
ymajorLocator = MultipleLocator(0.002) #将y轴主刻度标签设置为0.5的倍数
#ymajorFormatter = FormatStrFormatter('%d') #设置y轴标签文本的格式
    #设置主刻度标签的位置,标签文本的格式
ax1.xaxis.set_major_locator(xmajorLocator)
ax1.xaxis.set_major_formatter(xmajorFormatter)
ax1.yaxis.set_major_locator(ymajorLocator)
#ax3.yaxis.set_major_formatter(ymajorFormatter)
    
    #修改次刻度
xminorLocator = MultipleLocator(80) #将x轴次刻度标签设置为5的倍数
yminorLocator = MultipleLocator(0.0004) #将此y轴次刻度标签设置为0.1的倍数
    #设置次刻度标签的位置,没有标签文本格式
ax1.xaxis.set_minor_locator(xminorLocator)
ax1.yaxis.set_minor_locator(yminorLocator) 
ax1.tick_params(which='both', width=2.5)
ax1.tick_params(which='major', length=12)  
ax1.tick_params(which='minor',length=7)
ax1.legend(fontsize=40, frameon=False)
ax1.tick_params(axis='both', labelsize=35)
# 标题
#ax3.set_title('a            Network I', fontsize=28, x=0.31, fontweight='bold')
ax1.set_title('a', fontsize=56,x=-0.11, fontweight='bold',pad=10)
# ax3.xaxis.set_major_locator(MultipleLocator(400))  # 设置主刻度间隔
# ax3.xaxis.set_minor_locator(MultipleLocator(200))  # 设置副刻度间隔
# ax3.yaxis.set_major_locator(MultipleLocator(0.002))  # 设置主刻度间隔
# ax3.yaxis.set_minor_locator(MultipleLocator(0.001))  # 设置副刻度间隔

# # 设置刻度和标签
# ax3.tick_params(axis='both', which='major',  length=12,labelsize=35, color='black')
# ax3.tick_params(axis='both', which='minor', length=10, color='black')

# # 隐藏副刻度标签
# ax3.tick_params(axis='x', which='minor', labelbottom=False)
# ax3.tick_params(axis='y', which='minor', labelleft=False)
# 去掉不需要的边框（半框效果）
ax1.text(0.35, 1, r'Network I', fontsize=40, transform=ax1.transAxes)

# ========== 处理Network II数据 ==========
# 原始数据
raw_x3 = colddivergencedx
raw_y3 = colddivergencedy
raw_x2 = ramcolddivergencedx
raw_y2 = ramcolddivergencedy

# 应用平滑（新增部分）
smooth_x3, smooth_y3 = smooth_data(raw_x3, raw_y3, num=100)
smooth_x2, smooth_y2 = smooth_data(raw_x2, raw_y2, num=30)

# Network II 子图
ax2 = fig.add_subplot(gs[1])
setup_axis(ax2, 
          x_major=400, 
          x_minor=200,
          y_major=0.002,
          y_minor=0.001,
          xlabel='Divergence',
          ylabel='PDF')

ax2.plot(smooth_x3, smooth_y3, color='#C9352B', lw=4, zorder=5, marker='o', markersize=2)
ax2.plot(smooth_x2, smooth_y2, color='#999A9E', lw=2.5)  # 使用平滑数据
ax2.fill_between(smooth_x2, smooth_y2, color='#999A9E', alpha=0.3)  # 同步更新填充
ax2.fill_between(colddivergencedx, colddivergencedy, color='#DFB6BC', alpha=0.4)  # 同步更新填充
#add_hist_background(ax2, colddivergencedx, colddivergencedy)
#ax2.set_xlim(-1000, 1000)
#ax2.set_ylim(0, 0.009)
xmajorLocator = MultipleLocator(400) #将x主刻度标签设置为20的倍数
xmajorFormatter = FormatStrFormatter('%d') #设置x轴标签文本的格式
ymajorLocator = MultipleLocator(0.002) #将y轴主刻度标签设置为0.5的倍数
#ymajorFormatter = FormatStrFormatter('%d') #设置y轴标签文本的格式
    #设置主刻度标签的位置,标签文本的格式
ax2.xaxis.set_major_locator(xmajorLocator)
ax2.xaxis.set_major_formatter(xmajorFormatter)
ax2.yaxis.set_major_locator(ymajorLocator)
#ax3.yaxis.set_major_formatter(ymajorFormatter)
ax2.set_ylabel('PDF', fontsize=40)
ax2.set_xlabel('Degree', fontsize=40)
    #修改次刻度
xminorLocator = MultipleLocator(80) #将x轴次刻度标签设置为5的倍数
yminorLocator = MultipleLocator(0.0004) #将此y轴次刻度标签设置为0.1的倍数
    #设置次刻度标签的位置,没有标签文本格式
ax2.xaxis.set_minor_locator(xminorLocator)
ax2.yaxis.set_minor_locator(yminorLocator) 
ax2.tick_params(which='both', width=2.5)
ax2.tick_params(which='major', length=12)  
ax2.tick_params(which='minor',length=7)

# ax3.xaxis.set_major_locator(MultipleLocator(400))  # 设置主刻度间隔
# ax3.xaxis.set_minor_locator(MultipleLocator(200))  # 设置副刻度间隔
# ax3.yaxis.set_major_locator(MultipleLocator(0.002))  # 设置主刻度间隔
# ax3.yaxis.set_minor_locator(MultipleLocator(0.001))  # 设置副刻度间隔

# # 设置刻度和标签
# ax3.tick_params(axis='both', which='major',  length=12,labelsize=35, color='black')
# ax3.tick_params(axis='both', which='minor', length=10, color='black')

# # 隐藏副刻度标签
# ax3.tick_params(axis='x', which='minor', labelbottom=False)
# ax3.tick_params(axis='y', which='minor', labelleft=False)
# 去掉不需要的边框（半框效果）

ax2.set_title(r'Network II', fontsize=40,x=0.48, y=1)

plt.tight_layout()
plt.subplots_adjust(hspace=0.3)
plt.savefig('F:/heatwave_chaos/Fig2_a.svg', format='svg', bbox_inches='tight')
plt.show()


