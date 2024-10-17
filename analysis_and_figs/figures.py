"""
Author:Jaelyn Bos
"""
'''
Import packages
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy
from cartopy import crs as ccrs
import seaborn as sns

'''
Load data
'''

'Path to saved logger data'
data_path = '/projects/f_mlp195/bos_microclimate/'

'Path to save figures'
fig_path = '/home/jtb188/manuscript_figs/'

'Import training data'
train_dat = pd.read_csv(data_path + 'training_weeklydiffs_sst.csv')

'Calculate natural log of distance to shore'
train_dat['LOGDIST'] = np.log1p(train_dat['SHOREDIST'])
train_dat = train_dat.reset_index(drop=True)

'Import testing data'
test_dat = pd.read_csv(data_path + 'testing_weeklydiff_sst.csv')

'Calculate natural log of distance to shore'
test_dat['LOGDIST'] = np.log1p(test_dat['SHOREDIST'])
test_dat = test_dat.reset_index(drop=True)

'Combine training and testing data and calculate differences between in situ and remote temperature metrics'
all_dat = pd.concat([train_dat,test_dat])
all_dat['SST_DIFF'] = all_dat['TEMP_C'] - all_dat['SST']
all_dat = all_dat.reset_index(drop=True)
all_dat['Q99_DIFF'] = all_dat['Q99'] - all_dat['SST_MAX']

'''
Figure 1A
'''
fig= plt.figure(figsize=(11,8.5))
proj = ccrs.PlateCarree(central_longitude =180)
ax = plt.subplot(1,1,1,projection=proj)
ax.set_extent((-90,130,-40,40),crs=proj)
ax.coastlines()
ax.stock_img()
ax.scatter(test_dat['LONGITUDE'],test_dat['LATITUDE'],c='darkorange',marker='2',transform=ccrs.Geodetic())
ax.scatter(train_dat['LONGITUDE'],train_dat['LATITUDE'],c='darkorange',marker='2',transform=ccrs.Geodetic())
gl = ax.gridlines(draw_labels=True)
ax.text(0.05, 0.95, 'A)', transform=ax.transAxes, fontsize=28, fontweight='bold', va='top')
fig.savefig(fig_path + 'logger_map.png',transparent=False,bbox_inches='tight',dpi=2400)

'''
Figure 1B
'''
all_dat['REGION'] =all_dat['REGION'].astype('category')

yearsum = pd.DataFrame()

for i in range(2003, 2020):
    yri = all_dat[(pd.to_datetime(all_dat['START_TIME']) < np.datetime64(str(i+1) + '-01-01T00:00')) &  (pd.to_datetime(all_dat['END_TIME']) > np.datetime64(str(i) +'-01-01T00:00'))]
    yearsum.loc[:,i] =yri['REGION'].value_counts().sort_index()
mask= yearsum != 0
annotation = yearsum.astype(str).where(mask,' ')
yearsum.index = ['Australia','Caribbean  (NOAA)','Dongsha','Hawaii (NOAA and USGS)','Kiritimati','New Caledonia','Pacific Islands (NOAA)','Palau','Panama','Phoenix Islands']

fig, ax = plt.subplots(figsize=(21,8.5))
heat = sns.heatmap(yearsum,cmap='mako_r',annot=annotation,fmt="",annot_kws={"fontsize":20},cbar_kws={'label':'No. of recording loggers by region and year'})
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)

cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=21)

plt.xlabel('Year', fontsize=21)
plt.ylabel('Region', fontsize=21)
ax.text(0.05, 0.95, 'B)', transform=ax.transAxes, fontsize=48, fontweight='bold', va='top')

fig.savefig( fig_path +'logger_heatmap.png',transparent=False,bbox_inches='tight',dpi=1200)

'''
Figure 2
'''
# Create a 1x2 subplot figure
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# --- Panel A: Scatterplot (Logger MMM vs Remote MMM) ---
custom_bins = [0, 100, 1000, 10000, 100000, 250000]  
bin_labels = [f'{int(b1):,} - {int(b2):,} m' for b1, b2 in zip(custom_bins[:-1], custom_bins[1:])]

all_dat['DISTANCE_BINS'] = pd.cut(all_dat['SHOREDIST'], bins=custom_bins, labels=bin_labels)

# Scatterplot
scat = sns.scatterplot(x='TEMP_C', y='SST', hue='DISTANCE_BINS', data=all_dat, palette='viridis', ax=axes[0],s=90)
axes[0].legend(title="Distance to shore (m)", fontsize=14, loc='lower right',title_fontsize=16)
axes[0].grid(False)

# Line: SST == Logger temp
axes[0].plot([24, 31], [24, 31], linestyle='--', lw=1, color='black', label='SST == logger temp')

# Labels
axes[0].set_xlabel('Logger MMM °C',fontsize=18)
axes[0].set_ylabel('Remote MMM °C',fontsize=18)
axes[0].tick_params(axis='x', labelsize=18)
axes[0].tick_params(axis='y', labelsize=18)

axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)

# Add panel label A
axes[0].text(0.05, 0.95, 'A)', transform=axes[0].transAxes, fontsize=32, fontweight='bold', va='top')

# --- Panel B: Histogram (In situ - Remote MMM) ---
axes[1].hist(all_dat['SST_DIFF'], bins=20, color='skyblue', edgecolor='black')
axes[1].set_xlabel('In situ - Remote MMM °C',fontsize=18)
axes[1].set_ylabel('Frequency',fontsize=18)
axes[1].grid(False)
axes[1].tick_params(axis='x', labelsize=18)
axes[1].tick_params(axis='y', labelsize=18)

axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)

# Add panel label B
axes[1].text(0.05, 0.95, 'B)', transform=axes[1].transAxes, fontsize=32, fontweight='bold', va='top')

# Adjust layout and save
plt.tight_layout()
plt.savefig(fig_path + 'Figure2.png', transparent=False, bbox_inches='tight', dpi=1200)

'''
Figure 3
'''
#Define colors and order for geomorphic classes
rclass_ord = ['Terrestrial Reef Flat','Shallow Lagoon','Deep Lagoon','Back Reef Slope','Inner Reef Flat','Outer Reef Flat','Reef Crest','Sheltered Reef Slope','Reef Slope','Plateau']
rclass_col = {'Terrestrial Reef Flat':'pink','Shallow Lagoon':'dodgerblue','Deep Lagoon':'mediumblue','Back Reef Slope':'c','Inner Reef Flat':'orchid','Outer Reef Flat':'mediumpurple','Reef Crest':'purple','Sheltered Reef Slope':'darkgreen','Reef Slope':'seagreen','Plateau':'darkorange'}

# Create a 1x2 subplot figure
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# --- Panel A: Scatterplot (Depth vs In situ - Remote MMM by Temp) ---
scat = sns.scatterplot(x='DEPTH', y='SST_DIFF', hue='TEMP_C', data=all_dat, palette='viridis', ax=axes[0],s=70)
axes[0].legend(title="In situ MMM °C", fontsize=18, title_fontsize=18)
axes[0].set_xlabel('Depth (meters)', fontsize=18)
axes[0].set_ylabel('In situ - Remote MMM °C', fontsize=18)
axes[0].tick_params(axis='x', labelsize=18)
axes[0].grid(False)
axes[0].tick_params(axis='y', labelsize=18)

axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)

# Add panel label A
axes[0].text(0.05, 0.95, 'A)', transform=axes[0].transAxes, fontsize=32, fontweight='bold', va='top')

# --- Panel B: Boxplot and Stripplot (Geomorphic class vs In situ - Remote MMM) ---
boxes = sns.boxplot(y='SST_DIFF', x='RCLASS', data=all_dat, order=rclass_ord, dodge=False, boxprops={'facecolor': 'None'}, showfliers=False, showcaps=False, whiskerprops={'linewidth': 0}, ax=axes[1])
sns.stripplot(y='SST_DIFF', x='RCLASS', data=all_dat, order=rclass_ord, hue='RCLASS', palette=rclass_col, dodge=False, zorder=0.5, ax=axes[1],s=9)

# Remove legend from stripplot
boxes.get_legend().remove()

# Labels and tick adjustments
axes[1].set_xlabel('Geomorphic class', fontsize=18)
axes[1].set_ylabel('In situ - Remote MMM °C', fontsize=18)
axes[1].tick_params(axis='x', labelrotation=80,labelsize=18)
axes[1].tick_params(axis='y', labelsize=18)

axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)

# Add panel label B
axes[1].text(0.05, 0.95, 'B)', transform=axes[1].transAxes, fontsize=32, fontweight='bold', va='top')

# Adjust layout and save
plt.tight_layout()
plt.savefig(fig_path + 'Figure3.png', transparent=False, bbox_inches='tight', dpi=1200)

'''
Figure 4
'''
# Create a 1x2 subplot figure
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# --- Panel A: Scatterplot (Logger peak temp vs Remote peak temp) ---

# Scatterplot
scat = sns.scatterplot(x='Q99', y='SST_MAX', hue='DISTANCE_BINS', data=all_dat, palette='viridis', ax=axes[0],s=90)
axes[0].legend(title="Distance to shore (m)", fontsize=14,title_fontsize=16, loc='lower right')
axes[0].grid(False)

# Line: SST == Logger temp
axes[0].plot([25, 33], [25, 33], linestyle='--', lw=1, color='black', label='SST == logger temp')

# Labels
axes[0].set_xlabel('Logger Peak Temp °C',fontsize=18)
axes[0].set_ylabel('Remote Peak Temp °C',fontsize=18)
axes[0].tick_params(axis='x', labelsize=18)
axes[0].tick_params(axis='y', labelsize=18)

axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)

# Add panel label A
axes[0].text(0.05, 0.95, 'A)', transform=axes[0].transAxes, fontsize=32, fontweight='bold', va='top')

# --- Panel B: Histogram (In situ - Remote MMM) ---
axes[1].hist(all_dat['Q99_DIFF'], bins=20, color='skyblue', edgecolor='black')
axes[1].set_xlabel('In situ - Remote peak temp °C',fontsize=18)
axes[1].set_ylabel('Frequency',fontsize=18)
axes[1].grid(False)
axes[1].tick_params(axis='x', labelsize=18)
axes[1].tick_params(axis='y', labelsize=18)

axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)

# Add panel label B
axes[1].text(0.05, 0.95, 'B)', transform=axes[1].transAxes, fontsize=32, fontweight='bold', va='top')

# Adjust layout and save
plt.tight_layout()
plt.savefig(fig_path + 'Figure4.png', transparent=False, bbox_inches='tight', dpi=1200)

'''
Figure 5
'''
# Create a 1x2 subplot figure
fig, axes = plt.subplots(1, 2, figsize=(14, 8))

# --- Panel A: Scatterplot (Depth vs In situ - Remote MMM by Temp) ---
scat = sns.scatterplot(x='DEPTH', y='Q99_DIFF', hue='Q99', data=all_dat, palette='viridis', ax=axes[0],s=70)
axes[0].legend(title="In situ peak temp °C", fontsize=18, title_fontsize=18)
axes[0].set_xlabel('Depth (meters)', fontsize=18)
axes[0].set_ylabel('In situ - Remote peak temp  °C', fontsize=18)
axes[0].grid(False)
axes[0].tick_params(axis='x', labelsize=18)
axes[0].tick_params(axis='y', labelsize=18)

axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)

# Add panel label A
axes[0].text(0.05, 0.95, 'A)', transform=axes[0].transAxes, fontsize=32, fontweight='bold', va='top')

# --- Panel B: Boxplot and Stripplot (Geomorphic class vs In situ - Remote MMM) ---
boxes = sns.boxplot(y='Q99_DIFF', x='RCLASS', data=all_dat, order=rclass_ord, dodge=False, boxprops={'facecolor': 'None'}, showfliers=False, showcaps=False, whiskerprops={'linewidth': 0}, ax=axes[1])
sns.stripplot(y='Q99_DIFF', x='RCLASS', data=all_dat, order=rclass_ord, hue='RCLASS', palette=rclass_col, dodge=False, zorder=0.5, ax=axes[1],s=9)

# Remove legend from stripplot
boxes.get_legend().remove()

# Labels and tick adjustments
axes[1].set_xlabel('Geomorphic class', fontsize=18)
axes[1].set_ylabel('In situ - Remote peak temp °C', fontsize=18)
axes[1].tick_params(axis='x', labelrotation=80,labelsize=18)
axes[1].tick_params(axis='y', labelsize=18)

axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)

# Add panel label B
axes[1].text(0.05, 0.95, 'B)', transform=axes[1].transAxes, fontsize=32, fontweight='bold', va='top')

# Adjust layout and save
plt.tight_layout()
plt.savefig(fig_path + 'Figure5.png', transparent=False, bbox_inches='tight', dpi=1200)

'''
Supplemental figure 1
'''
all_dat['ABS_LAT'] = all_dat['LATITUDE'].abs()

fig, axes = plt.subplots(1, 2, figsize=(14, 8))
scat = sns.scatterplot(x = 'ABS_LAT', y = 'SST_DIFF', hue = 'TEMP_C', data= all_dat,palette='viridis', ax=axes[0],s=90)
axes[0].legend(title = "In situ MMM °C",loc = 'lower left',fontsize=18,title_fontsize=18)

axes[0].set_xlabel('Absolute latitude (degrees)', fontsize=18)
axes[0].set_ylabel('In situ MMM - remote MMM °C', fontsize=18)
axes[0].tick_params(axis='x', labelsize=18)
axes[0].tick_params(axis='y', labelsize=18)

axes[0].grid(False)

axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)

# Add panel label A
axes[0].text(0.05, 0.95, 'A)', transform=axes[0].transAxes, fontsize=32, fontweight='bold', va='top')

scat1 = sns.scatterplot(x = 'SHOREDIST', y = 'SST_DIFF', hue = 'TEMP_C', data= all_dat,palette='viridis', ax=axes[1],s=90)
axes[1].legend(title = "In situ MMM °C",fontsize=18,title_fontsize=18)

axes[1].set_xlabel('Distance to shore (meters)', fontsize=18)
axes[1].set_ylabel('In situ MMM - remote MMM °C', fontsize=18)
axes[1].tick_params(axis='x', labelsize=18,labelrotation=80)
axes[1].tick_params(axis='y', labelsize=18)

axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)

# Add panel label B
axes[1].text(0.05, 0.95, 'B)', transform=axes[1].transAxes, fontsize=32, fontweight='bold', va='top')

# Adjust layout and save
plt.tight_layout()
plt.savefig( fig_path + 'Supplemental1.png',transparent=False,bbox_inches='tight',dpi=1200)

'''
Supplemental figure 2
'''
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
scat = sns.scatterplot(x = 'ABS_LAT', y = 'Q99_DIFF', hue = 'Q99', data= all_dat,palette='viridis', ax=axes[0],s=90)
axes[0].legend(title = "In situ peak temp °C",loc = 'lower left',fontsize=14,title_fontsize=14)

axes[0].set_xlabel('Absolute latitude (degrees)', fontsize=18)
axes[0].set_ylabel('In situ peak temp - remote peak temp °C', fontsize=18)
axes[0].tick_params(axis='x', labelsize=18)
axes[0].tick_params(axis='y', labelsize=18)

axes[0].grid(False)

axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)

# Add panel label A
axes[0].text(0.05, 0.95, 'A)', transform=axes[0].transAxes, fontsize=32, fontweight='bold', va='top')

scat1 = sns.scatterplot(x = 'SHOREDIST', y = 'Q99_DIFF', hue = 'Q99', data= all_dat,palette='viridis', ax=axes[1],s=90)
axes[1].legend(title = "In situ peak temp °C",fontsize=14,title_fontsize=14)

axes[1].set_xlabel('Distance to shore (meters)', fontsize=18)
axes[1].set_ylabel('In situ peak temp - remote peak temp °C', fontsize=18)
axes[1].tick_params(axis='x', labelsize=18)
axes[1].tick_params(axis='y', labelsize=18)

axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)

# Add panel label B
axes[1].text(0.05, 0.95, 'B)', transform=axes[1].transAxes, fontsize=32, fontweight='bold', va='top')

# Adjust layout and save
plt.tight_layout()
plt.savefig( fig_path + 'Supplemental2.png',transparent=False,bbox_inches='tight',dpi=1200)

'''
Supplemental figure 3
'''
boxes = sns.boxplot(y='DAILY_DIFF',x='RCLASS',data=all_dat,order=rclass_ord,dodge=False,boxprops = {'facecolor':'None'},showfliers = False, showcaps = False,whiskerprops = {'linewidth': 0})
sns.stripplot(y='DAILY_DIFF',x='RCLASS',data=all_dat,order=rclass_ord,hue='RCLASS',palette=rclass_col,dodge=False,zorder = 0.5,s=9)
boxes.set_ylabel('Mean daily temperature range °C',fontsize=12)
boxes.set_xlabel('Geomorphic class',fontsize=12)
boxes.get_legend().remove()

boxes.tick_params(axis='x',labelrotation=80, labelsize=12)
boxes.tick_params(axis='y',labelsize=12)

boxes.spines['top'].set_visible(False)
boxes.spines['right'].set_visible(False)

boxes.figure.savefig(fig_path + 'Supplemental3.png',transparent=False,bbox_inches='tight',dpi=1200)
