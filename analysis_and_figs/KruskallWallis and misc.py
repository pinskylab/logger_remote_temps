"""
Author:Jaelyn Bos
"""
'''
Import packages
'''
import pandas as pd
import numpy as np
import scipy.stats as stats
from geopy.distance import great_circle

'''
Load data
'''
'Path to saved logger data'
data_path = '/projects/f_mlp195/bos_microclimate/'

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
all_dat['ABS_LAT'] = all_dat['LATITUDE'].abs()

'''
Summary statistics
'''
'Total logged hours'
starts = all_dat['START_TIME']
ends = all_dat['END_TIME']
starts=pd.to_datetime(starts)
ends = pd.to_datetime(ends)

times = ends - starts
hours = times /np.timedelta64(1,'h')

'Minimum distance between loggers'
pts = all_dat[['LATITUDE','LONGITUDE']]
great_circle(pts.iloc[0,:],pts.iloc[1,:]).meters

dist_mat = pd.DataFrame(index=all_dat['ID'])

for j in range(0,len(pts)):
     d = lambda k : great_circle(pts.iloc[k,:],pts.iloc[j,:]).meters
     dists = []
     for i in range(0,len(pts)):
             dists.append(d(i))
     dists = pd.Series(dists)
     dists.name = all_dat['ID'][j]
     dists.index = all_dat['ID']
     print(dists[0])
     dist_mat = dist_mat.join(dists)

dist_mat.replace(0,np.nan, inplace=True)

m = dist_mat.min().idxmin()
m1 = dist_mat[m].idxmin()

all_dat[all_dat['ID'] == m]['REGION']
all_dat[all_dat['ID'] == m1]['REGION']

'Mean DTR by geomorphic class'
all_dat.groupby(['RCLASS'])['DAILY_DIFF'].mean()

'''
Nonparametric Kruskall-Wallis tests
'''
'Remote vs in situ MMMs by geomorphic class'
gc = all_dat['RCLASS'].unique()
geos = [all_dat[all_dat['RCLASS']==gc[i]]['SST_DIFF'] for i in range(len(gc))]
stats.kruskal(*geos)

'Remote vs in situ peak temps by geomorphic class'
gc = all_dat['RCLASS'].unique()
geos = [all_dat[all_dat['RCLASS']==gc[i]]['Q99_DIFF'] for i in range(len(gc))]
stats.kruskal(*geos)

'Daily temperature range by geomorphic class'
gc = all_dat['RCLASS'].unique()
geos = [all_dat[all_dat['RCLASS']==gc[i]]['DAILY_DIFF'] for i in range(len(gc))]
stats.kruskal(*geos)

'''
Correlations with standard error
'''
'Remote and in situ MMM correlation'
cor = np.corrcoef(all_dat['TEMP_C'],all_dat['SST'])[1][0]
numerator = 1- cor**2
denominator = all_dat.shape[0] - 2
se = (numerator/denominator)**0.5

'Remote and in situ MMM offset v depth correlation'
cor = np.corrcoef(all_dat['DEPTH'],all_dat['SST_DIFF'])[1][0]
numerator = 1- cor**2
denominator = all_dat.shape[0] - 2
se = (numerator/denominator)**0.5

'Remote and in situ MMM offset v absolute latitude correlation'
cor = np.corrcoef(all_dat['ABSLAT'],all_dat['SST_DIFF'])[1][0]
numerator = 1- cor**2
denominator = all_dat.shape[0] - 2
se = (numerator/denominator)**0.5

'Remote and in situ MMM offset v distance to shore correlation'
cor = np.corrcoef(all_dat['SHOREDIST'],all_dat['SST_DIFF'])[1][0]
numerator = 1- cor**2
denominator = all_dat.shape[0] - 2
se = (numerator/denominator)**0.5

'Remote and in situ peak temps correlation'
cor = np.corrcoef(all_dat['Q99'],all_dat['SST_MAX'])[1][0]
numerator = 1- cor**2
denominator = all_dat.shape[0] - 2
se = (numerator/denominator)**0.5

'Remote and in situ peak annual temps offset v absolute latitude correlation'
cor = np.corrcoef(all_dat['ABS_LAT'],all_dat['Q99_DIFF'])[1][0]
numerator = 1- cor**2
denominator = all_dat.shape[0] - 2
se = (numerator/denominator)**0.5

'Remote and in situ  peak annual temps offset v distance to shore correlation'
cor = np.corrcoef(all_dat['SHOREDIST'],all_dat['Q99_DIFF'])[1][0]
numerator = 1- cor**2
denominator = all_dat.shape[0] - 2
se = (numerator/denominator)**0.5

'Remote and in situ peak temps below 31C'
low = all_dat.loc[all_dat['Q99']<31,]
low.corr()['SST_MAX']

'Remote and in situ peak temps above 31C'
high = all_dat.loc[all_dat['Q99']>=31,]
high.corr()['SST_MAX']

'Remote and in situ MMM offset vs depth correlation and daily temperature range vs depth correlation'
all_dat.corr()['DEPTH']
