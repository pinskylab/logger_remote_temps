"""
Script for processing in situ logger temperature data from the Palau Archipelago
Author: Jaelyn Bos
Data available at XXX/XX
2024
"""
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from scipy.io import loadmat

'Function to convert latitude and longitude coordinates to geopandas Point object'
def make_point(df):
    df['COORDS'] = df[['LONGITUDE','LATITUDE']].values.tolist()
    df['COORDS'] = df['COORDS'].apply(Point)
    df_gpd = gpd.GeoDataFrame(df,geometry='COORDS')
    return(df_gpd)

'Function to combine logger point data with geomorphic class data from the Allen Coral Atlas'
def agg_and_merge(region,geo):
    di = {'LATITUDE':['first'],'LONGITUDE':['first']}
    reg_agg = region.groupby('ID').agg(di).reset_index()
    reg_agg.columns = ['ID','LATITUDE','LONGITUDE']
    reg_agg = make_point(reg_agg)
    reg_agg.crs = geo.crs
    reg = pd.DataFrame(gpd.sjoin(reg_agg,geo,how='inner',op='within'))
    region = pd.merge(reg,region,on='ID') 
    region['RCLASS'] = region['class']
    region['LONGITUDE'] = region['LONGITUDE_x']
    region['LATITUDE'] = region['LATITUDE_x']
    region = region.drop(columns=['LONGITUDE_x','LATITUDE_x','LONGITUDE_y','LATITUDE_y','class','index_right'])
    return(region)

'Function to downsample temperature data to every 30 minutes'
def thirtyminsample(region,region_points):
    region = region.set_index('DATETIME')
    x = [region[region['ID'] == region_points[i]] for i in np.arange(0,len(region_points),1)]
    y = [x[i].resample('30T').first() for i in np.arange(0,len(x),1)]
    y = pd.concat(y)
    y = y.dropna(axis=0)
    return(y)

'Function to downsample data to one point every calendar day'
def onedaysample (region,region_points):
    region = region.set_index('DATETIME')
    x = [region[region['ID'] == region_points[i]] for i in np.arange(0,len(region_points),1)]
    y = [x[i].resample('1D').first() for i in np.arange(0,len(x),1)]
    y = pd.concat(y)
    y = y.dropna(axis=0)
    return(y)

'Function to return downsampled 30 min temperature data'
def downsampler (region):
    points_u = region['ID'].unique()
    region_ds = thirtyminsample(region,points_u)
    region_ds['DATETIME'] = region_ds.index
    region_ds['DATE'] = region_ds['DATETIME'].dt.date
    region_ds['YRMONTHPOINT'] = region_ds['DATETIME'].dt.year.astype('str') + '/' + region_ds['DATETIME'].dt.month.astype('str') +'_' + region_ds['ID'].astype('str')
    return(region_ds)

'Function to return one point per day per logger'
def downsampler2 (region):
    points_u = region['ID'].unique()
    region_ds = onedaysample(region,points_u)
    region_ds['DATETIME'] = region_ds.index
    region_ds['DATE'] = region_ds['DATETIME'].dt.date
    region_ds['YRMONTHPOINT'] = region_ds['DATETIME'].dt.year.astype('str') + '/' + region_ds['DATETIME'].dt.month.astype('str') +'_' + region_ds['ID'].astype('str')
    return(region_ds)

'Function to select a specific logger point'
def sitesel(region,point):
    return(region.loc[region['ID']==point].reset_index(drop=True))
  
'Function to select the highest temperature'
def maxsel(site):
   return(site.loc[(site['TEMP_C']==site['TEMP_C'].max()).values])

'Function to select max monthly mean from monthly means for each logger in a region'
def mmm_select(region,regional_points):
    return(pd.concat([maxsel(sitesel(region,regional_points[i])) for i in np.arange(0,len(regional_points),1)]).reset_index())

'Function to calculate 99th quantile for a series'
def quantile99(ser):
    return(ser.quantile(q=0.99))

'Function to return max monthly mean for each logger in region'    
def prep_data(region_ds):
        points_u= pd.Series(region_ds['ID'].unique())
        a = {'TEMP_C': ['mean'],'LONGITUDE':['median'],'LATITUDE':['median'],'RCLASS':['first'],'COORDS':['first'],'DEPTH':['median']}
        mms = region_ds.groupby(['MONTH','ID']).agg(a).reset_index() 
        MMMs = mmm_select(mms,points_u)
        return(MMMs)

'Function to return 99th quantile of hottest month for each logger in region'
def prep_q99(region_ds):
        points_u= pd.Series(region_ds['ID'].unique())
        q99 = region_ds.groupby(['MONTH','ID']).agg(np.quantile, q=0.99).reset_index() 
        Q99 = mmm_select(q99,points_u)
        Q99 = Q99.drop(columns = ['counts'])
        return(Q99)

'Function to return full datasets with start and end time for each logger'
def prep_data_more(region_ds,region_mmms):    
        starts = region_ds.groupby(['ID'])['DATETIME'].min().to_frame().reset_index()
        starts.columns = ['ID','START_TIME']
        ends = region_ds.groupby(['ID'])['DATETIME'].max().to_frame().reset_index()
        ends.columns = ['ID','END_TIME']
        pts = region_mmms.copy()
        pts = pd.merge(pts,starts)
        pts = pd.merge(pts,ends)
        pts = pts.reset_index(drop=True)
        return(pts)

'Columns to keep from original dataset'
keepcols = ['LATITUDE','LONGITUDE','ID','DATETIME','TEMP_C']

'Filepath to folder with dataset in matlab format'
path = '/projects/f_mlp195/bos_microclimate/loggers_testing/'

'Filepath to folder with geomorphic data from Allen Coral Atlas'
geom_path = '/projects/f_mlp195/bos_microclimate/geomorphology/'

'Filepath to folder to save processed data'
out_path = '/home/jtb188/'

'''
Points from Palau
'''
palau_mat = loadmat(path+ 'coral_temps_palau.mat')

palau = pd.DataFrame()
palau['LATITUDE'] = palau_mat['Latitude']
palau['LONGITUDE'] = palau_mat['Longitude']
palau['DATETIME'] = pd.to_datetime(palau_mat['Collection_date_time'],format='%-m/%-d/%y %-H:%M',infer_datetime_format=True)
palau['TEMP_C'] = palau_mat['Temperature']
palau['ID'] = palau_mat['Colony_tag']

palau = palau[palau['TEMP_C'] != 'nd']
palau = palau[palau['LATITUDE'] != 'nd']
palau['ID'] = palau['ID'].apply(str)
palau['TEMP_C'] = palau['TEMP_C'].apply(float)
palau['LATITUDE'] = palau['LATITUDE'].apply(float)
palau['LONGITUDE'] = palau['LONGITUDE'].apply(float)
palau = palau[keepcols]

'''
Combine with geomorphology from western micronesia
'''
palau_geo = gpd.read_file(geom_path+'Western-Micronesia.gpkg')
palau = agg_and_merge(palau,palau_geo)

'Downsample logger data to every 30 minutes'
palau = downsampler(palau)

'''
Remove data from after 2019 to align with SST data
'''
palau = palau[palau['DATETIME'] < pd.to_datetime('2020-01-01 00:00:00')]
palau = palau.reset_index(drop=True)

palau['MONTH'] =palau['DATETIME'].dt.month
d_agg = palau.groupby(['ID','MONTH']).first().reset_index()

'''
Remove days with fewer than 24 points
'''
palau['DAYPT'] =  palau['DATETIME'].dt.date.astype('str') +'_' + palau['ID'].astype('str')
dayagg= palau.groupby(['DAYPT'])['DAYPT'].count()
daykeep = pd.DataFrame()
daykeep['DAYCOUNT'] = dayagg.values
daykeep['DAYPT'] =dayagg.index

palau = pd.merge(palau,daykeep,how='left',on='DAYPT')
palau = palau[palau['DAYCOUNT'] > 23]
palau = palau.reset_index(drop=True)
palau = palau.drop(columns = ['DAYCOUNT','DAYPT'])

'''
Remove months with <15 days
'''
palau['MONTHPT'] =  palau['DATETIME'].dt.month.astype('str') + '_' + palau['DATETIME'].dt.year.astype('str') +'_' + palau['ID'].astype('str')
oneaday = downsampler2(palau)
monthagg= oneaday.groupby(['MONTHPT'])['MONTHPT'].count()
monthkeep = pd.DataFrame()
monthkeep['MONTHCOUNT'] = monthagg.values
monthkeep['MONTHPT'] =monthagg.index

palau = pd.merge(palau,monthkeep,how='left',on='MONTHPT')
palau = palau[palau['MONTHCOUNT'] > 15]
palau = palau.reset_index(drop=True)
palau = palau.drop(columns = ['MONTHCOUNT','MONTHPT'])

'''
Remove points with <12 months
'''
monthn= pd.DataFrame(d_agg['ID'].value_counts())
monthn['counts'] =monthn.values
monthn['ID'] =monthn.index
monthn = monthn.reset_index()

palau = pd.merge(palau,monthn,how='left',on='ID')
palau = palau[palau['counts'] > 11]
palau = palau.reset_index(drop=True)

'''
Calculate mean daily diff
'''
palau2 = palau.copy()
palau2['dmin'] = palau2.groupby(['ID','DATE']).TEMP_C.transform('min')
palau2['dmax'] = palau2.groupby(['ID','DATE']).TEMP_C.transform('max')
palau2['DAILY_DIFF'] = palau2['dmax'] - palau2['dmin']
palau2 = palau2.drop(columns=['dmin','dmax'])
a = {'DAILY_DIFF': ['mean']}
temps_a = palau2.groupby(['ID']).agg(a).reset_index() 

'''
Calculate MMMs
'''
palau['DEPTH'] = np.nan
palau_mmms = prep_data(palau)
palau_mmms.columns = ['index','MONTH','ID','TEMP_C','LONGITUDE','LATITUDE','RCLASS','COORDS','DEPTH']
palau_mmms = palau_mmms.drop(columns=['index'])

'''
Calculate q99
'''
palau_q99 = prep_q99(palau)
palau_q99.columns = ['index','MONTH', 'ID', 'Q99', 'LONGITUDE', 'LATITUDE', 'DATETIME', 'DATE','DEPTH']
palau_q99 = palau_q99.drop(columns=['index','MONTH','LONGITUDE','LATITUDE','DATETIME','DATE','DEPTH'])
palau_q99 = palau_q99.drop_duplicates()

'Combine MMM, peak temps, daily diffs'
palau = prep_data_more(palau,palau_mmms)
palau = pd.merge(palau,temps_a,on='ID')
palau = pd.merge(palau,palau_q99,on='ID')

'Rename columns'
palau.columns = ['MONTH','ID','TEMP_C','LONGITUDE','LATITUDE','RCLASS','COORDS','DEPTH','START_TIME','END_TIME','DAILY_DIFF','Q99']

'Reorder columns'
palau = palau[['MONTH','ID','TEMP_C','LONGITUDE','LATITUDE','RCLASS','COORDS','START_TIME','END_TIME','Q99','DAILY_DIFF','DEPTH']]

'Export data as csv'
outfile = open(out_path + 'palau.csv','wb')
palau.to_csv(outfile)
outfile.close()