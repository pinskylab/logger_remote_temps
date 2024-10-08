"""
Script for processing in situ logger temperature data from
ReefTEMPS - Reseau d'observation des eaux cotieres du pacifique insulaire
Author: Jaelyn Bos
Data available at XXX/XX
2024
"""
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from scipy.io import loadmat
import glob
import xarray as xr

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

'Filepath to folder with datasets in .nc format'
path = '/projects/f_mlp195/bos_microclimate/loggers_testing/Newcal_loggers1/reeftemps_nc_data_20220317/'

'Filepath to folder with geomorphic data from Allen Coral Atlas'
geom_path = '/projects/f_mlp195/bos_microclimate/geomorphology/'

'Filepath to folder to save processed data'
out_path = '/home/jtb188/'

'Read in all reeftemps data in netcdf form'
reeftemps_all = [xr.open_dataset(f) for f in glob.glob(path+ "*.nc")]

'Create Pandas dataframe from each set using only wanted columns'
reeftemps_pd = []
for i in range(0,len(reeftemps_all)):
    dat = pd.DataFrame()
    dat['TEMP_C'] = reeftemps_all[i]['TEMP'].to_series()
    dat['LATITUDE'] = reeftemps_all[i]['LATITUDE'].values[0]
    dat['LONGITUDE'] = reeftemps_all[i]['LONGITUDE'].values[0]
    dat['ID'] = reeftemps_all[i].platform_code
    dat = dat.reset_index()
    reeftemps_pd.append(dat)

'Concatenate all points to a single pandas dataframe'
reeftemps = pd.concat(reeftemps_pd).reset_index(drop=True)

'Rename time column to datetime to standardize with other datasets'
reeftemps = reeftemps.rename(columns={'TIME':'DATETIME'})

'''
Geomorphology
'''

'Add geomorphology from East Micronesia region'
emicro_geo = gpd.read_file(geom_path+'Eastern-Micronesia_W.gpkg')
emicro_geo = emicro_geo.rename(columns={'rclass':'class'})
emicro = agg_and_merge(reeftemps,emicro_geo)

'Add geomorphology from Central South Pacific region'
cs_pacific_geo = gpd.read_file(geom_path+'Central-South-Pacific.gpkg')
cs_pacific = agg_and_merge(reeftemps,cs_pacific_geo)

'Add geomorphology from Eastern Papua New Guinea region'
png_geo = gpd.read_file(geom_path+'Eastern-Papua-New-Guinea.gpkg')
png = agg_and_merge(reeftemps,png_geo)

'Add geomorphology from Southwestern Pacific region, west of International Date Line'
swpacific_w_geo = gpd.read_file(geom_path+'Southwestern-Pacific_W.gpkg')
swpacific_w_geo = swpacific_w_geo.rename(columns={'rclass':'class'})
swpacific_w = agg_and_merge(reeftemps,swpacific_w_geo)

'Add geomorphology from Southwestern Pacific region, east of International Date Line'
swpacific_e_geo = gpd.read_file(geom_path+'Southwestern-Pacific_E.gpkg')
swpacific_e_geo = swpacific_e_geo.rename(columns={'rclass':'class'})
swpacific_e = agg_and_merge(reeftemps,swpacific_e_geo)

'Add geomorphology from Western Micronesia'
wmicro_geo = gpd.read_file(geom_path+'Western-Micronesia.gpkg')
wmicro = agg_and_merge(reeftemps,wmicro_geo)

'Concat all geomorphic regions and downsample temperature data to every 30 minutes'
reeftemps = downsampler(pd.concat([emicro,cs_pacific,png,swpacific_w,swpacific_e,wmicro]).reset_index(drop=True))

'''
Remove data from before 2003
'''
reeftemps = reeftemps[reeftemps['DATETIME'] > pd.to_datetime('2003-01-01 00:00:00')]
reeftemps = reeftemps.reset_index(drop=True)

'''
Remove data from after 2019 to align with SST data
'''
reeftemps = reeftemps[reeftemps['DATETIME'] < pd.to_datetime('2020-01-01 00:00:00')]
reeftemps = reeftemps.reset_index(drop=True)

'''
Remove days with <24 samples
'''
reeftemps['DAYPT'] =  reeftemps['DATETIME'].dt.date.astype('str') +'_' + reeftemps['ID'].astype('str')
dayagg= reeftemps.groupby(['DAYPT'])['DAYPT'].count()
daykeep = pd.DataFrame()
daykeep['DAYCOUNT'] = dayagg.values
daykeep['DAYPT'] =dayagg.index

reeftemps = pd.merge(reeftemps,daykeep,how='left',on='DAYPT')
reeftemps = reeftemps[reeftemps['DAYCOUNT'] > 23]
reeftemps = reeftemps.reset_index(drop=True)
reeftemps = reeftemps.drop(columns = ['DAYCOUNT','DAYPT'])

'''
Remove months with <15 days
'''
reeftemps['MONTHPT'] =  reeftemps['DATETIME'].dt.month.astype('str') + '_' + reeftemps['DATETIME'].dt.year.astype('str') +'_' + reeftemps['ID'].astype('str')
oneaday = downsampler2(reeftemps)
monthagg= oneaday.groupby(['MONTHPT'])['MONTHPT'].count()
monthkeep = pd.DataFrame()
monthkeep['MONTHCOUNT'] = monthagg.values
monthkeep['MONTHPT'] =monthagg.index

reeftemps = pd.merge(reeftemps,monthkeep,how='left',on='MONTHPT')
reeftemps = reeftemps[reeftemps['MONTHCOUNT'] > 15]
reeftemps = reeftemps.reset_index(drop=True)
reeftemps = reeftemps.drop(columns = ['MONTHCOUNT','MONTHPT'])

'''
Remove points with fewer than 12 months of data
'''
reeftemps['MONTH'] =reeftemps['DATETIME'].dt.month
d_agg = reeftemps.groupby(['ID','MONTH']).first().reset_index()

monthn= pd.DataFrame(d_agg['ID'].value_counts())
monthn['counts'] =monthn.values
monthn['ID'] =monthn.index
monthn = monthn.reset_index()

reeftemps = pd.merge(reeftemps,monthn,how='left',on='ID')
reeftemps = reeftemps[reeftemps['counts'] > 11]
reeftemps = reeftemps.reset_index(drop=True)
'''
Calculate mean daily diff 
'''
rt = reeftemps.copy()
rt['dmin'] = rt.groupby(['ID','DATE']).TEMP_C.transform('min')
rt['dmax'] = rt.groupby(['ID','DATE']).TEMP_C.transform('max')
rt['DAILY_DIFF'] = rt['dmax'] - rt['dmin']
rt = rt.drop(columns=['dmin','dmax'])
a = {'DAILY_DIFF': ['mean']}
temps_a = rt.groupby(['ID']).agg(a).reset_index() 

'''
Calculate MMMs
'''
rt_mmms = prep_data(reeftemps)
rt_mmms.columns = ['index','MONTH','ID','TEMP_C','LONGITUDE','LATITUDE','RCLASS','COORDS','DEPTH']
rt_mmms = rt_mmms.drop(columns=['index'])

'''
Calculate q99
'''
rtemps_q99 = prep_q99(reeftemps)
rtemps_q99.columns = ['index','MONTH', 'ID','DEPTH', 'Q99',  'LONGITUDE', 'LATITUDE', 'DATETIME', 'DATE']
rtemps_q99 = rtemps_q99.drop(columns=['index','MONTH','DEPTH','LONGITUDE','LATITUDE','DATETIME','DATE'])
rtemps_q99 = rtemps_q99.drop_duplicates()

'Combine MMM, peak temps, daily diffs'
rtemps = prep_data_more(reeftemps,rt_mmms)
rtemps = pd.merge(rtemps,temps_a,on='ID')
rtemps = pd.merge(rtemps,rtemps_q99,on='ID')

'Rename columns'
rtemps.columns = ['MONTH','ID','TEMP_C','LONGITUDE','LATITUDE','RCLASS','COORDS','DEPTH','START_TIME','END_TIME','DAILY_DIFF','Q99']

'Reorder columns'
rtemps = rtemps[['MONTH','ID','TEMP_C','LONGITUDE','LATITUDE','RCLASS','COORDS','START_TIME','END_TIME','Q99','DAILY_DIFF','DEPTH']]

'Export data to csv'
outfile = open(out_path+'reeftemps.csv','wb')
rtemps.to_csv(outfile)
outfile.close()