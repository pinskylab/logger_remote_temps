"""
Script for processing in situ logger temperature data from the Phoenix Islands Protected Area
Author: Jaelyn Bos
Data available at XXX/XX
2024
"""
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
from shapely.geometry import Point

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

'Filepath to folder with dataset in netcdf format'
path = '/projects/f_mlp195/bos_microclimate/loggers_testing/'

'Filepath to folder with geomorphic data from Allen Coral Atlas'
geom_path = '/projects/f_mlp195/bos_microclimate/geomorphology/'

'Filepath to folder to save processed data'
out_path = '/home/jtb188/'

'Open data in netcdf form'
pipa_xr = xr.open_dataset(path+ 'PIPA_temperature.netcdf')

'Rename columns'
pipa = pd.DataFrame()
pipa['LATITUDE'] = pipa_xr['Latitude']
pipa['LONGITUDE'] = pipa_xr['Longitude']
pipa['TEMP_C'] = pipa_xr['Temperature']
pipa['Date'] = pipa_xr['Date']
pipa['Hour'] = pipa_xr['Hour']
pipa['island'] = pipa_xr['Island']

'Define little function to convert oddly formatted lats, lons, and temps to floating point numbers'
def toNum (series):
    series = series.apply(str)
    series = series.str.strip('b')
    series = series.str.strip("'")
    series = series.apply(float)
    return(series)

'Convert oddly formatted lats, lons, and temps to floating point numbers'
pipa['LATITUDE'] = toNum(pipa['LATITUDE'])
pipa['LONGITUDE'] = toNum(pipa['LONGITUDE'])
pipa['TEMP_C'] = toNum(pipa['TEMP_C'])

'Convert date and hour from irregular format to strings'
pipa['Date'] = pipa['Date'].apply(str)
pipa['Hour'] = pipa['Hour'].apply(str)

pipa['Date'] = pipa['Date'].str.strip('b')
pipa['Date'] = pipa['Date'].str.strip("'")

pipa['Hour'] = pipa['Hour'].str.strip('b')
pipa['Hour'] = pipa['Hour'].str.strip("'")

'Combine date and hour to create datetime variable, define proper format'
pipa['Datetime'] = pipa['Date'] + ':' + pipa['Hour']
pipa['DATETIME']  = pd.to_datetime(pipa['Datetime'],format='%Y-%m-%d:%H')

'Create ID column from lats and lons and define as string'
pipa['ID'] = pd.Series(pipa['LATITUDE'].apply(str) + pipa['LONGITUDE'].apply(str)).apply(str)

'Remove unwanted columns'
pipa = pipa[keepcols]

'''
Add geomorphology from Eastern Micronesia region
'''
pipa_geo = gpd.read_file(geom_path+ 'Eastern-Micronesia_E.gpkg')
pipa_geo.columns = ['class','geometry']

pipa = agg_and_merge(pipa,pipa_geo)

'''
Downsample data to every 30 minutes
'''
pipa = downsampler(pipa)
pipa['MONTH'] =pipa['DATETIME'].dt.month
d_agg = pipa.groupby(['ID','MONTH']).first().reset_index()

'''
Remove days with <24 samples
'''
pipa['DAYPT'] =  pipa['DATETIME'].dt.date.astype('str') +'_' + pipa['ID'].astype('str')
dayagg= pipa.groupby(['DAYPT'])['DAYPT'].count()
daykeep = pd.DataFrame()
daykeep['DAYCOUNT'] = dayagg.values
daykeep['DAYPT'] =dayagg.index

pipa = pd.merge(pipa,daykeep,how='left',on='DAYPT')
pipa = pipa[pipa['DAYCOUNT'] > 23]
pipa = pipa.reset_index(drop=True)
pipa = pipa.drop(columns = ['DAYCOUNT','DAYPT'])

'''
Remove months with <15 days
'''
pipa['MONTHPT'] =  pipa['DATETIME'].dt.month.astype('str') + '_' + pipa['DATETIME'].dt.year.astype('str') +'_' + pipa['ID'].astype('str')
oneaday = downsampler2(pipa)
monthagg= oneaday.groupby(['MONTHPT'])['MONTHPT'].count()
monthkeep = pd.DataFrame()
monthkeep['MONTHCOUNT'] = monthagg.values
monthkeep['MONTHPT'] =monthagg.index

pipa = pd.merge(pipa,monthkeep,how='left',on='MONTHPT')
pipa = pipa[pipa['MONTHCOUNT'] > 15]
pipa = pipa.reset_index(drop=True)
pipa = pipa.drop(columns = ['MONTHCOUNT','MONTHPT'])

'''
Remove points with <12 months
'''
monthn= pd.DataFrame(d_agg['ID'].value_counts())
monthn['counts'] =monthn.values
monthn['ID'] =monthn.index
monthn = monthn.reset_index()

pipa = pd.merge(pipa,monthn,how='left',on='ID')
pipa = pipa[pipa['counts'] > 11]
pipa = pipa.reset_index(drop=True)

'''
Calculate mean daily diff
'''
pipa2 = pipa.copy()
pipa2['dmin'] = pipa2.groupby(['ID','DATE']).TEMP_C.transform('min')
pipa2['dmax'] = pipa2.groupby(['ID','DATE']).TEMP_C.transform('max')
pipa2['DAILY_DIFF'] = pipa2['dmax'] - pipa2['dmin']
pipa2 = pipa2.drop(columns=['dmin','dmax'])
a = {'DAILY_DIFF': ['mean']}
temps_a = pipa2.groupby(['ID']).agg(a).reset_index() 

'''
Calculate MMMs
'''
pipa['DEPTH'] = 8
pipa_mmms = prep_data(pipa)
pipa_mmms.columns = ['index','MONTH','ID','TEMP_C','LONGITUDE','LATITUDE','RCLASS','COORDS','DEPTH']
pipa_mmms = pipa_mmms.drop(columns=['index'])

'''
Calculate q99
'''
pipa_q99 = prep_q99(pipa)
pipa_q99.columns = ['index','MONTH', 'ID', 'Q99', 'LONGITUDE', 'LATITUDE', 'DATETIME', 'DATE','DEPTH']
pipa_q99 = pipa_q99.drop(columns=['index','MONTH','LONGITUDE','LATITUDE','DATETIME','DATE','DEPTH'])
pipa_q99 = pipa_q99.drop_duplicates()

'Combine MMM, peak temps, daily diffs'
pipa = prep_data_more(pipa,pipa_mmms)
pipa = pd.merge(pipa,temps_a,on='ID')
pipa = pd.merge(pipa,pipa_q99,on='ID')

'Rename columns'
pipa.columns = ['MONTH','ID','TEMP_C','LONGITUDE','LATITUDE','RCLASS','COORDS','DEPTH','START_TIME','END_TIME','DAILY_DIFF','Q99']

'Reorder columns'
pipa = pipa[['MONTH','ID','TEMP_C','LONGITUDE','LATITUDE','RCLASS','COORDS','START_TIME','END_TIME','Q99','DAILY_DIFF','DEPTH']]

'Export data as csv'
outfile = open(out_path+'pipa.csv','wb')
pipa.to_csv(outfile)
outfile.close()