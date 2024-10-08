"""
Script for processing in situ logger temperature data from Dongsha Atoll National Park, South China Sea
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
keepcols = ['LATITUDE','LONGITUDE','ID','DATETIME','TEMP_C','DEPTH']

'Filepath to folder with dataset in netcdf format'
path = '/projects/f_mlp195/bos_microclimate/loggers_testing/'

'Filepath to folder with geomorphic data from Allen Coral Atlas'
geom_path = '/projects/f_mlp195/bos_microclimate/geomorphology/'

'Filepath to folder to save processed data'
out_path = '/home/jtb188/'

'''
Open data from netcdf format
'''
dongsha_xr = xr.open_dataset(path + 'dongsha_temps.netcdf')

'Create Pandas dataframe with desired columns'
dongsha = pd.DataFrame()
dongsha['LATITUDE'] = dongsha_xr['lat']
dongsha['LONGITUDE'] = dongsha_xr['lon']
dongsha['ISO_DateTime_UTC'] = dongsha_xr['ISO_DateTime_UTC']
dongsha['time'] = dongsha_xr['time']
dongsha['month'] = dongsha_xr['mon']
dongsha['year'] = dongsha_xr['year']
dongsha['day'] = dongsha_xr['day']
dongsha['TEMP_C'] = dongsha_xr['temp']
dongsha['ID'] = dongsha_xr['location']
dongsha['DEPTH'] = dongsha_xr['depth']

'''
Convert odd variable type ('bytes4104') to floating point numeric
'''
def toNum (series):
    series = series.apply(str)
    series = series.str.strip('b')
    series = series.str.strip("'")
    series = series.apply(float)
    return(series)

'Convert oddly formatted lats, lons, depths, temps, and partial date variables to floating point numbers'
dongsha['LATITUDE'] = toNum(dongsha['LATITUDE'])
dongsha['LONGITUDE'] = toNum(dongsha['LONGITUDE'])
dongsha['DEPTH'] = toNum(dongsha['DEPTH'])
dongsha['TEMP_C'] = toNum(dongsha['TEMP_C'])

dongsha['day'] = toNum(dongsha['day'])
dongsha['month'] = toNum(dongsha['month'])

'Change formatting on days, months, and years to move easily combine into datetime variable'
dongsha['year'] = dongsha['year'].apply(int)
dongsha['month'] = dongsha['month'].apply(int).apply(str).str.zfill(2)
dongsha['day'] = dongsha['day'].apply(int).apply(str).str.zfill(2)
dongsha['time'] = dongsha['time'].apply(int).apply(str).str.zfill(4)

'Divide time variable into hours and minutes in order to be able to combine into datetime variable'
times = dongsha['time'].apply(list)
minutes = dongsha['time'].str[2] + dongsha['time'].str[3]
hours = dongsha['time'].str[0] + dongsha['time'].str[1]

'Construct datetime variable from years, months, days, hours, and minutes'
dongsha['DATE'] = dongsha['year'].apply(str) + "-" + dongsha['month'].apply(str) + "-" + dongsha['day'].apply(str)
dongsha['TIME'] = hours + ":" + minutes + ":" + '00'
dongsha['DATETIME'] = pd.to_datetime((dongsha['DATE'] + " " + dongsha['time']),format='%Y-%m-%d %H:%M:%S')

'Drop unwanted columns'
dongsha = dongsha[keepcols]

'''
Merge with geomorphic from the South China Sea
'''
dongsha_geo = gpd.read_file(geom_path+'South-China-Sea.gpkg',layer='South China Sea')
dongsha = agg_and_merge(dongsha,dongsha_geo)

'''
Merge and temporal filter
'''
dongsha = downsampler(dongsha)
dongsha['MONTH'] =dongsha['DATETIME'].dt.month
d_agg = dongsha.groupby(['ID','MONTH']).first().reset_index()

'''
Remove days with fewer than 24 points
'''
dongsha['DAYPT'] =  dongsha['DATETIME'].dt.date.astype('str') +'_' + dongsha['ID'].astype('str')
dayagg= dongsha.groupby(['DAYPT'])['DAYPT'].count()
daykeep = pd.DataFrame()
daykeep['DAYCOUNT'] = dayagg.values
daykeep['DAYPT'] =dayagg.index

dongsha = pd.merge(dongsha,daykeep,how='left',on='DAYPT')
dongsha = dongsha[dongsha['DAYCOUNT'] > 23]
dongsha = dongsha.reset_index(drop=True)
dongsha = dongsha.drop(columns = ['DAYCOUNT','DAYPT'])

'''
Remove months with <15 days
'''
dongsha['MONTHPT'] =  dongsha['DATETIME'].dt.month.astype('str') + '_' + dongsha['DATETIME'].dt.year.astype('str') +'_' + dongsha['ID'].astype('str')
oneaday = downsampler2(dongsha)
monthagg= oneaday.groupby(['MONTHPT'])['MONTHPT'].count()
monthkeep = pd.DataFrame()
monthkeep['MONTHCOUNT'] = monthagg.values
monthkeep['MONTHPT'] =monthagg.index

dongsha = pd.merge(dongsha,monthkeep,how='left',on='MONTHPT')
dongsha = dongsha[dongsha['MONTHCOUNT'] > 15]
dongsha = dongsha.reset_index(drop=True)
dongsha = dongsha.drop(columns = ['MONTHCOUNT','MONTHPT'])

'''
Remove points with <12 months
'''
monthn= pd.DataFrame(d_agg['ID'].value_counts())
monthn['counts'] =monthn.values
monthn['ID'] =monthn.index
monthn = monthn.reset_index()

dongsha = pd.merge(dongsha,monthn,how='left',on='ID')
dongsha = dongsha[dongsha['counts'] > 11]
dongsha = dongsha.reset_index(drop=True)

'''
Calculate mean daily diff 
'''
d2 = dongsha.copy()
d2['dmin'] = d2.groupby(['ID','DATE']).TEMP_C.transform('min')
d2['dmax'] = d2.groupby(['ID','DATE']).TEMP_C.transform('max')
d2['DAILY_DIFF'] = d2['dmax'] - d2['dmin']
d2 = d2.drop(columns=['dmin','dmax'])
a = {'DAILY_DIFF': ['mean']}
temps_a = d2.groupby(['ID']).agg(a).reset_index() 

'''
Calculate MMMs
'''
dongsha_mmms = prep_data(dongsha)
dongsha_mmms.columns = ['index','MONTH','ID','TEMP_C','LONGITUDE','LATITUDE','RCLASS','COORDS','DEPTH']

'''
Calculate q99
'''
dongsha_q99 = prep_q99(dongsha)
dongsha_q99.columns = ['index','MONTH', 'ID', 'Q99', 'DEPTH', 'LONGITUDE', 'LATITUDE', 'DATETIME', 'DATE']
dongsha_q99 = dongsha_q99.drop(columns=['index','MONTH','DEPTH','LONGITUDE','LATITUDE','DATETIME','DATE'])
dongsha_q99 = dongsha_q99.drop_duplicates()

'Combine MMM, peak temps, daily diffs'
dongsha = prep_data_more(dongsha,dongsha_mmms)
dongsha = pd.merge(dongsha,temps_a,on='ID')
dongsha = pd.merge(dongsha,dongsha_q99,on='ID')

'Rename columns'
dongsha.columns = ['index','MONTH','ID','TEMP_C','LONGITUDE','LATITUDE','RCLASS','COORDS','DEPTH','START_TIME','END_TIME','DAILY_DIFF','Q99']

'Reorder columns'
dongsha = dongsha[['index','MONTH','ID','TEMP_C','LONGITUDE','LATITUDE','RCLASS','COORDS','START_TIME','END_TIME','Q99','DAILY_DIFF','DEPTH']]

'Export data to csv'
outfile = open(out_path + 'dongsha.csv','wb')
dongsha.to_csv(outfile)
outfile.close()
