"""
Script for processing in situ logger temperature data from Australian Institute of Marine Sciences (AIMS)
Author: Jaelyn Bos
Data available at XXX/XX
2024
"""
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import datetime as dt

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
        region_ds = region_ds[['ID','MONTH','TEMP_C']]
        q99 = region_ds.groupby(['MONTH','ID']).agg(np.quantile, q=0.99).reset_index() 
        Q99 = mmm_select(q99,points_u)
        Q99 = Q99.drop(columns = ['MONTH','index'])
        Q99 = Q99.drop_duplicates()
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

'Filepath to folder with dataset in csv format'
path = '/projects/f_mlp195/bos_microclimate/loggers_training/'

'Filepath to folder with geomorphic data from Allen Coral Atlas'
geom_path = '/projects/f_mlp195/bos_microclimate/geomorphology/'

'Filepath to folder to save processed data'
out_path = '/home/jtb188/'

'''
Import data as Pandas dataframe and move relevant columns to new pandas dataframe
'''
aims_data = pd.read_csv(path+'aims-logger-data.csv')

aims_data.columns

aims = pd.DataFrame()
aims['LATITUDE'] = aims_data['lat']
aims['LONGITUDE'] = aims_data['lon']
aims['DATETIME'] = pd.to_datetime(aims_data['time'],infer_datetime_format=True)
aims['TEMP_C'] = aims_data['qc_val']
aims['ID'] = aims_data['subsite_id']
aims['DEPTH'] = aims_data['depth']
'''
Filter out rows where time = NaN
'''
aims = aims[aims['TEMP_C']>0]

'''
Format datetime
'''
aims['DATETIME'] = aims['DATETIME'].dt.tz_convert(None)

'''
Filter out rows before the year 2003 (to agree with MUR SST data). Timezone is incorrect but matches input data
'''
aims = aims[aims['DATETIME']>pd.to_datetime('2003-01-01T00:00:00',infer_datetime_format=True)]

'''
Remove data from after 2019 to align with SST data
'''
aims = aims[aims['DATETIME'] < pd.to_datetime('2020-01-01T00:00:00',infer_datetime_format=True)]
aims = aims.reset_index(drop=True)

'''
Match to geomorphic data from six Allen Coral Atlas regions
'''
coral_geo = gpd.read_file(geom_path+'Coral-Sea.gpkg')
westaus_geo = gpd.read_file(geom_path+'Western-Australia.gpkg')
subeast_geo = gpd.read_file(geom_path+'Subtropical-Eastern-Australia.gpkg')
timor_geo = gpd.read_file(geom_path+'Timor-Arafura-Seas.gpkg')
enewguinea_geo = gpd.read_file(geom_path+'Eastern-Papua-New-Guinea.gpkg')
gbr_geo = gpd.read_file(geom_path+'Great-Barrier-Reef-and-Torres-Strait.gpkg')

coralsea = agg_and_merge(aims,coral_geo)
subeast = agg_and_merge(aims,subeast_geo)
westaus = agg_and_merge(aims,westaus_geo)
timor = agg_and_merge(aims,timor_geo)
gbr = agg_and_merge(aims,gbr_geo)

'Re concatenate loggers from each region'
aims2 = pd.concat([coralsea,subeast,westaus,timor,gbr])

'''
Downsample to every 30 minutes for each point
'''
aims2 = downsampler(aims2)

aims2['MONTH'] =aims2['DATETIME'].dt.month
d_agg = aims2.groupby(['ID','MONTH']).first().reset_index()
'''
Remove days with <24 samples
'''
aims2['DAYPT'] =  aims2['DATETIME'].dt.date.astype('str') +'_' + aims2['ID'].astype('str')
dayagg= aims2.groupby(['DAYPT'])['DAYPT'].count()
daykeep = pd.DataFrame()
daykeep['DAYCOUNT'] = dayagg.values
daykeep['DAYPT'] =dayagg.index

aims2 = pd.merge(aims2,daykeep,how='left',on='DAYPT')
aims2 = aims2[aims2['DAYCOUNT'] > 23]
aims2 = aims2.reset_index(drop=True)
aims2 = aims2.drop(columns = ['DAYCOUNT','DAYPT'])

'''
Remove months with <15 days
'''
aims2['MONTHPT'] =  aims2['DATETIME'].dt.month.astype('str') + '_' + aims2['DATETIME'].dt.year.astype('str') +'_' + aims2['ID'].astype('str')
oneaday = downsampler2(aims2)
monthagg= oneaday.groupby(['MONTHPT'])['MONTHPT'].count()
monthkeep = pd.DataFrame()
monthkeep['MONTHCOUNT'] = monthagg.values
monthkeep['MONTHPT'] =monthagg.index

aims2 = pd.merge(aims2,monthkeep,how='left',on='MONTHPT')
aims2 = aims2[aims2['MONTHCOUNT'] > 15]
aims2 = aims2.reset_index(drop=True)
aims2 = aims2.drop(columns = ['MONTHCOUNT','MONTHPT'])

'''
Remove points with <12 months
'''
monthn= pd.DataFrame(d_agg['ID'].value_counts())
monthn['counts'] =monthn.values
monthn['ID'] =monthn.index
monthn = monthn.reset_index()

aims2 = pd.merge(aims2,monthn,how='left',on='ID')
aims2 = aims2[aims2['counts'] > 11]
aims2 = aims2.reset_index(drop=True)

'''
Calculate mean daily diff
'''
aims3 = aims2.copy()
aims3['dmin'] = aims3.groupby(['ID','DATE']).TEMP_C.transform('min')
aims3['dmax'] = aims3.groupby(['ID','DATE']).TEMP_C.transform('max')
aims3['DAILY_DIFF'] = aims3['dmax'] - aims3['dmin']
aims3 = aims3.drop(columns=['dmin','dmax'])
a = {'DAILY_DIFF': ['mean']}
temps_a = aims3.groupby(['ID']).agg(a).reset_index() 

'''
Calculate MMMs
'''
aims_mmms = prep_data(aims2)
aims_mmms.columns = ['index','MONTH','ID','TEMP_C','LONGITUDE','LATITUDE','RCLASS','COORDS','DEPTH',]
aims_mmms = aims_mmms.drop(columns=['index'])

'''
Calculate q99
'''
aims_q99 = prep_q99(aims2)
aims_q99.columns =['ID','Q99']

'Combine MMM, peak temps, daily diffs'
aims = prep_data_more(aims2,aims_mmms)
aims = pd.merge(aims,temps_a,on='ID')
aims = pd.merge(aims,aims_q99,on='ID')

'Rename columns'
aims.columns = ['MONTH','ID','TEMP_C','LONGITUDE','LATITUDE','RCLASS','COORDS','DEPTH','START_TIME','END_TIME','DAILY_DIFF','Q99']

'Reorder columns'
aims = aims[['MONTH','ID','TEMP_C','LONGITUDE','LATITUDE','RCLASS','COORDS','START_TIME','END_TIME','Q99','DAILY_DIFF','DEPTH']]

'Export data as csv'
outfile = open(out_path+ 'aims.csv','wb')
aims.to_csv(outfile)
outfile.close()