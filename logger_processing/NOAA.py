"""
Script for processing in situ logger temperature data from
the National Oceanographic and Atmospheric Administration (NOAA) and the United States Geological Survey (USGS)
Author: Jaelyn Bos
Data available at XXX/XX
2024
"""
import pandas as pd
import glob
import numpy as np
import geopandas as gpd
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

'Function to return 99th quantile of hottest month for each logger in region'
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

'Filepath to folder with datasets in csv format'
path = '/projects/f_mlp195/bos_microclimate/loggers_training/NOAA_data/'

'Filepath to folder with geomorphic data from Allen Coral Atlas'
geom_path = '/projects/f_mlp195/bos_microclimate/geomorphology/'

'Filepath to folder to save processed data'
out_path = '/home/jtb188/'

'Columns to keep from original datasets'
keepcols = ['LATITUDE','LONGITUDE','ID','DATETIME','TEMP_C','DEPTH']

'''
Hawaii
'''

'Open three different NOAA datasets from Hawaii, all in csv form'
hawaii_1 = pd.read_csv(path + "hawaii_temps/hawaii_temps_1/2.2/data/0-data/unzipped/ESD_NCRMP_Temperature_2013_Hawaii.csv")
hawaii_2 = pd.read_csv(path + "hawaii_temps/hawaii_temps_2/ESD_NCRMP_Temperature_2016_Hawaii/ESD_NCRMP_Temperature_2016_Hawaii.csv")
hawaii_3 = pd.read_csv(path + "hawaii_temps/hawaii_temps_3/ESD_NCRMP_Temperature_2019_Hawaii/ESD_NCRMP_Temperature_2019_Hawaii.csv")

'Concatenate all NOAA-Hawaii data'
hawaii = pd.concat([hawaii_1,hawaii_2,hawaii_3],axis=0,ignore_index=True,join="outer")

'Reformate datetime'
hawaii['DATETIME'] = pd.to_datetime(hawaii['TIMESTAMP'], format='%Y-%m-%d %H:%M:%S')

'Rename logger ID column'
hawaii = hawaii.rename(columns={'OCC_SITEID':'ID'})

'Trim data to valid timeframe only'
hawaii = hawaii[hawaii['DATETIME'] > pd.to_datetime(hawaii['VALIDDATASTART'])]
hawaii = hawaii[hawaii['DATETIME'] < pd.to_datetime(hawaii['VALIDDATAEND'])]
hawaii = hawaii[keepcols]

'Delete separate datasets, keep concatenate dataset only'
del hawaii_1,hawaii_2,hawaii_3

'Add geomorphology from Allen Coral Atlas'
geom = gpd.read_file(geom_path+'Hawaiian-Islands.gpkg')

'Downsample logger data to every 30 minutes'
hawaii = downsampler(agg_and_merge(hawaii,geom))

'''
West Hawaii (from USGS)
'''

'Open three different USGS datasets from Hawaii, all in csv form'
hawaii_1 = pd.read_csv('/projects/f_mlp195/bos_microclimate/loggers_training/USGS_data/west_hawaii_temps/WaterTempTimeSeries_KAHO-KC.csv')
hawaii_2 = pd.read_csv('/projects/f_mlp195/bos_microclimate/loggers_training/USGS_data/west_hawaii_temps/WaterTempTimeSeries_KAHO.csv')
hawaii_3 = pd.read_csv('/projects/f_mlp195/bos_microclimate/loggers_training/USGS_data/west_hawaii_temps/WaterTempTimeSeries_westHawaii.csv')

'Concatenate all USGS Hawaii data'
west_hawaii = pd.concat([hawaii_1,hawaii_2,hawaii_3])

'Rename columns to match NOAA data'
west_hawaii = west_hawaii.rename(columns = {'Site':'ID'})
west_hawaii = west_hawaii.rename(columns = {'Depth_m':'DEPTH'})
west_hawaii = west_hawaii.rename(columns = {'Temperature_C':'TEMP_C'})
west_hawaii = west_hawaii.rename(columns = {'Latitude':'LATITUDE'})
west_hawaii = west_hawaii.rename(columns = {'Longitude':'LONGITUDE'})
west_hawaii['DATETIME'] = pd.to_datetime(west_hawaii['DateTime'])

'Keep only necessary columns'
west_hawaii = west_hawaii[keepcols]

'Delete intermediate datasets'
del hawaii_1,hawaii_2,hawaii_3

'Add geomorphology from Allen Coral Atlas and downsample to every 30 minutes'
west_hawaii = downsampler(agg_and_merge(west_hawaii,geom))
del geom

'Concatenate NOAA and USGS Hawaii data'
noaa_hawaii = pd.concat([hawaii,west_hawaii])

'''
Remove days with fewer than 24 points
'''
noaa_hawaii['DAYPT'] =  noaa_hawaii['DATETIME'].dt.date.astype('str') +'_' + noaa_hawaii['ID'].astype('str')
dayagg= noaa_hawaii.groupby(['DAYPT'])['DAYPT'].count()
daykeep = pd.DataFrame()
daykeep['DAYCOUNT'] = dayagg.values
daykeep['DAYPT'] =dayagg.index

noaa_hawaii = pd.merge(noaa_hawaii,daykeep,how='left',on='DAYPT')
noaa_hawaii = noaa_hawaii[noaa_hawaii['DAYCOUNT'] > 23]
noaa_hawaii = noaa_hawaii.reset_index(drop=True)
noaa_hawaii = noaa_hawaii.drop(columns = ['DAYCOUNT','DAYPT'])

'''
Remove months with <15 days
'''
noaa_hawaii['MONTHPT'] =  noaa_hawaii['DATETIME'].dt.month.astype('str') + '_' + noaa_hawaii['DATETIME'].dt.year.astype('str') +'_' + noaa_hawaii['ID'].astype('str')
oneaday = downsampler2(noaa_hawaii)
monthagg= oneaday.groupby(['MONTHPT'])['MONTHPT'].count()
monthkeep = pd.DataFrame()
monthkeep['MONTHCOUNT'] = monthagg.values
monthkeep['MONTHPT'] =monthagg.index

noaa_hawaii = pd.merge(noaa_hawaii,monthkeep,how='left',on='MONTHPT')
noaa_hawaii = noaa_hawaii[noaa_hawaii['MONTHCOUNT'] > 15]
noaa_hawaii = noaa_hawaii.reset_index(drop=True)
noaa_hawaii = noaa_hawaii.drop(columns = ['MONTHCOUNT','MONTHPT'])

'''
Remove points with <12 months
'''
noaa_hawaii['MONTH'] =noaa_hawaii['DATETIME'].dt.month
d_agg = noaa_hawaii.groupby(['ID','MONTH']).first().reset_index()

monthn= pd.DataFrame(d_agg['ID'].value_counts())
monthn['counts'] =monthn.values
monthn['ID'] =monthn.index
monthn = monthn.reset_index()

noaa_hawaii = pd.merge(noaa_hawaii,monthn,how='left',on='ID')
noaa_hawaii = noaa_hawaii[noaa_hawaii['counts'] > 11]
noaa_hawaii = noaa_hawaii.reset_index(drop=True)

'''
Calculate mean daily diff 
'''
nh = noaa_hawaii.copy()
nh['dmin'] = nh.groupby(['ID','DATE']).TEMP_C.transform('min')
nh['dmax'] = nh.groupby(['ID','DATE']).TEMP_C.transform('max')
nh['DAILY_DIFF'] = nh['dmax'] - nh['dmin']
nh = nh.drop(columns=['dmin','dmax'])
a = {'DAILY_DIFF': ['mean']}
temps_a = nh.groupby(['ID']).agg(a).reset_index() 

'''
Calculate MMMs
'''
hawaii_mmms = prep_data(noaa_hawaii)
hawaii_mmms.columns = ['index','MONTH','ID','TEMP_C','LONGITUDE','LATITUDE','RCLASS','COORDS','DEPTH']
hawaii_mmms = hawaii_mmms.drop(columns=['index'])

'''
Calculate q99
'''
hawaii_q99 = prep_q99(noaa_hawaii)
hawaii_q99.columns = ['index','MONTH', 'ID', 'Q99', 'DEPTH', 'LONGITUDE', 'LATITUDE', 'DATETIME', 'DATE']
hawaii_q99 = hawaii_q99.drop(columns=['index','MONTH','DEPTH','LONGITUDE','LATITUDE','DATETIME','DATE'])
hawaii_q99 = hawaii_q99.drop_duplicates()

'Combine MMM, peak temps, daily diffs'
hawaii = prep_data_more(noaa_hawaii,hawaii_mmms)
hawaii = pd.merge(hawaii,temps_a,on='ID')
hawaii = pd.merge(hawaii,hawaii_q99,on='ID')

'Rename columns'
hawaii.columns = ['MONTH','ID','TEMP_C','LONGITUDE','LATITUDE','RCLASS','COORDS','DEPTH','START_TIME','END_TIME','DAILY_DIFF','Q99']

'Reorder columns'
hawaii = hawaii[['MONTH','ID','TEMP_C','LONGITUDE','LATITUDE','RCLASS','COORDS','START_TIME','END_TIME','Q99','DAILY_DIFF','DEPTH']]

'Export all Hawaii dataset to csv'
outfile = open(out_path+'noaa_hawaii.csv','wb')
hawaii.to_csv(outfile)
outfile.close()
del d_agg, monthn, noaa_hawaii, temps_a

'''
NOAA data from various non-Hawaii Pacific islands
'''

'''
American Samoa
'''

'Import data from American Samoa in csv format'
amsam = pd.read_csv(path + 'american_samoa_temps/2.2/data/0-data/unzipped/ESD_NCRMP_Temperature_2015_AMSM.csv')
amsam = amsam.rename(columns={'OCC_SITEID':'ID'})
amsam['DATETIME'] = pd.to_datetime(amsam['TIMESTAMP'],infer_datetime_format=True)

'Trim data to valid timeframe only'
amsam = amsam[amsam['DATETIME'] > pd.to_datetime(amsam['VALIDDATASTART'])]
amsam = amsam[amsam['DATETIME'] < pd.to_datetime(amsam['VALIDDATAEND'])]
amsam = amsam[keepcols]

'Combine with geomorphology and downsample logger data to every 30 minutes'
geom = gpd.read_file(geom_path+'Southwestern-Pacific_E.gpkg')
geom = geom.rename(columns={'rclass':'class'})
amsam = downsampler(agg_and_merge(amsam,geom))
del geom

'''
Batangas
'''
batangas = pd.read_csv(path+ 'batangas_philippines_temps/1.1/data/0-data/MV_OCN_STR_TIMESERIES_Philippines.csv')
batangas = batangas.rename(columns={'TEMPERATURE':'TEMP_C'})

'Reformat datetime column to match other NOAA datasets'
dummy = batangas['YEAR'].astype(str)+ batangas['MONTH'].astype(str).str.zfill(2) + batangas['DAY'].astype(str).str.zfill(2) +batangas['HOUR'].astype(str).str.zfill(2) +batangas['MINUTE'].astype(str).str.zfill(2)+batangas['SECOND'].astype(str).str.zfill(2) 
batangas['DATETIME'] = pd.to_datetime(dummy,format='%Y%m%d%H%M%S')

'Rename ID column'
batangas = batangas.rename(columns={'SITETAG':'ID'})

'Trim data to valid timeframe only'
batangas = batangas[batangas['DATETIME'] > pd.to_datetime(batangas['VALIDDATASTART'])]
batangas = batangas[batangas['DATETIME'] < pd.to_datetime(batangas['VALIDDATAEND'])]
batangas = batangas[keepcols]

'Combine with geomorphic data and downsample logger data to every 30 minutes'
geom = gpd.read_file(geom_path+'Philippines.gpkg')
batangas = downsampler(agg_and_merge(batangas,geom))
del geom
'''
Marianas Islands
'''

'Open data from Marianas Islands in CSV format'
marianas = pd.read_csv(path + "marianas_temps/ESD_NCRMP_Temperature_2017_Marianas.csv")

'Reformat datetime column to match other NOAA data'
marianas['DATETIME'] = pd.to_datetime(marianas['TIMESTAMP'], format='%Y-%m-%d %H:%M:%S')

'Rename ID column'
marianas = marianas.rename(columns={'OCC_SITEID':'ID'})

'Trim data to valid timeframe only'
marianas = marianas[marianas['DATETIME'] > pd.to_datetime(marianas['VALIDDATASTART'])]
marianas = marianas[marianas['DATETIME'] < pd.to_datetime(marianas['VALIDDATAEND'])]
marianas = marianas[keepcols]

'Combine logger and geomorphic data and downsample logger data to every 30 minutes'
geom = gpd.read_file(geom_path+'Western-Micronesia.gpkg')
marianas = downsampler(agg_and_merge(marianas,geom))
del geom

'''
Wake Island
'''

'Import data from Wake Island in csv format'
wake = pd.read_csv(path+ "wake_island_temps/2.2/data/0-data/unzipped/ESD_NCRMP_Temperature_2014_PRIA.csv")
wake['DATETIME'] = pd.to_datetime(wake['TIMESTAMP'])

'Rename ID column'
wake = wake.rename(columns={'OCC_SITEID':'ID'})

'Trim data to valid timeframe only'
wake = wake[wake['DATETIME'] > pd.to_datetime(wake['VALIDDATASTART'])]
wake = wake[wake['DATETIME'] < pd.to_datetime(wake['VALIDDATAEND'])]
wake = wake[keepcols]

'Combine logger and geomorphology data and downsample logger data to every 30 minutes'
geom = gpd.read_file(geom_path+'Eastern-Micronesia_W.gpkg')
geom = geom.rename(columns={'rclass':'class'})
wake = downsampler(agg_and_merge(wake,geom))
del geom

'''
Pacific Remote Island Areas
'''

'Import other Pacific Remote Island Areas in csv format and rename ID columns'
path1 = path + "pria_temps/pria_temps/pria_temps_1/"
pria_1 = ([pd.read_csv(f) for f in glob.glob(path1+ "*.csv")])
pria_1 = pd.concat(pria_1,axis=0,ignore_index=True,join="outer")
pria_1 = pria_1.rename(columns={'SITETAG':'ID'})
path2 = path + "pria_temps/pria_temps/pria_temps_2/2.2/data/0-data/unzipped/split/"
pria_2 = ([pd.read_csv(f) for f in glob.glob(path2+ "*.csv")])
pria_2 = pd.concat(pria_2,axis=0,ignore_index=True,join="outer")
pria_2 = pria_2.rename(columns={'OCC_SITEID':'ID'})
pria = pd.concat([pria_1,pria_2], axis=0, ignore_index=True,join="outer")
del pria_1,pria_2
pria['DATETIME'] = pd.to_datetime(pria['TIMESTAMP'])

'Trim data to valid timeframe only'
pria = pria[pria['DATETIME'] > pd.to_datetime(pria['VALIDDATASTART'])]
pria = pria[pria['DATETIME'] < pd.to_datetime(pria['VALIDDATAEND'])]
pria = pria[keepcols]

'Combine logger points with geomorphology data from Eastern Micronesia, east of the IDL'
geom1 = pd.DataFrame(gpd.read_file(geom_path+'Eastern-Micronesia_E.gpkg'))
geom1 = geom1.rename(columns={'rclass':'class'})

'Combine logger points with geomorphology data from Eastern Micronesia, west of the IDL'
geom2 = pd.DataFrame(gpd.read_file(geom_path+'geomorphology/Eastern-Micronesia_W.gpkg'))
geom2 = geom2.rename(columns={'rclass':'class'})

'Combine logger points with geomorphology data from the Hawaiian Islands'
geom3 = pd.DataFrame(gpd.read_file(geom_path+'Hawaiian-Islands.gpkg'))

'Concatenate geomorphology data from Eastern Micronesia and Hawaii'
geoms = gpd.GeoDataFrame(pd.concat([geom1,geom2,geom3]))

'Combine logger and geomorphic data and '
pria = downsampler(agg_and_merge(pria,geoms))

'''
Timor Leste
'''
'Open data from Timor-Leste in csv format'
timor_leste = pd.read_csv(path+ "timor_leste_temps/1.1/data/0-data/STR_Temperature_Timor_2014.csv")

'Reformat datetime variable'
dummy = timor_leste['_DATE'].astype(str)+ "_" + timor_leste['HOUR'].astype(str).str.zfill(2)
timor_leste['DATETIME'] = pd.to_datetime(dummy,format='%m/%d/%Y_%H')

'Rename ID and depth columns'
timor_leste = timor_leste.rename(columns={'SITE_TAG':'ID'})
timor_leste = timor_leste.rename(columns={'DEPTH_M':'DEPTH'})

'Trim data to valid timeframe'
timor_leste = timor_leste[timor_leste['DATETIME'] > pd.to_datetime(timor_leste['DATA_START'])]
timor_leste = timor_leste[timor_leste['DATETIME'] < pd.to_datetime(timor_leste['DATA_END'])]
timor_leste = timor_leste[keepcols]
del dummy

'Combine logger and geomorphic data; downsample logger data to every 30 minutes'
geom = gpd.read_file(geom_path+'Southeast-Asian-Archipelago.gpkg')
timor_leste = downsampler(agg_and_merge(timor_leste,geom))

'Concatenate logger data from American Samoa, the Philippines, Marianas Islands, Wake Island, Timor Leste, and Pacific Remote Island Areas into NOAA_Pacific dataset'
noaa_pacific = pd.concat([amsam,batangas,marianas,wake,pria,timor_leste])

'''
Remove days with fewer than 24 points
'''
noaa_pacific['DAYPT'] =  noaa_pacific['DATETIME'].dt.date.astype('str') +'_' + noaa_pacific['ID'].astype('str')
dayagg= noaa_pacific.groupby(['DAYPT'])['DAYPT'].count()
daykeep = pd.DataFrame()
daykeep['DAYCOUNT'] = dayagg.values
daykeep['DAYPT'] =dayagg.index

noaa_pacific = pd.merge(noaa_pacific,daykeep,how='left',on='DAYPT')
noaa_pacific = noaa_pacific[noaa_pacific['DAYCOUNT'] > 23]
noaa_pacific = noaa_pacific.reset_index(drop=True)
noaa_pacific = noaa_pacific.drop(columns = ['DAYCOUNT','DAYPT'])

'''
Remove months with <15 days
'''
noaa_pacific['MONTHPT'] =  noaa_pacific['DATETIME'].dt.month.astype('str') + '_' + noaa_pacific['DATETIME'].dt.year.astype('str') +'_' + noaa_pacific['ID'].astype('str')
oneaday = downsampler2(noaa_pacific)
monthagg= oneaday.groupby(['MONTHPT'])['MONTHPT'].count()
monthkeep = pd.DataFrame()
monthkeep['MONTHCOUNT'] = monthagg.values
monthkeep['MONTHPT'] =monthagg.index

noaa_pacific = pd.merge(noaa_pacific,monthkeep,how='left',on='MONTHPT')
noaa_pacific = noaa_pacific[noaa_pacific['MONTHCOUNT'] > 15]
noaa_pacific = noaa_pacific.reset_index(drop=True)
noaa_pacific = noaa_pacific.drop(columns = ['MONTHCOUNT','MONTHPT'])

'''
Remove points with <12 months
'''
noaa_pacific['MONTH'] =noaa_pacific['DATETIME'].dt.month
d_agg = noaa_pacific.groupby(['ID','MONTH']).first().reset_index()

monthn= pd.DataFrame(d_agg['ID'].value_counts())
monthn['counts'] =monthn.values
monthn['ID'] =monthn.index
monthn = monthn.reset_index()

noaa_pacific = pd.merge(noaa_pacific,monthn,how='left',on='ID')
noaa_pacific = noaa_pacific[noaa_pacific['counts'] > 11]
noaa_pacific = noaa_pacific.reset_index(drop=True)

'''
Calculate mean daily diff
'''
nopac = noaa_pacific.copy()
nopac['dmin'] = nopac.groupby(['ID','DATE']).TEMP_C.transform('min')
nopac['dmax'] = nopac.groupby(['ID','DATE']).TEMP_C.transform('max')
nopac['DAILY_DIFF'] = nopac['dmax'] - nopac['dmin']
nopac = nopac.drop(columns=['dmin','dmax'])
a = {'DAILY_DIFF': ['mean']}
temps_a = nopac.groupby(['ID']).agg(a).reset_index() 

'''
Calculate MMMs
'''
pacific_mmms = prep_data(noaa_pacific)
pacific_mmms.columns = ['index','MONTH','ID','TEMP_C','LONGITUDE','LATITUDE','RCLASS','COORDS','DEPTH']
pacific_mmms = pacific_mmms.drop(columns=['index'])

'''
Calculate q99
'''
pacific_q99 = prep_q99(noaa_pacific)
pacific_q99.columns = ['index','MONTH', 'ID', 'Q99', 'DEPTH', 'LONGITUDE', 'LATITUDE', 'DATETIME', 'DATE']
pacific_q99 = pacific_q99.drop(columns=['index','MONTH','DEPTH','LONGITUDE','LATITUDE','DATETIME','DATE'])
pacific_q99 = pacific_q99.drop_duplicates()

'Combine MMM, peak temps, daily diffs'
pacific = prep_data_more(noaa_pacific,pacific_mmms)
pacific = pd.merge(pacific,temps_a,on='ID')
pacific = pd.merge(pacific,pacific_q99,on='ID')

'Rename columns'
pacific.columns = ['MONTH','ID','TEMP_C','LONGITUDE','LATITUDE','RCLASS','COORDS','DEPTH','START_TIME','END_TIME','DAILY_DIFF','Q99']

'Reorder columns'
pacific = pacific[['MONTH','ID','TEMP_C','LONGITUDE','LATITUDE','RCLASS','COORDS','START_TIME','END_TIME','Q99','DAILY_DIFF','DEPTH']]

'Export NOAA data for non-Hawaii Pacific islands in csv format'
outfile = open(out_path + 'noaa_pacific.csv','wb')
pacific.to_csv(outfile)
outfile.close()
del d_agg, monthn, noaa_pacific, temps_a

'''
NOAA logger data from Florida and the Caribbean
'''

''' 
Biscayne Bay
'''
'Import logger data from Biscayne Bay, Florida in csv format'
bpath = path + "biscayne_florida_temps/1.1/data/0-data/NCRMP_WaterTemperatureData_BiscayneNationalPark_2014-2017/NCRMP_STR_BNP/"
biscayne = pd.concat([pd.read_csv(f) for f in glob.glob(bpath+ "*.csv")], axis=0, ignore_index=True,join="outer")

'Reformat datetime variable'
biscayne['DATETIME'] = pd.to_datetime(biscayne['TIMESTAMP'],format='%m/%d/%Y %H:%M:%S')

'Rename ID column'
biscayne = biscayne.rename(columns={'SITETAG':'ID'})

'Trim data to valid timeframe only'
biscayne = biscayne[biscayne['DATETIME'] > pd.to_datetime(biscayne['VALIDDATASTART'])]
biscayne = biscayne[biscayne['DATETIME'] < pd.to_datetime(biscayne['VALIDDATAEND'])]
biscayne = biscayne[keepcols]

'Combine logger and geomorphic data and downsample logger data to every 30 minutes'
geom = gpd.read_file(geom_path+'Northern-Caribbean-Florida-Bahamas.gpkg')
biscayne = downsampler(agg_and_merge(biscayne,geom))

'''
Florida Keys
'''
'Import  datasets from the Florida Keys, all in csv format'
path1 = path + "florida_keys_temps/1.1/data/0-data/NCRMP_FloridaKeys_STR_Data_2013-12-03_to_2016-12-13/NCRMP_STR_FL_KEYS_LOW/"
keys_low =  pd.concat([pd.read_csv(f) for f in glob.glob(path1+ "*.csv")], axis=0, ignore_index=True,join="outer")
path2 = path + "florida_keys_temps/1.1/data/0-data/NCRMP_FloridaKeys_STR_Data_2013-12-03_to_2016-12-13/NCRMP_STR_FL_KEYS_MID/"
keys_mid =  pd.concat([pd.read_csv(f) for f in glob.glob(path2+ "*.csv")], axis=0, ignore_index=True,join="outer")
path3 = path + "florida_keys_temps/1.1/data/0-data/NCRMP_FloridaKeys_STR_Data_2013-12-03_to_2016-12-13/NCRMP_STR_FL_KEYS_UPP/"
keys_upp =  pd.concat([pd.read_csv(f) for f in glob.glob(path3+ "*.csv")], axis=0, ignore_index=True,join="outer")

'Concatenate all Florida Keys logger data'
florida_keys = pd.concat([keys_low,keys_mid,keys_upp],axis=0,ignore_index=True,join="outer")

'Reformat datetime variable'
dummy = florida_keys['TIMESTAMP'] + "_" + florida_keys['HOUR']
florida_keys['DATETIME'] = pd.to_datetime(dummy,format='%m/%d/%y_%H:%M:%S',errors='coerce')

'Rename ID column'
florida_keys = florida_keys.rename(columns={'SITETAG':'ID'})

'Trim data to valid timeframe only'
florida_keys = florida_keys[florida_keys['DATETIME'] > pd.to_datetime(florida_keys['VALIDDATASTART'])]
florida_keys = florida_keys[florida_keys['DATETIME'] < pd.to_datetime(florida_keys['VALIDDATAEND'])]
florida_keys = florida_keys[keepcols]

'Delete intermediates'
del keys_low,keys_mid,keys_upp,dummy

'Combine logger and geomorphic data and downsample logger data to every 30 minutes'
florida_keys = downsampler(agg_and_merge(florida_keys,geom))

'''
St Johns
'''

'Import logger data from St Johns in csv format'
path1 = path + "stthomas_stjohn_temps/1.1/data/0-data/NCRMP_WaterTemperatureData_STJOHN_STHOMAS_2014-2017/NCRMP_STR_STJOHN_EAST/"
stjohn_1 = pd.concat([pd.read_csv(f) for f in glob.glob(path1+ "*.csv")], axis=0, ignore_index=True,join="outer")
path2 = path + "stthomas_stjohn_temps/1.1/data/0-data/NCRMP_WaterTemperatureData_STJOHN_STHOMAS_2014-2017/NCRMP_STR_STJOHN_NORTH/"
stjohn_2 = pd.concat([pd.read_csv(f) for f in glob.glob(path2+ "*.csv")], axis=0, ignore_index=True,join="outer")
path3 = path + "stthomas_stjohn_temps/1.1/data/0-data/NCRMP_WaterTemperatureData_STJOHN_STHOMAS_2014-2017/NCRMP_STR_STJOHN_SOUTH/"
stjohn_3 = pd.concat([pd.read_csv(f) for f in glob.glob(path3+ "*.csv")], axis=0, ignore_index=True,join="outer")
path4 = path + "stthomas_stjohn_temps/1.1/data/0-data/NCRMP_WaterTemperatureData_STJOHN_STHOMAS_2014-2017/NCRMP_STR_STJOHN_WEST/"
stjohn_4 = pd.concat([pd.read_csv(f) for f in glob.glob(path4 + "*.csv")], axis=0, ignore_index=True,join="outer")

'Concatenate St Johns logger data'
stjohn = pd.concat([stjohn_1,stjohn_2,stjohn_3,stjohn_4],axis=0,ignore_index=True,join="outer")

'Reformat datetime variable'
stjohn['DATETIME'] = pd.to_datetime(stjohn['TIMESTAMP'], format='%m/%d/%Y %H:%M:%S')

'Rename ID column'
stjohn = stjohn.rename(columns={'SITETAG':'ID'})

'Trim data to valid timeframe only'
stjohn = stjohn[stjohn['DATETIME'] > pd.to_datetime(stjohn['VALIDDATASTART'])]
stjohn = stjohn[stjohn['DATETIME'] < pd.to_datetime(stjohn['VALIDDATAEND'])]
stjohn = stjohn[keepcols]
del stjohn_1,stjohn_2,stjohn_3,stjohn_4

'Combine logger and geomorphic data and downsample logger data to every 30 minutes'
geom = gpd.read_file(geom_path+'Southeastern-Caribbean.gpkg')
stjohn = downsampler(agg_and_merge(stjohn,geom))

'''
St Thomas
'''
'Import logger data from St Thomas in csv format'
path1 = path + "stthomas_stjohn_temps/1.1/data/0-data/NCRMP_WaterTemperatureData_STJOHN_STHOMAS_2014-2017/NCRMP_STR_STTHOMAS_NORTH/"
stthomas_1 = pd.concat([pd.read_csv(f) for f in glob.glob(path1+ "*.csv")], axis=0, ignore_index=True,join="outer")
path2 = path + "stthomas_stjohn_temps/1.1/data/0-data/NCRMP_WaterTemperatureData_STJOHN_STHOMAS_2014-2017/NCRMP_STR_STTHOMAS_SOUTH/"
stthomas_2 = pd.concat([pd.read_csv(f) for f in glob.glob(path2+ "*.csv")], axis=0, ignore_index=True,join="outer")
path3 = path + "stthomas_stjohn_temps/1.1/data/0-data/NCRMP_WaterTemperatureData_STJOHN_STHOMAS_2014-2017/NCRMP_STR_STTHOMAS_WEST/"
stthomas_3 = pd.concat([pd.read_csv(f) for f in glob.glob(path3+ "*.csv")], axis=0, ignore_index=True,join="outer")

'Concatenate at St Thomas datasets'
stthomas = pd.concat([stthomas_1,stthomas_2,stthomas_3],axis=0,ignore_index=True,join="outer")

'Reformate datetime variable'
stthomas['DATETIME'] = pd.to_datetime(stthomas['TIMESTAMP'], format='%m/%d/%Y %H:%M:%S')

'Rename ID column'
stthomas = stthomas.rename(columns={'SITETAG':'ID'})

'Trim data to valid timeframe only'
stthomas = stthomas[stthomas['DATETIME'] > pd.to_datetime(stthomas['VALIDDATASTART'])]
stthomas = stthomas[stthomas['DATETIME'] < pd.to_datetime(stthomas['VALIDDATAEND'])]
stthomas = stthomas[keepcols]
del stthomas_1,stthomas_2,stthomas_3

'Combine logger and geomorphic data and downsample logger data to every 30 minutes'
stthomas = downsampler(agg_and_merge(stthomas,geom))

'''
Puerto Rico
'''

'Import logger data from Puerto Rico'
path1 = path + "puerto_rico_temps/1.1/data/0-data/NCRMP_WaterTemperatureData_PuertoRico_2015-2017/NCRMP_STR_PR_EAST/"
pr_1 = pd.concat([pd.read_csv(f) for f in glob.glob(path1+ "*.csv")], axis=0, ignore_index=True,join="outer")
path2 = path + "puerto_rico_temps/1.1/data/0-data/NCRMP_WaterTemperatureData_PuertoRico_2015-2017/NCRMP_STR_PR_NORTH/"
pr_2 = pd.concat([pd.read_csv(f) for f in glob.glob(path2+ "*.csv")], axis=0, ignore_index=True,join="outer")
path3 = path + "puerto_rico_temps/1.1/data/0-data/NCRMP_WaterTemperatureData_PuertoRico_2015-2017/NCRMP_STR_PR_SOUTH/"
pr_3 = pd.concat([pd.read_csv(f) for f in glob.glob(path3+ "*.csv")], axis=0, ignore_index=True,join="outer")
path4 = path + "puerto_rico_temps/1.1/data/0-data/NCRMP_WaterTemperatureData_PuertoRico_2015-2017/NCRMP_STR_PR_WEST/"
pr_4 = pd.concat([pd.read_csv(f) for f in glob.glob(path4+ "*.csv")], axis=0, ignore_index=True,join="outer")
puerto_rico = pd.concat([pr_1,pr_2,pr_3,pr_4],axis=0,ignore_index=True,join="outer")

'Reformat datetime variable'
puerto_rico['DATETIME'] = pd.to_datetime(puerto_rico['TIMESTAMP'],errors='coerce')

'Rename ID variable'
puerto_rico = puerto_rico.rename(columns={'SITETAG':'ID'})

'Trim data to valid timeframe only'
puerto_rico = puerto_rico[puerto_rico['DATETIME'] > pd.to_datetime(puerto_rico['VALIDDATASTART'])]
puerto_rico = puerto_rico[puerto_rico['DATETIME'] < pd.to_datetime(puerto_rico['VALIDDATAEND'])]
puerto_rico = puerto_rico[keepcols]
del pr_1,pr_2,pr_3,pr_4

'Combine logger and geomorphic data and downsample logger data to every 30 minutes'
puerto_rico = downsampler(agg_and_merge(puerto_rico,geom))

'''
St Croix
'''

'Import logger data from St Croix in csv format'
path1 = path + "stcroix_temps/stcroix_temps_2/2.2/data/0-data/NCRMP_STR_Data_STX_2019/"
stcroix_1 = ([pd.read_csv(f) for f in glob.glob(path1+ "*.csv")])
stc_cols = ['SITETAG', 'LATITUDE', 'LONGITUDE', 'DEPTH', 'REGIONCODE',
       'LOCATIONCODE', 'INSTRUMENT', 'INSTRUMENTSN', 'DEPLOYCRUISE',
       'DEPLOYUTC', 'RETRIEVECRUISE', 'RETRIEVEUTC', 'VALIDDATASTART',
       'VALIDDATAEND', 'YEAR', 'MONTH', 'DAY', 'HOUR','TIMESTAMP','TEMP_C']
stcroix_1[6].columns  = stc_cols
stcroix_1[11].columns = stc_cols
stcroix_1[12].columns = stc_cols

stcroix_1[8].columns = ['SITETAG', 'LATITUDE', 'LONGITUDE', 'DEPTH', 'REGIONCODE',
       'LOCATIONCODE', 'INSTRUMENT', 'INSTRUMENTSN', 'DEPLOYCRUISE',
       'DEPLOYUTC', 'RETRIEVECRUISE', 'RETRIEVEUTC', 'VALIDDATASTART',
       'VALIDDATAEND', 'YEAR', 'MONTH', 'DAY', 'TIMESTAMP', 'HOUR', 'TEMP_C']

stcroix_1 = pd.concat(stcroix_1,axis=0,ignore_index=True,join="outer")

path2 = path + "stcroix_temps/stcroix_temps_1/1.1/data/0-data/NCRMP_STR_StCroix_WaterTemperatureData_2013-9-9_to_2016-9-14/NCRMP_STR_STCROIX_EAST/"
stcroix_2 = pd.concat([pd.read_csv(f) for f in glob.glob(path2+ "*.csv")], axis=0, ignore_index=True,join="outer")

path3 = path + "stcroix_temps/stcroix_temps_1/1.1/data/0-data/NCRMP_STR_StCroix_WaterTemperatureData_2013-9-9_to_2016-9-14/NCRMP_STR_STCROIX_NORTH/"
stcroix_3 = pd.concat([pd.read_csv(f) for f in glob.glob(path3+ "*.csv")], axis=0, ignore_index=True,join="outer")

path4 = path + "stcroix_temps/stcroix_temps_1/1.1/data/0-data/NCRMP_STR_StCroix_WaterTemperatureData_2013-9-9_to_2016-9-14/NCRMP_STR_STCROIX_SOUTH/"
stcroix_4 = pd.concat([pd.read_csv(f) for f in glob.glob(path4+ "*.csv")], axis=0, ignore_index=True,join="outer")

path5 = path + "stcroix_temps/stcroix_temps_1/1.1/data/0-data/NCRMP_STR_StCroix_WaterTemperatureData_2013-9-9_to_2016-9-14/NCRMP_STR_STCROIX_WEST/"
stcroix_5 = pd.concat([pd.read_csv(f) for f in glob.glob(path5+ "*.csv")], axis=0, ignore_index=True,join="outer")

'Concatenate St Croix logger datasets into one dataset'
stcroix = pd.concat([stcroix_1,stcroix_2,stcroix_3,stcroix_4,stcroix_5],axis=0,ignore_index=True,join="outer")

'Reformat datetime variable'
dummy = stcroix['TIMESTAMP'] + "_" + stcroix['HOUR']
stcroix['DATETIME'] = pd.to_datetime(dummy,format='%m/%d/%y_%H:%M:%S',errors='coerce')

'Rename ID variable'
stcroix = stcroix.rename(columns={'SITETAG':'ID'})

'Trim data to valid timeframe only'
stcroix = stcroix[stcroix['DATETIME'] > pd.to_datetime(stcroix['VALIDDATASTART'])]
stcroix = stcroix[stcroix['DATETIME'] < pd.to_datetime(stcroix['VALIDDATAEND'])]
stcroix = stcroix[keepcols]

'Delete intermediates'
del stcroix_1,stcroix_2,stcroix_3,stcroix_4,stcroix_5,dummy

'Combine logger and geomorphic data and downsample logger data to every 30 minutes'
stcroix = downsampler(agg_and_merge(stcroix,geom))

'Concatenate logger datasets from Florida, St John, St Thomas, Puerto Rico, and St Croix to make noaa_caribbean dataset'
noaa_caribbean = pd.concat([biscayne,florida_keys,stjohn,stthomas,puerto_rico,stcroix])

'''
Remove days with fewer than 24 points
'''
noaa_caribbean['DAYPT'] =  noaa_caribbean['DATETIME'].dt.date.astype('str') +'_' + noaa_caribbean['ID'].astype('str')
dayagg= noaa_caribbean.groupby(['DAYPT'])['DAYPT'].count()
daykeep = pd.DataFrame()
daykeep['DAYCOUNT'] = dayagg.values
daykeep['DAYPT'] =dayagg.index

noaa_caribbean = pd.merge(noaa_caribbean,daykeep,how='left',on='DAYPT')
noaa_caribbean = noaa_caribbean[noaa_caribbean['DAYCOUNT'] > 23]
noaa_caribbean = noaa_caribbean.reset_index(drop=True)
noaa_caribbean = noaa_caribbean.drop(columns = ['DAYCOUNT','DAYPT'])

'''
Remove months with <15 days
'''
noaa_caribbean['MONTHPT'] =  noaa_caribbean['DATETIME'].dt.month.astype('str') + '_' + noaa_caribbean['DATETIME'].dt.year.astype('str') +'_' + noaa_caribbean['ID'].astype('str')
oneaday = downsampler2(noaa_caribbean)
monthagg= oneaday.groupby(['MONTHPT'])['MONTHPT'].count()
monthkeep = pd.DataFrame()
monthkeep['MONTHCOUNT'] = monthagg.values
monthkeep['MONTHPT'] =monthagg.index

noaa_caribbean = pd.merge(noaa_caribbean,monthkeep,how='left',on='MONTHPT')
noaa_caribbean = noaa_caribbean[noaa_caribbean['MONTHCOUNT'] > 15]
noaa_caribbean = noaa_caribbean.reset_index(drop=True)
noaa_caribbean = noaa_caribbean.drop(columns = ['MONTHCOUNT','MONTHPT'])

'''
Remove points with <12 months
'''
noaa_caribbean['MONTH'] =noaa_caribbean['DATETIME'].dt.month
d_agg = noaa_caribbean.groupby(['ID','MONTH']).first().reset_index()

monthn= pd.DataFrame(d_agg['ID'].value_counts())
monthn['counts'] =monthn.values
monthn['ID'] =monthn.index
monthn = monthn.reset_index()

noaa_caribbean = pd.merge(noaa_caribbean,monthn,how='left',on='ID')
noaa_caribbean = noaa_caribbean[noaa_caribbean['counts'] > 11]
noaa_caribbean = noaa_caribbean.reset_index(drop=True)

'''
Calculate mean daily diff
'''
nocar = noaa_caribbean.copy()
nocar['dmin'] = nocar.groupby(['ID','DATE']).TEMP_C.transform('min')
nocar['dmax'] = nocar.groupby(['ID','DATE']).TEMP_C.transform('max')
nocar['DAILY_DIFF'] = nocar['dmax'] - nocar['dmin']
nocar = nocar.drop(columns=['dmin','dmax'])
a = {'DAILY_DIFF': ['mean']}
temps_a = nocar.groupby(['ID']).agg(a).reset_index() 

'''
Calculate MMMs
'''
caribbean_mmms = prep_data(noaa_caribbean)
caribbean_mmms.columns = ['index','MONTH','ID','TEMP_C','LONGITUDE','LATITUDE','RCLASS','COORDS','DEPTH']
caribbean_mmms = caribbean_mmms.drop(columns=['index'])

'''
Calculate q99
'''
carib_q99 = prep_q99(noaa_caribbean)
carib_q99.columns = ['index','MONTH', 'ID', 'Q99', 'DEPTH', 'LONGITUDE', 'LATITUDE', 'DATETIME', 'DATE']
carib_q99 = carib_q99.drop(columns=['index','MONTH','DEPTH','LONGITUDE','LATITUDE','DATETIME','DATE'])
carib_q99 = carib_q99.drop_duplicates()

'Combine MMM, peak temps, daily diffs'
noaa_caribbean = prep_data_more(noaa_caribbean,caribbean_mmms)
noaa_caribbean = pd.merge(noaa_caribbean,temps_a,on='ID')
noaa_caribbean = pd.merge(noaa_caribbean,carib_q99,on='ID')

'Rename columns'
noaa_caribbean.columns = ['MONTH','ID','TEMP_C','LONGITUDE','LATITUDE','RCLASS','COORDS','DEPTH','START_TIME','END_TIME','DAILY_DIFF','Q99']

'Reorder columns'
noaa_caribbean = noaa_caribbean[['MONTH','ID','TEMP_C','LONGITUDE','LATITUDE','RCLASS','COORDS','START_TIME','END_TIME','Q99','DAILY_DIFF','DEPTH']]

'Export data as csv'
outfile = open(out_path + 'noaa_caribbean.csv','wb')
noaa_caribbean.to_csv(outfile)
outfile.close()
del d_agg, monthn, noaa_caribbean