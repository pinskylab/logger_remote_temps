"""
Script for processing in situ logger temperature data from
the Smithsonian Tropical Research Institute - Panama
Author: Jaelyn Bos
Data available at XXX/XX
2024
"""
import numpy as np
import pandas as pd
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

'Filepath to folder with datasets in .csv format'
path = '/projects/f_mlp195/bos_microclimate/loggers_testing/Panama_loggers/'

'Filepath to folder with geomorphic data from Allen Coral Atlas'
geom_path = '/projects/f_mlp195/bos_microclimate/geomorphology/'

'Filepath to folder to save processed data'
out_path = '/home/jtb188/'

'''
San Blas, Bastimentos, and Mangrove Inn data were excluded from this analysis for being entirely before the year 2000
Isla Solarte exluded for being entirely before 2002
Latitudes, longitudes, and depth metadata found here: https://biogeodb.stri.si.edu/physical_monitoring/research/sst
'''
bahia_lasminas = pd.read_csv(path + 'Bahia Las Minas_wt/Bahia Las Minas_03m_wt.csv')
bahia_lasminas['LATITUDE'] = 9.397833
bahia_lasminas['LONGITUDE'] = -79.82295
bahia_lasminas['DEPTH'] = 3

bocas = pd.read_csv(path+ 'BocasPlatform/bocas_tower_wt_elect.csv')
bocas['LATITUDE'] = 9.35078
bocas['LONGITUDE'] = -82.25768
bocas['DEPTH'] = 1

chiriqui_ji0 = pd.read_csv(path+ 'Gulf_Chiriqui_Jicarita_wt/SST_cqjh_wt_elect.csv')
chiriqui_ji0['LATITUDE'] = 7.213639
chiriqui_ji0['LONGITUDE'] = -81.799167
chiriqui_ji0['DEPTH'] = 18.3

chiriqui_ji1 = pd.read_csv(path+ 'Gulf_Chiriqui_Jicarita_wt/SST_cqkh_wt_elect.csv')
chiriqui_ji1['LATITUDE'] =  7.20347
chiriqui_ji1['LONGITUDE'] = -81.80064
chiriqui_ji1['DEPTH'] = 3

chiriqui_oc = pd.read_csv(path+ 'Gulf_Chiriqui_OuterCanal_wt/SST_cqah_wt_elect.csv')
chiriqui_oc['LATITUDE'] = 7.696333
chiriqui_oc['LONGITUDE'] = -81.616028
chiriqui_oc['DEPTH'] = 2.4

chiriqui_ra = pd.read_csv(path+ 'Gulf_Chiriqui_Rancheria_wt/SST_cqrh_wt_elect.csv')
chiriqui_ra['LATITUDE'] = 7.63525
chiriqui_ra['LONGITUDE'] =  -81.75878
chiriqui_oc['DEPTH'] = 1.2

chiriqui_rp = pd.read_csv(path+ 'Gulf_Chiriqui_RocaPropsera_wt/SST_cqph_wt_elect.csv')
chiriqui_rp['LATITUDE'] = 7.7763
chiriqui_rp['LONGITUDE'] = -81.758783
chiriqui_rp['DEPTH'] = 15.2

isla_coiba_rh = pd.read_csv(path + 'Isla_Coiba_RocaHacha_wt/SST_cqhh_wt_elect.csv')
isla_coiba_rh['LATITUDE'] = 7.431944
isla_coiba_rh['LONGITUDE'] = -81.858056
isla_coiba_rh['DEPTH'] = 18.3

isla_coiba_fr = pd.read_csv(path + 'Isla_Coiba_Frijoles_wt/SST_cqfh_wt_elect.csv')
isla_coiba_fr['LATITUDE'] = 7.649889
isla_coiba_fr['LONGITUDE'] = -81.719278
isla_coiba_fr['DEPTH'] = 18.3

isla_coiba_ca = pd.read_csv(path + 'Isla_Coiba_Catedral_wt/SST_cqch_wt_elect.csv')
isla_coiba_ca['LATITUDE'] = 7.226028
isla_coiba_ca['LONGITUDE'] = -81.829278
isla_coiba_ca['DEPTH'] = 18.3

isla_coiba_ne = pd.read_csv(path + 'Isla_Coiba Noreste_ANAM_wt/SST_cbnh_wt_elect.csv')
isla_coiba_ne['LATITUDE'] = 9.133611
isla_coiba_ne['LONGITUDE'] = -82.040278
isla_coiba_ne['DEPTH'] = 3

isla_coiba_id = pd.read_csv(path + 'Isla Coiba_Isla Damas_wt/SST_cbih_wt_elect.csv')
isla_coiba_id['LATITUDE'] = 7.421944
isla_coiba_id['LONGITUDE'] = -81.695556
isla_coiba_id['DEPTH'] = 4.6

isla_sanjose = pd.read_csv(path + 'Isla San Jose_wt/Isla San Jose, 04.5m_wt.csv')
isla_sanjose['LATITUDE'] = 8.256889
isla_sanjose['LONGITUDE'] = -79.133889
isla_sanjose['DEPTH'] = 4.6

isla_pp= pd.read_csv(path + 'Isla Pedro-Pablo_wt/SST_ppah_wt_elect.csv')
isla_pp['LATITUDE'] = 8.459528
isla_pp['LONGITUDE'] = -78.852278
isla_pp['DEPTH'] = 5.5

isla_parida= pd.read_csv(path + 'Isla Parida_wt/SST_pdah_wt_elect.csv')
isla_parida['LATITUDE'] = 8.131389
isla_parida['LONGITUDE'] = -82.320833
isla_parida['DEPTH'] = 2.4

naos_pier= pd.read_csv(path + 'Naos_Main Pier_wt/SST_naph_wt_elect.csv')
naos_pier['LATITUDE'] = 8.917494
naos_pier['LONGITUDE'] = -79.532842
naos_pier['DEPTH'] = 2.4

galeta_dn = pd.read_csv(path + 'Galeta_Downstream_wt.csv')
galeta_dn['LATITUDE'] = 9.40336
galeta_dn['LONGITUDE'] = -79.86036
galeta_dn['DEPTH'] = 0.5

galeta_up = pd.read_csv(path + 'Galeta_Upstream_wt.csv')
galeta_up['LATITUDE'] = 9.40253
galeta_up['LONGITUDE'] = -79.86064
galeta_up['DEPTH'] = 0.5

'Concatenate all datasets and standardize column names'
panama0 = pd.concat([bahia_lasminas,bocas,chiriqui_ji0,chiriqui_ji1,chiriqui_oc,chiriqui_ra,chiriqui_rp,isla_coiba_rh,isla_coiba_fr,isla_coiba_ca,isla_coiba_ne,isla_coiba_id,isla_sanjose,isla_pp,isla_parida,naos_pier,galeta_dn,galeta_up],keys=['bahia_lasminas','bocas','chiriqui_ji0','chiriqui_ji1','chiriqui_oc','chiriqui_ra','chiriqui_rp','isla_coiba_rh','isla_coiba_fr','isla_coiba_ca','isla_coiba_ne','isla_coiba_id','isla_sanjose','isla_pp','isla_parida','naos_tank','naos_tanks']).reset_index()
panama0['TEMP_C'] = panama0['wt'].apply(float)
panama0['ID'] = panama0['level_0']
panama0['DATETIME'] = pd.to_datetime(panama0['datetime'])

'Remove datapoints marked as missing in the notes column'
panama0 = panama0[panama0['chk_note'] !='missing']

'Drop unneeded columns'
panama0 = panama0.drop(columns=['level_0','level_1','datetime','date','wt','raw','chk_note','chk_fail','cid'])

'Add geomorphology from Pacific side of Panama and downsample temperature data at those points to every 30 minutes'
geom = gpd.read_file(geom_path+'Eastern-Tropical-Pacific.gpkg')
panama_pacific = downsampler(agg_and_merge(panama0,geom))

'Add geomorphology from Caribbean side of Panama and downsample temperature data at those points to every 30 minutes'
geom = gpd.read_file(geom_path+'geomorphology/Mesoamerica.gpkg')
panama_carib = downsampler(agg_and_merge(panama0,geom))

'Combine Pacific and Caribbean side data'
panama = pd.concat([panama_pacific,panama_carib])

'Remove data from before 2003'
panama = panama[panama['DATETIME'] > pd.to_datetime('2003-01-01 00:00:00')]
panama = panama.reset_index(drop=True)

'Remove data from after 2019 to align with SST data'

panama = panama[panama['DATETIME'] < pd.to_datetime('2020-01-01 00:00:00')]
panama = panama.reset_index(drop=True)

panama['MONTH'] =panama['DATETIME'].dt.month
d_agg = panama.groupby(['ID','MONTH']).first().reset_index()
'''
Remove days with <24 samples
'''
panama['DAYPT'] =  panama['DATETIME'].dt.date.astype('str') +'_' + panama['ID'].astype('str')
dayagg= panama.groupby(['DAYPT'])['DAYPT'].count()
daykeep = pd.DataFrame()
daykeep['DAYCOUNT'] = dayagg.values
daykeep['DAYPT'] =dayagg.index

panama = pd.merge(panama,daykeep,how='left',on='DAYPT')
panama = panama[panama['DAYCOUNT'] > 23]
panama = panama.reset_index(drop=True)
panama = panama.drop(columns = ['DAYCOUNT','DAYPT'])

'''
Remove months with <15 days
'''
panama['MONTHPT'] =  panama['DATETIME'].dt.month.astype('str') + '_' + panama['DATETIME'].dt.year.astype('str') +'_' + panama['ID'].astype('str')
oneaday = downsampler2(panama)
monthagg= oneaday.groupby(['MONTHPT'])['MONTHPT'].count()
monthkeep = pd.DataFrame()
monthkeep['MONTHCOUNT'] = monthagg.values
monthkeep['MONTHPT'] =monthagg.index

panama = pd.merge(panama,monthkeep,how='left',on='MONTHPT')
panama = panama[panama['MONTHCOUNT'] > 15]
panama = panama.reset_index(drop=True)
panama = panama.drop(columns = ['MONTHCOUNT','MONTHPT'])

'''
Remove points with <12 months
'''
monthn= pd.DataFrame(d_agg['ID'].value_counts())
monthn['counts'] =monthn.values
monthn['ID'] =monthn.index
monthn = monthn.reset_index()

panama = pd.merge(panama,monthn,how='left',on='ID')
panama = panama[panama['counts'] > 11]
panama = panama.reset_index(drop=True)

'''
Calculate mean daily diff 
'''
pan = panama.copy()
pan['dmin'] = pan.groupby(['ID','DATE']).TEMP_C.transform('min')
pan['dmax'] = pan.groupby(['ID','DATE']).TEMP_C.transform('max')
pan['DAILY_DIFF'] = pan['dmax'] - pan['dmin']
pan = pan.drop(columns=['dmin','dmax'])
a = {'DAILY_DIFF': ['mean']}
temps_a = pan.groupby(['ID']).agg(a).reset_index() 

'''
Calculate MMMs
'''
panama_mmms = prep_data(panama)
panama_mmms.columns = ['index','MONTH','ID','TEMP_C','LONGITUDE','LATITUDE','RCLASS','COORDS','DEPTH']
panama_mmms = panama_mmms.drop(columns=['index'])

'''
Calculate q99
'''
panama_q99 = prep_q99(panama)
panama_q99.columns = ['index','MONTH', 'ID', 'DEPTH', 'Q99','LONGITUDE', 'LATITUDE', 'DATETIME', 'DATE']
panama_q99 = panama_q99.drop(columns=['index','MONTH','DEPTH','LONGITUDE','LATITUDE','DATETIME','DATE'])
panama_q99 = panama_q99.drop_duplicates()

'Combine MMM, peak temps, and daily temperature range data'
panama = prep_data_more(panama,panama_mmms)
panama = pd.merge(panama,temps_a,on='ID')
panama = pd.merge(panama,panama_q99,on='ID')

'Rename columns'
panama.columns = ['MONTH','ID','TEMP_C','LONGITUDE','LATITUDE','RCLASS','COORDS','DEPTH','START_TIME','END_TIME','DAILY_DIFF','Q99']

'Reorder columns'
panama = panama[['MONTH','ID','TEMP_C','LONGITUDE','LATITUDE','RCLASS','COORDS','START_TIME','END_TIME','Q99','DAILY_DIFF','DEPTH']]

'Export data to CSV'
outfile = open(out_path+'panama.csv','wb')
panama.to_csv(outfile)
outfile.close()