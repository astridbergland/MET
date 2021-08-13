from lxml import etree as et
import pandas as pd 
import xarray as xr
from datetime import datetime, timedelta
import glob
import numpy as np
import os 
import yaml

def parse_cfg(cfgfile):
    # Read config file
    print("Reading", cfgfile)
    with open(cfgfile, 'r') as ymlfile:
        cfgstr = yaml.full_load(ymlfile)

    return cfgstr
cfgstr = parse_cfg('config.cfg')
cfg_input = cfgstr['input']
indir = cfg_input['indir']
cfg_output = cfgstr['output']
destdir = cfg_output['destdir']
cfg_inv = cfgstr['investigator']

startdate = cfg_input['startdate']
enddate = cfg_input['enddate']
st_date = datetime.strptime(startdate, "%Y-%m-%d")
end_date = datetime.strptime(enddate, "%Y-%m-%d")
datasetstart4filename = str(st_date.year) + str(st_date.month) + str(st_date.day)
datasetend4filename = str(end_date.year) + str(end_date.month) + str(end_date.day)
dates = [st_date + timedelta(days = x) for x in range((end_date-st_date).days +1)]
date_str = [(str(dates[x].year)+str(dates[x].month)+str(dates[x].day)) for x in range(len(dates))]

files = [] 
for i in range(len(date_str)):
    file = glob.glob(str(indir) + date_str[i] + '*.xml')
    files += file

def get_keywords(name):
    keywords_dict = {
    'CTD' : 'EARTH SCIENCE > OCEANS > OCEAN CIRCULATION > BUOY POSITION, \n' + \
            'EARTH SCIENCE > OCEANS > SALINITY/DENSITY > CONDUCTIVITY, \n' + \
            'EARTH SCIENCE > OCEANS > SALINITY/DENSITY > DENSITY, \n' + \
            'EARTH SCIENCE > OCEANS > SALINITY/DENSITY > SALINITY, \n' + \
            'EARTH SCIENCE > OCEANS > OCEAN TEMPERATURE > WATER TEMPERATURE', 
    'Wave' : 'EARTH SCIENCE > OCEANS > OCEAN CIRCULATION > BUOY POSITION, \n' + \
            'EARTH SCIENCE > OCEANS > OCEAN WAVES > SIGNIFICANT WAVE HEIGHT,\n' + \
            'EARTH SCIENCE > OCEANS > OCEAN WAVES > WAVE HEIGHT, \n' + \
            'EARTH SCIENCE > OCEANS > OCEAN WAVES > WAVE LENGTH, \n' + \
            'EARTH SCIENCE > OCEANS > OCEAN WAVES > WAVE SPECTRA, \n' + \
            'EARTH SCIENCE > OCEANS > OCEAN WAVES > WAVE SPEED/DIRECTION, \n' + \
            'EARTH SCIENCE > OCEANS > OCEAN WAVES > WAVE PERIOD, \n' + \
            'EARTH SCIENCE > OCEANS > OCEAN WAVES > SWELLS, \n' + \
            'EARTH SCIENCE > OCEANS > OCEAN WINDS > SURFACE WINDS',
    'Spectrum' : 'EARTH SCIENCE > OCEANS > OCEAN CIRCULATION > BUOY POSITION', 
    'Weather' : 'EARTH SCIENCE > ATMOSPHERE > ATMOSPHERIC WATER VAPOR > WATER VAPOR INDICATORS > HUMIDITY > RELATIVE HUMIDITY, \n' + \
        'EARTH SCIENCE > ATMOSPHERE > ATMOSPHERIC WATER VAPOR > WATER VAPOR INDICATORS > DEW POINT TEMPERATURE, \n' + \
        'EARTH SCIENCE > ATMOSPHERE > ATMOSPHERIC WINDS > SURFACE WINDS > WIND DIRECTION, \n' + \
        'EARTH SCIENCE > ATMOSPHERE > ATMOSPHERIC WINDS > SURFACE WINDS > WIND SPEED, \n' + \
        'EARTH SCIENCE > ATMOSPHERE > ATMOSPHERIC TEMPERATURE > SURFACE TEMPERATURE > AIR TEMPERATURE,\n' + \
        'EARTH SCIENCE > ATMOSPHERE > ATMOSPHERIC PRESSURE > SURFACE PRESSURE'   
    }
    return keywords_dict.get(name)

def file_dict():
    """
    A dictionary to place the parameters in the NetCFDF-file
    """
    variables_dict = {
    'Wave': ['significant_wave_height_hm0', 'wave_peak_direction','wave_peak_direction_swell', 'wave_mean_direction',
       'wave_peak_direction_wind', 'wave_mean_period_tm02', 'wave_peak_period',
       'wave_peak_period_swell', 'wave_peak_period_wind',
       'wave_height_swell_hm0', 'wave_height_wind_hm0', 'wave_height_hmax',
       'wave_height_crest', 'wave_height_trough', 'wave_period_tmax',
       'wave_period_tz', 'significant_wave_height', 'mean_spreading_angle',
       'first_order_spread', 'long_crestedness_parameters', 'heading', 'pitch',
       'roll', 'stdev_heading', 'stdev_pitch', 'stdev_roll',
       'input_voltage_motus_wave_sensor', 'input_current',
       'memory_used_motus_wave_sensor'],
    'CTD': ['conductivity', 'temperature','salinity', 'density', 'soundspeed'],
    'Weather': ['average_corrected_direction','average_corrected_wind_speed', 'compass_corrected_gust_direction',
       'corrected_gust_speed', 'pressure', 'relative_humidity', 'dewpoint','compass_heading'],
    'SystemParameters': ['supply_voltage',
       'panel1_voltage', 'panel1_current', 'panel2_voltage', 'panel2_current',
       'panel3_voltage', 'panel3_current', 'panel4_voltage', 'panel4_current',
       'battery_voltage', 'boost_regulator_voltage', 'boost_regulator_current',
       'port1_current', 'port2_current', 'port3_current', 'port4_current',
       'port5_current', 'input_voltage_system_parameters', 'input_voltage_avg',
       'input_voltage_min', 'input_voltage_max', 'cpu_core_active',
       'memory_used_system_parameters', 'internal_temperature'],
    'Spectrum': ['energy_spectrum', 'directional_spectrum',
       'principal_directional_spectrum', 'orbital_ratio_spectrum',
       'fourier_coeff_a1', 'fourier_coeff_a2', 'fourier_coeff_b1',
       'fourier_coeff_b2']
        }
    return variables_dict

def units_dict(chosen_parameter):
    """
    A dictionary to assign units to the variables
    """
    units = {
        'significant_wave_height_hm0':'m', 'wave_peak_direction':'Deg.M','wave_peak_direction_swell':'Deg.M',
        'wave_mean_direction':'Deg.M','wave_peak_direction_wind':'Deg.M', 'wave_mean_period_tm02':'s', 
        'wave_peak_period':'s','wave_peak_period_swell':'s', 'wave_peak_period_wind':'s',
        'wave_height_swell_hm0':'m', 'wave_height_wind_hm0':'m', 'wave_height_hmax':'m', 
        'long_crestedness_parameters':'dimensionless',
        'wave_height_crest':'m', 'wave_height_trough':'m', 'wave_period_tmax':'s','wave_period_tz':'s',
        'significant_wave_height':'m', 'mean_spreading_angle':'Deg.M','first_order_spread':'Deg.M', 
        'heading':'Deg.M', 'pitch':'Deg','roll':'Deg', 'stdev_heading':'Deg.M', 
        'stdev_pitch':'Deg', 'stdev_roll':'Deg','input_voltage_motus_wave_sensor':'V', 
        'input_current':'mA','memory_used_motus_wave_sensor':'Bytes', 'conductivity':'mS/cm', 
        'temperature':'DegC','salinity':'PSU', 'density':'kg/m^3', 'soundspeed':'m/s', 
        'average_corrected_direction':'Deg','average_corrected_wind_speed':'m/s', 
        'compass_corrected_gust_direction':'Deg','corrected_gust_speed':'m/s', 'pressure':'hPa',
        'relative_humidity':'percent', 'dewpoint':'DegC','compass_heading':'Deg', 'latitude':'Deg_N', 
        'longitude':'Deg_E', 'supply_voltage':'V','panel1_voltage':'mV', 'panel1_current':'mV', 
        'panel2_voltage':'mA', 'panel2_current':'mA','panel3_voltage':'mV', 'panel3_current':'mA', 
        'panel4_voltage':'mV', 'panel4_current':'mA','battery_voltage':'mV', 'boost_regulator_voltage':'mV', 
        'boost_regulator_current':'mV','port1_current':'mA', 'port2_current':'mA', 'port3_current':'mA', 
        'port4_current':'mA','port5_current':'mA', 'input_voltage_system_parameters':'V', 'input_voltage_avg':'V',
        'input_voltage_min':'V', 'input_voltage_max':'V', 'cpu_core_active':'percent',
        'memory_used_system_parameters':'Bytes', 'internal_temperature':'Deg.C',
        'orbital_ratio_spectrum':'dimensionless', 'fourier_coeff_a1':'dimensionless', 
        'fourier_coeff_a2':'dimensionless','fourier_coeff_b1':'dimensionless', 
        'fourier_coeff_b2':'dimensionless',
        'energy_spectrum':'m^2/Hz', 'directional_spectrum':'Deg.M','principal_directional_spectrum' :'Deg.M', 
        }
    return units.get(chosen_parameter)

def dump_to_csv(files):
    """
    Dumps the pandas dataframe to a temporary csv file to avoid loading it 
    into memory.
    """
    first = True 
    for file in files:
        df = xml_to_df(file)
        if first: 
            df.to_csv('xml_dump.csv',header=True,mode='w')
            first = False
        else:
            df.to_csv('xml_dump.csv',header = False, mode='a')
    return 

def xml_to_df(file): 
    """
    Function to parse the XML file and convert it to a Pandas DataFrame
    """
    tree = et.parse(file)
    root = tree.getroot()
    namespace = '{' + str(tree.xpath('namespace-uri(.)')) + '}'
    new_dictionary = {}
    spectrum_name = []
    file_time = (root.find(namespace + 'Time')).text
    time_pd = (pd.to_datetime(file_time, utc = True) -pd.Timestamp("1970-01-01", tz='UTC')) // pd.Timedelta('1s')
    for child in root.iter(namespace+'Point'):
        var_name = str(child.get('Descr'))
        if var_name == 'Input Voltage':
            if str(child.get('ID')) == '0':
                var_name = var_name + ' System Parameters'
            else:
                var_name = var_name + ' Motus Wave Sensor'
        if var_name == 'Memory Used':
            if str(child.get('ID')) == '5':
                var_name = var_name + ' System Parameters'
            else: 
                var_name = var_name + ' Motus Wave Sensor'
        for i in child.iter('{}Value'.format(namespace)):
            new_dictionary[var_name] = [float(i.text)]
    for child in root.iter(namespace+'Spectrum'):
        var_name = child.get('Descr')
        spectrum_name.append(var_name)
        for i in child.iter('{}Value'.format(namespace)):
            new_dictionary[var_name] = i.text.replace(';',',') 
    df = pd.DataFrame.from_dict(new_dictionary)
    df['time'] = time_pd
    df = df.set_index('time')
    df.columns = df.columns.str.lower()
    df.columns = df.columns.str.replace(' ', '_')
    df = df.rename(columns={'significant_wave_height_h1/3':'significant_wave_height'})
    
    return df


dump_to_csv(files)

def make_ds(filename):
    """
    Converts the DataFrame to an xarray dataset 
    """
    df = pd.read_csv(filename)
    df = df.set_index('time')
    lon = df['gps_longitude'].to_xarray()
    lat = df['gps_latitude'].to_xarray()
    variables_dict = file_dict()

    for i,j in variables_dict.items(): 
        if i == 'Spectrum':
            df = df[j]
            spec_list = []
            for k in j:
                spec = df[k].str.split(',', expand=True).astype(float)
                time = spec.index
                coeffs = spec.columns # This acts as an dimension to make it possible to access the data
                da = xr.DataArray(spec, name=k, dims = ['time','coeffs'])
                spec_list.append(da)
            ds = xr.merge(spec_list)
            ds['coeffs'].attrs['long_name'] = 'the number corresponding to the array index for the spectrum data'   
        else:
            ds = xr.Dataset.from_dataframe(df[j])
        
        for n in j:
            ds[n].attrs['units'] = units_dict(n)
            
        
        ds = ds.assign_coords({'lat': lat.values, 'lon': lon.values})
    
        ds.time.attrs['standard_name'] = 'time'
        ds.time.attrs['units'] = 'seconds since 1970-01-01 00:00:00+0'
       
        lat_min = float(ds['lat'].min())
        lat_max = float(ds['lat'].max())
        lon_min = float(ds['lon'].min())
        lon_max = float(ds['lon'].max())
        
        
        ds.lon.attrs['standard_name'] = 'longitude'
        ds.lon.attrs['units'] = 'degrees'
        ds.lat.attrs['standard_name'] = 'latitude'
        ds.lat.attrs['units'] = 'degrees'
        
        ds.time.attrs['standard_name'] = 'time'
        ds.time.attrs['units'] = 'seconds since 1970-01-01 00:00:00+0'
        
        ds.attrs['featureType'] = 'timeSeries'
        #ds.attrs['title'] = ''  # Not quite sure what these should be in this case
        #ds.attrs['license'] = ''
        #ds.attrs['summary'] = ''
        
        ds.attrs['time_coverage_start'] = startdate
        ds.attrs['time_coverage_end'] = enddate
        ds.attrs['geospatial_lat_min'] = lat_min 
        ds.attrs['geospatial_lat_max'] = lat_max
        ds.attrs['geospatial_lon_min'] = lon_min
        ds.attrs['geospatial_lon_max'] = lon_max

        ds.attrs['creator_name'] = cfg_inv['name']
        ds.attrs['creator_email'] = cfg_inv['email']
        ds.attrs['creator_url'] = cfg_inv['url']
        ds.attrs['creator_institution'] = cfg_inv['organisation']
        ds.attrs['project'] = cfg_inv['project']
        if i == 'SystemParameters':
            continue
        else:
            ds.attrs['keywords'] = get_keywords(i)
        ds.attrs['keywords_vocabulary'] = 'GCMD'
        ds.attrs['Conventions'] = 'ACDD, CF-1.8'
        ds.attrs['featureType'] = 'timeSeries'
        

        # Dumps the file to netcdf
        # Maybe add the location to the filename as well
        outputfile = destdir+str(i)+'_'+datasetstart4filename+'-'+datasetend4filename+'.nc'
    
        ds.to_netcdf(outputfile, encoding={'time': {'dtype': 'int32'}})
        
    # Removes the temporary csv file
    if os.path.exists('xml_dump.csv'):
        os.remove('xml_dump.csv')
    else:
       print('The file does not exist')
    return ds

make_ds('xml_dump.csv')
