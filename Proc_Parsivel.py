#!/home/cpabla/anaconda3/envs/pysimba/bin/python
# coding: utf-8
import sys, os, glob
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from zipfile import ZipFile
import xarray as xr
import time

import Parsivel_Utilities as pu
import Plot_Disdrometer as pltdsd
import warnings
warnings.filterwarnings("ignore")

pd.set_option("display.max_columns", 1100)
np.set_printoptions(linewidth=160)

#############################################################################################
def proc_par():
#
# On web server...
# /d1/wallops-prf/Disdrometer/Parsivel/apu04/Plots/Rain/2021/WFF_apu04_2021_0212_rain.png
# /d1/wallops-prf/Disdrometer/Parsivel/apu04/Plots/DSD/2021/WFF_apu04_2021_0212_dsd.png  
# /d1/wallops-prf/Disdrometer/Parsivel/apu04/Text/2021/WFF_apu04_2021_0212.txt
#     site = 'WFF'
#     instrument = 'apu04'
#     year  = 2021 ; syear  = str(year).zfill(2)
#     month = 2    ; smonth = str(month).zfill(2)
#     day   = 12   ; sday   = str(day).zfill(2)

    beg_prog = time.time()
    #Out_Base_Dir = 'd1/wallops-prf/Disdrometer/Parsivel/' # Home
    #Out_Base_Dir = '/d1/wallops-prf/Field_Campaigns/IMPACTS/'
    Out_Base_Dir = '/home/cpabla/PyParsivel/'
    
    #In_Base_Dir = 'distro/apu/' # Home
    #In_Base_Dir = '/distro/apu/' # Web
    In_Base_Dir = '/gpm_raid/distro/apu/'
    
    if len(sys.argv) != 6:
        sys.exit("Usage: " + sys.argv[0] + " Site Inst Year Month Day")

    site  = sys.argv[1]
    inst  = sys.argv[2].lower()
    year  = int(sys.argv[3]) ; syear  = str(year).zfill(2)
    month = int(sys.argv[4]) ; smonth = str(month).zfill(2)
    day   = int(sys.argv[5]) ; sday   = str(day).zfill(2)
    
    # Read diameter and velocity bin values
    DVparms = pd.read_csv('Tables/parsivel_diameter_py.txt', sep='\s+',index_col=False)

    # Read Ali's D/V mask
    DF_Mask = pd.read_csv('Tables/parsivel_conditional_matrix.txt', header=None)
    
    # Locate the data
    in_dir = In_Base_Dir + inst + '/' + syear + smonth + '/'    

    wc = in_dir + inst + '_' + syear + smonth + sday + '??.zip'
    zfiles = sorted(glob.glob(wc))
    nf = len(zfiles)
    if(nf == 0):
        error_text = 'No files found!'
        pltdsd.plot_error(error_text, 'DSD', Out_Base_Dir, site, inst, syear, smonth, sday)
        pltdsd.plot_error(error_text, 'Rain', Out_Base_Dir, site, inst, syear, smonth, sday)
        flag = 'Stopped on Files not Found error!  ' + wc
        print(flag)
        print()
        sys.exit()

    # Unzip the files and return the name of the unzipped files
    print('Unzipping input files...')
    files, tmp_dir = pu.unzip_files(zfiles, inst, syear, smonth, sday)

    # Concatenate 10 minute hourly files into a single daily file
    print('Concatenating files...')
    in_file = pu.concatenate_files(files)
    #os.rmdir(tmp_dir)
    
    # Get xarray DataSet from Parsive input file
    print('Reading data and returning Xarray dataset...', end=" ")
    beg_time = time.time()
    DS, data2d = pu.get_dataset_from_parsivel(in_file, DVparms, Order='F')
    end_time = time.time()
    delt = end_time-beg_time
    print(f' It took {np.round(delt,1)} seconds to load file!')
    
    # Resample 10 s data to one minute data
    print('Resampling 10-second data to 1-minute data and summing...')
    DS_1min = DS.resample(time='1T').sum()
    DS_1min = DS_1min.fillna(0)
    del DS, data2d
    
    # Get integral parameters, PSD and Moments
    print('Getting Integral parameters, PSD and Moments from DataSet...')
    beg_time = time.time()
    Parms_DF, PSD_DF, Moments_DF = pu.get_integral_parameters(site, inst, DS_1min, DF_Mask, DVparms)
    end_time = time.time()
    delt = end_time-beg_time
    print(f' It took {np.round(delt,1)} seconds to retrieve DSD!')


    # Get integral parameters, PSD and Moments
    print('Getting Integral parameters, PSD and Moments from DataSet...')
    beg_time = time.time()
    Parms_DFx = pu.get_integral_parameters_xarray(site, inst, DS_1min, DF_Mask, DVparms)
    end_time = time.time()
    delt = end_time-beg_time
    print(f' It took {np.round(delt,1)} seconds to retrieve DSD via xarray!')

    
    # Do some plotting
    SMALL = 16; MEDIUM = 18; LARGE = 20
    pltdsd.set_plot_fontsizes(SMALL, MEDIUM, LARGE)
    
    # Generate and save plot with DSD, rain rate, reflectivity, LWC, Dm/Dmax, Total Drops and Concentration
    print('Generating rain image...')
    rain_png_file = pltdsd.plot_integral_parameters(Parms_DF, PSD_DF, Out_Base_Dir,
                                                    site, inst, syear, smonth, sday)
    
    # Create thumbnail of rain plot
    size_tuple = (200,400)
    rain_thumb_file = pltdsd.create_thumbnail(rain_png_file, size_tuple)

    # Generate standalone DSD plot
    print('Generating DSD image...')
    dsd_png_file = pltdsd.plot_dsd(Parms_DF, PSD_DF, Out_Base_Dir,
                                   site, inst, syear, smonth, sday)
    
    # Create thumbnail of DSD plot
    size_tuple = (400, 200)
    dsd_thumb_file = pltdsd.create_thumbnail(dsd_png_file, size_tuple)
    
    # Save Parms_DF, PSD_DF and Moments to CSV files
    print('Saving Parms, DSD and Moments to CSV files...')
    parms_file, psd_file, moms_file = pu.save_dataframes(Parms_DF, PSD_DF, Moments_DF, Out_Base_Dir,
                                                         site, inst, syear, smonth, sday)

    # Save the 1-minute DataSet that includes masking to a NetCDF file
    # print('Saving DS_1min DataSet to a netcdf file...')
    # ncdf_file = pu.save_netcdf(DS_1min, site, inst, syear, smonth, sday)
    print('Done.')
    end_prog = time.time()
    delt = end_prog-beg_prog
    print(f'It took {np.round(delt,1)} seconds to process {site}/{inst}')
    print()
    print()
    print()
    print()

#############################################################################################

if(__name__ == "__main__"):

    proc_par()
