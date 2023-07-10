import sys, os, glob
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import itertools
from datetime import datetime, timedelta
from zipfile import ZipFile
import xarray as xr
import time
import pdb

pd.set_option("display.max_columns", 1100)
np.set_printoptions(linewidth=160)

#############################################################################################
def save_netcdf(DS_1min, site, instrument, syear, smonth, sday):

    '''
    This function accepts a 1-minute Xarray DataSet containing Parsivel data and writes
    it to a NetCDF filee.  Note that the data has been masked using Ali's conditional
    matrix to mitigate unrealistic data in dropsize/velocity pairs.

    It returns the path/name of the written NetCDF file
    '''
    cdf_dir = Out_Base_Dir + instrument + '/NetCDF/' + syear + '/'
    os.makedirs(cdf_dir, exist_ok=True)

    cdf_file =  cdf_dir + site + '_' + instrument + '_' + syear + '_' + smonth + sday + '_1min.cdf'
    DS_1min.to_netcdf(cdf_file, mode='w', format='NETCDF4')
    print('    --> ' + cdf_file)
    return cdf_file

#############################################################################################
def save_dataframes(Parms_DF, PSD_DF, Moments_DF, Out_Base_Dir, site, instrument, syear, smonth, sday):

    # /d1/wallops-prf/Disdrometer/Parsivel/apu04/Text/2021/WFF_apu04_2021_0212.txt

    txt_dir = Out_Base_Dir + 'Text/' + instrument + '/' + syear + '/'
    os.makedirs(txt_dir, exist_ok=True)

    parms_file = txt_dir + site + '_' + instrument + '_' + syear + '_' + smonth + sday + '_parms.csv'
    Parms_DF.to_csv(parms_file)
    print('    --> ' + parms_file)

    psd_file = txt_dir + site + '_' + instrument + '_' + syear + '_' + smonth + sday + '_psd.csv'
    PSD_DF.to_csv(psd_file)
    print('    --> ' + psd_file)

    moms_file = txt_dir + site + '_' + instrument + '_' + syear + '_' + smonth + sday + '_moments.csv'
    Moments_DF.to_csv(moms_file)
    print('    --> ' + moms_file)

    return parms_file, psd_file, moms_file

#############################################################################################
def print_drop_matrix(data):
    np.set_printoptions(linewidth=160)
    x = data[:,:]
    for iv in range(32):
        line = x[iv,:].astype(int)
        print(line)
    return
#############################################################################################
def get_julday_from_datetime(dt):
    jday = dt.timetuple().tm_yday
    return jday

#############################################################################################
def get_julday_from_ymd(year, month, day):
    jday = datetime(year, month, day).timetuple().tm_yday
    return jday

#############################################################################################
def unzip_files(zfiles, instrument, syear, smonth, sday):
    tmp_dir = 'tmp/' + instrument + '/' + syear + smonth + sday + '/'
    os.makedirs(tmp_dir, exist_ok=True)
    files = []
    for i, zf in enumerate(zfiles):
        f = os.path.basename(zf)[:-4]+'.dat'
        #print(zf,' --> ',f)
        with ZipFile(zf, 'r') as zipObj:
            zipObj.extract(f, tmp_dir)
        files.append(tmp_dir + f)

    return files, tmp_dir

#############################################################################################
def concatenate_files(files):
    tmp_dir = 'tmp/tmp_dir_' + str(os.getpid()) + '/'
    os.makedirs(tmp_dir, exist_ok=True)

    fileb = os.path.basename(files[0])[:-4]
    in_file = tmp_dir + fileb + '.dat'

    with open(in_file, 'w') as f:
        for fname in files:
            print('Concat: ', fname)
            with open(fname, encoding="ISO-8859-1") as fs:
                for line in fs:
                    f.write(line)
    return in_file

#############################################################################################
def get_dataset_from_parsivel(file, DVparms, Order='C'):

    lines = []
    with open(file, 'r') as f:
        lines = f.readlines()

    dates = []
    nr = len(lines)
    nd = 32
    nv = 32
    data1d  = np.zeros([nd*nv, nr])
    data2d  = np.zeros([nd, nv, nr])
    datars  = np.zeros([nd, nv, nr])

    for ir in range(nr):
        x = lines[ir]
        dates.append(x[0:14])
        string_list = x[60:-2]
        x = np.array(string_list.split(','))

        if(len(x) == 1024):
            data1d[:,ir] = x
            x2 = np.reshape(x, (32, 32), order=Order)
            data2d[:, :, ir] = x2
        else:
            print('Bad record length: ', ir, len(x))

    diam = DVparms['Drop_bin'].values
    velo = DVparms['Measured_Vt'].values
    DT   = pd.to_datetime(dates)
    time = DT

    DS = xr.Dataset(
        data_vars=dict(
            drops=(["Diam", "Vel", "time"], data2d),
        ),
        attrs=dict(Description="Parsivel data."),
    )
    DS = DS.assign_coords(Diam=diam, Vel=velo, time=time)
    return DS, data2d

#############################################################################################
def get_integral_parameters(site, inst, DS_1min, DF_Mask, DVparms):

    Nsecs = 60
    missing = np.nan
    pi = np.pi
    missing = 0

    nd = 32
    nv = 32
    nrecs = len(DS_1min.time)

    # Load Ali's D/V Mask
    df_mask = DF_Mask.values[:,:-1]
    df1 = np.flip(df_mask, 1)
    df_mask = np.rot90(df1, 1)

    Total_Drops = np.zeros(nrecs).astype(np.int64)
    Conc        = np.zeros(nrecs).astype(np.float64)
    LWC         = np.zeros(nrecs).astype(np.float64)
    Z           = np.zeros(nrecs).astype(np.float64)
    dBZ         = np.zeros(nrecs).astype(np.float64)
    Rain        = np.zeros(nrecs).astype(np.float64)
    Dm          = np.zeros(nrecs).astype(np.float64)
    Accum       = np.zeros(nrecs).astype(np.float64)
    Dmax        = np.zeros(nrecs).astype(np.float64)
    Sigma_M     = np.zeros(nrecs).astype(np.float64)
    Moments     = np.zeros([nrecs, 8]).astype(np.float64)
    dsd         = np.zeros([nrecs, nd]).astype(np.float64)

    d_bin = DVparms['Drop_bin'].values
    v_bin = DVparms['Theoretical_Vt'].values
    #v_bin = DVparms['Measured_Vt'].values
    delta = DVparms['Delta-D'].values

    DT = DS_1min.time.values

    for ir in range(nrecs):
        Drops = DS_1min['drops'].isel(time=ir).values

        # Apply mask
        Drops[(df_mask == 0)] = 0

        Total_Drops[ir] = Drops.sum()
        #print('Total drops for ',DT[ir],'=', Total_Drops[ir])

        for idiam in range(nd):
            diam = d_bin[idiam]
            dt   = delta[idiam]

            for ivel in range(nv):
                print()
                vel  = v_bin[ivel]
                NumDrops = Drops[idiam, ivel]

                # Calculate the DSD
                cs2 = (180.*(30.-(diam/2.)))/100.
                bot2 = Nsecs * cs2 * vel * dt * 100
                dsd[ir,idiam] += (1.e6 * NumDrops)/bot2

                # Compute integral parameters
                cs  = (180.*(30.-(diam/2.)))
                bot = 60. * cs * vel
                vol = np.pi*(diam**3.)/6.

                if(NumDrops > 0): Dmax[ir] = diam
                Conc[ir] += NumDrops * 1.e6/bot
                LWC[ir]  += NumDrops * vol *1.e3/bot
                Z[ir]    += NumDrops * 1.e6 * diam**6. / bot
                #if Z[ir] >= 200.0: pdb.set_trace()
                Rain[ir] += NumDrops * vol * Nsecs / cs

                # Calculate eight first moments
                for im in range(8):
                    Moments[ir,im] += (1.e6 * NumDrops * diam**im)/bot

            # Calculate mass-weighted mean diameter and reflectivity
            Dm[ir] = Moments[ir,4]/Moments[ir,3]    # Ratio of 4th to 3rd moment
            if( (Dm[ir] < 0) | (Dm[ir] > 20) | np.isnan(Dm[ir]) ): Dm[ir] = missing

#           # Calculate sigma_m:  Sm^2 = sum((D - Dm)^2 N(D) D^3 dD) / sum(N(D) D^3 dD)
#           sig1 = np.double(0) ; sig2 = sig1
#           for idiam in range(nd):
#               diam = d_bin[idiam]
#               cs = 180. * (30. - diam/2.)
#               for ivel in range(nv):
#                   vel = v_bin[ivel]
#                   NumDrops = Drops[idiam, ivel]
#                   bot = 60. * cs * vel
#                   sig = (diam - Dm[ir])**2
#                   sig1 += sig * (diam**3 * NumDrops * 1.e6)/bot
#                   sig2 += (diam**3 * NumDrops * 1.e6)/bot
#
#           sig0 = (sig1/(sig2 * Dm[ir]**2))**0.5
#           sig0 *= Dm[ir]
#           Sigma_M[ir] = sig1/sig2
#           if( (Sigma_M[ir] < 0) | (Sigma_M[ir] > 20) ):   Sigma_M[ir] = missing
    # Place Parms into a DataFrame
    Parms = {'DateTime': DT, 'Total Drops': Total_Drops, 'Concentration': Conc,
             'LWC': LWC, 'Z': Z, 'dBZ': 10*np.log10(Z), 'Rain': Rain, 'Dm': Dm,
             'Dmax': Dmax, 'Sigma_M': Sigma_M}
    cols = ['Total Drops', 'Concentration', 'LWC', 'Z', 'dBZ', 'Rain', 'Dm', 'Dmax', 'Sigma_M']
    Parms_DF = pd.DataFrame(data=Parms, index=DT, columns=cols)
    #pdb.set_trace()
    # Place PSD dictionary into a Pandas DataFrame
    PSD = {'DateTime': DT, 'DropSize': d_bin, 'DSD': dsd}
    PSD_DF = pd.DataFrame(data=PSD['DSD'], index=Parms['DateTime'], columns=PSD['DropSize'])

    # Place Moments into a DataFrame
    moms = ['M0', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7']
    Moments = {'DateTime': DT, 'Moments': Moments, 'Mom#': moms}
    Moments_DF = pd.DataFrame(data=Moments['Moments'], index=Moments['DateTime'], columns=Moments['Mom#'])


    return Parms_DF, PSD_DF, Moments_DF

#############################################################################################
def get_integral_parameters_xarray(site, inst, DS_1min, DF_Mask, DVparms):

    nd = 32
    nv = 32
    nrecs = len(DS_1min.time.values)

    DT = DS_1min.time.values

    Total_Drops = np.zeros(nrecs).astype(np.int64)
    Conc        = np.zeros(nrecs).astype(np.float64)
    LWC         = np.zeros(nrecs).astype(np.float64)
    Z           = np.zeros(nrecs).astype(np.float64)
    dBZ         = np.zeros(nrecs).astype(np.float64)
    Rain        = np.zeros(nrecs).astype(np.float64)
    Dm          = np.zeros(nrecs).astype(np.float64)
    Accum       = np.zeros(nrecs).astype(np.float64)
    Dmax        = np.zeros(nrecs).astype(np.float64)
    Sigma_M     = np.zeros(nrecs).astype(np.float64)
    Moments     = np.zeros([nrecs, 8]).astype(np.float64)
    dsd         = np.zeros([nrecs, nd]).astype(np.float64)

    drop_bins = DVparms['Drop_bin'].values
    vel_bins = DVparms['Theoretical_Vt'].values

    for ir in range(nrecs):

        Drops = DS_1min['drops'].isel(time=ir).values
        Drops = np.rot90(Drops)

        NumDrops = DS_1min['drops'].isel(time=ir).values.sum()
        Total_Drops[ir] = NumDrops

        Nsecs = 60
        cs  = (180.*(30.-(drop_bins/2.)))
        bot = 60. * cs * vel_bins
        vol = np.pi*(drop_bins**3.)/6.

        Rain[ir] = np.sum((Drops * vol * Nsecs)/cs)
        Conc[ir] = np.sum((Drops * 1.e6)/bot)
        LWC[ir]  = np.sum((Drops * vol *1.e3)/bot)
        Z[ir]    = np.sum((Drops * 1.e6 * drop_bins**6)/bot)
        dBZ[ir] = 10*np.log10(Z[ir])

        # print(ir, Total_Drops[ir], np.round(Conc[ir], 5), np.round(LWC[ir], 5),
        #       np.round(Z[ir], 5), np.round(dBZ[ir], 5), np.round(Rain[ir], 5))


    # Place Parms into a DataFrame
    Parms = {'DateTime': DT, 'Total Drops': Total_Drops, 'Concentration': Conc,
             'LWC': LWC, 'Z': Z, 'dBZ': 10*np.log10(Z), 'Rain': Rain, 'Dm': Dm,
             'Dmax': Dmax, 'Sigma_M': Sigma_M}
    cols = ['Total Drops', 'Concentration', 'LWC', 'Z', 'dBZ', 'Rain', 'Dm', 'Dmax', 'Sigma_M']
    Parms_DF = pd.DataFrame(data=Parms, index=DT, columns=cols)

    return Parms_DF

