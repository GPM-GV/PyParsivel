#!/home/cpabla/anaconda3/envs/pysimba/bin/python
# coding: utf-8
import sys, glob, os
import subprocess
from datetime import date
from datetime import timedelta

import warnings
warnings.filterwarnings("ignore")

def get_ymd_from_command_line(argv):
    nargs = len(argv)

    if(nargs == 2):
        theDate = date.today()

        if(argv[1] == 'TODAY'):
            year  = theDate.year
            month = theDate.month
            day   = theDate.day

        if(argv[1] == 'YESTERDAY'):
            theDate = date.today() - timedelta(days=1)
            year  = theDate.year
            month = theDate.month
            day   = theDate.day

    if(nargs == 4):
        year  = int(argv[1])
        month = int(argv[2])
        day   = int(argv[3])

    if( (nargs != 2) & (nargs !=4)):
        sys.exit('Must provide TODAY, YESTERDAY or year, month, day')

    return year, month, day
# ****************************************************************************************
if __name__ == "__main__":

    site = 'WFF'
    prog = 'Proc_Parsivel.py'
    # Parse command line to get date for execution
    year, month, day = get_ymd_from_command_line(sys.argv)
    syear  = str(year).zfill(4)
    smonth = str(month).zfill(2)
    sday   = str(day).zfill(2)
    sdate = smonth + '/' + sday + '/' + syear
    
    #insts = ['apu01', 'apu04', 'apu11', 'apu15',
    #         'apu17', 'apu18']
    #insts = ['apu02']
    insts = ['apu02', 'apu07', 'apu16', 'apu21']


    for i,inst in enumerate(insts):
        c = prog + ' ' + site + ' ' + inst + ' ' + syear + ' ' + smonth + ' ' + sday
        clist = c.split()
        print(c)
        subprocess.run(clist)
