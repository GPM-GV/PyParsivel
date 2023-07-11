import os, sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.colors as colors
from matplotlib.ticker import LogFormatterExponent
from matplotlib.ticker import LogFormatterSciNotation
import Parsivel_Utilities as pu

#############################################################################################
def set_plot_fontsizes(SMALL, MEDIUM, LARGE):
    plt.rc('font',   size=SMALL)       # Default text
    plt.rc('axes',   titlesize=SMALL)  # Axes title
    plt.rc('axes',   labelsize=MEDIUM) # x and y labels
    plt.rc('xtick',  labelsize=SMALL)  # Tick labels
    plt.rc('ytick',  labelsize=SMALL)  # Tick labels
    plt.rc('legend', fontsize =SMALL)  # Legend
    plt.rc('figure', titlesize=LARGE)  # Figure title
    return

#############################################################################################
def plot_error(error_text, plot_type, Out_Base_Dir, site, inst, syear, smonth, sday,
               savefig=True):
    """
        Code to plot a blank image with an error_text annotation
        plot_type = 'Rain' or 'DSD'
    """
    fig = plt.figure(figsize=(10,5), facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot([0,0], [0,0])
    ax.set_xlim((0,1))
    ax.set_xticks([])
    ax.set_ylim((0,1))
    ax.set_yticks([])
    ax.annotate(error_text, (0.5, 0.5), horizontalalignment='center', color='red', fontsize=24)

    # Save the plot
    if( (plot_type != 'Rain') & (plot_type != 'DSD')):
        return

    title = site + '/' + inst + ' ' + smonth + '/' + sday + '/' + syear
    plt.suptitle(title, fontsize=30)

    if(savefig):
        png_dir = Out_Base_Dir + '/Plots/' + inst + '/' + plot_type + '/' + syear + '/'
        os.makedirs(png_dir, exist_ok=True)

        sdate = syear + '_' + smonth + sday
        png_file = png_dir + site + '_' + inst + '_' + sdate + '_' + plot_type.lower() + '.png'
        plt.savefig(png_file)
        print('    --> ERROR: ' + png_file)
        plt.close()
    else:
        plt.show()

    return

#############################################################################################
def create_thumbnail(png_file, size_tuple):
    thumb_file = png_file[:-4] + '_thumb.png'
    try:
        image = Image.open(png_file)
        image.thumbnail(size_tuple)
        image.save(thumb_file)
    except IOError:
        pass
    return thumb_file
#############################################################################################
def plot_dsd(Parms_DF, PSD_DF, Out_Base_Dir, site, instrument, syear, smonth, sday,
             savefig=True):

    color = 'black'
    levels = np.logspace(-4, 4, base=10, num=17)
    cb_levels = np.logspace(-4, 4, base=10, num=9)

    fig = plt.figure(figsize=(10, 8), facecolor='white')

    ax0 = fig.add_subplot(111)
    cf = ax0.contourf(PSD_DF.index.values, PSD_DF.columns.values, PSD_DF.values.T,
                     cmap='jet', levels=levels, norm=colors.LogNorm())

    ax0.set_ylabel('Drop Diameter [mm]')
    ax0.set_ylim((0, 10))

    cb = plt.colorbar(cf, ax=ax0, location='bottom', fraction=0.125, ticks=cb_levels,
                      label='Drops per ($m^{3}$/mm)', pad=0.175, aspect=40)

    cb.formatter = LogFormatterSciNotation(base=10)
    cb.update_ticks()

    plt.xticks(rotation=45)
    plt.grid(True)

    title = site + '/' + instrument + ' ' + smonth + '/' + sday + '/' + syear
    plt.suptitle(title, fontsize=30, y=0.93)
    plt.tight_layout()

    dsd_dir = Out_Base_Dir + '/Plots/' + instrument + '/DSD/' + syear + '/'
    dsd_file = dsd_dir + site + '_' + instrument + '_' + syear + '_' + smonth + sday + '_dsd.png'
    if(savefig):
        print('PNG_DIR: ', png_dir)
        os.makedirs(png_dir, exist_ok=True)
        plt.savefig(dsd_file, bbox_inches='tight')
        print('    --> ' + dsd_file)
        plt.close()
    else:
        plt.show()

    return dsd_file
#############################################################################################
def plot_integral_parameters(Parms_DF, PSD_DF, Out_Base_Dir,
                             site, instrument, syear, smonth, sday,
                             figsize=(10,20), savefig=True):
    color = 'black'
    fig = plt.figure(figsize=figsize, facecolor='white')

    ax1 = fig.add_subplot(711)
    ax1.plot(Parms_DF.index, Parms_DF['Rain'].values, color=color, linewidth=1.0, label='Rain Rate')
    ax1.set_ylabel('Rain Rate')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()

    ax2 = fig.add_subplot(712)
    ax2.plot(Parms_DF.index, Parms_DF['dBZ'].values, color=color, linewidth=1.0, label='Reflectivity')
    ax2.set_ylabel('Reflectivity')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()

    ax3 = fig.add_subplot(713)
    ax3.plot(Parms_DF.index, Parms_DF['LWC'].values, color=color, linewidth=1.0, label='LWC')
    ax3.set_ylabel('LWC')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()

    ax4 = fig.add_subplot(714)
    ax4.plot(Parms_DF.index, Parms_DF['Dm'].values, alpha=1.0, color=color, linewidth=1.0, label='Dm')
    ax4.plot(Parms_DF.index, Parms_DF['Dmax'].values, alpha=0.5, color='red', linewidth=1.0, label='Dmax')
    ax4.set_ylabel('Dm & DMax')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()

    ax5 = fig.add_subplot(715)
    ax5.plot(Parms_DF.index, Parms_DF['Total Drops'].values, color=color, linewidth=1.0, label='Total Drops')
    ax5.set_ylabel('Total Drops')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()

    ax6 = fig.add_subplot(716)
    ax6.plot(Parms_DF.index, Parms_DF['Concentration'].values, color=color, linewidth=1.0, label='Concentration')
    ax6.set_ylabel('Concentration')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()

    title = site + '/' + instrument + ' ' + smonth + '/' + sday + '/' + syear
    plt.suptitle(title, fontsize=30)
    plt.tight_layout()

    png_dir = Out_Base_Dir + 'Plots/' + instrument + '/Rain/' + syear + '/'
    png_file = png_dir + site + '_' + instrument + '_' + syear + '_' + smonth + sday + '_rain.png'
    if(savefig):
        # Save the plot
        os.makedirs(png_dir, exist_ok=True)
        plt.savefig(png_file, bbox_inches='tight')
        print('    --> ' + png_file)
        plt.close()
    else:
        plt.show()

    return png_file
