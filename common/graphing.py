#/usr/bin/env python3
from scipy import stats
import pandas as pd
import numpy as np
import matplotlib.backends.backend_pdf
from matplotlib import rcParams
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")

def figures_to_pdf(fig_list, pdf_path):
    pdf = matplotlib.backends.backend_pdf.PdfPages(pdf_path)
    for fig in fig_list:
        ax_list = fig.axes
        for ax in ax_list:
            ax.set_rasterized(True)
        pdf.savefig(fig, orientation='portrait', dpi=300)
    pdf.close()

def histogram_boxcox_plot(var_name, var_data):
    shift = np.amin(var_data)
    shift = min([shift, 0])
    shift_output_var = [x + shift for x in var_data]
    #Histogram and boxcox plot for output
    fig = plt.figure(figsize=(8,11), dpi=300)
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    #histogram of original data
    ax1.set_title("Distribution of " + var_name)
    ax1.set_xlabel(var_name)
    var_data.hist(ax=ax1, rasterized=True)
    #boxcox norm plot of shifted data
    lmbdas, ppcc = stats.boxcox_normplot(shift_output_var, -10, 10, plot=ax2)
    ax2.plot(lmbdas, ppcc, 'bo')
    shift_output_var_t, maxlog = stats.boxcox(shift_output_var)
    boxcox_output_var_t = [x - shift for x in shift_output_var_t]
    #adding vertical line to plot
    ax2.axvline(maxlog, color='r')
    ax2.text(maxlog + 0.1, 0, s=str(maxlog))
    return fig

def hist_prob_plot(var_name, var_data):
    fig2 = plt.figure(figsize=(8,11), dpi=300)
    ax3 = fig2.add_subplot(211)
    ax4 = fig2.add_subplot(212)
    ax3.set_title("Distribution of " + var_name)
    ax3.set_xlabel(var_name)
    pd.Series(var_data).hist(ax=ax3, rasterized=True)
    #measures of normalness
    (osm, osr), (slope, intercept, r) = stats.probplot(var_data, dist="norm",plot=ax4)
    r2 = r ** 2
    r2 = round(r2, 3)
    curr_kurtosis = stats.kurtosis(var_data, fisher=False)
    curr_kurtosis = round(curr_kurtosis, 3)
    ax3.text(s='kurtosis= ' + str(curr_kurtosis), transform=ax3.transAxes,
            x=0.8,y=0.8)
    ax4.text(s='r^2= ' + str(r2), transform=ax4.transAxes,
             x=0.8,y=0.2)
    return fig2


