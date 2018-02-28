#/usr/bin/env python3
from scipy import stats
import pandas as pd
import numpy as np
import matplotlib.backends.backend_pdf
from matplotlib import rcParams
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")

#function to return axis & figures for boxplots
def grab_new_ax_array():
    boxplot_fig = plt.figure(figsize=(8, 17))
    gs1 = gridspec.GridSpec(5, 1)
    ax1 = boxplot_fig.add_subplot(gs1[0])
    ax2 = boxplot_fig.add_subplot(gs1[1])
    ax3 = boxplot_fig.add_subplot(gs1[2])
    ax4 = boxplot_fig.add_subplot(gs1[3])
    ax5 = boxplot_fig.add_subplot(gs1[4])
    ax_array1 = [ax1, ax2, ax3, ax4, ax5]
    return gs1, boxplot_fig, ax_array1

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

#dynamic fontsize function created specifically for seaborn factorplots
def suggest_fontsize(num_bars):
    m = -(8/15)
    intercept = 19
    fontsize = (num_bars * m) + intercept
    if fontsize < 7:
        fontsize = 7
    if fontsize > 16:
        fontsize = 16
    return fontsize

def bargraph_from_db(modeldb, x, y, seperate_by, hue=None, y_title=None):
    if modeldb.empty:
        return []

    total_bars = len(modeldb[x].unique())
    if total_bars > 15:
        #seperate modeldb into current 15 & next whatever excess
        curr_modeldb = modeldb.query(str(x) + " <= 15")
        next_modeldb = modeldb.query(str(x) + " > 15")
        curr_figs = bargraph_from_db(curr_modeldb, x, y, seperate_by, hue, y_title)
        next_figs = bargraph_from_db(next_modeldb, x, y, seperate_by, hue, y_title)
        return curr_figs + next_figs

    #otherwise continue to create graph
    g = sns.factorplot(data=modeldb, x=x, y=y, row=seperate_by, kind='bar', hue=hue, size=3, aspect=6, ci=None,
                       legend=False)
    g.set_titles("{row_name}")
    if y_title:
        g.set_axis_labels(str(x), y_title)

    #calculating fontsize for bargraph based on # of bars
    fontsize = suggest_fontsize(total_bars)
    axes = g.axes.flat

    #first make space for each axis, then add the legend
    for ax in axes:
        box = ax.get_position() # get position of figure
        ax.set_position([box.x0, box.y0, box.width * 0.9, box.height]) # resize position
        #now add a legend to the right most axes
        ax.legend(loc='right', bbox_to_anchor=(1.12, 0.5), ncol=1)

    #reload axes
    axes = g.axes.flat
    #annotate each bar
    for ax in axes:
        for p in ax.patches:
            ax.annotate(str(p.get_height())[:5], (p.get_x() + (0 * p.get_width()), p.get_height() * 1.005),
                        fontsize=fontsize)

    return [g.fig]


#function to graph importances from DF that has var column & other columns with importances
#returns list of figures
#an imp_df has the following form: one column is titled 'var' and has all variables, other columns 
#    have importance values with the column title being the algo
def feature_importance_bargraphs(imp_df, tag="", annotations=None):
    #set autolayout to true to accomodate long var names
    rcParams.update({'figure.autolayout': True})

    all_figs = []
    imp_cols = imp_df.columns.tolist()
    imp_cols.remove("var")

    #handling if too big
    if len(imp_df) > 100:
        for col in imp_cols:
            #get max/min of column for scale
            curr_max = imp_df[col].max()
            curr_min = min(imp_df[col].min(), 0)
            imp_df.sort_values(col, ascending=False, inplace=True)
            #split into chunks of rows; each row is a variable
            chunks = [imp_df[x:x+100] for x in range(0, len(imp_df), 100)]
            for idx, chunk in enumerate(chunks):
                y_title = col
                g = sns.factorplot(x='var', y=col, data=chunk, size=7.5, aspect=2.3, kind='bar', ci=None)
                g.set_xticklabels(rotation=45,ha='right')
                #set titles for graph
                axes = g.axes.flat
                for ax in axes:
                    ax.set_title(tag + " Feature Importances for " + col + " Part " + str(idx))
                    ax.set_ylim([curr_min, curr_max])
                    if annotations:
                        ax.text(s=annotations, transform=ax.transAxes, x=0.9,y=0.8)
                g.set_axis_labels("", y_title)
                curr_fig = g.fig
                all_figs = all_figs + [curr_fig]

    else:
        for col in imp_cols:
            y_title = col
            #fig,ax = plt.subplots()
            imp_df = imp_df.sort_values(col, ascending=False)
            g = sns.factorplot(x='var', y=col, data=imp_df, size=7.5, aspect=2.3, kind='bar')
            g.set_xticklabels(rotation=45,ha='right')
            #set titles for graph
            axes = g.axes.flat
            for ax in axes:
                ax.set_title(tag + " Feature Importances for " + col)
                if annotations:
                    ax.text(s=annotations, transform=ax.transAxes, x=0.9,y=0.8)
            g.set_axis_labels("", y_title)
            curr_fig = g.fig
            all_figs = all_figs + [curr_fig]

    return all_figs

def linegraph_from_db(modeldb, x, y):
    g = sns.FacetGrid(modeldb, size=4)
    g = g.map(sns.pointplot, x, y)
    return g.fig

def paint_declines_red(g):
    axes = g.axes.flat
    for ax in axes:
        patch_list = ax.patches
        for first, second in zip(patch_list, patch_list[1:]):
            next_height = second.get_height()
            curr_height = first.get_height()
            curr_diff = next_height - curr_height
            if curr_diff < 0:
                second.set_facecolor('red')
                ax.annotate(str(round(curr_diff, 2)),(second.get_x() + (0.25 * second.get_width()), second.get_height() * 1.005))
    return g

def bargraph_fsa(mydf, y, title, paint_red_declines=False):
    rcParams.update({'figure.autolayout': True})
    g = sns.factorplot(data=mydf, x='var', y=y, col=None , kind='bar', size=4, aspect=3, ci=None)
    g.set_xticklabels(rotation=45,ha='right')
    axes = g.axes.flat
    for ax in axes:
        ax.set_title(title)
    if paint_red_declines:
        g = paint_declines_red(g)
    return g.fig


