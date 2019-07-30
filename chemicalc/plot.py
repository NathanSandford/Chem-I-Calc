import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import StrMethodFormatter
import plotly.graph_objs as go


def plotly_crlb(crlb):
    labs = crlb.index.values
    cols = crlb.columns
    c = plt.cm.get_cmap('viridis', len(cols))
    data = []
    for j, col in enumerate(cols):
        data.append(go.Scatter(
                        name=col,
                        x=crlb[col].loc[labs].index,
                        y=crlb[col].loc[labs].values,
                        marker={'size': 12,
                                'line': {'width': 2}},
                        mode='lines+markers',
                        line={'color': f'rgba{c(j)}'},
                        connectgaps=True))

    layout = go.Layout(legend=dict(x=0.99, y=0.01,
                                   xanchor='right',
                                   yanchor='bottom',
                                   font={'size': 16},
                                   orientation='h'),
                       yaxis=dict(range=[-3, 0],
                                  type='log',
                                  title=dict(text='[X/Fe]',
                                             font=dict(size=24)),
                                  ticks='inside',
                                  ticktext=np.concatenate([[0.001],
                                                           np.repeat("", 8),
                                                           [0.01],
                                                           np.repeat("", 8),
                                                           [0.1],
                                                           np.repeat("", 8),
                                                           [1.0]]),
                                  tickvals=np.concatenate([np.linspace(0.001, 0.009, 9),
                                                           np.linspace(0.01, 0.09, 9),
                                                           np.linspace(0.1, 1.0, 10)]),
                                  showline=True,
                                  linewidth=2,
                                  linecolor='black',
                                  gridcolor='rgba(0,0,0,0.25)',
                                  zeroline=False,
                                  mirror='ticks'),
                       xaxis=dict(range=[-0.5, len(labs)-0.5],
                                  ticks='inside',
                                  tickangle=45,
                                  showline=True,
                                  linewidth=2,
                                  linecolor='black',
                                  gridcolor='rgba(0,0,0,0.25)',
                                  zeroline=False,
                                  mirror='ticks'),
                       plot_bgcolor='white')
    return go.Figure(data=data, layout=layout)


def multiplot_crlb(crlb_list, **kwargs):
    if 'crlb_min' in kwargs.keys():
        ymin = kwargs['crlb_min']
    else:
        ymin = 9e-4
    if 'crlb_max' in kwargs.keys():
        ymax = kwargs['crlb_max']
    else:
        ymax = 3
    if 'ncol_legend' in kwargs.keys():
        ncol = kwargs['ncol_legend']
    else:
        ncol = 1

    if type(crlb_list) is not list:
        crlb_list = [crlb_list]

    lead_crlb = np.argmax([len(crlb.index) for crlb in crlb_list])
    all_labs = crlb_list[lead_crlb].index
    nlabs = all_labs.shape[0]
    cols = crlb_list[0].columns
    c = plt.cm.get_cmap('viridis', len(cols))

    fig = plt.figure(figsize=(25, len(crlb_list) * 10))
    gs = GridSpec(len(crlb_list), 1)
    gs.update(hspace=0.0)

    for i, crlb in enumerate(crlb_list):
        if i == 0:
            ax = plt.subplot(gs[i, 0])
        else:
            ax = plt.subplot(gs[i, 0], sharex=ax)
        labs = crlb.index
        for j, col in enumerate(cols):
            mask = pd.notnull(crlb[col].loc[labs].values)
            plt.plot(crlb[col].loc[labs].index[mask],
                     crlb[col].loc[labs].values[mask],
                     marker='o', markersize=16, markeredgewidth=2,
                     linestyle='-', linewidth=2,
                     color=c(j), markeredgecolor='k',
                     label=col)
        plt.grid(True, 'both', 'both', linewidth=0.5, alpha=0.5)
        ax.set_ylabel(r'$\sigma$[X/Fe]', size=36)
        ax.set_xlim(-0.5, nlabs - 0.5)
        ax.set_ylim(ymin, ymax)
        ax.set_yscale('log')
        plt.legend(fontsize=30, ncol=ncol, loc='lower right')
        ax.tick_params(axis='both', which='minor', length=5)
        ax.tick_params(axis='both', which='both', direction='in', labelsize=30)
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:.3f}"))
        if i == len(crlb_list) - 1:
            ax.tick_params(axis='x', which='major', rotation=-45, length=10)
        else:
            ax.tick_params(axis='x', length=10, labelsize=0)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment('left')
        if 'labels' in kwargs.keys():
            plt.text(0.975, 0.95, s=kwargs['labels'][i], fontsize=30,
                     horizontalalignment='right',
                     verticalalignment='top',
                     transform=ax.transAxes,
                     bbox=dict(facecolor='white', edgecolor='black', pad=10.0))
    return fig


def overplot_crlb(crlb_list, **kwargs):
    if 'crlb_min' in kwargs.keys():
        ymin = kwargs['crlb_min']
    else:
        ymin = 9e-4
    if 'crlb_max' in kwargs.keys():
        ymax = kwargs['crlb_max']
    else:
        ymax = 3
    if 'ncol_legend' in kwargs.keys():
        ncol = kwargs['ncol_legend']
    else:
        ncol = 1

    lead_crlb = np.argmax([len(crlb.index) for crlb in crlb_list])
    labs = crlb_list[lead_crlb].index
    nlabs = labs.shape[0]
    cols = crlb_list[0].columns

    fig = plt.figure(figsize=(20, 10))
    ax = plt.subplot(111)
    c = plt.cm.get_cmap('viridis', len(cols))
    lines = ['-', '--', ':', '-.']

    for i, crlb in enumerate(crlb_list):
        for j, col in enumerate(cols):
            if i == 0:
                label = col
            else:
                label = '_nolegend_'
            mask = pd.notnull(crlb[col].loc[labs].values)
            plt.plot(crlb[col].loc[labs].index[mask],
                     crlb[col].loc[labs].values[mask],
                     marker='o', markersize=16, markeredgewidth=2,
                     linestyle=lines[i], linewidth=2,
                     color=c(j), markeredgecolor='k',
                     label=label)
        plt.grid(True, 'both', 'both', linewidth=0.5, alpha=0.5)
        plt.ylabel(r'$\sigma$[X/Fe]', size=36)
        plt.xlim(-0.5, nlabs - 0.5)
        plt.ylim(ymin, ymax)
        plt.yscale('log')
        plt.legend(fontsize=30, ncol=ncol, loc='lower right')
        plt.xticks(fontsize=30, rotation=-45)
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:.3f}"))
        for label in ax.get_xticklabels():
            label.set_horizontalalignment('left')
        plt.yticks(fontsize=30)
    return fig
