from typing import List, Tuple, Optional, Union
import numpy as np
import pandas as pd
from chemicalc.reference_spectra import ReferenceSpectra
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import StrMethodFormatter
from matplotlib.lines import Line2D


def plot_gradients(
    star: ReferenceSpectra,
    inst_name: str,
    labels: List[str],
    panel_height: float = 3,
    panel_width: float = 8,
    inset_ylabel: bool = False,
    inset_ylabel_xoffset: float = 0,
    inset_ylabel_yoffset: float = 0,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = (-0.11, 0.05),
    ylim_spec: Tuple[float, float] = (0.35, 1.15),
    yticks: Optional[List[float]] = None,
    yticks_spec: Optional[List[float]] = None,
    xtick_size: float = 18,
    ytick_size: float = 18,
    xlabel_size: float = 26,
    ylabel_size: float = 26,
    ylabel_pad: float = 10,
    include_spec: bool = True,
) -> plt.figure:
    """
    Plot gradients of a spectrum with respect to its stellar labels.

    :param ReferenceSpectra star: Reference star object
    :param str inst_name: Instrument name
    :param List[str] labels: List of labels
    :param float panel_height: Height of each subplot
    :param float panel_width: Width of figure
    :param bool inset_ylabel: Include label as inset annotation instead of in y-axis (better for large #s of labels)
    :param float inset_ylabel_xoffset: Relative x position of label annotation
    :param float inset_ylabel_yoffset: Relative y position of label annotation
    :param Optional[Tuple[float,float]] xlim: Bounds on the x-axis
    :param Optional[Tuple[float,float]] ylim: Bounds on the y-axis (for gradients)
    :param Tuple[float,float] ylim_spec: Bounds on the y-axis (for the spetrum if included)
    :param Optional[List[float]] yticks: Manual y-axis ticks (for gradients)
    :param Optional[List[float]] yticks_spec: Manual y-axis ticks (for the spetrum if included)
    :param float xtick_size: Fontsize of x-axis tick labels
    :param float ytick_size: Fontsize of y-axis tick labels
    :param float xlabel_size: Fontsize of x-axis labels
    :param float ylabel_size: Fontsize of y-axis labels
    :param float ylabel_pad: Pad between placeholder y-axis label and y-axis when using inset_ylabel
    :param bool include_spec: Include spectrum in top panel
    :return plt.figure: Matplotlib figure
    """
    nlabels = len(labels)
    if include_spec:
        nfigures = nlabels + 1
    else:
        nfigures = nlabels
    wave = star.wavelength[inst_name]
    if xlim is None:
        xlim = (np.min(wave), np.max(wave))
    fig = plt.figure(figsize=(panel_width, panel_height * nfigures))
    gs = GridSpec(nfigures, 1)
    gs.update(hspace=0.0)
    i = 0
    if include_spec:  # Plot spectrum in top panel
        ax = plt.subplot(gs[0, 0])
        ax.plot(wave, star.spectra[inst_name][0], c="k", lw=1)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim_spec)
        ax.set_ylabel(r"$f_\mathrm{norm}$", size=ylabel_size)
        ax.tick_params(axis="x", labelsize=0)
        if yticks_spec is not None:
            ax.set_yticks(yticks_spec)
        ax.tick_params(axis="y", labelsize=ytick_size)
        i += 1
    for label in labels:  # Plot gradients in individual panels
        ax = plt.subplot(gs[i, 0])
        ax.plot(star.gradients[inst_name].loc[label], c="k", lw=1)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        if inset_ylabel:  # Include label names as annotations
            ax.set_ylabel(
                r"$\frac{\partial f}{\partial X}}$",
                size=ylabel_size,
                rotation=0,
                va="center",
                labelpad=ylabel_pad,
            )
            ax.text(
                ax.get_xlim()[0] + inset_ylabel_xoffset,
                ax.get_ylim()[0] + inset_ylabel_yoffset,
                f"[{label}/H]",
                fontsize=18,
            )
        else:  # Include label names in y-axis labels
            ylabel = (
                "$\\frac{\partial f_\mathrm{norm}}{\partial \mathrm{"
                + f"[{label}/H]"
                + "}}$"
            )
            ax.set_ylabel(fr"{ylabel}", size=ylabel_size)
        if yticks is not None:
            ax.set_yticks(yticks)
        ax.tick_params(axis="y", labelsize=ytick_size)
        ax.tick_params(axis="x", labelsize=xtick_size)
        i += 1
        if i == nfigures:
            ax.set_xlabel(r"Wavelength ($\AA$)", size=xlabel_size)
        else:
            ax.tick_params(axis="x", labelsize=0)
    plt.tight_layout()
    return fig


def plot_crlb(
    crlb_list: Union[pd.DataFrame, List[pd.DataFrame]],
    cutoff: Optional[float] = None,
    labels: Union[str, List[str]] = None,
    label_loc: Tuple[float, float] = (0.98, 0.95),
    panel_height: float = 3,
    panel_width: float = 8,
    cutoff_label_xoffset: float = 3,
    cutoff_label_yoffset: float = 0.05,
    ylim: Optional[Tuple[float, float]] = (0.009, 1.7),
    yticks: Optional[List[float]] = None,
    ytick_ndecimal: int = 2,
    legend_ncol: int = 1,
    legend_loc: str = "lower right",
    reverse_legend: bool = False,
    color_palette: str = "plasma",
) -> plt.figure:
    """
    Plots standard presentation of CRLBs

    :param Union[pd.DataFrame,List[pd.DataFrame]] crlb_list: CRLB dataframe or list of CRLB dataframes
    :param Optional[float] cutoff: Cutoff precision for abundances
    :param Union[str,List[str]] labels: List of additional text to include in each panel.
                                        Must be same length as the number of CRLB dataframes
    :param Tuple[float,float] label_loc: Location of additional text box
    :param float panel_height: Height of each subplot
    :param float panel_width: Width of each subplot
    :param float cutoff_label_xoffset: Relative x position of cutoff label (increases to the left)
    :param float cutoff_label_yoffset: Relative y position of cutoff label
    :param Optional[Tuple[float,float]] ylim: Bound on y-axis
    :param Optional[List[float]] yticks: Manual y-axis ticks.
                                         Helpful when log-spacing yields only one tick on the y-axis.
    :param int ytick_ndecimal: Number of decimal places to include in y-axis ticks.
    :param int legend_ncol: Number of legend columns
    :param str legend_loc: Location of legend (standard matplotlib inputs)
    :param bool reverse_legend: Reverse order of legend items
    :param str color_palette: Color palette of lines and markers (standard matplotlib selection)
    :return plt.figure: Matplotlib figure
    """
    if type(crlb_list) is not list:
        crlb_list = [crlb_list]

    # Sort sets of CRLBs
    # ToDo: Thoroughly check that sorting works as intended
    order = np.argsort([-len(crlb.index) for crlb in crlb_list])
    sorted_crlb_list = [crlb_list[i] for i in order]
    all_crlb = pd.concat(sorted_crlb_list, axis=1, sort=False)
    all_labs = all_crlb.index
    all_cols = all_crlb.columns
    nlabs = all_labs.shape[0]
    npanels = len(crlb_list)

    # Initialize Figure
    fig = plt.figure(figsize=(panel_width, panel_height * npanels))
    gs = GridSpec(npanels, 1)
    gs.update(hspace=0.0)

    # Iterate through panels
    for i, crlb in enumerate(crlb_list):
        if i == 0:
            ax = plt.subplot(gs[i, 0])
        else:
            ax = plt.subplot(gs[i, 0], sharex=ax)
        labs = all_crlb.index
        cols = crlb.columns
        crlb_sorted = crlb.reindex(labs)
        c = plt.cm.get_cmap(color_palette, len(cols))
        # Iterate through CRLBs w/in panel
        for j, col in enumerate(
            all_cols
        ):  # Placeholder to make x-axis match between panels
            mask = np.isfinite(all_crlb.iloc[:, j].values)
            ax.plot(
                all_crlb.iloc[:, j].index[mask],
                all_crlb.iloc[:, j].values[mask],
                marker="",
                markersize=0,
                linestyle="",
                linewidth=0,
            )
        for j, col in enumerate(crlb):
            mask = np.isfinite(crlb_sorted.loc[:, col].values)
            plt.plot(
                crlb_sorted.loc[:, col].index[mask],
                crlb_sorted.loc[:, col].values[mask],
                marker="o",
                markersize=8,
                markeredgewidth=1,
                linestyle="-",
                linewidth=1,
                color=c(j),
                markeredgecolor="k",
                label=col,
            )
        # Plot cutoff line
        if cutoff:
            ax.axhline(cutoff, ls="--", lw=1, c="k")
            plt.text(
                s=f"{cutoff:01.1f} dex",
                x=nlabs - cutoff_label_xoffset,
                y=cutoff + cutoff_label_yoffset,
                fontsize=12,
            )
        # Axes
        ax.set_ylabel(r"$\sigma$[X/H]", size=16)
        # ToDo: replace StrMethodFormatter with FuncFormatter
        ax.yaxis.set_major_formatter(
            StrMethodFormatter("{x:." + f"{ytick_ndecimal}" + "f}")
        )
        ax.set_xlim(-0.5, nlabs - 0.5)
        ax.set_ylim(ylim)
        ax.set_yscale("log")
        plt.grid(True, "both", "both")
        if i == npanels - 1:
            ax.tick_params(axis="x", which="major", rotation=-45)
        else:
            ax.tick_params(axis="x", labelsize=0)
        for j, label in enumerate(ax.get_xticklabels()):
            label.set_horizontalalignment("left")
        # ToDo: replace StrMethodFormatter with FuncFormatter
        ax.yaxis.set_major_formatter(
            StrMethodFormatter("{x:." + f"{ytick_ndecimal}" + "f}")
        )
        # Add Label
        if labels is not None:
            if type(labels) is not list:
                labels = [labels]
            plt.text(
                label_loc[0],
                label_loc[1],
                s=labels[i],
                fontsize=10,
                horizontalalignment="right",
                verticalalignment="top",
                transform=ax.transAxes,
                bbox=dict(fc="white", ec="black", lw=1, pad=5.0),
            )
        # Legend
        plt.legend(fontsize=10, ncol=legend_ncol, loc=legend_loc)
        if reverse_legend:
            leg_handles, leg_labels = fig.axes[i].get_legend_handles_labels()
            fig.axes[i].legend(
                leg_handles[::-1],
                leg_labels[::-1],
                fontsize=10,
                ncol=legend_ncol,
                loc=legend_loc,
            )
        if yticks is not None:
            fig.axes[i].set_yticks(yticks)

    plt.tight_layout()
    return fig


def overplot_crlb(
    crlb_list: List[pd.DataFrame],
    names: List[str],
    cutoff: Optional[float] = None,
    labels: Union[str, List[str]] = None,
    label_loc: Tuple[float, float] = (0.98, 0.95),
    panel_height: float = 3,
    panel_width: float = 8,
    cutoff_label_xoffset: float = 3,
    cutoff_label_yoffset: float = 0.05,
    ylim: Optional[Tuple[float, float]] = (0.009, 1.7),
    yticks: Optional[List[float]] = None,
    legend_ncol: int = 1,
    legend_loc: str = "lower right",
    reverse_legend: bool = False,
    legend2_ncol: int = 1,
    legend2_loc: Tuple[float, float] = (1, 0.425),
    reverse_legend2: bool = False,
    color_palette: str = "plasma",
) -> plt.figure:
    """
    Overplots two groups of CRLBs using different line styles and marker shapes

    :param List[pd.DataFrame] crlb_list: List of CRLB dataframes
    :param List[str] names: Labels to show in second legend
    :param Optional[float] cutoff: Cutoff precision for abundances
    :param Union[str,List[str]] labels: List of additional text to include in each panel
    :param Tuple[float,float] label_loc: Location of additional text box
    :param float panel_height: Height of each subplot
    :param float panel_width: Width of each subplot
    :param float cutoff_label_xoffset: Relative x position of cutoff label (increases to the left)
    :param float cutoff_label_yoffset: Relative y position of cutoff label
    :param Optional[Tuple[float,float]] ylim: Bound on y-axis
    :param Optional[List[float]] yticks: Manual y-axis ticks
    :param int legend_ncol: Number of legend columns
    :param str legend_loc: Location of legend (standard matplotlib inputs)
    :param bool reverse_legend: Reverse order of legend items
    :param int  legend2_ncol: Number of legend columns for second legend
    :param Tuple[float, float] legend2_loc: Location of legend for second legend (axis coords)
    :param bool reverse_legend2: Reverse order of legend items for second legend
    :param str color_palette: Color palette of lines and markers
    :return plt.figure: Matplotlib figure
    """
    # ToDo: Thoroughly check that sorting works as intended
    # Determin CRLBs with most labels
    lead_crlb = np.argmax([len(crlb.index) for crlb in crlb_list])
    all_labs = crlb_list[lead_crlb].index
    nlabs = all_labs.shape[0]
    # Initialize Figure
    fig = plt.figure(figsize=(panel_width, panel_height))
    gs = GridSpec(1, 1)
    gs.update(hspace=0.0)
    ax = plt.subplot(gs[0, 0])
    c = plt.cm.get_cmap(color_palette, np.max([crlb.shape[1] for crlb in crlb_list]))
    lines = ["-", "--", ":", "-."]
    markers = ["s", "o", "^", "*"]
    # Iterate through panels
    for i, crlb in enumerate(crlb_list):
        labs = crlb.index
        cols = crlb_list[i].columns
        # Iterate through CRLBs w/in set
        for j, col in enumerate(cols):
            if i == 0:
                label = col
            else:
                label = "_nolegend_"
            mask = pd.notnull(crlb[col].loc[labs].values)
            plt.plot(
                crlb[col].loc[labs].index[mask],
                crlb[col].loc[labs].values[mask],
                marker=markers[i],
                markersize=8,
                markeredgewidth=1,
                linestyle=lines[i],
                linewidth=1,
                color=c(j),
                markeredgecolor="k",
                label=label,
            )
    # Plot cutoff line
    if cutoff:
        ax.axhline(cutoff, ls="--", lw=1, c="k")
        plt.text(
            s=f"{cutoff:01.1f} dex",
            x=nlabs - cutoff_label_xoffset,
            y=cutoff + cutoff_label_yoffset,
            fontsize=12,
        )
    plt.text(
        s="0.3 dex",
        x=nlabs - cutoff_label_xoffset,
        y=cutoff + cutoff_label_yoffset,
        fontsize=12,
    )
    # Axes
    ax.set_ylabel(r"$\sigma$[X/H]", size=16)
    # ToDo: replace StrMethodFormatter with FuncFormatter
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:.2f}"))
    ax.set_xlim(-0.5, nlabs - 0.5)
    ax.set_ylim(ylim)
    ax.set_yscale("log")
    plt.grid(True, "both", "both")
    ax.tick_params(axis="x", which="major", rotation=-45)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("left")
    # ToDo: replace StrMethodFormatter with FuncFormatter
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:.2f}"))
    # Add Label
    if labels is not None:
        for i, label in enumerate(labels):
            plt.text(
                label_loc[0],
                label_loc[1],
                s=labels[i],
                fontsize=10,
                horizontalalignment="right",
                verticalalignment="top",
                transform=ax.transAxes,
                bbox=dict(fc="white", ec="black", lw=1, pad=5.0),
            )
    # Legend
    custom_lines = []
    for i, name in enumerate(names):
        custom_lines.append(
            Line2D(
                [0], [0], color="k", lw=1, ls=lines[i], marker=markers[i], markersize=8
            )
        )
    if reverse_legend2:
        custom_lines = custom_lines[::-1]
        names = names[::-1]
    leg2 = fig.axes[0].legend(
        custom_lines,
        names,
        fontsize=10,
        ncol=legend2_ncol,
        bbox_to_anchor=legend2_loc,
        edgecolor="k",
    )
    plt.legend(fontsize=10, ncol=legend_ncol, loc=legend_loc)
    if reverse_legend:
        leg_handles, leg_labels = fig.axes[0].get_legend_handles_labels()
        fig.axes[0].legend(
            leg_handles[::-1],
            leg_labels[::-1],
            fontsize=10,
            ncol=legend_ncol,
            loc=legend_loc,
        )
    fig.axes[0].add_artist(leg2)
    if yticks is not None:
        fig.axes[0].set_yticks(yticks)
    plt.tight_layout()
    return fig


def gridplot_crlb(
    crlb: pd.DataFrame,
    xlabel: str,
    figsize: Tuple[float, float] = (8, 9),
    label_fontsize: float = 20,
    tick_fontsize: float = 10,
    xtick_rotation: float = -70,
    color_palette: str = "plasma",
) -> plt.figure:
    """
    Plots grid representation of the CRLBs for many instrumental specifications.

    :param pd.DataFrame crlb: CRLB DataFrame
    :param str xlabel: X-axis Label
    :param Tuple[float,float] figsize: X- and y-dimensions of figure
    :param float label_fontsize: Fontsize of x- and y-axis labels
    :param float tick_fontsize: Fontsize of x- and y-axis tick labels
    :param float xtick_rotation: Rotation of x-axis tick labels
    :param str color_palette: Color palette of figure
    :return plt.figure: Matplotlib figure
    """
    # Initialize Figure
    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_subplot(111)
    # Plot CRLBs
    cax = ax1.imshow(crlb, aspect="auto", cmap=plt.get_cmap(color_palette))
    # Add Axis Labels
    plt.xticks(
        np.arange(len(crlb.columns)),
        crlb.columns,
        fontsize=tick_fontsize,
        rotation=xtick_rotation,
        ha="left",
        rotation_mode="anchor",
    )
    plt.yticks(np.arange(len(crlb.index)), crlb.index, fontsize=tick_fontsize)
    plt.xlabel(xlabel, size=label_fontsize)
    # Draw Grid
    ax1.grid(False, "both", "both")
    for i in np.arange(len(crlb.columns)):
        ax1.axvline(i + 0.5, c="grey", lw=0.5)
    for i in np.arange(len(crlb.index)):
        ax1.axhline(i + 0.5, c="grey", lw=0.5)
    ax1.tick_params(axis="both", which="major", pad=5)
    # Add Colorbar
    cbar = fig.colorbar(cax)
    cbar.set_label(r"$\sigma$[X/H]", size=label_fontsize)
    cbar.ax.tick_params(labelsize=tick_fontsize)
    plt.tight_layout()
    return fig
