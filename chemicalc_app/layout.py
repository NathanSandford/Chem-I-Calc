import dash_html_components as html
import dash_core_components as dcc
import dash_table

from chemicalc.reference_spectra import reference_stars
from chemicalc.s2n import keck_options
from .utils import spectrographs, prior_tooltips, snr_options, etc_options
from .utils import sample_labels, lab_tab_text

import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objs as go

UC_colors = {'blue': 'rgb(0, 50, 98)',
             'white': 'rgb(255, 255, 255)'}

n_combo = 3
n_inst = 3


def create_instrument_selection(combo_no, inst_no):
    inst_presets = html.Div([
        dcc.Dropdown(id=f'preset-{combo_no}-{inst_no}',
                     placeholder='Load from presets',
                     options=[{'label': spec_name, 'value': spec_name} for spec_name in spectrographs.keys()],
                     style={'marginRight': 10})
    ])
    inst_name = html.Div([
        html.Label('Name: ',
                   style={'float': 'Left',
                          'padding': '10px 0px',
                          'display': 'tableCell',
                          'verticalAlign': 'Bottom',
                          'marginTop': 3}),
        dcc.Input(id=f'instname-{combo_no}-{inst_no}',
                  placeholder='-',
                  value='',
                  style={'width': '75%',
                         'float': 'Right',
                         'textAlign': 'Right',
                         'display': 'tableCell',
                         'verticalAlign': 'Bottom',
                         'marginTop': 3,
                         'marginRight': 10}
    )
        ], style={'display': 'block'})
    inst_wave = html.Div([
        html.Label(u'Wavelength (\u212B):',
                   style={'float': 'left',
                          'padding': '10px 10px 0px 0px'}),
        html.Div([
            dcc.Input(id=f'wavemin-{combo_no}-{inst_no}',
                      type='number', inputMode='numeric',
                      step=10, placeholder="start",
                      style={'width': 75,
                             'display': 'inline-block'}),
            html.P(u'Min: 3000 \u212B',
                   style={'fontStyle': 'italic',
                          'fontSize': 12,
                          'padding': '0 0'})
        ], style={'display': 'inline-block',
                  'float': 'Left'}),
        html.Div(html.Label('-'),
                 style={'float': 'Left',
                        'padding': '10px 0px',
                        'display': 'inline_block'}),
        html.Div([
            dcc.Input(id=f'wavemax-{combo_no}-{inst_no}',
                      type='number', inputMode='numeric',
                      step=10, placeholder="end",
                      style={'width': 75,
                             'display': 'inline-block'}),
            html.P(u'Max: 18000 \u212B',
                   style={'fontStyle': 'italic',
                          'fontSize': 12})
        ], style={'display': 'inline-block',
                  'float': 'Left',
                  'marginRight': 10})
    ])
    inst_res = html.Div([
        html.Label(u'Res. Power (\u0394\u03BB/\u03BB):',
                   style={'float': 'Left',
                          'verticalAlign': 'Middle',
                          'padding': '5px 0px'}),
        dcc.Input(id=f'res-{combo_no}-{inst_no}',
                  type='number', inputMode='numeric',
                  placeholder='-',
                  min=1e2, max=1e5, step=1e2,
                  style={'width': '40%',
                         'float': 'Right',
                         'marginTop': -3,
                         'marginRight': 10,
                         'textAlign': 'Right'}),
    ], style={'display': 'block'})
    inst_samp = html.Div([
        html.Label(r'Pixels / FWHM:',
                   style={'float': 'Left',
                          'verticalAlign': 'Middle',
                          'padding': '5px 0px'}),
        dcc.Input(id=f'samp-{combo_no}-{inst_no}',
                  type='number', inputMode='numeric',
                  placeholder='-',
                  min=1, max=10, step=1,
                  style={'width': '40%',
                         'float': 'Right',
                         'marginTop': 3,
                         'marginRight': 10,
                         'textAlign': 'Right'}),
    ], style={'display': 'block'})
    inst_snr = html.Details([
        html.Summary('SNR Configuration', style={'textAlign': 'Left'}),
        dcc.Dropdown(id=f'snr-type-{combo_no}-{inst_no}',
                     placeholder='Select SNR type',
                     options=[{'label': opt, 'value': opt} for opt in snr_options],
                     value=snr_options[0],
                     style={'textAlign': 'Left',
                            'marginTop': 5,
                            'marginRight': 10,}),
        html.Div(id=f'constant-snr-{combo_no}-{inst_no}', children=[
            html.Label(r'SNR / pixel:',
                       style={'float': 'Left',
                              'verticalAlign': 'Middle',
                              'padding': '10px 0px'}),
            dcc.Input(id=f'snr-const-value-{combo_no}-{inst_no}',
                      type='number', inputMode='numeric',
                      placeholder='-',
                      value=100,
                      min=1, max=1000, step=10,
                      style={'width': '40%',
                             'float': 'Right',
                             'marginTop': 3,
                             'marginRight': 10,
                             'textAlign': 'Right'})
        ]),
        dcc.Dropdown(id=f'etc-dropdown-{combo_no}-{inst_no}',
                     placeholder='Choose ETC...',
                     options=[{'label': opt, 'value': opt} for opt in etc_options],
                     clearable=True,
                     style={'textAlign': 'Left',
                            'marginRight': 10,
                            'display': 'None'}),
        dcc.Dropdown(id=f'wmko-inst-{combo_no}-{inst_no}',
                     placeholder='Select Instrument...',
                     options=[{'label': opt, 'value': opt} for opt in keck_options['instrument']],
                     style={'textAlign': 'Left',
                            'marginRight': 10,
                            'display': 'None'}),

        dcc.Dropdown(id=f'wmko-grating-{combo_no}-{inst_no}',
                     placeholder='Select Grating...',
                     options=[],
                     style={'textAlign': 'Left',
                            'marginRight': 10,
                            'display': 'None'}),
        dcc.Dropdown(id=f'wmko-grism-{combo_no}-{inst_no}',
                     placeholder='Select Grism...',
                     options=[{'label': opt, 'value': opt} for opt in keck_options['grism (LRIS)']],
                     style={'textAlign': 'Left',
                            'marginRight': 10,
                            'display': 'None'}),
        dcc.Dropdown(id=f'wmko-slitwidth-{combo_no}-{inst_no}',
                     placeholder='Select Slitwidth...',
                     options=[],
                     style={'textAlign': 'Left',
                            'marginRight': 10,
                            'display': 'None'}),
        dcc.Dropdown(id=f'wmko-dichroic-{combo_no}-{inst_no}',
                     placeholder='Select Dichroic...',
                     options=[{'label': opt, 'value': opt} for opt in keck_options['dichroic (LRIS)']],
                     style={'textAlign': 'Left',
                            'marginRight': 10,
                            'display': 'None'}),
        dcc.Dropdown(id=f'wmko-binning-{combo_no}-{inst_no}',
                     placeholder='Select Binning...',
                     options=[],
                     style={'textAlign': 'Left',
                            'marginRight': 10,
                            'display': 'None'}),
        dcc.Dropdown(id=f'wmko-cwave-{combo_no}-{inst_no}',
                     placeholder='Select Central Wavelength...',
                     options=[{'label': opt, 'value': opt} for opt in keck_options['central wavelength (DEIMOS)']],
                     style={'textAlign': 'Left',
                            'marginRight': 10,
                            'display': 'None'}),

        html.Div(id=f'wmko-exptime-{combo_no}-{inst_no}', children=[
            html.Label(r'Exposure Time (s):',
                       style={'float': 'Left',
                              'verticalAlign': 'Middle',
                              'padding': '10px 0px'}),
            dcc.Input(id=f'wmko-exptime-value-{combo_no}-{inst_no}',
                      type='number', inputMode='numeric',
                      placeholder='-',
                      value=3600,
                      min=1, step=1000,
                      style={'width': '40%',
                             'float': 'Right',
                             'marginTop': 3,
                             'marginRight': 10,
                             'textAlign': 'Right'})
        ], style={'display': 'None'}),
        html.Div(style={'clear': 'both'}),
        html.Div(id=f'wmko-mag-{combo_no}-{inst_no}', children=[
            html.Label(r'Apparent Magnitude:',
                       style={'float': 'Left',
                              'verticalAlign': 'Middle',
                              'padding': '10px 0px'}),
            dcc.Input(id=f'wmko-mag-value-{combo_no}-{inst_no}',
                      type='number', inputMode='numeric',
                      placeholder='-',
                      value=19,
                      min=0, max=30, step=0.5,
                      style={'width': '40%',
                             'float': 'Right',
                             'marginTop': 3,
                             'marginRight': 10,
                             'textAlign': 'Right'})
        ], style={'display': 'None'}),
        html.Div(style={'clear': 'both'}),
        dcc.Dropdown(id=f'wmko-magtype-{combo_no}-{inst_no}',
                     placeholder='AB or Vega...',
                     options=[{'label': opt, 'value': opt} for opt in keck_options['mag type']],
                     style={'textAlign': 'Left',
                            'marginRight': 10,
                            'display': 'None'}),
        dcc.Dropdown(id=f'wmko-filter-{combo_no}-{inst_no}',
                     placeholder='Select Filter...',
                     options=[{'label': opt, 'value': opt} for opt in keck_options['filter']],
                     style={'textAlign': 'Left',
                            'marginRight': 10,
                            'display': 'None'}),
        dcc.Dropdown(id=f'wmko-star-{combo_no}-{inst_no}',
                     placeholder='Select Stellar Template...',
                     options=[{'label': opt, 'value': opt} for opt in keck_options['template']],
                     style={'textAlign': 'Left',
                            'marginRight': 10,
                            'display': 'None'}),

        html.Div(style={'clear': 'both'}),
        html.Div(id=f'wmko-airmass-{combo_no}-{inst_no}', children=[
            html.Label(r'Airmass:',
                       style={'float': 'Left',
                              'verticalAlign': 'Middle',
                              'padding': '10px 0px'}),
            dcc.Input(id=f'wmko-airmass-value-{combo_no}-{inst_no}',
                      type='number', inputMode='numeric',
                      placeholder='-',
                      value=1.1,
                      min=1, max=5, step=0.1,
                      style={'width': '40%',
                             'float': 'Right',
                             'marginTop': 3,
                             'marginRight': 10,
                             'textAlign': 'Right'})
        ], style={'display': 'None'}),
        html.Div(style={'clear': 'both'}),
        html.Div(id=f'wmko-seeing-{combo_no}-{inst_no}', children=[
            html.Label(r'Seeing ("):',
                       style={'float': 'Left',
                              'verticalAlign': 'Middle',
                              'padding': '10px 0px'}),
            dcc.Input(id=f'wmko-seeing-value-{combo_no}-{inst_no}',
                      type='number', inputMode='numeric',
                      placeholder='-',
                      value=0.75,
                      min=0.5, max=2, step=0.1,
                      style={'width': '40%',
                             'float': 'Right',
                             'marginTop': 3,
                             'marginRight': 10,
                             'textAlign': 'Right'})
        ], style={'display': 'None'}),
        html.Div(style={'clear': 'both'}),
    ])

    inst_selection = html.Div([
        inst_presets,
        inst_name,
        html.Div(style={'clear': 'both'}),
        inst_wave,
        html.Div(style={'clear': 'both'}),
        inst_res,
        html.Div(style={'clear': 'both'}),
        inst_samp,
        html.Div(style={'clear': 'both'}),
        inst_snr,
    ])
    return inst_selection


def create_combination_selection(combo_no):
    combo_name = html.Div([
        html.H4('Instrument Combination: ',
                style={'display': 'inline-block'}),
        dcc.Input(id=f'comboname-{combo_no}',
                  value=f'Combo-{combo_no}',
                  style={'height': 25,
                         'width': 200,
                         'fontSize': 20,
                         'marginLeft': 20,
                         'display': 'inline-block'}),
    ], style={'display': 'block'})

    instruments = []
    for i in range(n_inst):
        if i == 0:
            disp = 'inline-block'
            button = 'Remove'
        else:
            disp = 'none'
            button = 'Add'
        instruments.append(
            html.Div([
                html.Div(id=f'instrument-selection-{combo_no}-{i}',
                         children=create_instrument_selection(combo_no, i),
                         style={'display': disp}),
                html.Button(id=f'inst-button-{combo_no}-{i}',
                            children=button,
                            style={'display': 'inline-block'}),

                ], style={'display': 'inline-block',
                          'verticalAlign': 'Top',
                          'width': '33%',
                          'height': '100%',
                          'borderRight': '1px solid gray',
                          'textAlign': 'Center'})
        )

    combo_selection = html.Div([
        combo_name,
        html.Div(instruments,
                 style={'margin': '0 auto',
                        'width': '99%'}),
    ], style={'display': 'block'})

    return combo_selection




###################
# Initialize Plots
###################
crlb_plot_layout = go.Layout(
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

                       xaxis=dict(range=[-0.5, len(sample_labels)-0.5],
                                  ticks='inside',
                                  ticktext=sample_labels,
                                  tickvals=np.linspace(0, len(sample_labels), len(sample_labels)-1),
                                  tickangle=45,
                                  showline=True,
                                  linewidth=2,
                                  linecolor='black',
                                  gridcolor='rgba(0,0,0,0.25)',
                                  zeroline=False,
                                  mirror='ticks'),
                       plot_bgcolor='white')

grad_plot_layout = go.Layout(width=1000, height=600,
                             yaxis=dict(title=dict(text='df/dX',
                                                   font=dict(size=24)),
                                        range=[-0.5, 0.5],
                                        ticks='inside',
                                        showline=True,
                                        linewidth=2,
                                        linecolor='black',
                                        gridcolor='rgba(0,0,0,0.25)',
                                        zeroline=False,
                                        mirror='ticks',
                                        domain=[0.5, 1.0]),
                             yaxis2=dict(title=dict(text='SNR / pixel',
                                                    font=dict(size=24)),
                                         range=[0.0, 100],
                                         ticks='inside',
                                         showline=True,
                                         linewidth=2,
                                         linecolor='black',
                                         gridcolor='rgba(0,0,0,0.25)',
                                         zeroline=False,
                                         mirror='ticks',
                                         domain=[0.0, 0.5]),
                             xaxis=dict(title=dict(text=u'Wavelength (\u212B)',
                                                   font=dict(size=24)),
                                        range=[3e3, 18e3],
                                        ticks='inside',
                                        tickangle=0,
                                        showline=True,
                                        linewidth=2,
                                        linecolor='black',
                                        gridcolor='rgba(0,0,0,0.25)',
                                        zeroline=False,
                                        mirror='ticks'),
                             xaxis2=dict(title=dict(text=u'Wavelength (\u212B)',
                                                   font=dict(size=24)),
                                        range=[3e3, 18e3],
                                        ticks='inside',
                                        tickangle=0,
                                        showline=True,
                                        linewidth=2,
                                        linecolor='black',
                                        gridcolor='rgba(0,0,0,0.25)',
                                        zeroline=False,
                                        mirror='ticks'),
                             plot_bgcolor='white'
                             )
grad_plot_data = [go.Scatter(x=[], y=[], yaxis='y1', xaxis='x1'),
                  go.Scatter(x=[], y=[], yaxis='y2', xaxis='x2'),
                  ]
fig = make_subplots(rows=2, cols=1, vertical_spacing=0, shared_xaxes=True)
fig.append_trace(go.Scatter(x=[], y=[]), row=1, col=1)
fig.append_trace(go.Scatter(x=[], y=[]), row=2, col=1)
fig.update_layout(width=1000, height=800,
                  margin=dict(l=75, r=50, t=30, b=50),
                  yaxis=dict(title=dict(text='df/dX',
                                        font=dict(size=24)),
                             range=[-0.5, 0.5],
                             ticks='inside',
                             showline=True,
                             linewidth=2,
                             linecolor='black',
                             gridcolor='rgba(0,0,0,0.25)',
                             zeroline=False,
                             mirror='ticks',),
                  yaxis2=dict(title=dict(text='SNR (pixel<sup>-1</sup>)',
                                         font=dict(size=24)),
                              range=[-0.5, 0.5],
                              ticks='inside',
                              showline=True,
                              linewidth=2,
                              linecolor='black',
                              gridcolor='rgba(0,0,0,0.25)',
                              zeroline=False,
                              mirror='ticks',),
                  xaxis=dict(#title=dict(text=u'Wavelength (\u212B)',
                             #           font=dict(size=24)),
                             range=[3e3, 18e3],
                             ticks='inside',
                             tickangle=0,
                             showline=True,
                             linewidth=2,
                             linecolor='black',
                             gridcolor='rgba(0,0,0,0.25)',
                             zeroline=False,
                             mirror='ticks'),
                  xaxis2=dict(title=dict(text=u'Wavelength (\u212B)',
                                         font=dict(size=24)),
                              range=[3e3, 18e3],
                              ticks='inside',
                              tickangle=0,
                              showline=True,
                              linewidth=2,
                              linecolor='black',
                              gridcolor='rgba(0,0,0,0.25)',
                              zeroline=False,
                              mirror='ticks'),
                  plot_bgcolor='white')


###################
# Initialize Layout
###################

# -----------
# Header
# -----------
header = html.Div([
    html.H1(children='ChemiCalc - Chemical Information Calculator',
            style={'backgroundColor': UC_colors['blue'],
                   'color': UC_colors['white'],
                   'margin': 0,
                   'padding': 0}),
    ], style={'display': 'block',
              'width': '100%',
              'position': 'fixed',
              'left': 0,
              'top': 0,
              'zIndex': 10})

# -----------
# Star Select
# -----------
reference_star_select = html.Div([
    html.H4('Reference Star(s):',
            style={'display': 'block'}),
    dcc.Dropdown(id='reference-select',
                 options=[{'label': ref, 'value': ref} for ref in reference_stars],
                 placeholder='Select one or more reference stars',
                 multi=False)
])

# -----------
# Inst Select
# -----------
combinations = []
for i in range(n_combo):
    if i == 0:
        disp = 'block'
        button = 'Remove Instrument Combination'
    else:
        disp = 'none'
        button = 'Add Instrument Combination'
    combinations.append(
        html.Div(id=f'instrument-combination-{i}',
                 children=create_combination_selection(i),
                 style={'display': disp}),
    )
    combinations.append(
        html.Button(id=f'combo-button-{i}',
                    children=button,
                    style={'display': 'block',
                           'width': '50%',
                           'margin': 'auto',
                           'marginTop': 10}),
    )
    combinations.append(html.Hr())
combo_init = html.Div(combinations)

# -----------
# Results
# -----------
crlb_init = html.Div([
    # ............
    # Summary
    # ............
    html.Div([
        html.H4('Reference Star Summary:'),
        dash_table.DataTable(id='ref-table',
                             columns=[{'name': i, 'id': i} for i
                                      in ['name', 'Teff', 'logg', 'v_micro',
                                          '[Fe/H]', '[alpha/Fe]']],
                             sort_action='native',
                             sort_by=[{'column_id': 'name',
                                       'direction': 'asc'}],
                             style_as_list_view=True,
                             style_cell={'minWidth': f'{45/6}%', 'width': f'{45/6}%', 'maxWidth': f'{45/6}%'},
                             style_header={'backgroundColor': 'white',
                                           'fontWeight': 'bold'},
                             ),
        html.H4('Instrument Summary:'),
        dash_table.DataTable(id='inst-table',
                             columns=[{'name': i, 'id': i} for i
                                      in ['name', 'combo',
                                          'start', 'end',
                                          'R_res', 'R_samp']],
                             sort_action='native',
                             sort_by=[{'column_id': 'combo',
                                       'direction': 'asc'}],
                             style_as_list_view=True,
                             style_cell={'minWidth': f'{45/5}%', 'width': f'{45/5}%', 'maxWidth': f'{45/5}%'},
                             style_header={'backgroundColor': 'white',
                                           'fontWeight': 'bold'},
                             ),
    ], style={'display': 'inline-block',
              'width': '45%'}),
    # ............
    # Priors
    # ............
    html.Div([
        html.H4('Priors:',
                style={'paddingTop': '0px',
                       'height': '100%'}),
        dash_table.DataTable(id='prior-table',
                             columns=[{'name': i, 'id': i, 'editable': (i == 'Priors')}
                                      for i
                                      in ['Labels', 'Priors']],
                             data=[{'Labels': 'Teff', 'Priors': ''},
                                   {'Labels': 'log(g)', 'Priors': ''},
                                   {'Labels': '[Fe/H]', 'Priors': ''}],
                             tooltip_data=[{'Labels': '', 'Priors': prior_tooltips[i], 'type':'text'}
                                           for i in range(len(prior_tooltips))],
                             tooltip_delay=0,
                             style_as_list_view=True,
                             style_cell={'minWidth': '100px', 'width': '100px', 'maxWidth': '100px'},
                             style_header={'backgroundColor': 'white',
                                           'fontWeight': 'bold'},
                             )
    ], style={'display': 'inline-block',
              'width': '45%',
              'verticalAlign': 'Top',
              'paddingLeft': '50px'}),
    html.Div([
        html.Button(id='crlb-button',
                    disabled=True,
                    children='No instrument or star provided',
                    style={'width': '100%',
                           'color': '#ff8080',
                           'borderColor': '#ff8080'}),

    ], style={'display': 'block',
              'width': '50%',
              'margin': '30px auto'}),
    # ............
    # Results Tabs
    # ............
    dcc.Tabs(id='crlb-tabs', value='crlb',
             children=[
                 # CRLB Tab #
                 dcc.Tab(label='CRLB', value='crlb',
                         children=[
                             dcc.Graph(id='crlb-graph',
                                       style={'height': 500,
                                              'width': '100%'},
                                       figure=go.Figure(data=[], layout=crlb_plot_layout)),
                             html.Label(id='crlb-table-label',
                                        children='Cramer-Rao Lower Bounds:',
                                        style={'display': 'none'}),
                             dash_table.DataTable(id='crlb-table',
                                                  sort_action='native',
                                                  style_as_list_view=True),
                         ]),
                 # Grad Tab #
                 dcc.Tab(label='Gradients', value='grad',
                         children=[
                             html.Div('', style={'marginTop': 25}),
                             html.Label('Instruments:'),
                             dcc.Dropdown(id='grad-graph-inst',
                                          placeholder='Select instruments to plot',
                                          options=[],
                                          multi=True,
                                          ),
                             html.Label('Stellar Labels:'),
                             dcc.Dropdown(id='grad-graph-labels',
                                          placeholder='Select elements to plot',
                                          options=[{'label': lab, 'value': lab} for lab in lab_tab_text],
                                          multi=True,
                                          value=['Fe']
                                          ),
                             dcc.Graph(id='grad-graph',
                                       style={'height': '100%',
                                              'width': '100%'},
                                       #figure=go.Figure(data=grad_plot_data, layout=grad_plot_layout)),
                                       figure=fig),
                             html.Div('', style={'marginTop': 25}),

                         ]),
             ], style={'margin': 0,
                       'padding': 0}),
], style={'display': 'block'})

