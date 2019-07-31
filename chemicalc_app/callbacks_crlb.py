from .server import app
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from dash import callback_context

from chemicalc import reference_spectra as ref
from chemicalc.instruments import InstConfig
from chemicalc import s2n
from chemicalc.utils import calc_crlb, sort_crlb
from .plot import plotly_crlb, plotly_grads_snr

from .utils import tmp_data_dir

import pandas as pd


@app.callback(
    [Output('crlb-button', 'disabled'),
     Output('crlb-button', 'children'),
     Output('crlb-button', 'style'),
     ],
    [Input('ref-store', 'data'),
     Input('inst-store', 'data'),
     Input('snr-missing', 'children'),
     Input('crlb-button', 'n_clicks'),
     Input('crlb-complete', 'children')
     ]
)
def check_crlb_button(ref_data, inst_data, snr_missing, n_clicks, flag):
    if ref_data and inst_data:
        inst_name_conflict = 0
        name_set = {tmp['name'] for tmp in inst_data}
        for name in name_set:
            config_set = {f"{tmp['start']}-{tmp['end']}_R{tmp['R_res']}({tmp['R_samp']})"
                          for tmp
                          in inst_data
                          if tmp['name'] == name}
            if len(config_set) != 1:
                inst_name_conflict += 1
        if inst_name_conflict:
            children = 'Instrument Name Conflict'
            style = {'width': '100%',
                     'color': '#ff8080',
                     'borderColor': '#ff8080'}
            disabled = True
        elif snr_missing:
            children = 'Instrument Missing SNR'
            style = {'width': '100%',
                     'color': '#ff8080',
                     'borderColor': '#ff8080'}
            disabled = True
        else:
            children = 'Calculate CRLB!'
            style = {'width': '100%',
                     'borderWidth': '3px',
                     'fontSize': 16,
                     'color': 'green',
                     'borderColor': 'green'}
            disabled = False
    elif ref_data and not inst_data:
        children = 'No instrument provided'
        style = {'width': '100%',
                 'color': '#ff8080',
                 'borderColor': '#ff8080'}
        disabled = True
    elif inst_data and not ref_data:
        children = 'No  star provided'
        style = {'width': '100%',
                 'color': '#ff8080',
                 'borderColor': '#ff8080'}
        disabled = True
    else:
        children = 'No instrument or star provided'
        style = {'width': '100%',
                 'color': '#ff8080',
                 'borderColor': '#ff8080'}
        disabled = True

    context = callback_context.triggered[0]['prop_id'].split('.')[0]
    if context == 'crlb-complete':
        if n_clicks:
            if flag == 1:
                children = 'Calculate CRLB!'
                style = {'width': '100%',
                         'borderWidth': '3px',
                         'fontSize': 16,
                         'color': 'green',
                         'borderColor': 'green'}
                disabled = False
        else:
            children = 'No instrument or star provided'
            style = {'width': '100%',
                     'color': '#ff8080',
                     'borderColor': '#ff8080'}
            disabled = True
    elif context == 'crlb-button':
        if n_clicks:
            children = 'Calculating CRLB...'
            style = {'width': '100%',
                     'borderWidth': '3px',
                     'fontSize': 16,
                     'color': 'green',
                     'borderColor': 'green'}
            disabled = True
        else:
            children = 'No instrument or star provided'
            style = {'width': '100%',
                     'color': '#ff8080',
                     'borderColor': '#ff8080'}
            disabled = True

    return disabled, children, style


@app.callback(
    [Output('crlb-store', 'data'),
     Output('crlb-complete', 'children'),
     Output('grad-key-div', 'children'),
     Output('snr-key-div', 'children')
     ],
    [Input('crlb-button', 'n_clicks')],
    [State('inst-store', 'data'),
     State('ref-store', 'data'),
     State('prior-table', 'data'),
     State('snr-store', 'data')
     ]
)
def calculate_crlb(n_clicks, inst_data, ref_data, prior_data, snr_data):
    if not n_clicks:
        raise PreventUpdate
    print('Calculating CRLBs')
    if all(tmp['Priors'] == '' for tmp in prior_data):
        priors = None
    else:
        priors = {}
        for i, label in enumerate(['Teff', 'logg', 'Fe']):
            prior = prior_data[i]['Priors']
            priors[label] = float(prior) if prior != '' else None
    snr_names = []
    # max_res = max([inst_dict['R_res'] * inst_dict['R_samp'] for inst_dict in inst_data])
    for i, ref_dict in enumerate(ref_data):
        ref_name = ref_dict['name']
        star = ref.ReferenceSpectra(reference=ref_name, res='high')
        if i == 0:
            crlb_df = pd.DataFrame(index=star.labels.index)
            combo_dict = {tmp['combo']: [] for tmp in inst_data}
        for j, inst_dict in enumerate(inst_data):
            inst = InstConfig(name=inst_dict['name'],
                              start=inst_dict['start'],
                              end=inst_dict['end'],
                              res=inst_dict['R_res'],
                              samp=inst_dict['R_samp'])
            snr_dict = snr_data[j]
            if snr_dict['type'] == "Constant":
                snr = snr_dict['const']
            elif snr_dict['type'] == 'From ETC':
                if snr_dict['etc'] == 'WMKO':
                    if snr_dict['inst'] == 'deimos':
                        snr = s2n.Sig2NoiseDEIMOS(grating=snr_dict['grating'], exptime=snr_dict['exptime'],
                                                  mag=snr_dict['mag'], magtype=snr_dict['magtype'],
                                                  band=snr_dict['bandpass'], template=snr_dict['star'],
                                                  cwave=snr_dict['cwave'], slitwidth=snr_dict['slitwidth'],
                                                  binning=snr_dict['binning'], airmass=snr_dict['airmass'],
                                                  seeing=snr_dict['seeing'], redshift=0)
                    elif snr_dict['inst'] == 'lris':
                        snr = s2n.Sig2NoiseLRIS(grating=snr_dict['grating'], grism=snr_dict['grism'],
                                                dichroic=snr_dict['dichroic'], exptime=snr_dict['exptime'],
                                                mag=snr_dict['mag'], magtype=snr_dict['magtype'],
                                                band=snr_dict['bandpass'], template=snr_dict['star'],
                                                slitwidth=snr_dict['slitwidth'], binning=snr_dict['binning'],
                                                airmass=snr_dict['airmass'], seeing=snr_dict['seeing'], redshift=0)
                    elif snr_dict['inst'] == 'hires':
                        snr = s2n.Sig2NoiseHIRES(slitwidth=snr_dict['slitwidth'], exptime=snr_dict['exptime'],
                                                 mag=snr_dict['mag'], magtype=snr_dict['magtype'],
                                                 band=snr_dict['bandpass'], template=snr_dict['star'],
                                                 binning=snr_dict['binning'], airmass=snr_dict['airmass'],
                                                 seeing=snr_dict['seeing'], redshift=0)
                    else:
                        snr = 0
                else:
                    snr = 0
            else:
                snr = 0
            inst.set_snr(snr)
            snr_df = pd.DataFrame(inst.snr, index=inst.wave)
            snr_df.to_hdf(tmp_data_dir.joinpath('tmp_snr.h5'),
                          key=f"{inst_dict['name']} ({inst_dict['combo']})", mode='a')
            snr_names.append(f"{inst_dict['name']} ({inst_dict['combo']})")

            combo_dict[inst_dict['combo']].append(inst)

            star.convolve(inst)
            star.calc_gradient(inst.name, symmetric=True, ref_included=True, v_micro_scaling=1)
        for combo in combo_dict:
            crlb_df[combo] = calc_crlb(star, combo_dict[combo], priors=priors).round(3)

    crlb_df = sort_crlb(crlb_df, cutoff=0.3, sort_by='default')
    crlb_df.replace(1000, 0, inplace=True)
    crlb_df['Labels'] = crlb_df.index

    grad_names = star.get_names()
    grad_names.remove('init')
    for name in grad_names:
        if name == 'init':
            continue
        star.gradients[name].T.to_hdf(tmp_data_dir.joinpath('tmp_grad.h5'), key=name, mode='a')

    return crlb_df.to_dict('records'), 1, grad_names, snr_names


@app.callback(
    [Output('crlb-table', 'data'),
     Output('crlb-table', 'columns'),
     Output('crlb-table-label', 'style')
     ],
    [Input('crlb-store', 'data')]
)
def update_crlb_table(data):
    if data is None:
        raise PreventUpdate
    new_columns = [{'name': 'Labels', 'id': 'Labels'}]
    new_columns += [{'name': i, 'id': i} for i
                    in list(data[0].keys())
                    if i != 'Labels']
    style = {'display': 'block'}
    return data, new_columns, style


@app.callback(
    Output('crlb-graph', 'figure'),
    [Input('crlb-store', 'data')]
)
def update_crlb_graph(data):
    if data is None:
        raise PreventUpdate
    crlb = pd.DataFrame(data)
    crlb.set_index('Labels', drop=True, inplace=True)
    crlb.index = ['T<sub>eff</sub> (100 K)', 'log(g)', 'v<sub>micro</sub> (km/s)'] + list(crlb.index[3:])
    fig = plotly_crlb(crlb)
    return fig


@app.callback(
    Output('grad-graph', 'figure'),
    [Input('grad-graph-inst', 'value'),
     Input('snr-key-div', 'children'),
     Input('grad-graph-labels', 'value')]
)
def update_grad_graph(grad_names, snr_names, labels):
    if not grad_names:
        raise PreventUpdate
    grad_dict = {}
    snr_dict = {}
    for name in grad_names:
        grad_dict[name] = pd.read_hdf(tmp_data_dir.joinpath('tmp_grad.h5'), key=name)
    for name in snr_names:
        snr_dict[name] = pd.read_hdf(tmp_data_dir.joinpath('tmp_snr.h5'), key=name)

    fig = plotly_grads_snr(grad_dict, snr_dict, grad_names, snr_names, list(labels))
    return fig


@app.callback(
    [Output('grad-graph-inst', 'options'),
     Output('grad-graph-inst', 'value')],
    [Input('grad-key-div', 'children')]
)
def update_grad_graph_inst(names):
    if not names:
        raise PreventUpdate
    return [{'label': name, 'value': name} for name in names], names
