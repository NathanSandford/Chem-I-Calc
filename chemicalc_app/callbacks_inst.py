import pandas as pd
import numpy as np

from .server import app
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from .layout import n_inst, n_combo
from .utils import load_preset_specs

from chemicalc.s2n import keck_options

for i in range(n_combo):
    @app.callback(
        [Output(f'combo-button-{i}', 'children'),
         Output(f'instrument-combination-{i}', 'style')],
        [Input(f'combo-button-{i}', 'n_clicks')],
        [State(f'combo-button-{i}', 'children'),
         State(f'combo-button-{i}', 'id')
         ])
    def add_remove_combinations(n_clicks, button, combo_id):
        combo_number = int(combo_id[-1])
        if not n_clicks:
            new_button = button
            if combo_number == 0:
                new_inst_style = {'display': 'block'}
            else:
                new_inst_style = {'display': 'None'}
        else:
            if button == 'Remove Instrument Combination':
                new_button = 'Add Instrument Combination'
                new_inst_style = {'display': 'None'}
            else:
                new_button = 'Remove Instrument Combination'
                new_inst_style = {'display': 'block'}
        return new_button, new_inst_style


    for j in range(n_inst):
        @app.callback(
            [Output(f'inst-button-{i}-{j}', 'children'),
             Output(f'instrument-selection-{i}-{j}', 'style')],
            [Input(f'inst-button-{i}-{j}', 'n_clicks')],
            [State(f'inst-button-{i}-{j}', 'children'),
             State(f'inst-button-{i}-{j}', 'id')
             ])
        def add_remove_instrument(n_clicks, button, inst_id):
            inst_number = int(inst_id[-1])
            if not n_clicks:
                new_button = button
                if inst_number == 0:
                    new_inst_style = {'display': 'inline-block'}
                else:
                    new_inst_style = {'display': 'None'}
            else:
                if button == 'Remove':
                    new_button = 'Add'
                    new_inst_style = {'display': 'None'}
                else:
                    new_button = 'Remove'
                    new_inst_style = {'display': 'inline-block'}
            return new_button, new_inst_style

        @app.callback(
            [Output(f'instname-{i}-{j}', 'value'),
             Output(f'wavemin-{i}-{j}', 'value'),
             Output(f'wavemax-{i}-{j}', 'value'),
             Output(f'res-{i}-{j}', 'value'),
             Output(f'samp-{i}-{j}', 'value'),
             ],
            [Input(f'preset-{i}-{j}', 'value')]
        )
        def load_preset(preset_name):
            if not preset_name:
                return '', '', '', '', ''
            if preset_name:
                inst = load_preset_specs(preset_name)
                return (inst['name'],
                        inst['start_wavelength'], inst['end_wavelength'],
                        inst['R_res'], inst['R_samp'])

        @app.callback(
            [Output(f'constant-snr-{i}-{j}', 'style'),
             Output(f'etc-dropdown-{i}-{j}', 'style'),
             Output(f'wmko-inst-{i}-{j}', 'style'),
             Output(f'wmko-grating-{i}-{j}', 'style'),
             Output(f'wmko-grism-{i}-{j}', 'style'),
             Output(f'wmko-slitwidth-{i}-{j}', 'style'),
             Output(f'wmko-dichroic-{i}-{j}', 'style'),
             Output(f'wmko-binning-{i}-{j}', 'style'),
             Output(f'wmko-cwave-{i}-{j}', 'style'),
             Output(f'wmko-exptime-{i}-{j}', 'style'),
             Output(f'wmko-mag-{i}-{j}', 'style'),
             Output(f'wmko-magtype-{i}-{j}', 'style'),
             Output(f'wmko-filter-{i}-{j}', 'style'),
             Output(f'wmko-star-{i}-{j}', 'style'),
             Output(f'wmko-airmass-{i}-{j}', 'style'),
             Output(f'wmko-seeing-{i}-{j}', 'style'),
             ],
            [Input(f'snr-type-{i}-{j}', 'value'),
             Input(f'etc-dropdown-{i}-{j}', 'value'),
             Input(f'wmko-inst-{i}-{j}', 'value')]
        )
        def create_snr_input(snr_type, etc, inst):
            basic_show = {'textAlign': 'Left',
                          'marginRight': 10,
                          'display': 'block'}
            basic_hide = {'textAlign': 'Left',
                          'marginRight': 10,
                          'display': 'None'}

            constant_snr_style = {'display': 'None'}
            etc_dropdown_style = basic_hide
            wmko_inst_style = basic_hide
            wmko_grating_style = basic_hide
            wmko_grism_style = basic_hide
            wmko_slitwidth_style = basic_hide
            wmko_dichroic_style = basic_hide
            wmko_binning_style = basic_hide
            wmko_cwave_style = basic_hide
            wmko_exptime_style = {'display': 'None'}
            wmko_mag_style = {'display': 'None'}
            wmko_magtype_style = basic_hide
            wmko_filter_style = basic_hide
            wmko_star_style = basic_hide
            wmko_airmass_style = {'display': 'None'}
            wmko_seeing_style = {'display': 'None'}
            if snr_type == "Constant":
                constant_snr_style = {'display': 'block'}
            elif snr_type == "From ETC":
                etc_dropdown_style = basic_show
                if etc == "WMKO":
                    wmko_inst_style = basic_show
                    if inst == 'deimos':
                        wmko_grating_style = basic_show
                        wmko_slitwidth_style = basic_show
                        wmko_binning_style = basic_show
                        wmko_cwave_style = basic_show
                        wmko_exptime_style = basic_show
                        wmko_mag_style = basic_show
                        wmko_magtype_style = basic_show
                        wmko_filter_style = basic_show
                        wmko_star_style = basic_show
                        wmko_airmass_style = basic_show
                        wmko_seeing_style = basic_show
                    elif inst == 'lris':
                        wmko_grating_style = basic_show
                        wmko_grism_style = basic_show
                        wmko_slitwidth_style = basic_show
                        wmko_dichroic_style = basic_show
                        wmko_binning_style = basic_show
                        wmko_exptime_style = basic_show
                        wmko_mag_style = basic_show
                        wmko_magtype_style = basic_show
                        wmko_filter_style = basic_show
                        wmko_star_style = basic_show
                        wmko_airmass_style = basic_show
                        wmko_seeing_style = basic_show
                    elif inst == 'hires':
                        wmko_slitwidth_style = basic_show
                        wmko_binning_style = basic_show
                        wmko_exptime_style = basic_show
                        wmko_mag_style = basic_show
                        wmko_magtype_style = basic_show
                        wmko_filter_style = basic_show
                        wmko_star_style = basic_show
                        wmko_airmass_style = basic_show
                        wmko_seeing_style = basic_show
            return (constant_snr_style, etc_dropdown_style, wmko_inst_style,
                    wmko_grating_style, wmko_grism_style, wmko_slitwidth_style,
                    wmko_dichroic_style, wmko_binning_style, wmko_cwave_style,
                    wmko_exptime_style, wmko_mag_style, wmko_magtype_style,
                    wmko_filter_style, wmko_star_style, wmko_airmass_style,
                    wmko_seeing_style)

        @app.callback(
            [Output(f'wmko-grating-{i}-{j}', 'options'),
             Output(f'wmko-grating-{i}-{j}', 'value'),
             Output(f'wmko-slitwidth-{i}-{j}', 'options'),
             Output(f'wmko-slitwidth-{i}-{j}', 'value'),
             Output(f'wmko-binning-{i}-{j}', 'options'),
             Output(f'wmko-binning-{i}-{j}', 'value'),
             ],
            [Input(f'wmko-inst-{i}-{j}', 'value')]
        )
        def set_wmko_inst_options(inst):
            if inst == 'deimos':
                grating_options = [{'label': opt, 'value': opt} for opt in keck_options['grating (DEIMOS)']]
                slitwidth_options = [{'label': opt, 'value': opt} for opt in keck_options['slitwidth (DEIMOS)']]
                binning_options = [{'label': opt, 'value': opt} for opt in keck_options['binning (DEIMOS)']]
            elif inst == 'lris':
                grating_options = [{'label': opt, 'value': opt} for opt in keck_options['grating (LRIS)']]
                slitwidth_options = [{'label': opt, 'value': opt} for opt in keck_options['slitwidth (LRIS)']]
                binning_options = [{'label': opt, 'value': opt} for opt in keck_options['binning (LRIS)']]
            elif inst == 'hires':
                grating_options = []
                slitwidth_options = [{'label': opt, 'value': opt} for opt in keck_options['slitwidth (HIRES)']]
                binning_options = [{'label': opt, 'value': opt} for opt in keck_options['binning (HIRES)']]
            else:
                grating_options, slitwidth_options, binning_options = [], [], []
            return grating_options, None, slitwidth_options, None, binning_options, None


inst_states = []
for i in range(n_combo):
    for j in range(n_inst):
        inst_states.append(Input(f'combo-button-{i}', 'children'))
        inst_states.append(Input(f'inst-button-{i}-{j}', 'children'))
        inst_states.append(Input(f'instname-{i}-{j}', 'value')),
        inst_states.append(Input(f'comboname-{i}', 'value')),
        inst_states.append(Input(f'wavemin-{i}-{j}', 'value')),
        inst_states.append(Input(f'wavemax-{i}-{j}', 'value')),
        inst_states.append(Input(f'res-{i}-{j}', 'value')),
        inst_states.append(Input(f'samp-{i}-{j}', 'value'))


@app.callback(
    Output('inst-store', 'data'),
    inst_states
)
def update_inst_store(*inputs):
    combo_active = [True if state[:6] == 'Remove'
                    else False
                    for state in inputs[0::8]]
    inst_active = [True if state[:6] == 'Remove'
                   else False
                   for state in inputs[1::8]]

    df = pd.DataFrame(index=inputs[2::8])
    df['name'] = inputs[2::8]
    df['combo'] = inputs[3::8]
    df['active'] = [True if combo_active[i] and inst_active[i]
                    else False
                    for i in range(n_combo*n_inst)]
    df['start'] = inputs[4::8]
    df['end'] = inputs[5::8]
    df['R_res'] = inputs[6::8]
    df['R_samp'] = inputs[7::8]
    active_df = df[df['active']]
    final_df = active_df.replace(to_replace=['None', ''], value=np.nan).dropna()

    return final_df.to_dict('records')


@app.callback(
    Output('inst-table', 'data'),
    [Input('inst-store', 'data')]
)
def update_inst_table(data):
    if data is None:
        raise PreventUpdate
    data = [dict(t) for t in {tuple(d.items()) for d in data}]
    return data


snr_states = [Input('inst-store', 'data')]
for i in range(n_combo):
    for j in range(n_inst):
        snr_states.append(Input(f'combo-button-{i}', 'children'))
        snr_states.append(Input(f'inst-button-{i}-{j}', 'children'))
        snr_states.append(Input(f'instname-{i}-{j}', 'value'))
        snr_states.append(Input(f'comboname-{i}', 'value'))
        snr_states.append(Input(f'snr-type-{i}-{j}', 'value'))
        snr_states.append(Input(f'snr-const-value-{i}-{j}', 'value'))
        snr_states.append(Input(f'etc-dropdown-{i}-{j}', 'value'))
        snr_states.append(Input(f'wmko-inst-{i}-{j}', 'value'))
        snr_states.append(Input(f'wmko-grating-{i}-{j}', 'value'))
        snr_states.append(Input(f'wmko-grism-{i}-{j}', 'value'))
        snr_states.append(Input(f'wmko-slitwidth-{i}-{j}', 'value'))
        snr_states.append(Input(f'wmko-dichroic-{i}-{j}', 'value'))
        snr_states.append(Input(f'wmko-binning-{i}-{j}', 'value'))
        snr_states.append(Input(f'wmko-cwave-{i}-{j}', 'value'))
        snr_states.append(Input(f'wmko-exptime-value-{i}-{j}', 'value'))
        snr_states.append(Input(f'wmko-mag-value-{i}-{j}', 'value'))
        snr_states.append(Input(f'wmko-magtype-{i}-{j}', 'value'))
        snr_states.append(Input(f'wmko-filter-{i}-{j}', 'value'))
        snr_states.append(Input(f'wmko-star-{i}-{j}', 'value'))
        snr_states.append(Input(f'wmko-airmass-value-{i}-{j}', 'value'))
        snr_states.append(Input(f'wmko-seeing-value-{i}-{j}', 'value'))


@app.callback(
    [Output('snr-missing', 'children'),
     Output('snr-store', 'data')],
    snr_states
)
def check_snr_states(data, *inputs):
    if data is None:
        raise PreventUpdate
    combo_active = np.array([True if state[:6] == 'Remove'
                             else False
                             for state in inputs[0::21]])
    inst_active = np.array([True if state[:6] == 'Remove'
                            else False
                            for state in inputs[1::21]])
    active_instruments = combo_active * inst_active

    df = pd.DataFrame()
    df['name'] = inputs[2::21]
    df['combo'] = inputs[3::21]
    df['active'] = active_instruments
    df['type'] = inputs[4::21]
    df['const'] = inputs[5::21]
    df['etc'] = inputs[6::21]
    df['inst'] = inputs[7::21]
    df['grating'] = inputs[8::21]
    df['grism'] = inputs[9::21]
    df['slitwidth'] = inputs[10::21]
    df['dichroic'] = inputs[11::21]
    df['binning'] = inputs[12::21]
    df['cwave'] = inputs[13::21]
    df['exptime'] = inputs[14::21]
    df['mag'] = inputs[15::21]
    df['magtype'] = inputs[16::21]
    df['bandpass'] = inputs[17::21]
    df['star'] = inputs[18::21]
    df['airmass'] = inputs[19::21]
    df['seeing'] = inputs[20::21]
    active_df = df[df['active']]

    wmko_snr_requirements = ['slitwidth', 'binning', 'exptime', 'mag', 'magtype',
                             'bandpass', 'star', 'airmass', 'seeing']
    deimos_snr_requirements = wmko_snr_requirements + ['grating', 'cwave']
    lris_snr_requirements = wmko_snr_requirements + ['grating', 'grism', 'dichroic']
    hires_snr_requirements = wmko_snr_requirements
    missing_snr = 0
    for i in active_df.index:
        if active_df.loc[i, 'type'] == 'Constant':
            missing_snr += active_df['const'] is None
        elif active_df.loc[i, 'type'] == 'From ETC':
            if active_df.loc[i, 'inst'] == 'deimos':
                missing_snr += np.any(active_df.loc[i, deimos_snr_requirements].values == None)
            elif active_df.loc[i, 'inst'] == 'lris':
                missing_snr += np.any(active_df.loc[i, lris_snr_requirements].values == None)
            elif active_df.loc[i, 'inst'] == 'hires':
                missing_snr += np.any(active_df.loc[i, hires_snr_requirements].values == None)
            else:
                missing_snr += True
        else:
            missing_snr += True
    if missing_snr:
        snr_store = []
    else:
        snr_store = active_df.to_dict('records')
    return missing_snr, snr_store

