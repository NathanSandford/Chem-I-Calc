from .server import app
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from chemicalc.utils import data_dir

import pandas as pd


@app.callback(
    Output('ref-store', 'data'),
    [Input('reference-select', 'value')]
)
def update_ref_store(refs):
    name = []
    Teff = []
    logg = []
    v_micro = []
    FeH = []
    alphaFe = []
    if not refs:
        return pd.DataFrame().to_dict('records')
    elif type(refs) != list:
        refs = [refs]
    for reference in refs:
        label_df = pd.read_hdf(data_dir.joinpath('reference_labels.h5'), reference)
        name.append(reference)
        Teff.append(label_df.loc['Teff', 'aaaaa'])
        logg.append(label_df.loc['logg', 'aaaaa'])
        v_micro.append(label_df.loc['v_micro', 'aaaaa'])
        FeH.append(label_df.loc['Fe', 'aaaaa'])
        alphaFe.append(label_df.loc['Mg', 'aaaaa'] - label_df.loc['Fe', 'aaaaa'])
    df = pd.DataFrame()
    df['name'] = name
    df['Teff'] = Teff
    df['logg'] = logg
    df['v_micro'] = v_micro
    df['[Fe/H]'] = FeH
    df['[alpha/Fe]'] = alphaFe
    return df.to_dict('records')


@app.callback(
    Output('ref-table', 'data'),
    [Input('ref-store', 'data')]
)
def update_ref_table(data):
    if data is None:
        raise PreventUpdate
    return data
