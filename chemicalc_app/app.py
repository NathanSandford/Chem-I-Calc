from .server import app, server

import dash_html_components as html
import dash_core_components as dcc

from .layout import header, reference_star_select, combo_init, crlb_init

app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>Chem-I-Calc</title>
        {%favicon%}
        {%css%}
    </head>
    <body>
        <div>If the page fails to load, try refreshing.</div>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

app.layout = html.Div([
    header,
    html.Div('', style={'marginTop': 75}),
    html.Hr(),
    html.Details([
        html.Summary('1. Select Reference Star', style={'fontSize': 30}),
        reference_star_select
    ], open=True),
    html.Hr(),
    html.Details([
        html.Summary('2. Set Instrument Specifications', style={'fontSize': 30}),
        combo_init
    ], open=True),
    html.Hr(),
    html.Details([
        html.Summary('3. Calculate Cramer-Rao Bounds', style={'fontSize': 30}),
        crlb_init
    ], open=True),
    html.Hr(),
    html.Div('', style={'marginTop': 25}),

    # Storage Divs
    dcc.Store(id='ref-store', storage_type='memory'),
    dcc.Store(id='inst-store', storage_type='memory'),
    dcc.Store(id='crlb-store', storage_type='memory'),
    dcc.Store(id='snr-store', storage_type='memory'),
    html.Div(id='crlb-complete', children=0, style=dict(display='none')),
    html.Div(id='grad-key-div', children=[], style=dict(display='none')),
    html.Div(id='snr-key-div', children=[], style=dict(display='none')),
    html.Div(id='snr-missing', children=0, style=dict(display='none')),
])
