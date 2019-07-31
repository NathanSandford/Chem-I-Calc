from flask import Flask
from dash import Dash

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

server = Flask('chemicalc_app')
app = Dash(server=server, external_stylesheets=external_stylesheets)

app.config['suppress_callback_exceptions'] = False