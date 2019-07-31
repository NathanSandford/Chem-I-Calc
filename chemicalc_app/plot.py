import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
import plotly.graph_objs as go
from plotly.subplots import make_subplots


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


def plotly_grads_snr(grad_dict, snr_dict, grad_names, snr_names, labels):
    c1 = plt.cm.get_cmap('viridis', len(labels)*len(grad_names))
    linecycler = cycle(['solid', 'dash', 'dashdot', 'dot'])

    fig = make_subplots(rows=2, cols=1, vertical_spacing=0, shared_xaxes=True)
    annotations = []
    shapes = []
    n = 0
    for i, lab in enumerate(labels):
        for j, name in enumerate(grad_names):
            fig.append_trace(go.Scatter(
                name=f'{name} ({lab})',
                x=grad_dict[name].index,
                y=grad_dict[name][lab],
                text=f'{name} ({lab})',
                hoverinfo='y+x+text',
                mode='lines',
                line={'color': f'rgba{c1(i+j)[:3] + (0.6,)}',
                      'dash': 'solid'}),
                row=1, col=1)
            snr_names2 = [snr_names[l] for l in range(len(snr_names)) if grad_names[j] in snr_names[l]]
            for k, snr_name in enumerate(snr_names2):
                print(snr_names2, snr_name)
                line = next(linecycler)
                fig.append_trace(go.Scatter(
                    name=snr_name,
                    x=snr_dict[snr_name].index,
                    y=snr_dict[snr_name].T.values[0],
                    showlegend=False,
                    text=snr_name,
                    hoverinfo='y+x+text',
                    mode='lines',
                    line={'color': f'rgba{c1(i+j)[:3] + (0.6,)}',
                          'dash': line}),
                    row=2, col=1)
                annotations.append(
                    go.layout.Annotation(
                        x=0.90, y=0.01+0.05*n,
                        showarrow=False,
                        text=snr_name,
                        font=dict(
                            color=f'rgba{c1(i+j)}',
                            size=16
                        ),
                        xanchor='right',
                        yanchor='bottom',
                        xref='paper', yref='paper')
                    )
                shapes.append(
                    go.layout.Shape(
                        type="line",
                        xref="paper",
                        yref="paper",
                        x0=0.92, x1=0.99,
                        y0=0.01 + 0.05 * k, y1=0.01 + 0.05*n,
                        line=dict(
                            color=f'rgba{c1(i+j)}',
                            dash=line,
                        )
                    )
                )
                n += 1

    fig.update_layout(annotations=annotations,
                      width=1000, height=800,
                      margin=dict(l=75, r=50, t=30, b=50),
                      legend=dict(x=0.99, y=0.51,
                                  xanchor='right',
                                  yanchor='bottom',
                                  font={'size': 16}),
                      yaxis=dict(title=dict(text='df/dX',
                                            font=dict(size=24)),
                                 ticks='inside',
                                 showline=True,
                                 linewidth=2,
                                 linecolor='black',
                                 gridcolor='rgba(0,0,0,0.25)',
                                 zeroline=False,
                                 mirror='ticks',
                                 hoverformat=".4f"),
                      yaxis2=dict(title=dict(text='SNR (pixel<sup>-1</sup>)',
                                             font=dict(size=24)),
                                  ticks='inside',
                                  showline=True,
                                  linewidth=2,
                                  linecolor='black',
                                  gridcolor='rgba(0,0,0,0.25)',
                                  zeroline=False,
                                  mirror='ticks',
                                  hoverformat=".1f"),
                      xaxis=dict(
                          ticks='inside',
                          tickangle=0,
                          showline=True,
                          linewidth=2,
                          linecolor='black',
                          gridcolor='rgba(0,0,0,0.25)',
                          zeroline=False,
                          mirror='ticks',
                          hoverformat=".0f"),
                      xaxis2=dict(title=dict(text=u'Wavelength (\u212B)',
                                             font=dict(size=24)),
                                  ticks='inside',
                                  tickangle=0,
                                  showline=True,
                                  linewidth=2,
                                  linecolor='black',
                                  gridcolor='rgba(0,0,0,0.25)',
                                  zeroline=False,
                                  mirror='ticks',
                                  hoverformat=".0f"),
                      plot_bgcolor='white')
    return fig
