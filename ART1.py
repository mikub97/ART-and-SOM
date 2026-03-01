import dash
from dash import dcc, html, Input, Output, State, ctx
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# -----------------------------------------------------------------------------
# ORIGINAL ART1 IMPLEMENTATION
# -----------------------------------------------------------------------------
original_code_string = """class ART1:
    ''' ART class '''

    def __init__(self, n=5, m=10, rho=.5):
        # Comparison layer
        self.F1 = np.ones(n)
        # Recognition layer
        self.F2 = np.ones(m)
        # Feed-forward weights
        self.Wf = np.zeros((m,n))
        # Feed-back weights
        self.Wb = np.ones((n,m))
        # Vigilance
        self.rho = rho
        # Number of active units in F2
        self.active = 0

    def learn(self, X):
        ''' Learn X '''
        # Compute F2 output and sort them (I)
        self.F2[...] = np.dot(self.Wf, X)
        I = np.argsort(self.F2[:self.active].ravel())[::-1]

        for i in I:
            # Check if nearest memory is above the vigilance level
            d = (self.Wb[:,i] * X).sum() / X.sum()
            if d >= self.rho:
                # Learn data
                self.Wb[:,i] *= X
                self.Wf[i,:] = self.Wb[:,i]/(0.5+self.Wb[:,i].sum())
                return self.Wb[:,i], i

        # No match found, increase the number of active units
        # and make the newly active unit to learn data
        if self.active < self.F2.size:
            i = self.active
            self.Wb[:,i] *= X
            self.Wf[i,:] = self.Wb[:,i]/(0.5+self.Wb[:,i].sum())
            self.active += 1
            return self.Wb[:,i], i

        return None,None"""

# Data from Example 1
simple_data_strings = [
    "   O ",
    "  O O",
    "    O",
    "  O O",
    "    O",
    "  O O",
    "    O",
    " OO O",
    " OO  ",
    " OO O"
]

simple_samples = []
for d in simple_data_strings:
    X = np.zeros(len(d))
    for j in range(len(d)):
        X[j] = (d[j] == 'O')
    simple_samples.append(X)

# -----------------------------------------------------------------------------
# GLOBAL STATE MANAGER
# -----------------------------------------------------------------------------
class AppState:
    def __init__(self):
        self.reset()

    def reset(self):
        self.input_idx = 0
        self.n_inputs = 5
        self.m_categories = 10

        self.Wf = np.zeros((self.m_categories, self.n_inputs))
        self.Wb = np.ones((self.n_inputs, self.m_categories))
        self.F2 = np.zeros(self.m_categories)
        self.active = 0

        self.phase = "INIT"
        self.phase_executed = False   # True after computation runs; advance on next click
        self.transition_to = None     # Phase to move to on the advance click

        self.current_sample = np.zeros(self.n_inputs)
        self.I_sorted = []
        self.candidate_idx = 0
        self.winning_cat = None
        self.is_new = False
        self.status_log = "System initialized.\n"

art1_state = AppState()

def generate_code_display(phase):
    """Maps the current execution phase to specific lines of code for highlighting."""
    lines = original_code_string.split('\n')
    highlight_indices = []

    if phase == "INIT":
        highlight_indices = list(range(3, 16))
    elif phase == "LOAD_INPUT":
        highlight_indices = [17, 18]
    elif phase == "BOTTOM_UP":
        highlight_indices = [19, 20, 21]
    elif phase == "VIGILANCE_CHECK":
        highlight_indices = [23, 24, 25, 26]
    elif phase == "CREATE_NEW":
        highlight_indices = list(range(32, 40))  # allocation + weight update as one block
    elif phase == "UPDATE_WEIGHTS":
        highlight_indices = [27, 28, 29, 30]     # resonance weight update only

    components = []
    for i, line in enumerate(lines):
        style = {'paddingLeft': '10px', 'margin': '0', 'whiteSpace': 'pre', 'fontFamily': 'monospace', 'fontSize': '13px'}
        if i in highlight_indices:
            style['backgroundColor'] = '#ffeeba'
            style['fontWeight'] = 'bold'
            style['borderLeft'] = '4px solid #ffc107'
        else:
            style['borderLeft'] = '4px solid transparent'
            style['color'] = '#555'

        components.append(html.Div(line, style=style))

    return components


# -----------------------------------------------------------------------------
# LAYOUT (importable)
# -----------------------------------------------------------------------------
layout = html.Div(
    style={'display': 'flex', 'flexDirection': 'row', 'fontFamily': 'Arial, sans-serif',
           'padding': '20px', 'gap': '30px', 'height': '85vh', 'boxSizing': 'border-box'},
    children=[
        # LEFT PANEL: Original Code
        html.Div(style={'flex': '0 0 400px', 'display': 'flex', 'flexDirection': 'column'}, children=[
            html.H3("Original ART1 Implementation Code",
                    style={'marginTop': '0', 'borderBottom': '1px solid #ccc', 'paddingBottom': '10px'}),
            html.Div(id='art1-code-area', style={
                'backgroundColor': '#f8f9fa', 'padding': '15px 0', 'border': '1px solid #ddd',
                'overflowY': 'auto', 'flex': '1', 'lineHeight': '1.6'
            })
        ]),

        # RIGHT PANEL: Interactive Dashboard
        html.Div(style={'flex': '1', 'display': 'flex', 'flexDirection': 'column'}, children=[
            html.H1("Live ART1 Sequential Dashboard",
                    style={'marginTop': '0', 'borderBottom': '1px solid #ccc', 'paddingBottom': '10px'}),

            # Controls
            html.Div(style={'display': 'flex', 'gap': '20px', 'marginBottom': '20px',
                            'alignItems': 'center', 'paddingTop': '10px'}, children=[
                html.Div([
                    html.Label("Vigilance (Rho): ", style={'fontWeight': 'bold'}),
                    dcc.Slider(id='art1-rho-slider', min=0, max=1, step=0.05, value=0.5,
                               marks={i/10: str(i/10) for i in range(11)})
                ], style={'flex': '1'}),
                html.Button("▶ Execute Step", id='art1-next-btn', n_clicks=0,
                            style={'padding': '10px 20px', 'fontSize': '16px', 'backgroundColor': '#007BFF',
                                   'color': 'white', 'border': 'none', 'cursor': 'pointer', 'borderRadius': '4px',
                                   'minWidth': '180px'}),
                html.Button("Reset Network", id='art1-reset-btn', n_clicks=0,
                            style={'padding': '10px 20px', 'fontSize': '16px', 'backgroundColor': '#DC3545',
                                   'color': 'white', 'border': 'none', 'cursor': 'pointer', 'borderRadius': '4px'}),
            ]),

            # Visualizations and Logs
            html.Div(style={'display': 'flex', 'gap': '20px', 'flex': '1'}, children=[
                dcc.Textarea(
                    id='art1-log-area',
                    value=art1_state.status_log,
                    readOnly=True,
                    style={'width': '280px', 'height': '100%', 'fontFamily': 'monospace',
                           'fontSize': '12px', 'padding': '10px', 'boxSizing': 'border-box',
                           'resize': 'none', 'border': '1px solid #ccc'}
                ),
                dcc.Graph(id='art1-main-graph', style={'flex': '1', 'height': '100%'})
            ])
        ])
    ]
)


# -----------------------------------------------------------------------------
# CALLBACKS
# -----------------------------------------------------------------------------
def register_callbacks(app):
    @app.callback(
        [Output('art1-main-graph', 'figure'),
         Output('art1-log-area', 'value'),
         Output('art1-code-area', 'children'),
         Output('art1-next-btn', 'children')],
        [Input('art1-next-btn', 'n_clicks'),
         Input('art1-reset-btn', 'n_clicks')],
        [State('art1-rho-slider', 'value')]
    )
    def update_dashboard(next_clicks, reset_clicks, current_rho):
        triggered_id = ctx.triggered_id

        if triggered_id == 'art1-reset-btn':
            art1_state.reset()
            art1_state.status_log += "Network reset.\n"

        elif triggered_id == 'art1-next-btn':

            if not art1_state.phase_executed:
                # ── EXECUTE: run this phase's computation, update matrices ──────────
                if art1_state.phase == "INIT":
                    art1_state.status_log += "Initialization complete. Ready for inputs.\n"
                    art1_state.transition_to = "LOAD_INPUT"

                elif art1_state.phase == "LOAD_INPUT":
                    if art1_state.input_idx >= len(simple_samples):
                        art1_state.status_log += "No more inputs.\n"
                        art1_state.transition_to = None  # stay
                    else:
                        art1_state.current_sample = simple_samples[art1_state.input_idx]
                        art1_state.status_log += f"\n--- Input {art1_state.input_idx}: '{simple_data_strings[art1_state.input_idx]}' ---\n"
                        art1_state.F2 = np.zeros(art1_state.m_categories)
                        art1_state.winning_cat = None
                        art1_state.is_new = False
                        art1_state.transition_to = "BOTTOM_UP"

                elif art1_state.phase == "BOTTOM_UP":
                    art1_state.F2[...] = np.dot(art1_state.Wf, art1_state.current_sample)
                    art1_state.I_sorted = np.argsort(art1_state.F2[:art1_state.active].ravel())[::-1]
                    art1_state.candidate_idx = 0
                    art1_state.status_log += f"F2 computed. Sorted queue: {art1_state.I_sorted.tolist()}\n"
                    art1_state.transition_to = "VIGILANCE_CHECK" if len(art1_state.I_sorted) > 0 else "CREATE_NEW"

                elif art1_state.phase == "VIGILANCE_CHECK":
                    if art1_state.candidate_idx < len(art1_state.I_sorted):
                        i = art1_state.I_sorted[art1_state.candidate_idx]
                        X = art1_state.current_sample
                        d = (art1_state.Wb[:, i] * X).sum() / X.sum()
                        art1_state.status_log += f"Check Cat {i}: Match {d:.2f} (Req: {current_rho})\n"
                        if d >= current_rho:
                            art1_state.status_log += f"-> RESONANCE with Cat {i}. Weights updated.\n"
                            art1_state.winning_cat = i
                            art1_state.is_new = False
                            art1_state.Wb[:, i] *= X
                            art1_state.Wf[i, :] = art1_state.Wb[:, i] / (0.5 + art1_state.Wb[:, i].sum())
                            art1_state.transition_to = "UPDATE_WEIGHTS"
                        else:
                            art1_state.status_log += f"-> RESET Cat {i}.\n"
                            art1_state.candidate_idx += 1
                            if art1_state.candidate_idx >= len(art1_state.I_sorted):
                                art1_state.status_log += "All active categories failed.\n"
                                art1_state.transition_to = "CREATE_NEW"
                            else:
                                art1_state.transition_to = None  # stay: check next candidate
                    else:
                        art1_state.transition_to = "CREATE_NEW"

                elif art1_state.phase == "CREATE_NEW":
                    if art1_state.active < art1_state.m_categories:
                        art1_state.winning_cat = art1_state.active
                        art1_state.active += 1
                        art1_state.is_new = True
                        i = art1_state.winning_cat
                        X = art1_state.current_sample
                        art1_state.Wb[:, i] *= X
                        art1_state.Wf[i, :] = art1_state.Wb[:, i] / (0.5 + art1_state.Wb[:, i].sum())
                        art1_state.status_log += f"Allocated New Cat {art1_state.winning_cat}. Weights updated.\n"
                        art1_state.transition_to = "LOAD_INPUT"
                    else:
                        art1_state.status_log += "Capacity full. Skipping input.\n"
                        art1_state.input_idx += 1
                        art1_state.transition_to = "LOAD_INPUT"

                elif art1_state.phase == "UPDATE_WEIGHTS":
                    art1_state.status_log += "Step complete.\n"
                    art1_state.input_idx += 1
                    art1_state.transition_to = "LOAD_INPUT"

                art1_state.phase_executed = True

            else:
                # ── ADVANCE: move highlight to the next phase, no computation ────────
                if art1_state.transition_to is not None:
                    art1_state.phase = art1_state.transition_to
                    art1_state.transition_to = None
                # else: stay in current phase (e.g. VIGILANCE_CHECK, more candidates remain)
                art1_state.phase_executed = False

        # ── Determine button label ─────────────────────────────────────────────
        if art1_state.phase_executed:
            btn_label = "→ Next Phase"
        else:
            btn_label = "▶ Execute Step"

        # ── Generate Figure ────────────────────────────────────────────────────
        annotation_font = dict(size=10)
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Current Input X", "F2 Activations (Wf * X)",
                            "Feed-forward Weights (Wf)", "Feed-back Prototypes (Wb)"),
            vertical_spacing=0.15, horizontal_spacing=0.1
        )

        fig.add_trace(go.Heatmap(
            z=art1_state.current_sample.reshape(1, -1), colorscale='Greys',
            text=art1_state.current_sample.reshape(1, -1), texttemplate="%{text:.0f}",
            showscale=False, textfont=annotation_font, xgap=1, ygap=1
        ), row=1, col=1)

        fig.add_trace(go.Heatmap(
            z=art1_state.F2.reshape(1, -1), colorscale='Purples',
            text=art1_state.F2.reshape(1, -1), texttemplate="%{text:.2f}",
            showscale=False, textfont=annotation_font, xgap=1, ygap=1
        ), row=1, col=2)

        fig.add_trace(go.Heatmap(
            z=art1_state.Wf, colorscale='Greens',
            text=art1_state.Wf, texttemplate="%{text:.2f}",
            showscale=False, textfont=annotation_font, xgap=1, ygap=1
        ), row=2, col=1)

        fig.add_trace(go.Heatmap(
            z=art1_state.Wb.T, colorscale='Blues',
            text=art1_state.Wb.T, texttemplate="%{text:.0f}",
            showscale=False, textfont=annotation_font, xgap=1, ygap=1
        ), row=2, col=2)

        # Red highlight on the active category (winning or next candidate to check)
        active_cat = None
        if art1_state.phase in ["VIGILANCE_CHECK", "CREATE_NEW", "UPDATE_WEIGHTS"]:
            if art1_state.winning_cat is not None:
                active_cat = art1_state.winning_cat
            elif art1_state.phase == "VIGILANCE_CHECK" and art1_state.candidate_idx < len(art1_state.I_sorted):
                active_cat = art1_state.I_sorted[art1_state.candidate_idx]

        if active_cat is not None:
            fig.add_shape(type="rect", x0=active_cat-0.5, y0=-0.5, x1=active_cat+0.5, y1=0.5,
                          line=dict(color="red", width=3), row=1, col=2)
            fig.add_shape(type="rect", x0=-0.5, y0=active_cat-0.5, x1=art1_state.n_inputs-0.5, y1=active_cat+0.5,
                          line=dict(color="red", width=3), row=2, col=1)
            fig.add_shape(type="rect", x0=-0.5, y0=active_cat-0.5, x1=art1_state.n_inputs-0.5, y1=active_cat+0.5,
                          line=dict(color="red", width=3), row=2, col=2)

        fig.update_xaxes(title_text="Features (n)", row=1, col=1, tickfont=dict(size=9))
        fig.update_yaxes(showticklabels=False, row=1, col=1)
        fig.update_xaxes(title_text="Categories (m)", row=1, col=2, tickfont=dict(size=9), dtick=1)
        fig.update_yaxes(showticklabels=False, row=1, col=2)
        fig.update_xaxes(title_text="Features (n)", row=2, col=1, tickfont=dict(size=9))
        fig.update_yaxes(title_text="Categories (m)", row=2, col=1, tickfont=dict(size=9), dtick=1)
        fig.update_xaxes(title_text="Features (n)", row=2, col=2, tickfont=dict(size=9))
        fig.update_yaxes(title_text="Categories (m)", row=2, col=2, tickfont=dict(size=9), dtick=1)
        fig.update_layout(margin=dict(l=20, r=40, t=40, b=20), autosize=True)

        code_html = generate_code_display(art1_state.phase)
        return fig, art1_state.status_log, code_html, btn_label


if __name__ == '__main__':
    _app = dash.Dash(__name__)
    _app.layout = layout
    register_callbacks(_app)
    _app.run(debug=True)
