import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo

# ─────────────────────────────────────────────
# DATA LOADING & PREPROCESSING
# ─────────────────────────────────────────────
chronic_kidney_disease = fetch_ucirepo(id=336)
X = chronic_kidney_disease.data.features
y = chronic_kidney_disease.data.targets
df = pd.concat([X, y], axis=1)

# Type fixes
cols_entiers = ['age', 'bu', 'bp', 'bgr', 'sod', 'pcv', 'wbcc']
df[cols_entiers] = df[cols_entiers].round(0)

# Target encoding
df['target'] = df['class'].map({'ckd': 1, 'notckd': 0})
df['Diagnostic'] = df['class'].map({'ckd': 'CKD', 'notckd': 'Non-CKD'})

# Categorical encoding pour graphique metier
df['htn_label'] = df['htn'].astype(str).str.strip().str.lower().map({'yes': 'Hypertension', 'no': "Pas d'hypertension"})
df['dm_label']  = df['dm'].astype(str).str.strip().str.lower().map({'yes': 'Diabète', 'no': 'Pas de diabète'})

df = df.dropna(subset=['target'])
df['target'] = df['target'].astype(int)
df['age'] = pd.to_numeric(df['age'], errors='coerce')

# Variables numeriques
var_num = ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'wbcc', 'rbcc']

# Slider range
age_min = int(df['age'].min())
age_max = int(df['age'].max())

# Correlation vars
corr_vars = ['al', 'bgr', 'sod', 'rbcc', 'pcv', 'sg', 'target']

# ─────────────────────────────────────────────
# COLOR PALETTE
# ─────────────────────────────────────────────
COLORS = {
    'bg':      '#0a0e1a',
    'card':    '#111827',
    'card2':   '#1a2236',
    'accent':  '#00d4ff',
    'accent2': '#7c3aed',
    'ckd':     '#f43f5e',
    'notckd':  '#10b981',
    'text':    '#e2e8f0',
    'muted':   '#64748b',
    'border':  '#1e293b',
}

# ─────────────────────────────────────────────
# APP INIT
# ─────────────────────────────────────────────
app = dash.Dash(
    __name__,
    title="CKD Analytics Dashboard",
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}]
)

# ─────────────────────────────────────────────
# LAYOUT
# ─────────────────────────────────────────────
app.layout = html.Div(style={
    'backgroundColor': COLORS['bg'],
    'minHeight': '100vh',
    'fontFamily': '"DM Sans", "Segoe UI", sans-serif',
    'color': COLORS['text'],
}, children=[

    # Google Font
    html.Link(rel='stylesheet', href='https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;700&family=Space+Mono:wght@400;700&display=swap'),

    # ── HEADER ──
    html.Div(style={
        'background': f'linear-gradient(135deg, {COLORS["card"]} 0%, {COLORS["card2"]} 100%)',
        'borderBottom': f'1px solid {COLORS["border"]}',
        'padding': '20px 40px',
        'display': 'flex',
        'alignItems': 'center',
        'justifyContent': 'space-between',
    }, children=[
        html.Div([
            html.Span("⬡ ", style={'color': COLORS['accent'], 'fontSize': '26px'}),
            html.Span("CKD", style={'fontFamily': '"Space Mono", monospace', 'fontSize': '24px', 'fontWeight': '700', 'color': COLORS['accent'], 'letterSpacing': '3px'}),
            html.Span(" Analytics", style={'fontFamily': '"Space Mono", monospace', 'fontSize': '24px', 'fontWeight': '400', 'color': COLORS['text'], 'letterSpacing': '2px'}),
        ]),
        html.Div("Chronic Kidney Disease · UCI ML Repository · 400 patients",
                 style={'color': COLORS['muted'], 'fontSize': '12px', 'fontFamily': '"Space Mono", monospace'}),
    ]),

    # ── MAIN ──
    html.Div(style={'padding': '28px 40px', 'maxWidth': '1400px', 'margin': '0 auto'}, children=[

        # ── FILTRES ──
        html.Div(style={
            'background': COLORS['card'],
            'border': f'1px solid {COLORS["border"]}',
            'borderRadius': '12px',
            'padding': '20px 28px',
            'marginBottom': '24px',
            'display': 'flex',
            'alignItems': 'center',
            'gap': '32px',
            'flexWrap': 'wrap',
        }, children=[
            html.Div([
                html.Div("FILTRES GLOBAUX", style={'fontSize': '10px', 'color': COLORS['muted'], 'letterSpacing': '2px', 'fontFamily': '"Space Mono", monospace', 'marginBottom': '4px'}),
                html.Div("Tous les graphiques se mettent à jour automatiquement",
                         style={'fontSize': '12px', 'color': COLORS['muted']}),
            ], style={'minWidth': '220px'}),

            # Dropdown Variable X
            html.Div([
                html.Div("Variable X (Scatter)", style={'fontSize': '11px', 'color': COLORS['muted'], 'marginBottom': '6px'}),
                dcc.Dropdown(
                    id='scatter-x',
                    options=[{'label': v, 'value': v} for v in var_num],
                    value='sc',
                    clearable=False,
                    style={'width': '140px', 'fontSize': '13px'},
                ),
            ]),

            # Dropdown Variable Y
            html.Div([
                html.Div("Variable Y (Scatter)", style={'fontSize': '11px', 'color': COLORS['muted'], 'marginBottom': '6px'}),
                dcc.Dropdown(
                    id='scatter-y',
                    options=[{'label': v, 'value': v} for v in var_num],
                    value='bgr',
                    clearable=False,
                    style={'width': '140px', 'fontSize': '13px'},
                ),
            ]),

            # Dropdown Catégorie
            html.Div([
                html.Div("Catégorie", style={'fontSize': '11px', 'color': COLORS['muted'], 'marginBottom': '6px'}),
                dcc.Dropdown(
                    id='diag-filter',
                    options=[
                        {'label': 'Tous', 'value': 'Tous'},
                        {'label': 'CKD', 'value': 'CKD'},
                        {'label': 'Non-CKD', 'value': 'Non-CKD'},
                    ],
                    value='Tous',
                    clearable=False,
                    style={'width': '130px', 'fontSize': '13px'},
                ),
            ]),

            # Slider âge
            html.Div([
                html.Div(id='slider-label', style={'fontSize': '11px', 'color': COLORS['muted'], 'marginBottom': '6px'}),
                dcc.RangeSlider(
                    id='age-slider',
                    min=age_min,
                    max=age_max,
                    step=1,
                    value=[age_min, age_max],
                    marks={age_min: str(age_min), age_max: str(age_max)},
                    tooltip={"placement": "bottom", "always_visible": True},
                ),
            ], style={'minWidth': '260px', 'flex': '1'}),
        ]),

        # ── KPI CARDS ──
        html.Div(style={
            'display': 'grid',
            'gridTemplateColumns': 'repeat(4, 1fr)',
            'gap': '16px',
            'marginBottom': '20px',
        }, children=[
            html.Div(id='kpi-total', style={'background': f'linear-gradient(135deg, {COLORS["card"]} 0%, {COLORS["card2"]} 100%)', 'border': f'1px solid {COLORS["border"]}', 'borderTop': f'3px solid {COLORS["accent"]}', 'borderRadius': '12px', 'padding': '20px'}),
            html.Div(id='kpi-ckd',   style={'background': f'linear-gradient(135deg, {COLORS["card"]} 0%, {COLORS["card2"]} 100%)', 'border': f'1px solid {COLORS["border"]}', 'borderTop': f'3px solid {COLORS["ckd"]}',    'borderRadius': '12px', 'padding': '20px'}),
            html.Div(id='kpi-notckd',style={'background': f'linear-gradient(135deg, {COLORS["card"]} 0%, {COLORS["card2"]} 100%)', 'border': f'1px solid {COLORS["border"]}', 'borderTop': f'3px solid {COLORS["notckd"]}', 'borderRadius': '12px', 'padding': '20px'}),
            html.Div(id='kpi-age',   style={'background': f'linear-gradient(135deg, {COLORS["card"]} 0%, {COLORS["card2"]} 100%)', 'border': f'1px solid {COLORS["border"]}', 'borderTop': f'3px solid {COLORS["accent2"]}','borderRadius': '12px', 'padding': '20px'}),
        ]),

        # ── ROW 1 : Histogramme + Boxplot ──
        html.Div(style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '16px', 'marginBottom': '16px'}, children=[

            html.Div(style={'background': COLORS['card'], 'border': f'1px solid {COLORS["border"]}', 'borderRadius': '12px', 'padding': '20px'}, children=[
                html.Div("HISTOGRAMME", style={'fontSize': '10px', 'color': COLORS['muted'], 'letterSpacing': '2px', 'fontFamily': '"Space Mono", monospace', 'marginBottom': '4px'}),
                html.Div(style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center', 'marginBottom': '12px'}, children=[
                    html.Div("Distribution par variable", style={'fontSize': '15px', 'fontWeight': '600'}),
                    dcc.Dropdown(id='hist-var', options=[{'label': v, 'value': v} for v in var_num], value='age', clearable=False, style={'width': '110px', 'fontSize': '13px'}),
                ]),
                dcc.Graph(id='hist-plot', style={'height': '300px'}, config={'displayModeBar': False}),
            ]),

            html.Div(style={'background': COLORS['card'], 'border': f'1px solid {COLORS["border"]}', 'borderRadius': '12px', 'padding': '20px'}, children=[
                html.Div("BOXPLOT", style={'fontSize': '10px', 'color': COLORS['muted'], 'letterSpacing': '2px', 'fontFamily': '"Space Mono", monospace', 'marginBottom': '4px'}),
                html.Div(style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center', 'marginBottom': '12px'}, children=[
                    html.Div("Distribution par diagnostic", style={'fontSize': '15px', 'fontWeight': '600'}),
                    dcc.Dropdown(id='box-var', options=[{'label': v, 'value': v} for v in var_num], value='sc', clearable=False, style={'width': '110px', 'fontSize': '13px'}),
                ]),
                dcc.Graph(id='box-plot', style={'height': '300px'}, config={'displayModeBar': False}),
            ]),
        ]),

        # ── ROW 2 : Scatter + Heatmap ──
        html.Div(style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '16px', 'marginBottom': '16px'}, children=[

            html.Div(style={'background': COLORS['card'], 'border': f'1px solid {COLORS["border"]}', 'borderRadius': '12px', 'padding': '20px'}, children=[
                html.Div("SCATTER PLOT", style={'fontSize': '10px', 'color': COLORS['muted'], 'letterSpacing': '2px', 'fontFamily': '"Space Mono", monospace', 'marginBottom': '4px'}),
                html.Div("Relation entre variables · filtré par âge et catégorie", style={'fontSize': '15px', 'fontWeight': '600', 'marginBottom': '12px'}),
                dcc.Graph(id='scatter-plot', style={'height': '320px'}, config={'displayModeBar': False}),
            ]),

            html.Div(style={'background': COLORS['card'], 'border': f'1px solid {COLORS["border"]}', 'borderRadius': '12px', 'padding': '20px'}, children=[
                html.Div("HEATMAP", style={'fontSize': '10px', 'color': COLORS['muted'], 'letterSpacing': '2px', 'fontFamily': '"Space Mono", monospace', 'marginBottom': '4px'}),
                html.Div("Matrice de corrélation · filtrée par sélection", style={'fontSize': '15px', 'fontWeight': '600', 'marginBottom': '12px'}),
                dcc.Graph(id='heatmap', style={'height': '320px'}, config={'displayModeBar': False}),
            ]),
        ]),

        # ── ROW 3 : Graphique métier ──
        html.Div(style={'background': COLORS['card'], 'border': f'1px solid {COLORS["border"]}', 'borderRadius': '12px', 'padding': '20px', 'marginBottom': '20px'}, children=[
            html.Div("GRAPHIQUE MÉTIER", style={'fontSize': '10px', 'color': COLORS['muted'], 'letterSpacing': '2px', 'fontFamily': '"Space Mono", monospace', 'marginBottom': '4px'}),
            html.Div("Effet combiné Diabète × Hypertension sur la CKD", style={'fontSize': '15px', 'fontWeight': '600', 'marginBottom': '12px'}),
            dcc.Graph(id='business-chart', style={'height': '340px'}, config={'displayModeBar': False}),
        ]),

        # ── FOOTER ──
        html.Div("CKD Dashboard · Plotly Dash · UCI ML Repository ID=336",
                 style={'textAlign': 'center', 'padding': '16px 0', 'color': COLORS['muted'], 'fontSize': '11px', 'fontFamily': '"Space Mono", monospace'}),
    ]),
])

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def apply_filters(age_range, diag):
    """Filtre le DataFrame selon le slider d'âge et le dropdown diagnostic."""
    filtered = df[(df['age'] >= age_range[0]) & (df['age'] <= age_range[1])].copy()
    if diag != 'Tous':
        filtered = filtered[filtered['Diagnostic'] == diag]
    return filtered

def dark_layout(fig):
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='"DM Sans", sans-serif', color=COLORS['text'], size=12),
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor=COLORS['border'], font=dict(size=12)),
        xaxis=dict(gridcolor=COLORS['border'], zerolinecolor=COLORS['border']),
        yaxis=dict(gridcolor=COLORS['border'], zerolinecolor=COLORS['border']),
    )
    return fig

def kpi_card(label, value, sub, color):
    return [
        html.Div(label, style={'fontSize': '10px', 'color': COLORS['muted'], 'letterSpacing': '2px', 'fontFamily': '"Space Mono", monospace', 'marginBottom': '8px'}),
        html.Div(str(value), style={'fontSize': '38px', 'fontWeight': '700', 'color': color, 'lineHeight': '1'}),
        html.Div(sub, style={'fontSize': '11px', 'color': COLORS['muted'], 'marginTop': '6px'}),
    ]

# ─────────────────────────────────────────────
# CALLBACKS
# ─────────────────────────────────────────────

# Slider label
@app.callback(Output('slider-label', 'children'), Input('age-slider', 'value'))
def update_slider_label(age_range):
    return f"Tranche d'âge : {age_range[0]} – {age_range[1]} ans"

# KPI cards
@app.callback(
    Output('kpi-total', 'children'),
    Output('kpi-ckd', 'children'),
    Output('kpi-notckd', 'children'),
    Output('kpi-age', 'children'),
    Input('age-slider', 'value'),
    Input('diag-filter', 'value'),
)
def update_kpis(age_range, diag):
    d = apply_filters(age_range, diag)
    total = len(d)
    ckd = int(d['target'].sum())
    notckd = total - ckd
    avg_age = d['age'].mean()
    return (
        kpi_card("TOTAL PATIENTS", total, "dans la sélection", COLORS['text']),
        kpi_card("PATIENTS CKD", ckd, f"{ckd/total*100:.1f}% CKD" if total > 0 else "–", COLORS['ckd']),
        kpi_card("PATIENTS SAINS", notckd, f"{notckd/total*100:.1f}% Non-CKD" if total > 0 else "–", COLORS['notckd']),
        kpi_card("ÂGE MOYEN", f"{avg_age:.0f}" if total > 0 else "–", "ans · sélection active", COLORS['accent2']),
    )

# Histogramme
@app.callback(
    Output('hist-plot', 'figure'),
    Input('age-slider', 'value'),
    Input('diag-filter', 'value'),
    Input('hist-var', 'value'),
)
def update_hist(age_range, diag, var):
    d = apply_filters(age_range, diag).dropna(subset=[var]).copy()
    d[var] = d[var].astype(float)
    fig = px.histogram(
        d, x=var, color='Diagnostic',
        color_discrete_map={'CKD': COLORS['ckd'], 'Non-CKD': COLORS['notckd']},
        barmode='overlay', opacity=0.75, nbins=30,
    )
    return dark_layout(fig)

# Boxplot
@app.callback(
    Output('box-plot', 'figure'),
    Input('age-slider', 'value'),
    Input('diag-filter', 'value'),
    Input('box-var', 'value'),
)
def update_boxplot(age_range, diag, var):
    d = apply_filters(age_range, diag).dropna(subset=[var]).copy()
    d[var] = d[var].astype(float)
    fig = px.box(
        d, x='Diagnostic', y=var,
        color='Diagnostic',
        color_discrete_map={'CKD': COLORS['ckd'], 'Non-CKD': COLORS['notckd']},
        points='outliers', notched=True,
    )
    fig.update_traces(line_width=2, marker=dict(size=5, opacity=0.6))
    return dark_layout(fig)

# Scatter plot
@app.callback(
    Output('scatter-plot', 'figure'),
    Input('age-slider', 'value'),
    Input('diag-filter', 'value'),
    Input('scatter-x', 'value'),
    Input('scatter-y', 'value'),
)
def update_scatter(age_range, diag, x_var, y_var):
    d = apply_filters(age_range, diag).dropna(subset=[x_var, y_var]).copy()
    d[x_var] = d[x_var].astype(float)
    d[y_var] = d[y_var].astype(float)
    fig = px.scatter(
        d, x=x_var, y=y_var,
        color='Diagnostic',
        color_discrete_map={'CKD': COLORS['ckd'], 'Non-CKD': COLORS['notckd']},
        opacity=0.75, hover_data=['age'],
        trendline='ols', trendline_scope='overall',
        trendline_color_override=COLORS['accent'],
    )
    fig.update_traces(marker=dict(size=7, line=dict(width=0)))
    return dark_layout(fig)

# Heatmap
@app.callback(
    Output('heatmap', 'figure'),
    Input('age-slider', 'value'),
    Input('diag-filter', 'value'),
)
def update_heatmap(age_range, diag):
    d = apply_filters(age_range, diag)
    cm = d[corr_vars].corr().round(2)
    fig = go.Figure(go.Heatmap(
        z=cm.values,
        x=cm.columns.tolist(),
        y=cm.index.tolist(),
        colorscale=[[0.0, COLORS['ckd']], [0.5, COLORS['card2']], [1.0, COLORS['notckd']]],
        zmin=-1, zmax=1,
        text=cm.values.round(2),
        texttemplate='%{text}',
        textfont=dict(size=11),
        hovertemplate='%{x} × %{y}: %{z:.2f}<extra></extra>',
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=COLORS['text'], size=11),
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(tickangle=-30),
    )
    return fig

# Business chart
@app.callback(
    Output('business-chart', 'figure'),
    Input('age-slider', 'value'),
    Input('diag-filter', 'value'),
)
def update_business(age_range, diag):
    d = apply_filters(age_range, diag).dropna(subset=['htn_label', 'dm_label']).copy()
    grouped = d.groupby(['htn_label', 'dm_label', 'Diagnostic']).size().reset_index(name='count')
    fig = px.bar(
        grouped, x='htn_label', y='count',
        color='Diagnostic', facet_col='dm_label',
        barmode='group',
        color_discrete_map={'CKD': COLORS['ckd'], 'Non-CKD': COLORS['notckd']},
        text='count',
    )
    fig.update_traces(textposition='outside', textfont=dict(size=12))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=COLORS['text'], size=12),
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(bgcolor='rgba(0,0,0,0)'),
        xaxis=dict(gridcolor=COLORS['border']),
        yaxis=dict(gridcolor=COLORS['border']),
    )
    fig.for_each_annotation(lambda a: a.update(
        text=a.text.split("=")[-1],
        font=dict(size=13, color=COLORS['accent']),
    ))
    return fig

# ─────────────────────────────────────────────
if __name__ == '__main__':
    app.run(debug=True)
