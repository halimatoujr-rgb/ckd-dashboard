import dash
from dash import dcc, html, Input, Output, callback
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

# Categorical encoding
df['rbc_enc'] = df['rbc'].map({'normal': 1, 'abnormal': 0})
df['appet_enc'] = df['appet'].map({'good': 1, 'poor': 0})
df['htn_enc'] = df['htn'].str.strip().str.lower().map({'yes': 1, 'no': 0})
df['dm_enc'] = df['dm'].str.strip().str.lower().map({'yes': 1, 'no': 0})
df['ane_enc'] = df['ane'].str.strip().str.lower().map({'yes': 1, 'no': 0})

df = df.dropna(subset=['target'])
df['target'] = df['target'].astype(int)

# Numeric vars
var_num = ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wbcc', 'rbcc']

# KPIs
total = len(df)
ckd_count = df['target'].sum()
notckd_count = total - ckd_count
avg_age = df['age'].mean()
avg_hemo = df['hemo'].mean()

# Correlation matrix
corr_vars = ['al', 'bgr', 'sod', 'hemo', 'rbcc', 'pcv', 'sg', 'target']
corr_matrix = df[corr_vars].corr().round(2)

# ─────────────────────────────────────────────
# COLOR PALETTE
# ─────────────────────────────────────────────
COLORS = {
    'bg': '#0a0e1a',
    'card': '#111827',
    'card2': '#1a2236',
    'accent': '#00d4ff',
    'accent2': '#7c3aed',
    'ckd': '#f43f5e',
    'notckd': '#10b981',
    'text': '#e2e8f0',
    'muted': '#64748b',
    'border': '#1e293b',
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
    'padding': '0',
}, children=[

    # ── Google Font ──
    html.Link(rel='stylesheet', href='https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;700&family=Space+Mono:wght@400;700&display=swap'),

    # ── HEADER ──
    html.Div(style={
        'background': f'linear-gradient(135deg, {COLORS["card"]} 0%, {COLORS["card2"]} 100%)',
        'borderBottom': f'1px solid {COLORS["border"]}',
        'padding': '24px 40px',
        'display': 'flex',
        'alignItems': 'center',
        'justifyContent': 'space-between',
    }, children=[
        html.Div([
            html.Span("⬡ ", style={'color': COLORS['accent'], 'fontSize': '28px'}),
            html.Span("CKD", style={
                'fontFamily': '"Space Mono", monospace',
                'fontSize': '26px',
                'fontWeight': '700',
                'color': COLORS['accent'],
                'letterSpacing': '3px',
            }),
            html.Span(" Analytics", style={
                'fontFamily': '"Space Mono", monospace',
                'fontSize': '26px',
                'fontWeight': '400',
                'color': COLORS['text'],
                'letterSpacing': '2px',
            }),
        ]),
        html.Div("Chronic Kidney Disease · UCI ML Repository · 400 patients", style={
            'color': COLORS['muted'],
            'fontSize': '13px',
            'fontFamily': '"Space Mono", monospace',
        })
    ]),

    # ── MAIN CONTENT ──
    html.Div(style={'padding': '32px 40px', 'maxWidth': '1400px', 'margin': '0 auto'}, children=[

        # ── KPI CARDS ──
        html.Div(style={
            'display': 'grid',
            'gridTemplateColumns': 'repeat(4, 1fr)',
            'gap': '20px',
            'marginBottom': '32px',
        }, children=[
            # Card 1
            html.Div(style={
                'background': f'linear-gradient(135deg, {COLORS["card"]} 0%, {COLORS["card2"]} 100%)',
                'border': f'1px solid {COLORS["border"]}',
                'borderTop': f'3px solid {COLORS["accent"]}',
                'borderRadius': '12px',
                'padding': '24px',
            }, children=[
                html.Div("TOTAL PATIENTS", style={'fontSize': '11px', 'color': COLORS['muted'], 'letterSpacing': '2px', 'fontFamily': '"Space Mono", monospace', 'marginBottom': '8px'}),
                html.Div(f"{total}", style={'fontSize': '42px', 'fontWeight': '700', 'color': COLORS['text'], 'lineHeight': '1'}),
                html.Div("dans le dataset", style={'fontSize': '12px', 'color': COLORS['muted'], 'marginTop': '6px'}),
            ]),
            # Card 2
            html.Div(style={
                'background': f'linear-gradient(135deg, {COLORS["card"]} 0%, {COLORS["card2"]} 100%)',
                'border': f'1px solid {COLORS["border"]}',
                'borderTop': f'3px solid {COLORS["ckd"]}',
                'borderRadius': '12px',
                'padding': '24px',
            }, children=[
                html.Div("PATIENTS CKD", style={'fontSize': '11px', 'color': COLORS['muted'], 'letterSpacing': '2px', 'fontFamily': '"Space Mono", monospace', 'marginBottom': '8px'}),
                html.Div(f"{ckd_count}", style={'fontSize': '42px', 'fontWeight': '700', 'color': COLORS['ckd'], 'lineHeight': '1'}),
                html.Div(f"{ckd_count/total*100:.1f}% du total", style={'fontSize': '12px', 'color': COLORS['muted'], 'marginTop': '6px'}),
            ]),
            # Card 3
            html.Div(style={
                'background': f'linear-gradient(135deg, {COLORS["card"]} 0%, {COLORS["card2"]} 100%)',
                'border': f'1px solid {COLORS["border"]}',
                'borderTop': f'3px solid {COLORS["notckd"]}',
                'borderRadius': '12px',
                'padding': '24px',
            }, children=[
                html.Div("PATIENTS SAINS", style={'fontSize': '11px', 'color': COLORS['muted'], 'letterSpacing': '2px', 'fontFamily': '"Space Mono", monospace', 'marginBottom': '8px'}),
                html.Div(f"{notckd_count}", style={'fontSize': '42px', 'fontWeight': '700', 'color': COLORS['notckd'], 'lineHeight': '1'}),
                html.Div(f"{notckd_count/total*100:.1f}% du total", style={'fontSize': '12px', 'color': COLORS['muted'], 'marginTop': '6px'}),
            ]),
            # Card 4
            html.Div(style={
                'background': f'linear-gradient(135deg, {COLORS["card"]} 0%, {COLORS["card2"]} 100%)',
                'border': f'1px solid {COLORS["border"]}',
                'borderTop': f'3px solid {COLORS["accent2"]}',
                'borderRadius': '12px',
                'padding': '24px',
            }, children=[
                html.Div("ÂGE MOYEN", style={'fontSize': '11px', 'color': COLORS['muted'], 'letterSpacing': '2px', 'fontFamily': '"Space Mono", monospace', 'marginBottom': '8px'}),
                html.Div(f"{avg_age:.0f}", style={'fontSize': '42px', 'fontWeight': '700', 'color': COLORS['accent2'], 'lineHeight': '1'}),
                html.Div("ans · population totale", style={'fontSize': '12px', 'color': COLORS['muted'], 'marginTop': '6px'}),
            ]),
        ]),

        # ── ROW 1 : Scatter + Donut ──
        html.Div(style={
            'display': 'grid',
            'gridTemplateColumns': '2fr 1fr',
            'gap': '20px',
            'marginBottom': '20px',
        }, children=[

            # Scatter plot
            html.Div(style={
                'background': COLORS['card'],
                'border': f'1px solid {COLORS["border"]}',
                'borderRadius': '12px',
                'padding': '24px',
            }, children=[
                html.Div(style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center', 'marginBottom': '16px'}, children=[
                    html.Div([
                        html.Div("SCATTER PLOT", style={'fontSize': '11px', 'color': COLORS['muted'], 'letterSpacing': '2px', 'fontFamily': '"Space Mono", monospace'}),
                        html.Div("Relation entre variables", style={'fontSize': '16px', 'fontWeight': '600', 'color': COLORS['text']}),
                    ]),
                    html.Div(style={'display': 'flex', 'gap': '12px', 'alignItems': 'center'}, children=[
                        html.Span("X:", style={'color': COLORS['muted'], 'fontSize': '13px'}),
                        dcc.Dropdown(
                            id='scatter-x',
                            options=[{'label': v, 'value': v} for v in var_num],
                            value='hemo',
                            clearable=False,
                            style={'width': '110px', 'fontSize': '13px'},
                        ),
                        html.Span("Y:", style={'color': COLORS['muted'], 'fontSize': '13px'}),
                        dcc.Dropdown(
                            id='scatter-y',
                            options=[{'label': v, 'value': v} for v in var_num],
                            value='pcv',
                            clearable=False,
                            style={'width': '110px', 'fontSize': '13px'},
                        ),
                    ]),
                ]),
                dcc.Graph(id='scatter-plot', style={'height': '340px'}, config={'displayModeBar': False}),
            ]),

            # Donut chart
            html.Div(style={
                'background': COLORS['card'],
                'border': f'1px solid {COLORS["border"]}',
                'borderRadius': '12px',
                'padding': '24px',
            }, children=[
                html.Div("RÉPARTITION", style={'fontSize': '11px', 'color': COLORS['muted'], 'letterSpacing': '2px', 'fontFamily': '"Space Mono", monospace', 'marginBottom': '4px'}),
                html.Div("CKD vs Non-CKD", style={'fontSize': '16px', 'fontWeight': '600', 'color': COLORS['text'], 'marginBottom': '16px'}),
                dcc.Graph(id='donut-chart', style={'height': '340px'}, config={'displayModeBar': False}),
            ]),
        ]),

        # ── ROW 2 : Heatmap + Boxplot ──
        html.Div(style={
            'display': 'grid',
            'gridTemplateColumns': '1fr 1fr',
            'gap': '20px',
            'marginBottom': '20px',
        }, children=[

            # Heatmap
            html.Div(style={
                'background': COLORS['card'],
                'border': f'1px solid {COLORS["border"]}',
                'borderRadius': '12px',
                'padding': '24px',
            }, children=[
                html.Div("HEATMAP", style={'fontSize': '11px', 'color': COLORS['muted'], 'letterSpacing': '2px', 'fontFamily': '"Space Mono", monospace', 'marginBottom': '4px'}),
                html.Div("Matrice de corrélation", style={'fontSize': '16px', 'fontWeight': '600', 'color': COLORS['text'], 'marginBottom': '16px'}),
                dcc.Graph(id='heatmap', style={'height': '360px'}, config={'displayModeBar': False}),
            ]),

            # Boxplot dynamique
            html.Div(style={
                'background': COLORS['card'],
                'border': f'1px solid {COLORS["border"]}',
                'borderRadius': '12px',
                'padding': '24px',
            }, children=[
                html.Div(style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center', 'marginBottom': '16px'}, children=[
                    html.Div([
                        html.Div("BOXPLOT", style={'fontSize': '11px', 'color': COLORS['muted'], 'letterSpacing': '2px', 'fontFamily': '"Space Mono", monospace'}),
                        html.Div("Distribution par diagnostic", style={'fontSize': '16px', 'fontWeight': '600', 'color': COLORS['text']}),
                    ]),
                    dcc.Dropdown(
                        id='box-var',
                        options=[{'label': v, 'value': v} for v in var_num],
                        value='hemo',
                        clearable=False,
                        style={'width': '120px', 'fontSize': '13px'},
                    ),
                ]),
                dcc.Graph(id='box-plot', style={'height': '360px'}, config={'displayModeBar': False}),
            ]),
        ]),

        # ── ROW 3 : Business Chart ──
        html.Div(style={
            'background': COLORS['card'],
            'border': f'1px solid {COLORS["border"]}',
            'borderRadius': '12px',
            'padding': '24px',
            'marginBottom': '20px',
        }, children=[
            html.Div("GRAPHIQUE MÉTIER", style={'fontSize': '11px', 'color': COLORS['muted'], 'letterSpacing': '2px', 'fontFamily': '"Space Mono", monospace', 'marginBottom': '4px'}),
            html.Div("Effet combiné Diabète × Hypertension sur la CKD", style={'fontSize': '16px', 'fontWeight': '600', 'color': COLORS['text'], 'marginBottom': '16px'}),
            dcc.Graph(id='business-chart', style={'height': '360px'}, config={'displayModeBar': False}),
        ]),

        # ── FOOTER ──
        html.Div(style={'textAlign': 'center', 'padding': '20px 0', 'color': COLORS['muted'], 'fontSize': '12px', 'fontFamily': '"Space Mono", monospace'}, children=[
            "CKD Dashboard · Built with Plotly Dash · UCI ML Repository ID=336"
        ]),
    ]),
])

# ─────────────────────────────────────────────
# CHART THEME BASE
# ─────────────────────────────────────────────
def dark_layout(fig, height=None):
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='"DM Sans", sans-serif', color=COLORS['text'], size=12),
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(
            bgcolor='rgba(0,0,0,0)',
            bordercolor=COLORS['border'],
            font=dict(size=12),
        ),
        xaxis=dict(gridcolor=COLORS['border'], zerolinecolor=COLORS['border']),
        yaxis=dict(gridcolor=COLORS['border'], zerolinecolor=COLORS['border']),
    )
    return fig

# ─────────────────────────────────────────────
# CALLBACKS
# ─────────────────────────────────────────────

# Scatter plot
@app.callback(
    Output('scatter-plot', 'figure'),
    Input('scatter-x', 'value'),
    Input('scatter-y', 'value'),
)
def update_scatter(x_var, y_var):
    temp = df.dropna(subset=[x_var, y_var]).copy()
    temp[x_var] = temp[x_var].astype(float)
    temp[y_var] = temp[y_var].astype(float)

    fig = px.scatter(
        temp, x=x_var, y=y_var,
        color='Diagnostic',
        color_discrete_map={'CKD': COLORS['ckd'], 'Non-CKD': COLORS['notckd']},
        opacity=0.75,
        hover_data=['age'],
        trendline='ols',
        trendline_scope='overall',
        trendline_color_override=COLORS['accent'],
    )
    fig.update_traces(marker=dict(size=7, line=dict(width=0)))
    return dark_layout(fig)


# Donut chart
@app.callback(Output('donut-chart', 'figure'), Input('scatter-x', 'value'))
def update_donut(_):
    fig = go.Figure(go.Pie(
        labels=['CKD', 'Non-CKD'],
        values=[ckd_count, notckd_count],
        hole=0.65,
        marker=dict(colors=[COLORS['ckd'], COLORS['notckd']], line=dict(color=COLORS['bg'], width=3)),
        textfont=dict(size=13),
        hovertemplate='%{label}: %{value} patients (%{percent})<extra></extra>',
    ))
    fig.add_annotation(
        text=f"<b>{ckd_count/total*100:.0f}%</b><br>CKD",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=20, color=COLORS['ckd']),
        align='center',
    )
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=COLORS['text']),
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(size=13)),
        showlegend=True,
    )
    return fig


# Heatmap
@app.callback(Output('heatmap', 'figure'), Input('scatter-x', 'value'))
def update_heatmap(_):
    fig = go.Figure(go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns.tolist(),
        y=corr_matrix.index.tolist(),
        colorscale=[
            [0.0, COLORS['ckd']],
            [0.5, COLORS['card2']],
            [1.0, COLORS['notckd']],
        ],
        zmin=-1, zmax=1,
        text=corr_matrix.values.round(2),
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


# Boxplot
@app.callback(Output('box-plot', 'figure'), Input('box-var', 'value'))
def update_boxplot(var):
    temp = df.dropna(subset=[var]).copy()
    temp[var] = temp[var].astype(float)

    fig = px.box(
        temp, x='Diagnostic', y=var,
        color='Diagnostic',
        color_discrete_map={'CKD': COLORS['ckd'], 'Non-CKD': COLORS['notckd']},
        points='outliers',
        notched=True,
    )
    fig.update_traces(line_width=2, marker=dict(size=5, opacity=0.6))
    return dark_layout(fig)


# Business chart
@app.callback(Output('business-chart', 'figure'), Input('scatter-x', 'value'))
def update_business(_):
    temp = df.dropna(subset=['htn', 'dm']).copy()
    temp['htn_label'] = temp['htn'].str.strip().str.lower().map({'yes': 'Hypertension', 'no': 'Pas d\'hypertension'})
    temp['dm_label'] = temp['dm'].str.strip().str.lower().map({'yes': 'Diabète', 'no': 'Pas de diabète'})

    grouped = temp.groupby(['htn_label', 'dm_label', 'Diagnostic']).size().reset_index(name='count')

    fig = px.bar(
        grouped,
        x='htn_label', y='count',
        color='Diagnostic',
        facet_col='dm_label',
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
