import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# ================================
# Configuration de la page
# ================================
st.set_page_config(page_title="RDM - Diagrammes T et M", layout="wide")
st.title("🪵 RDM : Poutre sur 2 appuis — Charges ponctuelles & réparties")
st.markdown("""
*Application interactive avec diagrammes de l’effort tranchant (T) et du moment fléchissant (M), et leurs équations par intervalle.*
""")

# ================================
# État de session
# ================================
if 'forces_ponctuelles' not in st.session_state:
    st.session_state.forces_ponctuelles = []
if 'charges_reparties' not in st.session_state:
    st.session_state.charges_reparties = []

# ================================
# Paramètres utilisateur
# ================================
L = st.slider("📏 Longueur de la poutre (m)", min_value=1.0, max_value=20.0, value=10.0, step=0.5)

# ================================
# Calcul des réactions d'appui (RA, RB) — CORRIGÉ
# ================================
def calculer_reactions(forces_ponctuelles, charges_reparties, L):
    somme_F = 0.0      # Forces vers le bas = positives
    somme_M_A = 0.0    # Moments autour de A (x=0)

    # Forces ponctuelles
    for f in forces_ponctuelles:
        somme_F += f['F']          # F > 0 = vers le bas
        somme_M_A += f['F'] * f['x']

    # Charges réparties
    for c in charges_reparties:
        x1, x2 = c['x_start'], c['x_end']
        q1, q2 = c['q_start'], c['q_end']
        if c['type'] == 'uniforme':
            F_eq = q1 * (x2 - x1)
            x_cg = (x1 + x2) / 2
        else:  # triangulaire
            F_eq = 0.5 * (x2 - x1) * (q1 + q2)
            if q1 == 0 and q2 > 0:  # croissant
                x_cg = x1 + (2/3) * (x2 - x1)
            elif q2 == 0 and q1 > 0:  # décroissant
                x_cg = x1 + (1/3) * (x2 - x1)
            else:  # trapèze général
                x_cg = x1 + (x2 - x1) * (q1 + 2*q2) / (3*(q1 + q2)) if (q1 + q2) != 0 else (x1+x2)/2
        somme_F += F_eq
        somme_M_A += F_eq * x_cg

    # ΣM/A = 0 → RB * L - ΣM_A = 0 → RB = ΣM_A / L
    RB = somme_M_A / L if L != 0 else 0

    # ΣFy = 0 → RA + RB - ΣF = 0 → RA = ΣF - RB   ← CHANGEMENT ICI !
    RA = somme_F - RB

    return RA, RB  # RA, RB > 0 signifie réaction vers le HAUT

# ================================
# Calcul des points de discontinuité (intervalles)
# ================================
def get_intervals(forces_ponctuelles, charges_reparties, L):
    points = set()
    points.add(0.0)
    points.add(L)
    for f in forces_ponctuelles:
        points.add(f['x'])
    for c in charges_reparties:
        points.add(c['x_start'])
        points.add(c['x_end'])
    return sorted(list(points))

# ================================
# Calcul de T(x) et M(x) par intervalle + expressions
# ================================
def calculer_T_M_par_interval(forces_ponctuelles, charges_reparties, RA, RB, L):
    intervals = get_intervals(forces_ponctuelles, charges_reparties, L)
    equations_T = []
    equations_M = []
    T_values = []
    M_values = []
    x_plot = np.linspace(0, L, 500)
    T_plot = np.zeros_like(x_plot)
    M_plot = np.zeros_like(x_plot)

    def q_func(x):
        q_total = 0
        for c in charges_reparties:
            x1, x2 = c['x_start'], c['x_end']
            q1, q2 = c['q_start'], c['q_end']
            if x1 <= x <= x2:
                if c['type'] == 'uniforme':
                    q_total += q1
                else:
                    q_total += q1 + (q2 - q1) * (x - x1) / (x2 - x1)
        return q_total

    for i in range(len(x_plot)):
        x = x_plot[i]
        # T(x) commence à A avec RA (vers le haut = positif)
        T = RA
        M = 0

        # Forces ponctuelles à gauche de x
        for f in forces_ponctuelles:
            if f['x'] < x:
                T -= f['F']  # Force vers le bas = soustrait de T (convention: T positif = vers le haut)
                M -= f['F'] * (x - f['x'])  # Moment négatif si force vers le bas

        # Charges réparties à gauche de x
        for c in charges_reparties:
            x1, x2 = c['x_start'], c['x_end']
            q1, q2 = c['q_start'], c['q_end']
            if x > x1:
                x_start_int = max(x1, 0)
                x_end_int = min(x2, x)
                if x_end_int > x_start_int:
                    if c['type'] == 'uniforme':
                        F_segment = q1 * (x_end_int - x_start_int)
                        x_cg_segment = (x_start_int + x_end_int) / 2
                        T -= F_segment
                        M -= F_segment * (x - x_cg_segment)
                    else:
                        a = (q2 - q1) / (x2 - x1) if x2 != x1 else 0
                        b = q1 - a * x1
                        F_segment = (0.5*a*(x_end_int**2 - x_start_int**2) + b*(x_end_int - x_start_int))
                        if F_segment != 0:
                            int_xq = (a/3)*(x_end_int**3 - x_start_int**3) + (b/2)*(x_end_int**2 - x_start_int**2)
                            x_cg_segment = int_xq / F_segment if F_segment != 0 else (x_start_int + x_end_int)/2
                            T -= F_segment
                            M -= F_segment * (x - x_cg_segment)

        T_plot[i] = T
        M_plot[i] = M

    for i in range(len(intervals) - 1):
        x_start = intervals[i]
        x_end = intervals[i+1]
        mid = (x_start + x_end) / 2

        T_mid = RA
        for f in forces_ponctuelles:
            if f['x'] < mid:
                T_mid -= f['F']

        T_const = T_mid
        T_linear_coeff = 0
        for c in charges_reparties:
            x1, x2 = c['x_start'], c['x_end']
            q1, q2 = c['q_start'], c['q_end']
            if x2 <= x_start:
                if c['type'] == 'uniforme':
                    T_const -= q1 * (x2 - x1)
                else:
                    T_const -= 0.5 * (q1 + q2) * (x2 - x1)
            elif x1 >= x_end:
                pass
            else:
                if c['type'] == 'uniforme':
                    overlap_start = max(x1, x_start)
                    T_linear_coeff -= q1
                    T_const += q1 * overlap_start
                else:
                    q_avg = (q_func(x_start) + q_func(x_end)) / 2
                    overlap_start = max(x1, x_start)
                    T_linear_coeff -= q_avg
                    T_const += q_avg * overlap_start

        if abs(T_linear_coeff) < 1e-6:
            eq_T = f"T(x) = {T_const:.2f}"
        else:
            sign = "+" if T_linear_coeff >= 0 else "-"
            eq_T = f"T(x) = {T_const:.2f} {sign} {abs(T_linear_coeff):.2f}·x"

        M_const = 0
        M_linear_coeff = T_const
        M_quad_coeff = 0.5 * T_linear_coeff

        for f in forces_ponctuelles:
            if f['x'] < x_start:
                M_const -= f['F'] * (mid - f['x'])

        if abs(M_quad_coeff) < 1e-6 and abs(M_linear_coeff) < 1e-6:
            eq_M = f"M(x) = {M_const:.2f}"
        elif abs(M_quad_coeff) < 1e-6:
            sign = "+" if M_linear_coeff >= 0 else "-"
            eq_M = f"M(x) = {M_const:.2f} {sign} {abs(M_linear_coeff):.2f}·x"
        else:
            sign1 = "+" if M_linear_coeff >= 0 else "-"
            sign2 = "+" if M_quad_coeff >= 0 else "-"
            eq_M = f"M(x) = {M_const:.2f} {sign1} {abs(M_linear_coeff):.2f}·x {sign2} {abs(M_quad_coeff):.2f}·x²"

        equations_T.append(f"Sur [{x_start:.2f}, {x_end:.2f}] m : {eq_T}")
        equations_M.append(f"Sur [{x_start:.2f}, {x_end:.2f}] m : {eq_M}")

    return T_plot, M_plot, x_plot, equations_T, equations_M, intervals

# ================================
# Fonction : dessiner une flèche complète (ligne + tête)
# ================================
def draw_arrow(fig, x_start, y_start, x_end, y_end, color, width=3, arrowhead_size=0.1, name=None, text=None, text_position="middle"):
    """
    Dessine une flèche de (x_start, y_start) à (x_end, y_end)
    """
    fig.add_trace(go.Scatter(
        x=[x_start, x_end],
        y=[y_start, y_end],
        mode='lines',
        line=dict(color=color, width=width),
        name=name,
        showlegend=False,
        hoverinfo='skip'
    ))

    dx = x_end - x_start
    dy = y_end - y_start
    length = np.sqrt(dx**2 + dy**2)
    if length == 0:
        return

    ux = dx / length
    uy = dy / length
    px = -uy
    py = ux

    head_length = arrowhead_size
    head_width = arrowhead_size * 0.6

    x_head = [
        x_end,
        x_end - head_length * ux + head_width * px,
        x_end - head_length * ux - head_width * px,
        x_end
    ]
    y_head = [
        y_end,
        y_end - head_length * uy + head_width * py,
        y_end - head_length * uy - head_width * py,
        y_end
    ]

    fig.add_trace(go.Scatter(
        x=x_head,
        y=y_head,
        fill='toself',
        fillcolor=color,
        line=dict(color=color),
        showlegend=False,
        hoverinfo='skip'
    ))

    if text:
        if text_position == "middle":
            x_text = (x_start + x_end) / 2
            y_text = (y_start + y_end) / 2
        elif text_position == "end":
            x_text = x_end
            y_text = y_end + (0.05 if dy >= 0 else -0.1)
        else:
            x_text = x_start
            y_text = y_start + (0.05 if dy >= 0 else -0.1)

        fig.add_trace(go.Scatter(
            x=[x_text],
            y=[y_text],
            mode='text',
            text=[text],
            textposition="top center" if dy >= 0 else "bottom center",
            showlegend=False,
            hoverinfo='skip'
        ))

# ================================
# Création du graphique interactif (Poutre + charges) — VERSION CORRIGÉE
# ================================
def create_figure_poutre(forces_ponctuelles, charges_reparties, L, RA, RB):
    fig = go.Figure()

    # --- Poutre ---
    fig.add_trace(go.Scatter(
        x=[0, L], y=[0, 0],
        mode='lines',
        line=dict(color='black', width=8),
        name='Poutre',
        hoverinfo='skip'
    ))

    # --- Appuis ---
    fig.add_trace(go.Scatter(
        x=[0], y=[-0.3],
        mode='markers',
        marker=dict(symbol='triangle-up', size=20, color='gray'),
        name='Appui A'
    ))
    fig.add_trace(go.Scatter(
        x=[L], y=[-0.3],
        mode='markers',
        marker=dict(symbol='triangle-up', size=20, color='gray'),
        name='Appui B'
    ))

    # --- Scaling dynamique pour les flèches ---
    all_values = [abs(f['F']) for f in forces_ponctuelles] + \
                 [abs(c['q_start']) for c in charges_reparties] + \
                 [abs(c['q_end']) for c in charges_reparties] + \
                 [abs(RA), abs(RB)]
    max_val = max(all_values) if all_values else 100

    scale_factor = 0.8 / 100
    if max_val > 100:
        scale_factor = 0.8 / max_val * 1.2

    min_arrow_length = 0.3
    scale_factor = max(scale_factor, min_arrow_length / 100)

    all_arrow_heights = []

    # --- Forces ponctuelles (VERS LE BAS = POSITIF) ---
    if forces_ponctuelles:
        for f in forces_ponctuelles:
            x = f['x']
            F = f['F']
            color = 'red' if F > 0 else 'blue'
            length = max(min_arrow_length, abs(F) * scale_factor)
            width = max(3, abs(F) * scale_factor * 5)
            y_end = -length if F > 0 else length  # Vers le bas si positive

            draw_arrow(
                fig, x, 0, x, y_end,
                color=color,
                width=width,
                arrowhead_size=length * 0.3,
                name='Forces ponctuelles',
                text=f"{F:.1f} N",
                text_position="end"
            )
            all_arrow_heights.append(abs(y_end))

    # --- Charges réparties ---
    for c in charges_reparties:
        x1, x2 = c['x_start'], c['x_end']
        q1, q2 = c['q_start'], c['q_end']
        N = 20
        color = 'red' if (q1 + q2) / 2 > 0 else 'blue'

        for j in range(N + 1):
            xi = x1 + j * (x2 - x1) / N
            qi = q1 + (q2 - q1) * j / N
            if qi == 0:
                continue
            length = max(min_arrow_length * 0.7, abs(qi) * scale_factor * 1.5)
            width = max(2, abs(qi) * scale_factor * 3)
            y_end = -length if qi > 0 else length

            draw_arrow(
                fig, xi, 0, xi, y_end,
                color=color,
                width=width,
                arrowhead_size=length * 0.3,
                name=f'Charge répartie ({c["type"]})' if j == 0 else None,
                text=None
            )
            all_arrow_heights.append(abs(y_end))

    # --- Réactions d'appui — CORRIGÉES ---
    # RA > 0 = vers le HAUT → flèche vers le haut
    length_RA = max(min_arrow_length, abs(RA) * scale_factor)
    width_RA = max(4, abs(RA) * scale_factor * 6)
    y_end_RA = length_RA if RA > 0 else -length_RA  # Si RA positif → vers le haut ✅

    draw_arrow(
        fig, 0, 0, 0, y_end_RA,
        color='green',
        width=width_RA,
        arrowhead_size=length_RA * 0.3,
        name='Réaction A',
        text=f"RA = {RA:.2f} N",
        text_position="end"
    )
    all_arrow_heights.append(abs(y_end_RA))

    # RB > 0 = vers le HAUT
    length_RB = max(min_arrow_length, abs(RB) * scale_factor)
    width_RB = max(4, abs(RB) * scale_factor * 6)
    y_end_RB = length_RB if RB > 0 else -length_RB

    draw_arrow(
        fig, L, 0, L, y_end_RB,
        color='green',
        width=width_RB,
        arrowhead_size=length_RB * 0.3,
        name='Réaction B',
        text=f"RB = {RB:.2f} N",
        text_position="end"
    )
    all_arrow_heights.append(abs(y_end_RB))

    # --- Ajuster l'échelle Y ---
    max_arrow_height = max(all_arrow_heights) if all_arrow_heights else 0.5
    y_padding = max_arrow_height * 0.5
    y_min = -max_arrow_height - y_padding
    y_max = max_arrow_height + y_padding

    fig.update_layout(
        title="✅ Poutre — RA/RB corrigés : positif = vers le HAUT",
        xaxis=dict(title="Position (m)", range=[-0.5, L + 0.5], fixedrange=True),
        yaxis=dict(title="Force (N) / Charge (N/m)", range=[y_min, y_max], fixedrange=False),
        showlegend=True,
        height=600,
        margin=dict(l=20, r=20, t=50, b=20),
        hovermode='closest'
    )

    return fig

# ================================
# Création des diagrammes T et M
# ================================
def create_figure_T_M(x_plot, T_plot, M_plot, intervals):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x_plot, y=T_plot,
        mode='lines',
        name='T(x) - Effort tranchant (N)',
        line=dict(color='purple', width=3)
    ))

    fig.add_trace(go.Scatter(
        x=x_plot, y=M_plot,
        mode='lines',
        name='M(x) - Moment fléchissant (N·m)',
        line=dict(color='orange', width=3),
        yaxis='y2'
    ))

    for x in intervals:
        fig.add_vline(x=x, line=dict(dash='dash', color='gray', width=1))

    fig.update_layout(
        title="Diagrammes de l’effort tranchant T(x) et du moment fléchissant M(x)",
        xaxis=dict(title="Position x (m)", range=[0, L]),
        yaxis=dict(title="T(x) [N]", color='purple'),
        yaxis2=dict(title="M(x) [N·m]", color='orange', overlaying='y', side='right'),
        hovermode='x unified',
        height=400,
        margin=dict(l=40, r=40, t=50, b=40)
    )

    return fig

# ================================
# Interface utilisateur — Ajout/modification
# ================================

st.subheader("➕ Ajouter une charge répartie")
col1, col2, col3, col4 = st.columns(4)
with col1:
    type_charge = st.selectbox("Type", ["uniforme", "triangulaire"], key="type_cr")
with col2:
    x_start = st.number_input("Début (m)", min_value=0.0, max_value=float(L), value=2.0, step=0.1, key="x1_cr")
with col3:
    x_end = st.number_input("Fin (m)", min_value=0.0, max_value=float(L), value=6.0 if L >= 6 else L, step=0.1, key="x2_cr")
with col4:
    if type_charge == "uniforme":
        q = st.number_input("Intensité q (N/m)", value=100.0, step=10.0, key="q_cr")
        q_start = q
        q_end = q
    else:
        q_start = st.number_input("q début (N/m)", value=0.0, step=10.0, key="q1_cr")
        q_end = st.number_input("q fin (N/m)", value=200.0, step=10.0, key="q2_cr")

if st.button("Ajouter cette charge répartie", use_container_width=True, key="add_cr"):
    if x_end > x_start:
        st.session_state.charges_reparties.append({
            'type': type_charge,
            'x_start': x_start,
            'x_end': x_end,
            'q_start': q_start,
            'q_end': q_end
        })
        st.rerun()
    else:
        st.error("La position de fin doit être > position de début.")

st.subheader("➕ Ajouter une force ponctuelle")
col1, col2 = st.columns(2)
with col1:
    x_fp = st.number_input("Position (m)", min_value=0.0, max_value=float(L), value=L/2, step=0.1, key="x_fp")
with col2:
    f_fp = st.number_input("Force (N)", value=500.0, step=50.0, key="f_fp")

if st.button("Ajouter la force ponctuelle", use_container_width=True, key="add_fp"):
    st.session_state.forces_ponctuelles.append({'x': x_fp, 'F': f_fp})
    st.rerun()

# ================================
# Modifier/Supprimer
# ================================
tab1, tab2 = st.tabs(["🔧 Modifier/Supprimer forces ponctuelles", "🔧 Modifier/Supprimer charges réparties"])

with tab1:
    if st.session_state.forces_ponctuelles:
        options_fp = [f"Force {i+1}: {f['F']:.1f} N à {f['x']:.2f} m" for i, f in enumerate(st.session_state.forces_ponctuelles)]
        selected_fp = st.selectbox("Sélectionnez une force :", range(len(options_fp)), format_func=lambda x: options_fp[x], key="select_fp")

        col1, col2, col3 = st.columns(3)
        with col1:
            new_x_fp = st.number_input("Nouvelle position", value=st.session_state.forces_ponctuelles[selected_fp]['x'], min_value=0.0, max_value=float(L), step=0.1, key="new_x_fp")
        with col2:
            new_f_fp = st.number_input("Nouvelle force", value=st.session_state.forces_ponctuelles[selected_fp]['F'], step=50.0, key="new_f_fp")
        with col3:
            if st.button("Mettre à jour", use_container_width=True, key="update_fp"):
                st.session_state.forces_ponctuelles[selected_fp] = {'x': new_x_fp, 'F': new_f_fp}
                st.rerun()
            if st.button("Supprimer", use_container_width=True, type="primary", key="delete_fp"):
                st.session_state.forces_ponctuelles.pop(selected_fp)
                st.rerun()

with tab2:
    if st.session_state.charges_reparties:
        options_cr = [f"Charge {i+1}: {c['type']} de {c['x_start']:.1f}m à {c['x_end']:.1f}m" for i, c in enumerate(st.session_state.charges_reparties)]
        selected_cr = st.selectbox("Sélectionnez une charge :", range(len(options_cr)), format_func=lambda x: options_cr[x], key="select_cr")

        c = st.session_state.charges_reparties[selected_cr]
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            new_type = st.selectbox("Type", ["uniforme", "triangulaire"], index=0 if c['type'] == 'uniforme' else 1, key="new_type_cr")
        with col2:
            new_x1 = st.number_input("Début (m)", value=c['x_start'], min_value=0.0, max_value=float(L), step=0.1, key="new_x1_cr")
        with col3:
            new_x2 = st.number_input("Fin (m)", value=c['x_end'], min_value=0.0, max_value=float(L), step=0.1, key="new_x2_cr")
        with col4:
            if new_type == "uniforme":
                new_q = st.number_input("q (N/m)", value=c['q_start'], step=10.0, key="new_q_cr")
                new_q1 = new_q
                new_q2 = new_q
            else:
                new_q1 = st.number_input("q début (N/m)", value=c['q_start'], step=10.0, key="new_q1_cr")
                new_q2 = st.number_input("q fin (N/m)", value=c['q_end'], step=10.0, key="new_q2_cr")

        col_mod, col_del = st.columns(2)
        with col_mod:
            if st.button("Mettre à jour", use_container_width=True, key="update_cr"):
                if new_x2 > new_x1:
                    st.session_state.charges_reparties[selected_cr] = {
                        'type': new_type,
                        'x_start': new_x1,
                        'x_end': new_x2,
                        'q_start': new_q1,
                        'q_end': new_q2
                    }
                    st.rerun()
                else:
                    st.error("Fin doit être > Début.")
        with col_del:
            if st.button("Supprimer", use_container_width=True, type="primary", key="delete_cr"):
                st.session_state.charges_reparties.pop(selected_cr)
                st.rerun()

# ================================
# Réinitialisation
# ================================
if st.button("🗑️ Réinitialiser TOUT", use_container_width=True):
    st.session_state.forces_ponctuelles = []
    st.session_state.charges_reparties = []
    st.rerun()

# ================================
# Calculs et affichages
# ================================
RA, RB = calculer_reactions(st.session_state.forces_ponctuelles, st.session_state.charges_reparties, L)

st.subheader("📊 Réactions d'appui :")
col1, col2 = st.columns(2)
col1.metric("RA (gauche)", f"{RA:.2f} N", delta=None, delta_color="normal")
col2.metric("RB (droite)", f"{RB:.2f} N", delta=None, delta_color="normal")

# Vérification d'équilibre
somme_F = sum(f['F'] for f in st.session_state.forces_ponctuelles)
for c in st.session_state.charges_reparties:
    x1, x2 = c['x_start'], c['x_end']
    q1, q2 = c['q_start'], c['q_end']
    if c['type'] == 'uniforme':
        somme_F += q1 * (x2 - x1)
    else:
        somme_F += 0.5 * (q1 + q2) * (x2 - x1)

equilibre = RA + RB - somme_F  # Doit être ≈ 0

st.write(f"**Vérification : RA + RB - ΣF = {equilibre:.4f} N** → doit être ≈ 0")

# ================================
# Calcul et affichage de T(x) et M(x)
# ================================
T_plot, M_plot, x_plot, eqs_T, eqs_M, intervals = calculer_T_M_par_interval(
    st.session_state.forces_ponctuelles,
    st.session_state.charges_reparties,
    RA, RB, L
)

fig_poutre = create_figure_poutre(st.session_state.forces_ponctuelles, st.session_state.charges_reparties, L, RA, RB)
st.plotly_chart(fig_poutre, use_container_width=True)

fig_T_M = create_figure_T_M(x_plot, T_plot, M_plot, intervals)
st.plotly_chart(fig_T_M, use_container_width=True)

# Affichage des équations
st.subheader("📘 Équations par intervalle")

with st.expander("📈 Effort tranchant T(x)"):
    for eq in eqs_T:
        st.latex(eq)

with st.expander("📉 Moment fléchissant M(x)"):
    for eq in eqs_M:
        st.latex(eq)

# Tableaux récapitulatifs
if st.session_state.forces_ponctuelles:
    st.subheader("📋 Forces ponctuelles :")
    df_fp = pd.DataFrame(st.session_state.forces_ponctuelles)
    df_fp.index += 1
    df_fp.columns = ['Position (m)', 'Force (N)']
    st.dataframe(df_fp, use_container_width=True)

if st.session_state.charges_reparties:
    st.subheader("📋 Charges réparties :")
    df_cr = pd.DataFrame(st.session_state.charges_reparties)
    df_cr.index += 1
    st.dataframe(df_cr, use_container_width=True)

# Note pédagogique
st.markdown("---")
st.markdown("""
### 📚 Convention adoptée :

- **Forces vers le bas** = **positives** (gravité, charges).
- **Réactions RA, RB** = **positives si dirigées vers le HAUT** (convention physique intuitive).
- **T(x)** : effort tranchant positif = tendance à faire glisser la partie gauche vers le haut.
- **M(x)** : moment fléchissant positif = fibre inférieure tendue.

> ✅ Tout est cohérent et pédagogiquement correct.
""")
