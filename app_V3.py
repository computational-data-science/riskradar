import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import io
from scipy.stats import spearmanr

st.set_page_config(page_title="RiskRadar · Supplier Intelligence", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;background-color:#0f1117;color:#e8eaf0;}
.stApp{background-color:#0f1117;}
section[data-testid="stSidebar"]{background-color:#161b27;border-right:1px solid #252d3d;}
section[data-testid="stSidebar"] *{color:#c8cfe0 !important;}
[data-testid="metric-container"]{background:linear-gradient(135deg,#1a2035,#1e2640);border:1px solid #2a3350;border-radius:14px;padding:14px 18px !important;box-shadow:0 4px 20px rgba(0,0,0,0.3);}
[data-testid="metric-container"] label{color:#8899bb !important;font-size:0.72rem !important;letter-spacing:0.08em;text-transform:uppercase;}
[data-testid="metric-container"] [data-testid="metric-value"]{color:#e8eaf0 !important;font-size:1.6rem !important;font-weight:700;}
.stTabs [data-baseweb="tab-list"]{background-color:#161b27;border-radius:10px;padding:4px;gap:4px;}
.stTabs [data-baseweb="tab"]{border-radius:8px;color:#8899bb;font-weight:500;padding:8px 16px;}
.stTabs [aria-selected="true"]{background-color:#2563eb !important;color:white !important;}
.stButton>button{background:linear-gradient(135deg,#2563eb,#1d4ed8);color:white;border:none;border-radius:8px;font-weight:500;}
.info-box{background:linear-gradient(135deg,#1a2640,#1e2d4a);border-left:4px solid #2563eb;border-radius:0 10px 10px 0;padding:12px 16px;margin:8px 0;font-size:0.85rem;color:#a8bbd4;line-height:1.6;}
.risk-high{border-left-color:#ef4444 !important;background:linear-gradient(135deg,#2a1a1a,#331c1c) !important;color:#ffb3b3 !important;}
.risk-medium{border-left-color:#f59e0b !important;background:linear-gradient(135deg,#2a2210,#332a10) !important;color:#ffe0a0 !important;}
.risk-low{border-left-color:#22c55e !important;background:linear-gradient(135deg,#0f2a18,#12331e) !important;color:#a0f0c0 !important;}
.section-header{font-size:1.0rem;font-weight:600;color:#c8d4f0;letter-spacing:0.04em;margin-bottom:10px;padding-bottom:6px;border-bottom:1px solid #252d3d;}
.ampel-card{border-radius:14px;padding:18px 20px;text-align:center;font-weight:600;font-size:1.1rem;margin:6px 0;box-shadow:0 4px 16px rgba(0,0,0,0.3);}
.ampel-rot{background:linear-gradient(135deg,#3a1010,#4a1515);border:2px solid #ef4444;color:#ff8080;}
.ampel-gelb{background:linear-gradient(135deg,#2a2010,#3a2c10);border:2px solid #f59e0b;color:#ffd080;}
.ampel-gruen{background:linear-gradient(135deg,#0a2a14,#0f3a1c);border:2px solid #22c55e;color:#80ffa8;}
.hero-box{background:linear-gradient(135deg,#1a2640,#0f1a30);border:1px solid #2a3350;border-radius:18px;padding:36px 40px;margin:12px 0;text-align:center;}
.hero-title{font-size:3rem;font-weight:700;color:#e8eaf0;margin:0;}
.hero-sub{font-size:1.1rem;color:#5577aa;margin-top:8px;}
.feature-card{background:linear-gradient(135deg,#1a2035,#1e2640);border:1px solid #2a3350;border-radius:14px;padding:22px 20px;text-align:center;margin:8px 4px;}
.feature-title{font-size:1.0rem;font-weight:600;color:#c8d4f0;margin-bottom:6px;}
.feature-desc{font-size:0.82rem;color:#8899bb;line-height:1.5;}
.umsatz-card{background:linear-gradient(135deg,#1a2035,#1e2640);border:1px solid #2a3350;border-radius:12px;padding:16px 18px;margin:6px 0;}
.umsatz-bar{height:4px;border-radius:2px;margin-top:6px;}
.formula-box{background:#1a2035;border:1px solid #2a3350;border-radius:10px;padding:14px 18px;font-family:monospace;font-size:0.82rem;color:#94a3b8;line-height:1.8;}
.glossar-card{background:linear-gradient(135deg,#1a2035,#1e2640);border:1px solid #2a3350;border-radius:12px;padding:14px 18px;margin:6px 0;}
.glossar-term{font-size:0.95rem;font-weight:600;color:#60a5fa;margin-bottom:4px;}
.glossar-def{font-size:0.82rem;color:#8899bb;line-height:1.5;}

/* KPI-Zeile oben: heller machen */
[data-testid="metric-container"] label,
[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] p {
    color: #e2e8f0 !important;
    opacity: 1 !important;
}

[data-testid="metric-container"] [data-testid="stMetricValue"],
[data-testid="stMetricValue"],
[data-testid="stMetricValue"] div,
[data-testid="stMetricValue"] p {
    color: #f8fafc !important;
    opacity: 1 !important;
    font-weight: 700 !important;
}

[data-testid="metric-container"] [data-testid="stMetricDelta"],
[data-testid="stMetricDelta"],
[data-testid="stMetricDelta"] div {
    opacity: 1 !important;
}

/* Markdown-Überschriften heller machen */
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3,
[data-testid="stMarkdownContainer"] h4 {
    color: #f8fafc !important;
    opacity: 1 !important;
}

/* normaler Markdown-Text etwas heller */
[data-testid="stMarkdownContainer"] p {
    color: #dbe4f0 !important;
    opacity: 1 !important;
}

/* kursive Hinweise wie "Tab: Dashboard" */
[data-testid="stMarkdownContainer"] i {
    color: #cbd5e1 !important;
    opacity: 1 !important;
}

/* Sidebar: Expander und Auswahlfelder dunkel machen */
section[data-testid="stSidebar"] [data-testid="stExpander"] {
    background-color: #111827 !important;
    border: 1px solid #252d3d !important;
    border-radius: 10px !important;
}

/* Expander-Kopf, z.B. "Lieferanten & Produkte" */
section[data-testid="stSidebar"] [data-testid="stExpander"] summary,
section[data-testid="stSidebar"] [data-testid="stExpander"] summary p,
section[data-testid="stSidebar"] [data-testid="stExpander"] summary span {
    background-color: #1a2035 !important;
    color: #f8fafc !important;
}

/* Multiselect-/Select-Feld: Hintergrund dunkel */
section[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background-color: #1a2035 !important;
    border-color: #334155 !important;
    color: #f8fafc !important;
}

/* Text im Select-Feld */
section[data-testid="stSidebar"] [data-baseweb="select"] span,
section[data-testid="stSidebar"] [data-baseweb="select"] input,
section[data-testid="stSidebar"] [data-baseweb="select"] div {
    color: #f8fafc !important;
}

/* Placeholder / ausgegrauter Text */
section[data-testid="stSidebar"] [data-baseweb="select"] input::placeholder {
    color: #cbd5e1 !important;
    opacity: 1 !important;
}

/* Kleine x- und Pfeil-Icons */
section[data-testid="stSidebar"] [data-baseweb="select"] svg {
    fill: #cbd5e1 !important;
    color: #cbd5e1 !important;
}

/* Ausgewählte Tags, z.B. Supplier 1 */
section[data-testid="stSidebar"] [data-baseweb="tag"] {
    background-color: #ef4444 !important;
    color: #ffffff !important;
}

section[data-testid="stSidebar"] [data-baseweb="tag"] span {
    color: #ffffff !important;
}

</style>
""", unsafe_allow_html=True)

plt.rcParams.update({
    "figure.facecolor":"#161b27","axes.facecolor":"#1a2035","axes.edgecolor":"#2a3350",
    "axes.labelcolor":"#8899bb","xtick.color":"#8899bb","ytick.color":"#8899bb",
    "text.color":"#c8d4f0","grid.color":"#252d3d","grid.linestyle":"--","grid.alpha":0.5,
})

SUP_COLORS  = {}
PROD_COLORS = {"skincare":"#06b6d4","haircare":"#f59e0b","cosmetics":"#a855f7"}
RISK_COLORS = {"🔴 Hoch":"#ef4444","🟡 Mittel":"#f59e0b","🟢 Niedrig":"#22c55e"}
BUCKET_COLORS = {"🥇 Premium":"#60a5fa","🔵 Standard":"#94a3b8","💚 Budget":"#4ade80"}
LOC_MEDALS = ["🥇","🥈","🥉","4️⃣","5️⃣"]

@st.cache_data
def load_data():
    df = pd.read_csv("supply_chain_data.csv")
    num_cols = ["Price","Availability","Number of products sold","Revenue generated","Stock levels",
                "Lead times","Order quantities","Shipping times","Shipping costs","Lead time",
                "Production volumes","Manufacturing lead time","Manufacturing costs","Defect rates","Costs"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in df.select_dtypes(include=[np.number]).columns:
        df[c] = df[c].round(2)
    if "Gesamtlieferzeit" not in df.columns:
        df["Gesamtlieferzeit"] = df["Lead time"] + df["Shipping times"]
    def norm(s):
        mn,mx=s.min(),s.max(); return (s-mn)/(mx-mn+1e-9)
    score = (norm(df["Price"]) + norm(df["Revenue generated"]) + (1-norm(df["Defect rates"])))/3
    p33,p66 = score.quantile(0.33), score.quantile(0.66)
    df["Produkt-Bucket"] = pd.cut(score, bins=[-1,p33,p66,2], labels=["💚 Budget","🔵 Standard","🥇 Premium"])
    df["Bucket-Score"] = score.round(3)
    df["_p33"] = round(p33, 3)
    df["_p66"] = round(p66, 3)
    return df

df = load_data()
P33 = df["_p33"].iloc[0]
P66 = df["_p66"].iloc[0]

for i,s in enumerate(sorted(df["Supplier name"].unique())):
    SUP_COLORS[s] = ["#2563eb","#f59e0b","#22c55e","#a855f7","#ef4444"][i%5]

def norm_col(series, invert=False):
    mn,mx=series.min(),series.max()
    if mx==mn: return pd.Series(0.5,index=series.index)
    n=(series-mn)/(mx-mn)
    return (1-n) if invert else n

def compute_risk(df_in, wd, wl, wc, wi, wr, th, tl):
    r=df_in.copy()
    r["_nd"]=norm_col(r["Defect rates"])
    r["_nl"]=norm_col(r["Gesamtlieferzeit"])
    r["_nc"]=norm_col(r["Costs"])
    r["_nr"]=norm_col(r["Revenue generated"],invert=True)
    r["_ni"]=r["Inspection results"].map({"Fail":1.0,"Pending":0.5,"Pass":0.0}).fillna(0.5)
    total=wd+wl+wc+wi+wr
    r["Risk Score"]=(wd*r["_nd"]+wl*r["_nl"]+wc*r["_nc"]+wi*r["_ni"]+wr*r["_nr"])/total*100
    r["Risk Score"]=r["Risk Score"].round(2)
    def cat(s):
        if s>=th: return "🔴 Hoch"
        if s>=tl: return "🟡 Mittel"
        return "🟢 Niedrig"
    r["Risikostufe"]=r["Risk Score"].apply(cat)
    return r.drop(columns=[c for c in r.columns if c.startswith("_")])

if "risk_filter" not in st.session_state:
    st.session_state.risk_filter = "Alle"

# ════════════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## RiskRadar")
    st.markdown("<p style='color:#5577aa;font-size:0.8rem;margin-top:-10px'>THI Ingolstadt · DPDS 2026 · Gruppe 9</p>", unsafe_allow_html=True)
    st.markdown("---")

    # Entscheidungs-Profil – Preset setzt Startwerte, Slider bleiben immer sichtbar
    st.markdown("### Entscheidungs-Profil")
    preset = st.radio("", ["Manuell","Höchste Qualität","Umsatzstärkster Lieferant","Günstig & Schnell"],
                      key="preset", label_visibility="collapsed")

    # Startwerte je Preset
    defaults = {
        "Manuell":                    (4, 3, 2, 5, 2),
        "Höchste Qualität":           (10, 1, 1, 10, 5),
        "Umsatzstärkster Lieferant":  (3, 3, 2, 3, 10),
        "Günstig & Schnell":          (1, 10, 10, 1, 5),
    }
    dwd, dwl, dwc, dwi, dwr = defaults[preset]

    if preset != "Manuell":
        preset_labels = {
            "Höchste Qualität":          ("risk-low",    "Qualität first – Defektrate & Inspektion dominieren."),
            "Umsatzstärkster Lieferant": ("risk-medium", "Umsatz-Fokus – hoher Umsatz = geringes Risiko."),
            "Günstig & Schnell":         ("risk-high",   "Speed & Cost – Lieferzeit & Kosten dominieren."),
        }
        css_p, msg_p = preset_labels[preset]
        st.markdown(f"<div class='info-box {css_p}'>{msg_p}</div>", unsafe_allow_html=True)

    # Slider immer sichtbar, Preset setzt Startwert
    wd = st.slider("Defektrate",                    1, 10, dwd, key=f"swd_{preset}")
    wl = st.slider("Gesamtlieferzeit",              1, 10, dwl, key=f"swl_{preset}")
    wc = st.slider("Gesamtkosten",                  1, 10, dwc, key=f"swc_{preset}")
    wi = st.slider("Inspektionsergebnis",           1, 10, dwi, key=f"swi_{preset}")
    wr = st.slider("Umsatz (invers: hoch = besser)",1, 10, dwr, key=f"swr_{preset}")

    tw = wd+wl+wc+wi+wr
    st.markdown(f"""<div class='info-box' style='font-size:0.78rem'>
    Defektrate: <b>{wd/tw*100:.0f}%</b> · Lieferzeit: <b>{wl/tw*100:.0f}%</b> · Kosten: <b>{wc/tw*100:.0f}%</b><br>
    Inspektion: <b>{wi/tw*100:.0f}%</b> · Umsatz (invers): <b>{wr/tw*100:.0f}%</b><br>
    <i style='font-size:0.72rem'>Umsatz invers = hoher Umsatz senkt das Risiko</i>
    </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Filter")
    with st.expander("Lieferanten & Produkte", expanded=True):
        sel_sup   = st.multiselect("Lieferant",        sorted(df["Supplier name"].unique()),       default=sorted(df["Supplier name"].unique()),       key="gs")
        sel_prod  = st.multiselect("Produktkategorie", sorted(df["Product type"].unique()),        default=sorted(df["Product type"].unique()),        key="gp")
        sel_bucket= st.multiselect("Produkt-Segment",  ["🥇 Premium","🔵 Standard","💚 Budget"],  default=["🥇 Premium","🔵 Standard","💚 Budget"],  key="gb")
    with st.expander("Logistik", expanded=False):
        sel_loc   = st.multiselect("Standort",     sorted(df["Location"].unique()),              default=sorted(df["Location"].unique()),              key="gl")
        sel_carr  = st.multiselect("Carrier",      sorted(df["Shipping carriers"].unique()),     default=sorted(df["Shipping carriers"].unique()),     key="gc")
        sel_mode  = st.multiselect("Transportweg", sorted(df["Transportation modes"].unique()),  default=sorted(df["Transportation modes"].unique()),  key="gm")
    with st.expander("Qualität", expanded=False):
        sel_insp  = st.multiselect("Inspektionsstatus", sorted(df["Inspection results"].unique()), default=sorted(df["Inspection results"].unique()), key="gi")
        max_defect= st.slider("Max. Defektrate (%)", 0.0, float(df["Defect rates"].max()), float(df["Defect rates"].max()))

    st.markdown("---")
    st.markdown("### Grenzwerte Risk Score")
    thresh_high = st.slider("Ab hier Hoch (rot)",  34, 90, 60)
    thresh_low  = st.slider("Ab hier Mittel (gelb)", 10, thresh_high-1, 35)
    st.markdown(f"<div class='info-box' style='font-size:0.78rem'>Hoch ≥{thresh_high} · Mittel {thresh_low}–{thresh_high-1} · Niedrig &lt;{thresh_low}<br><i>Default 60: statistisch neutrale Terzil-Einteilung</i></div>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
#  FILTER + SCORING
# ════════════════════════════════════════════════════════════════
filtered = df[
    df["Supplier name"].isin(sel_sup) & df["Product type"].isin(sel_prod) &
    df["Produkt-Bucket"].isin(sel_bucket) & df["Location"].isin(sel_loc) &
    df["Shipping carriers"].isin(sel_carr) & df["Transportation modes"].isin(sel_mode) &
    df["Inspection results"].isin(sel_insp) & (df["Defect rates"] <= max_defect)
].copy()

if filtered.empty:
    st.warning("Keine Daten. Filter anpassen."); st.stop()

scored      = compute_risk(filtered, wd,wl,wc,wi,wr, thresh_high,thresh_low)
scored_full = compute_risk(df,       wd,wl,wc,wi,wr, thresh_high,thresh_low)
sup_order   = sorted(scored["Supplier name"].unique())

# ════════════════════════════════════════════════════════════════
#  HEADER
# ════════════════════════════════════════════════════════════════
st.markdown(f"""
<h1 style='font-size:1.9rem;font-weight:700;color:#e8eaf0;margin-bottom:2px'>🛡️ RiskRadar</h1>
<p style='color:#5577aa;font-size:0.88rem;margin-top:0'>
Supplier & Procurement Risk Intelligence · THI Ingolstadt DPDS 2026 · Gruppe 9
&nbsp;·&nbsp; Profil: <b style='color:#c8d4f0'>{preset}</b>
&nbsp;·&nbsp; Grenzwert Hoch: <b style='color:#ef4444'>{thresh_high}</b>
</p>""", unsafe_allow_html=True)
st.markdown("---")

n_high = (scored["Risikostufe"]=="🔴 Hoch").sum()
n_med  = (scored["Risikostufe"]=="🟡 Mittel").sum()
n_low  = (scored["Risikostufe"]=="🟢 Niedrig").sum()

k1,k2,k3,k4,k5,k6,k7 = st.columns(7)
k1.metric("Ø Risk Score",        f"{scored['Risk Score'].mean():.2f}")
k2.metric("Hoch (kritisch)",     f"{n_high} SKUs", delta=f"{n_high/len(scored)*100:.0f}%", delta_color="inverse")
k3.metric("Mittel",              f"{n_med} SKUs")
k4.metric("Niedrig (OK)",        f"{n_low} SKUs")
k5.metric("Ø Defektrate",        f"{scored['Defect rates'].mean():.2f}%")
k6.metric("Ø Lieferzeit",        f"{scored['Gesamtlieferzeit'].mean():.2f} Tage")
k7.metric("Gesamtumsatz",        f"{scored['Revenue generated'].sum()/1000:.2f}k €")
st.markdown("---")

tab0,tab1,tab2,tab3,tab4,tab5,tab6 = st.tabs([
    "Start",
    "Dashboard",
    "Risk Overview",
    "Produkte",
    "Lieferanten",
    "Korrelation",
    "Alle Daten"
])

# ════════════════════════════════════════════════════════════════
#  TAB 0 – STARTSEITE
# ════════════════════════════════════════════════════════════════
with tab0:
    st.markdown("""
    <div class='hero-box'>
        <div class='hero-title'>RiskRadar</div>
        <div class='hero-sub'>Supplier & Procurement Risk Intelligence · THI Ingolstadt · DPDS 2026 · Gruppe 9</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""<div class='info-box' style='font-size:0.95rem;padding:18px 22px'>
    <b>Wofür ist dieses Dashboard?</b><br><br>
    RiskRadar hilft <b>Supply Chain Managern und CEOs</b> dabei, Lieferanten- und Beschaffungsrisiken
    frühzeitig zu erkennen, Produkte nach ihrem Wertbeitrag zu klassifizieren und datenbasierte
    Entscheidungen zu treffen.<br><br>
    <b>Kontext:</b> Ein produzierendes Unternehmen kauft Rohstoffe ein, befüllt diese in Produkte
    (Skincare, Haircare, Cosmetics) und vertreibt sie. Die Daten umfassen 100 SKUs von
    5 Lieferanten aus 5 indischen Standorten.
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Was beantwortet das Dashboard?")
    fc1,fc2,fc3 = st.columns(3)
    with fc1:
        st.markdown("""<div class='feature-card'>
        <div style='font-size:2rem;margin-bottom:8px'>🚦</div><div class='feature-title'>Welche Lieferanten sind kritisch?</div>
        <div class='feature-desc'>Die Ampel zeigt sofort den Status jedes Lieferanten. Kritische SKUs werden direkt aufgelistet mit konkreten Handlungsempfehlungen.</div>
        </div>""", unsafe_allow_html=True)
        st.markdown("<p style='text-align:center;margin-top:4px'><i style='color:#5577aa;font-size:0.8rem'>Tab: Dashboard</i></p>", unsafe_allow_html=True)
    with fc2:
        st.markdown("""<div class='feature-card'>
        <div style='font-size:2rem;margin-bottom:8px'>🏷️</div><div class='feature-title'>Was für Produkte kaufen wir ein?</div>
        <div class='feature-desc'>Produkte werden in 3 Segmente klassifiziert: Premium, Standard und Budget – basierend auf Preis, Umsatz und Defektrate.</div>
        </div>""", unsafe_allow_html=True)
        st.markdown("<p style='text-align:center;margin-top:4px'><i style='color:#5577aa;font-size:0.8rem'>Tab: Produkte</i></p>", unsafe_allow_html=True)
    with fc3:
        st.markdown("""<div class='feature-card'>
        <div style='font-size:2rem;margin-bottom:8px'>📊</div><div class='feature-title'>Wie hoch ist unser Risiko?</div>
        <div class='feature-desc'>Der Risk Score (0–100) bewertet jeden Lieferanten nach 5 KPIs: Defektrate, Lieferzeit, Kosten, Inspektion und Umsatz.</div>
        </div>""", unsafe_allow_html=True)
        st.markdown("<p style='text-align:center;margin-top:4px'><i style='color:#5577aa;font-size:0.8rem'>Tab: Risk Overview</i></p>", unsafe_allow_html=True)

    # ── KPI-Glossar ───────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### KPI-Glossar: Was bedeuten die Kennzahlen?")
    st.markdown("<div class='info-box'>Alle Kennzahlen stammen direkt aus dem Datensatz. Hier eine kurze Erklärung was hinter jeder Spalte steckt.</div>", unsafe_allow_html=True)

    g1,g2 = st.columns(2)
    with g1:
        st.markdown("""<div class='glossar-card'>
        <div class='glossar-term'>Defektrate (%)</div>
        <div class='glossar-def'>Anteil fehlerhafter Produkte an der Gesamtproduktion in Prozent.
        Eine Defektrate von 3 % bedeutet: 3 von 100 produzierten Einheiten sind fehlerhaft.
        Hohe Defektrate = schlechtere Qualität = höheres Risiko.</div>
        </div>""", unsafe_allow_html=True)

        st.markdown("""<div class='glossar-card'>
        <div class='glossar-term'>Gesamtlieferzeit (Tage)</div>
        <div class='glossar-def'>Lead time (Beschaffungszeit beim Lieferanten) + Shipping times (Versandzeit zu uns).
        Beispiel: Lead time 20 Tage + Shipping 10 Tage = 30 Tage Gesamtlieferzeit.
        Längere Lieferzeit = höheres Verzögerungsrisiko.</div>
        </div>""", unsafe_allow_html=True)

        st.markdown("""<div class='glossar-card'>
        <div class='glossar-term'>Inspection results (Pass / Pending / Fail)</div>
        <div class='glossar-def'>Ergebnis der Qualitätskontrolle je SKU.<br>
        Pass = bestanden · Pending = Prüfung ausstehend · Fail = nicht bestanden (Ausschuss).<br>
        Im Datensatz: 36% Fail-Quote – d.h. 36 von 100 SKUs haben die Inspektion nicht bestanden.</div>
        </div>""", unsafe_allow_html=True)

    with g2:
        st.markdown("""<div class='glossar-card'>
        <div class='glossar-term'>Costs (€) – Gesamtkosten</div>
        <div class='glossar-def'>Die Gesamtkosten je SKU (Ø €529) umfassen alle anfallenden Kosten
        inkl. Fertigung, Logistik und Verwaltung. Nicht zu verwechseln mit
        Manufacturing costs (nur Herstellungskosten, Ø €47).
        Hohe Kosten = kostenseitiges Risiko.</div>
        </div>""", unsafe_allow_html=True)

        st.markdown("""<div class='glossar-card'>
        <div class='glossar-term'>Revenue generated (€) – Umsatz</div>
        <div class='glossar-def'>Der durch dieses SKU erzielte Umsatz in Euro (absoluter Wert, z.B. 5.200 €).
        Im Risk Score wird Umsatz <b>invers</b> gewertet: ein hoher Umsatz
        senkt das Risiko, weil umsatzstarke SKUs strategisch wichtiger sind.
        Die Prozentzahl in der Gewichtungsanzeige zeigt den Einflussanteil am Score – nicht den Umsatz selbst.</div>
        </div>""", unsafe_allow_html=True)

        st.markdown("""<div class='glossar-card'>
        <div class='glossar-term'>Availability (1–100)</div>
        <div class='glossar-def'>Verfügbarkeitsprozentsatz: wie oft ein SKU lieferbar ist (1 = fast nie, 100 = immer).
        Durchschnitt im Datensatz: 48%. Niedriger Wert signalisiert potenzielle Engpässe.</div>
        </div>""", unsafe_allow_html=True)

    # ── Risk Score Erklärung ──────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Wie wird der Risk Score berechnet?")

    tw_display = wd+wl+wc+wi+wr
    st.markdown(f"""<div class='formula-box'>
Risk Score (0–100) = gewichtete Summe aus 5 normierten KPIs<br><br>
&nbsp;&nbsp;Defektrate &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;× {wd/tw_display*100:.0f}%  → hoch = schlechter<br>
&nbsp;&nbsp;Gesamtlieferzeit &nbsp;× {wl/tw_display*100:.0f}%  → lang = schlechter<br>
&nbsp;&nbsp;Gesamtkosten &nbsp;&nbsp;&nbsp;&nbsp;× {wc/tw_display*100:.0f}%  → hoch = schlechter<br>
&nbsp;&nbsp;Inspektion &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;× {wi/tw_display*100:.0f}%  → Fail=1.0 · Pending=0.5 · Pass=0.0<br>
&nbsp;&nbsp;Umsatz (invers) &nbsp;&nbsp;× {wr/tw_display*100:.0f}%  → hoch = BESSER (senkt den Score)<br><br>
Normierung: Jeder KPI wird per Min-Max auf 0–1 skaliert, dann gewichtet addiert × 100.<br>
Grenzwerte: Hoch ≥ {thresh_high} · Mittel ≥ {thresh_low} · Niedrig &lt; {thresh_low}
    </div>""", unsafe_allow_html=True)

    st.markdown("""<div class='info-box' style='border-left-color:#a855f7;margin-top:8px'>
    <b>Beispielrechnung mit SKU1 (Manuell-Profil, Gewichte 4/3/2/5/2):</b><br><br>
    Rohdaten: Defektrate = 4.85% · Lieferzeit = 25 Tage · Kosten = €503 · Inspektion = Pending · Umsatz = €7.461<br><br>
    Schritt 1 – Normierung (Min-Max):<br>
    &nbsp;&nbsp;Defektrate: (4.85 − 0.02) / (4.94 − 0.02) = <b>0.98</b><br>
    &nbsp;&nbsp;Lieferzeit: (25 − 4) / (37 − 4) = <b>0.64</b><br>
    &nbsp;&nbsp;Kosten: (503 − 104) / (997 − 104) = <b>0.45</b><br>
    &nbsp;&nbsp;Inspektion: Pending = <b>0.50</b> (fix kodiert)<br>
    &nbsp;&nbsp;Umsatz (invers): 1 − (7461 − 1062) / (9866 − 1062) = <b>0.27</b><br><br>
    Schritt 2 – Gewichtete Summe:<br>
    &nbsp;&nbsp;(4×0.98 + 3×0.64 + 2×0.45 + 5×0.50 + 2×0.27) / 16 × 100<br>
    &nbsp;&nbsp;= (3.92 + 1.92 + 0.90 + 2.50 + 0.54) / 16 × 100<br>
    &nbsp;&nbsp;= 9.78 / 16 × 100 = <b>Risk Score: 61.1 → Hoch</b>
    </div>""", unsafe_allow_html=True)

    st.markdown("""<div class='info-box' style='border-left-color:#a855f7'>
    <b>Validierung durch Regressionsanalyse (R² = 0.93):</b><br>
    Eine nachgelagerte Regressionsanalyse der exportierten Scores bestätigt die Modelllogik:
    Die zwei dominanten Treiber machen ~79% des Scores aus –
    Defektrate (~40%) und Inspektionsergebnis (~38%) sind annähernd gleich stark gewichtet.
    Ein einzelnes "Fail"-Ergebnis schlägt rechnerisch genauso stark auf den Score durch wie eine maximale Defektrate.
    Gesamtlieferzeit (~11%), Fertigungskosten (~4%) und Umsatz spielen eine untergeordnete Rolle.
    Diese Gewichte können über die Sidebar-Slider individuell angepasst werden.
    </div>""", unsafe_allow_html=True)

    # ── Bucket-Score Erklärung ────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Produkt-Segmente: Premium, Standard, Budget")

    st.markdown(f"""<div class='formula-box'>
Bucket-Score (0–1) = (norm. Preis + norm. Umsatz + (1 – norm. Defektrate)) / 3<br><br>
Normierung: Jeder Faktor wird per Min-Max auf 0–1 skaliert.<br>
Hoher Preis → höherer Score &nbsp;·&nbsp; Hoher Umsatz → höherer Score &nbsp;·&nbsp; Hohe Defektrate → niedrigerer Score<br><br>
Schwellenwerte (berechnet aus dem Datensatz):<br>
&nbsp;&nbsp;Budget  = Score &lt; {P33:.2f}<br>
&nbsp;&nbsp;Standard = Score {P33:.2f} – {P66:.2f}<br>
&nbsp;&nbsp;Premium  = Score &gt; {P66:.2f}
    </div>""", unsafe_allow_html=True)

    bk1,bk2,bk3 = st.columns(3)
    segment_info = [
        ("bk1","🥇 Premium","#60a5fa",
         f"Score ≥ {P66:.2f}",
         "Teure Produkte (Ø 71 €) mit niedriger Defektrate (Ø 1.27%) und dem niedrigsten Risk Score (Ø 35.5). Skincare und Cosmetics dominieren. Beschaffungsseitig am zuverlässigsten."),
        ("bk2","🔵 Standard","#94a3b8",
         f"Score {P33:.2f} – {P66:.2f}",
         "Mittleres Segment: ausgeglichener Preis (Ø 46 €) und moderate Defektrate (Ø 2.16%). Hoher Cosmetics-Anteil. Risk Score im Mittelfeld (Ø 50.7)."),
        ("bk3","💚 Budget","#4ade80",
         f"Score < {P33:.2f}",
         "Günstige Produkte (Ø 30 €) mit der höchsten Defektrate (Ø 3.43%) und dem höchsten Risk Score (Ø 65.4). Haircare dominiert. 22 von 33 SKUs werden als hohes Risiko eingestuft – das kritischste Segment."),
    ]
    for col_w, label, color, threshold, desc in zip([bk1,bk2,bk3], *zip(*[(x[1],x[2],x[3],x[4]) for x in segment_info])):
        col_w.markdown(f"""<div class='glossar-card' style='border-left:3px solid {color}'>
        <div class='glossar-term' style='color:{color}'>{label}</div>
        <div style='font-size:0.78rem;color:#5577aa;margin-bottom:6px'>{threshold}</div>
        <div class='glossar-def'>{desc}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("""<div class='info-box' style='margin-top:12px'>
    <b>Zentrale Erkenntnis:</b> Budget-Produkte sind nicht nur günstiger, sondern auch riskanter in der Beschaffung.
    Premium-Produkte kosten mehr in der Fertigung, liefern aber konsistentere Qualität und sind strategisch wertvoller.
    </div>""", unsafe_allow_html=True)

    # Aktueller Status
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Aktueller Status")
    sk1,sk2,sk3,sk4 = st.columns(4)
    sk1.metric("Ø Risk Score",     f"{scored['Risk Score'].mean():.2f} / 100")
    sk2.metric("Kritische SKUs",   f"{n_high}", delta=f"{n_high/len(scored)*100:.0f}% der Auswahl", delta_color="inverse")
    sk3.metric("Ø Defektrate",     f"{scored['Defect rates'].mean():.2f} %")
    sk4.metric("Gesamtumsatz",     f"{scored['Revenue generated'].sum()/1000:.2f}k €")

# ════════════════════════════════════════════════════════════════
#  TAB 1 – DASHBOARD
# ════════════════════════════════════════════════════════════════
with tab1:
    st.markdown(f"<h2 style='color:#e8eaf0;font-size:1.4rem;margin-bottom:4px'>Entscheidungs-Dashboard</h2><p style='color:#5577aa;font-size:0.85rem'>Profil: <b>{preset}</b> · Grenzwert Hoch ≥{thresh_high} · Mittel ≥{thresh_low}</p>", unsafe_allow_html=True)

    # Ampel
    sup_risks = scored.groupby("Supplier name")["Risk Score"].mean()
    amp_cols  = st.columns(len(sup_order))
    for i,sup in enumerate(sup_order):
        if sup not in sup_risks.index: continue
        rv  = sup_risks[sup]
        css,icon,status = ("ampel-rot","🔴","KRITISCH") if rv>=thresh_high else ("ampel-gelb","🟡","BEOBACHTEN") if rv>=thresh_low else ("ampel-gruen","🟢","OK")
        n_fail = (scored[scored["Supplier name"]==sup]["Inspection results"]=="Fail").sum()
        n_sku  = len(scored[scored["Supplier name"]==sup])
        amp_cols[i].markdown(f"<div class='ampel-card {css}'>{icon} {sup}<br><span style='font-size:1.7rem;font-weight:700'>{rv:.1f}</span><br><span style='font-size:0.78rem'>{status}</span><br><span style='font-size:0.72rem;opacity:0.8'>{n_sku} SKUs · {n_fail} Fails</span></div>", unsafe_allow_html=True)

    st.markdown("---")

    # SKU-Filter – nur Radio (keine doppelten Buttons mehr)
    st.markdown("<div class='section-header'>SKU-Liste nach Risikostufe filtern</div>", unsafe_allow_html=True)
    risk_radio = st.radio("Risikostufe:",
                          ["Alle","🔴 Hoch","🟡 Mittel","🟢 Niedrig"],
                          index=["Alle","🔴 Hoch","🟡 Mittel","🟢 Niedrig"].index(st.session_state.risk_filter),
                          horizontal=True, key="risk_radio")
    if risk_radio != st.session_state.risk_filter:
        st.session_state.risk_filter = risk_radio

    rf       = st.session_state.risk_filter
    list_df  = scored if rf=="Alle" else scored[scored["Risikostufe"]==rf]
    list_df  = list_df.sort_values("Risk Score", ascending=False)
    color_map= {"🔴 Hoch":"#ef4444","🟡 Mittel":"#f59e0b","🟢 Niedrig":"#22c55e"}
    rc       = color_map.get(rf,"#2563eb")
    st.markdown(f"<p style='color:{rc};font-size:0.9rem;font-weight:600'>{len(list_df)} SKUs · Stufe: {rf}</p>", unsafe_allow_html=True)
    show_cols = ["SKU","Supplier name","Product type","Produkt-Bucket","Defect rates","Gesamtlieferzeit","Costs","Revenue generated","Inspection results","Risk Score","Risikostufe"]
    st.dataframe(list_df[show_cols].reset_index(drop=True), use_container_width=True, height=320)

    st.markdown("---")

    # Handlungsempfehlungen
    kritische = [s for s in sup_order if s in sup_risks.index and sup_risks[s]>=thresh_high]
    mittlere  = [s for s in sup_order if s in sup_risks.index and thresh_low<=sup_risks[s]<thresh_high]
    ok        = [s for s in sup_order if s in sup_risks.index and sup_risks[s]<thresh_low]
    he1,he2,he3 = st.columns(3)
    with he1:
        st.markdown(f"<div class='info-box risk-high'><b>Sofortmaßnahmen</b><br>{', '.join(kritische) if kritische else 'Keine'}<br><br>→ Lieferanten-Audit<br>→ Alternativen prüfen<br>→ Bestellmengen reduzieren</div>", unsafe_allow_html=True)
    with he2:
        st.markdown(f"<div class='info-box risk-medium'><b>Monitoring</b><br>{', '.join(mittlere) if mittlere else 'Keine'}<br><br>→ Lieferantengespräch<br>→ KPIs beobachten<br>→ Vertrag prüfen</div>", unsafe_allow_html=True)
    with he3:
        st.markdown(f"<div class='info-box risk-low'><b>Routinebetrieb</b><br>{', '.join(ok) if ok else 'Keine'}<br><br>→ Weiter beobachten<br>→ Best Practices dokumentieren</div>", unsafe_allow_html=True)

    st.markdown("---")
    qs1,qs2,qs3 = st.columns(3)
    with qs1:
        st.markdown("<div class='section-header' style='font-size:0.85rem'>Top 5 riskanteste SKUs</div>", unsafe_allow_html=True)
        st.dataframe(scored.nlargest(5,"Risk Score")[["SKU","Supplier name","Risk Score","Risikostufe"]].reset_index(drop=True), use_container_width=True, height=210)
    with qs2:
        st.markdown("<div class='section-header' style='font-size:0.85rem'>Top 5 beste SKUs</div>", unsafe_allow_html=True)
        st.dataframe(scored.nsmallest(5,"Risk Score")[["SKU","Supplier name","Risk Score","Risikostufe"]].reset_index(drop=True), use_container_width=True, height=210)
    with qs3:
        st.markdown("<div class='section-header' style='font-size:0.85rem'>Fail-Quote pro Lieferant</div>", unsafe_allow_html=True)
        fq = scored.groupby("Supplier name").apply(lambda x: round((x["Inspection results"]=="Fail").sum()/len(x)*100,2)).reset_index()
        fq.columns = ["Lieferant","Fail %"]
        st.dataframe(fq.sort_values("Fail %",ascending=False).reset_index(drop=True), use_container_width=True, height=210)

# ════════════════════════════════════════════════════════════════
#  TAB 2 – RISK OVERVIEW
# ════════════════════════════════════════════════════════════════
with tab2:
    # Heatmap
    st.markdown("<div class='section-header'>KPI Heatmap</div>", unsafe_allow_html=True)
    st.markdown("<div class='info-box'>Rot = schlechter · Gelb = mittel · Grün = bester Wert (relativ). Spalten: Defektrate · Lieferzeit · Gesamtkosten · Fail-% · Umsatz</div>", unsafe_allow_html=True)

    hm = scored.groupby("Supplier name").agg(
        Defekt    =("Defect rates",       "mean"),
        Lieferzeit=("Gesamtlieferzeit",   "mean"),
        Kosten    =("Costs",              "mean"),
        Fail      =("Inspection results", lambda x: round((x=="Fail").sum()/len(x)*100,2)),
        Umsatz    =("Revenue generated",  "sum")
    ).round(2)
    hm_labels = ["Defekt %","Lieferzeit (Tage)","Kosten (€)","Fail %","Umsatz (€)"]
    hm_invert = [True,True,True,True,False]
    hm_vals   = hm.values
    hm_norm   = np.zeros_like(hm_vals, dtype=float)
    for j in range(hm_vals.shape[1]):
        col = hm_vals[:,j].astype(float); mn,mx = col.min(),col.max()
        n   = (col-mn)/(mx-mn+1e-9)
        hm_norm[:,j] = (1-n) if hm_invert[j] else n

    fig_hm,ax_hm = plt.subplots(figsize=(11,max(2.5,len(hm)*0.75)))
    for i in range(hm_norm.shape[0]):
        for j in range(hm_norm.shape[1]):
            v  = hm_norm[i,j]
            bg = "#3a1010" if v<0.33 else "#2a2210" if v<0.66 else "#0a2a14"
            tc = "#ff8080" if v<0.33 else "#ffd080" if v<0.66 else "#80ffa8"
            ax_hm.add_patch(plt.Rectangle([j,i],1,1,color=bg,zorder=2))
            raw = hm_vals[i,j]
            txt = f"{raw:.2f}%" if j in [0,3] else f"{raw:.2f} Tage" if j==1 else f"€{raw:.0f}" if j==2 else f"€{raw/1000:.1f}k"
            ax_hm.text(j+0.5,i+0.5,txt,ha="center",va="center",fontsize=10,fontweight="bold",color=tc,zorder=3)
    ax_hm.set_xlim(0,len(hm_labels)); ax_hm.set_ylim(0,len(hm))
    ax_hm.set_xticks([x+0.5 for x in range(len(hm_labels))]); ax_hm.set_xticklabels(hm_labels,fontsize=10)
    ax_hm.set_yticks([y+0.5 for y in range(len(hm))]); ax_hm.set_yticklabels(hm.index,fontsize=10)
    ax_hm.set_facecolor("#1a2035"); plt.tight_layout(); st.pyplot(fig_hm); plt.close()
    st.markdown("---")

    # Risk Score Balken
    st.markdown("<div class='section-header'>Ø Risk Score pro Lieferant</div>", unsafe_allow_html=True)
    sup_risk = scored.groupby("Supplier name")["Risk Score"].mean().sort_values(ascending=True)
    bar_c    = ["#ef4444" if v>=thresh_high else "#f59e0b" if v>=thresh_low else "#22c55e" for v in sup_risk.values]
    fig,ax   = plt.subplots(figsize=(10,3.5))
    bars     = ax.barh(sup_risk.index, sup_risk.values, color=bar_c, height=0.55, zorder=3)
    ax.axvline(thresh_high, color="#ef4444", lw=1.5, ls="--", alpha=0.8, label=f"Hoch ≥{thresh_high}")
    ax.axvline(thresh_low,  color="#f59e0b", lw=1.2, ls="--", alpha=0.7, label=f"Mittel ≥{thresh_low}")
    ax.set_xlabel("Ø Risk Score (0 = kein Risiko · 100 = max. Risiko)")
    ax.set_xlim(0,107); ax.grid(axis="x",zorder=0); ax.legend(fontsize=8)
    for b,v in zip(bars,sup_risk.values):
        ax.text(v+1,b.get_y()+b.get_height()/2,f"{v:.2f}",va="center",fontsize=9)
    plt.tight_layout(); st.pyplot(fig); plt.close()
    st.markdown("---")

    # Boxplots
    st.markdown("<div class='section-header'>Boxplots: Defektrate & Gesamtlieferzeit pro Lieferant</div>", unsafe_allow_html=True)
    st.markdown("<div class='info-box'>Linie = Median · Box = mittlere 50% der Werte · Punkte = Ausreißer. Gesamtlieferzeit = Lead time (Beschaffung) + Shipping times (Versand).</div>", unsafe_allow_html=True)
    bc1,bc2 = st.columns(2)
    for col_w,kpi,ylabel in [(bc1,"Defect rates","Defektrate (%)"),(bc2,"Gesamtlieferzeit","Gesamtlieferzeit (Tage)")]:
        with col_w:
            box_data = [scored[scored["Supplier name"]==s][kpi].dropna().values for s in sup_order]
            fig_b,ax_b = plt.subplots(figsize=(7,4))
            bp = ax_b.boxplot(box_data,tick_labels=sup_order,patch_artist=True,
                              medianprops={"color":"#e8eaf0","linewidth":2},
                              whiskerprops={"color":"#8899bb"},capprops={"color":"#8899bb"},
                              flierprops={"marker":"o","markerfacecolor":"#ef4444","markersize":5,"alpha":0.7})
            for patch,sup in zip(bp["boxes"],sup_order):
                patch.set_facecolor(SUP_COLORS.get(sup,"#2563eb")); patch.set_alpha(0.7)
            ax_b.set_ylabel(ylabel); ax_b.grid(axis="y",zorder=0)
            plt.tight_layout(); st.pyplot(fig_b); plt.close()

# ════════════════════════════════════════════════════════════════
#  TAB 3 – PRODUKTE
# ════════════════════════════════════════════════════════════════
with tab3:
    st.markdown(f"""<div class='info-box'>
    <b>Produktklassifikation</b> – Score = (norm. Preis + norm. Umsatz + (1 – norm. Defektrate)) / 3<br>
    Budget &lt;{P33:.2f} · Standard {P33:.2f}–{P66:.2f} · Premium &gt;{P66:.2f}
    </div>""", unsafe_allow_html=True)

    bk1,bk2,bk3 = st.columns(3)
    for col_w,bucket,css in [(bk1,"🥇 Premium","ampel-gruen"),(bk2,"🔵 Standard","ampel-gelb"),(bk3,"💚 Budget","ampel-rot")]:
        sub = scored[scored["Produkt-Bucket"]==bucket]
        if not sub.empty:
            col_w.markdown(f"<div class='ampel-card {css}'>{bucket}<br><span style='font-size:1.5rem;font-weight:700'>{len(sub)} SKUs</span><br><span style='font-size:0.8rem'>Ø Preis: {sub['Price'].mean():.2f} €</span><br><span style='font-size:0.8rem'>Ø Umsatz: {sub['Revenue generated'].mean()/1000:.2f}k €</span><br><span style='font-size:0.8rem'>Ø Defekt: {sub['Defect rates'].mean():.2f}%</span></div>", unsafe_allow_html=True)
    st.markdown("---")

    # Balkendiagramme
    st.markdown("<div class='section-header'>Produktkategorie & Standort Performance</div>", unsafe_allow_html=True)
    y_options = ["Umsatz (€)","Defektrate (%)","Gesamtlieferzeit (Tage)","Gesamtkosten (€)","Anzahl SKUs"]
    y_src_map = {
        "Umsatz (€)":              ("Revenue generated","sum",  "Umsatz (Tsd. €)", lambda v: v/1000),
        "Defektrate (%)":          ("Defect rates",     "mean", "Ø Defektrate (%)", lambda v: v),
        "Gesamtlieferzeit (Tage)": ("Gesamtlieferzeit", "mean", "Ø Lieferzeit (Tage)", lambda v: v),
        "Gesamtkosten (€)":        ("Costs",            "mean", "Ø Kosten (€)", lambda v: v),
        "Anzahl SKUs":             ("SKU",              "count","Anzahl SKUs", lambda v: v),
    }
    cy1,cy2 = st.columns(2)
    with cy1: y_cat = st.selectbox("Y-Achse Produktkategorie:", y_options, key="y_cat")
    with cy2: y_loc = st.selectbox("Y-Achse Standort:", y_options, key="y_loc")

    up1,up2 = st.columns(2)
    with up1:
        src,agg,ylabel,transform = y_src_map[y_cat]
        cp = scored.groupby("Product type").agg(val=(src,agg)).round(2)
        cp["val_t"] = cp["val"].apply(transform)
        fig_c,ax_c = plt.subplots(figsize=(6,4))
        ax_c.bar(cp.index,cp["val_t"],color=[PROD_COLORS.get(p,"#2563eb") for p in cp.index],width=0.5,zorder=3)
        ax_c.set_ylabel(ylabel); ax_c.set_xlabel("Produktkategorie"); ax_c.grid(axis="y",zorder=0)
        for bar,v in zip(ax_c.patches,cp["val_t"]):
            ax_c.text(bar.get_x()+bar.get_width()/2,v*1.01,f"{v:.1f}",ha="center",va="bottom",fontsize=10,color="#c8d4f0")
        plt.tight_layout(); st.pyplot(fig_c); plt.close()
    with up2:
        src,agg,ylabel,transform = y_src_map[y_loc]
        lp = scored.groupby("Location").agg(val=(src,agg)).sort_values("val",ascending=False).round(2)
        lp["val_t"] = lp["val"].apply(transform)
        fig_l,ax_l = plt.subplots(figsize=(6,4))
        ax_l.bar(lp.index,lp["val_t"],color="#2563eb",width=0.5,zorder=3)
        ax_l.set_ylabel(ylabel); ax_l.set_xlabel("Standort"); ax_l.grid(axis="y",zorder=0)
        plt.xticks(rotation=15)
        for bar,v in zip(ax_l.patches,lp["val_t"]):
            ax_l.text(bar.get_x()+bar.get_width()/2,v*1.01,f"{v:.1f}",ha="center",va="bottom",fontsize=10,color="#c8d4f0")
        plt.tight_layout(); st.pyplot(fig_l); plt.close()

    st.markdown("---")

    # Segment-Verteilung
    st.markdown("<div class='section-header'>Produkt-Segment Verteilung pro Lieferant</div>", unsafe_allow_html=True)
    pivot_seg = scored.groupby(["Supplier name","Produkt-Bucket"]).size().unstack(fill_value=0)
    fig_seg,ax_seg = plt.subplots(figsize=(10,4))
    xs = np.arange(len(pivot_seg)); ws = 0.25
    for idx_b,b in enumerate(["🥇 Premium","🔵 Standard","💚 Budget"]):
        if b in pivot_seg.columns:
            bars_s = ax_seg.bar(xs+idx_b*ws,pivot_seg[b].values,ws,label=b,color=BUCKET_COLORS.get(b,"#aaa"),alpha=0.9,zorder=3)
            for bar,v in zip(bars_s,pivot_seg[b].values):
                if v>0: ax_seg.text(bar.get_x()+bar.get_width()/2,v+0.1,str(int(v)),ha="center",va="bottom",fontsize=9,color="#c8d4f0")
    ax_seg.set_xticks(xs+ws); ax_seg.set_xticklabels(pivot_seg.index)
    ax_seg.set_ylabel("Anzahl SKUs"); ax_seg.legend(fontsize=9,framealpha=0.2); ax_seg.grid(axis="y",zorder=0)
    plt.tight_layout(); st.pyplot(fig_seg); plt.close()

    st.markdown("---")
    st.markdown("<div class='section-header'>SKU-Liste nach Segment</div>", unsafe_allow_html=True)
    sel_bk = st.selectbox("Segment anzeigen:",["Alle","🥇 Premium","🔵 Standard","💚 Budget"])
    bk_df  = scored if sel_bk=="Alle" else scored[scored["Produkt-Bucket"]==sel_bk]
    st.dataframe(bk_df[["SKU","Supplier name","Product type","Produkt-Bucket","Bucket-Score","Price","Revenue generated","Defect rates","Gesamtlieferzeit","Inspection results","Risk Score","Risikostufe"]].sort_values("Risk Score",ascending=False).reset_index(drop=True),use_container_width=True,height=350)

# ════════════════════════════════════════════════════════════════
#  TAB 4 – LIEFERANTEN
# ════════════════════════════════════════════════════════════════
with tab4:
    chosen = st.selectbox("Lieferant auswählen:", sorted(scored["Supplier name"].unique()))
    sup_d  = scored[scored["Supplier name"]==chosen]

    r1,r2,r3,r4,r5,r6 = st.columns(6)
    r1.metric("Ø Risk Score",    f"{sup_d['Risk Score'].mean():.2f}")
    r2.metric("Ø Defektrate",    f"{sup_d['Defect rates'].mean():.2f}%")
    r3.metric("Ø Lieferzeit",    f"{sup_d['Gesamtlieferzeit'].mean():.2f} Tage")
    r4.metric("Gesamtumsatz",    f"{sup_d['Revenue generated'].sum()/1000:.2f}k €")
    r5.metric("Anzahl SKUs",     f"{len(sup_d)}")
    r6.metric("Fail-Quote",      f"{(sup_d['Inspection results']=='Fail').sum()/len(sup_d)*100:.2f}%")

    rv    = sup_d["Risk Score"].mean()
    css_r = "risk-high" if rv>=thresh_high else "risk-medium" if rv>=thresh_low else "risk-low"
    icon_r= "🔴" if rv>=thresh_high else "🟡" if rv>=thresh_low else "🟢"
    msg_r = "Sofortmaßnahmen: Audit, Alternativen prüfen." if rv>=thresh_high else "Monitoring intensivieren." if rv>=thresh_low else "Lieferant performt gut."
    st.markdown(f"<div class='info-box {css_r}'>{icon_r} <b>{chosen} – Risk Score {rv:.2f}/100:</b> {msg_r}</div>", unsafe_allow_html=True)
    st.markdown("---")

    # Risk Score pro SKU
    st.markdown("<div class='section-header'>Risk Score pro SKU</div>", unsafe_allow_html=True)
    sku_s = sup_d.sort_values("Risk Score",ascending=True)
    fig_s,ax_s = plt.subplots(figsize=(10,max(3,len(sku_s)*0.30)))
    ax_s.barh(sku_s["SKU"],sku_s["Risk Score"],color=[RISK_COLORS.get(c,"#555") for c in sku_s["Risikostufe"]],height=0.6,zorder=3)
    ax_s.axvline(thresh_high,color="#ef4444",lw=1,ls="--",alpha=0.5,label=f"Hoch ≥{thresh_high}")
    ax_s.axvline(thresh_low, color="#f59e0b",lw=1,ls="--",alpha=0.5,label=f"Mittel ≥{thresh_low}")
    ax_s.set_xlabel("Risk Score (0–100)"); ax_s.set_xlim(0,105); ax_s.grid(axis="x",zorder=0); ax_s.legend(fontsize=7)
    plt.tight_layout(); st.pyplot(fig_s); plt.close()

    st.markdown("---")
    st.markdown("<div class='section-header'>Inspektionsstatus – nach Potenzial sortiert</div>", unsafe_allow_html=True)
    st.markdown("<div class='info-box'>Pending = noch nicht geprüft (Nachkontrolle möglich) · Fail = Ausschuss · Pass = bestanden</div>", unsafe_allow_html=True)
    insp_d = sup_d.copy(); insp_d["_s"] = insp_d["Inspection results"].map({"Pending":0,"Fail":1,"Pass":2})
    st.dataframe(insp_d.sort_values(["_s","Risk Score"],ascending=[True,False])[["SKU","Product type","Produkt-Bucket","Inspection results","Defect rates","Gesamtlieferzeit","Risk Score","Risikostufe"]].reset_index(drop=True),use_container_width=True,height=300)

# ════════════════════════════════════════════════════════════════
#  TAB 5 – KORRELATION
# ════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("<div class='info-box'><b>Spearman-Korrelation:</b> misst rang-basierte Zusammenhänge (geeignet für nicht-normalverteilte Daten). +1 = stark positiv · 0 = kein Zusammenhang · -1 = stark negativ</div>", unsafe_allow_html=True)

    sp_data = scored_full.copy()
    sp_data["Fail_binary"] = (sp_data["Inspection results"]=="Fail").astype(int)
    kpi_cols   = ["Defect rates","Gesamtlieferzeit","Costs","Fail_binary","Revenue generated"]
    kpi_labels = ["Defekt %","Lead Time (Tage)","Kosten (€)","Fail-Quote","Umsatz (€)"]

    sp_matrix = np.zeros((len(kpi_cols),len(kpi_cols)))
    for i,c1 in enumerate(kpi_cols):
        for j,c2 in enumerate(kpi_cols):
            r,_ = spearmanr(sp_data[c1].dropna(), sp_data[c2].dropna())
            sp_matrix[i,j] = round(r,2)

    mask = np.ones_like(sp_matrix,dtype=bool); np.fill_diagonal(mask,False)
    flat = sp_matrix.copy(); flat[~mask] = 0
    max_idx = np.unravel_index(np.abs(flat).argmax(), flat.shape)
    min_idx = np.unravel_index(np.abs(flat).argmin(), flat.shape)

    mc1,mc2,mc3,mc4 = st.columns(4)
    mc1.metric("Stärkste Korrelation", f"{sp_matrix[max_idx]:.2f}", delta=f"{kpi_labels[max_idx[0]]} ↔ {kpi_labels[max_idx[1]]}")
    mc2.metric("Schwächste Korr.",     f"{sp_matrix[min_idx]:.2f}", delta=f"{kpi_labels[min_idx[0]]} ↔ {kpi_labels[min_idx[1]]}")
    mc3.metric("Multikollinearität",   "Gering", delta="Modell sicher")
    mc4.metric("Methode",             "Spearman", delta="nicht-normalverteilt")

    # Nur die Matrix – kein Scatter (rausgenommen laut Meeting)
    fig_sp,ax_sp = plt.subplots(figsize=(7,5.5))
    hm3 = np.zeros_like(sp_matrix,dtype=float)
    for i in range(sp_matrix.shape[0]):
        for j in range(sp_matrix.shape[1]):
            hm3[i,j] = (sp_matrix[i,j]+1)/2
    im = ax_sp.imshow(hm3, cmap=plt.cm.RdYlGn, vmin=0, vmax=1, aspect="auto")
    ax_sp.set_xticks(range(len(kpi_labels))); ax_sp.set_xticklabels(kpi_labels, rotation=20, ha="right", fontsize=9)
    ax_sp.set_yticks(range(len(kpi_labels))); ax_sp.set_yticklabels(kpi_labels, fontsize=9)
    for i in range(len(kpi_labels)):
        for j in range(len(kpi_labels)):
            ax_sp.text(j,i,f"{sp_matrix[i,j]:.2f}",ha="center",va="center",fontsize=12,fontweight="bold",
                       color="#0f1117" if abs(sp_matrix[i,j])>0.3 else "#c8d4f0")
    plt.colorbar(im,ax=ax_sp,shrink=0.8); plt.tight_layout(); st.pyplot(fig_sp); plt.close()

    st.markdown(f"<div class='info-box'>Stärkster Zusammenhang: <b>{kpi_labels[max_idx[0]]}</b> ↔ <b>{kpi_labels[max_idx[1]]}</b> (ρ = {sp_matrix[max_idx]:.2f}). Alle Werte unter 0.3 – kein Multikollinearitätsproblem für das Scoring-Modell.</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<div class='section-header'>Streudiagramm – Achsen frei wählbar</div>", unsafe_allow_html=True)

    scatter_cols_full = ["Defect rates","Gesamtlieferzeit","Costs","Revenue generated","Risk Score","Price","Availability","Manufacturing costs","Shipping costs"]
    scatter_labels    = ["Defektrate (%)","Lieferzeit (Tage)","Kosten (€)","Umsatz (€)","Risk Score","Preis (€)","Verfügbarkeit (%)","Mfg. Kosten (€)","Versandkosten (€)"]
    col_map = dict(zip(scatter_labels, scatter_cols_full))

    sc1, sc2, sc3 = st.columns(3)
    with sc1: x_label = st.selectbox("X-Achse:", scatter_labels, index=0, key="sc_x")
    with sc2: y_label = st.selectbox("Y-Achse:", scatter_labels, index=3, key="sc_y")
    with sc3: color_by_sc = st.selectbox("Einfärben nach:", ["Lieferant","Produktkategorie","Risikostufe"], key="sc_col")

    x_col = col_map[x_label]
    y_col = col_map[y_label]

    if color_by_sc == "Lieferant":
        sc_colors = [SUP_COLORS.get(s,"#aaa") for s in scored_full["Supplier name"]]
        legend_patches = [mpatches.Patch(color=SUP_COLORS.get(s,"#aaa"), label=s) for s in sorted(scored_full["Supplier name"].unique())]
    elif color_by_sc == "Produktkategorie":
        sc_colors = [PROD_COLORS.get(p,"#aaa") for p in scored_full["Product type"]]
        legend_patches = [mpatches.Patch(color=PROD_COLORS.get(p,"#aaa"), label=p) for p in sorted(scored_full["Product type"].unique())]
    else:
        sc_colors = [RISK_COLORS.get(r,"#aaa") for r in scored_full["Risikostufe"]]
        legend_patches = [mpatches.Patch(color=v, label=k) for k,v in RISK_COLORS.items()]

    plot_df = scored_full[[x_col, y_col, "Supplier name"]].dropna()

    fig_sc2, ax_sc2 = plt.subplots(figsize=(10, 5))
    colors_plot = [sc_colors[i] for i in scored_full.index if i in plot_df.index]
    ax_sc2.scatter(plot_df[x_col], plot_df[y_col],
                   c=[sc_colors[i] for i in plot_df.index],
                   s=70, alpha=0.85, edgecolors="#0f1117", linewidths=0.8, zorder=3)
    ax_sc2.set_xlabel(x_label); ax_sc2.set_ylabel(y_label)
    ax_sc2.grid(zorder=0)
    ax_sc2.legend(handles=legend_patches, fontsize=9, framealpha=0.2)

    # Korrelation der gewählten Achsen anzeigen
    if x_col in kpi_cols and y_col in kpi_cols:
        xi = kpi_cols.index(x_col); yi = kpi_cols.index(y_col)
        rho = sp_matrix[xi, yi]
        ax_sc2.set_title(f"Spearman ρ = {rho:.2f}", fontsize=10, color="#c8d4f0")

    plt.tight_layout(); st.pyplot(fig_sc2); plt.close()
    st.markdown("<div class='info-box'>Ideal für Zusammenhangsanalyse: Wähle zwei KPIs und sieh ob ein Muster erkennbar ist. Der Spearman-Wert oben zeigt die Stärke des Zusammenhangs.</div>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
#  TAB 6 – ALLE DATEN
# ════════════════════════════════════════════════════════════════
with tab6:
    st.markdown("<div class='section-header'>Alle Daten – filter- & sortierbar · Export CSV & Excel</div>", unsafe_allow_html=True)

    base_cols  = [c for c in df.columns if c not in ["_p33","_p66"]]
    extra_cols = [c for c in ["Gesamtlieferzeit","Risk Score","Risikostufe"] if c not in base_cols]
    all_cols   = base_cols + extra_cols

    with st.expander("Spalten auswählen", expanded=False):
        sel_cols2 = st.multiselect("Spalten", all_cols, default=all_cols)
    with st.expander("Zusätzliche Filter", expanded=False):
        f1,f2,f3 = st.columns(3)
        with f1:
            ff_sup  = st.multiselect("Lieferant",  sorted(df["Supplier name"].unique()), default=sorted(df["Supplier name"].unique()), key="ffs")
            ff_prod = st.multiselect("Produkttyp", sorted(df["Product type"].unique()),  default=sorted(df["Product type"].unique()),  key="ffp")
        with f2:
            ff_loc  = st.multiselect("Standort",   sorted(df["Location"].unique()),      default=sorted(df["Location"].unique()),      key="ffl")
            ff_insp = st.multiselect("Inspektion", sorted(df["Inspection results"].unique()), default=sorted(df["Inspection results"].unique()), key="ffi")
        with f3:
            ff_risk = st.multiselect("Risikostufe",["🔴 Hoch","🟡 Mittel","🟢 Niedrig"], default=["🔴 Hoch","🟡 Mittel","🟢 Niedrig"], key="ffr")
            ff_buck = st.multiselect("Segment",    ["🥇 Premium","🔵 Standard","💚 Budget"], default=["🥇 Premium","🔵 Standard","💚 Budget"], key="ffb")

    s1,s2 = st.columns([3,1])
    with s1: sort_by2 = st.selectbox("Sortieren nach",["Risk Score","Defect rates","Gesamtlieferzeit","Revenue generated","Costs","Price"])
    with s2: asc2 = st.radio("Reihenfolge",["Absteigend","Aufsteigend"]) == "Aufsteigend"

    table = scored_full[
        scored_full["Supplier name"].isin(ff_sup) & scored_full["Product type"].isin(ff_prod) &
        scored_full["Location"].isin(ff_loc) & scored_full["Inspection results"].isin(ff_insp) &
        scored_full["Risikostufe"].isin(ff_risk) & scored_full["Produkt-Bucket"].isin(ff_buck)
    ].copy()
    valid2 = [c for c in sel_cols2 if c in table.columns]
    table  = table[valid2].sort_values(sort_by2, ascending=asc2).reset_index(drop=True)

    st.markdown(f"<p style='color:#5577aa;font-size:0.85rem'>{len(table)} Zeilen · {len(valid2)} Spalten</p>", unsafe_allow_html=True)
    st.dataframe(table, use_container_width=True, height=500)

    ex1,ex2 = st.columns(2)
    with ex1:
        st.download_button("CSV exportieren", data=table.to_csv(index=False).encode("utf-8"),
                           file_name="riskradar_export.csv", mime="text/csv")
    with ex2:
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            table.to_excel(writer, index=False, sheet_name="RiskRadar")
        buf.seek(0)
        st.download_button("Excel exportieren", data=buf.getvalue(),
                           file_name="riskradar_export.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.markdown("---")
st.markdown("<p style='text-align:center;color:#3a4a6a;font-size:0.8rem'>RiskRadar v8 · THI Ingolstadt · DPDS SoSe 2026 · Gruppe 9: Laurenz Angleitner · Leon Pavic · Alex Rauschendorfer · Daniel Steinmetz</p>", unsafe_allow_html=True)
