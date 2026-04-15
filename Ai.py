import streamlit as st
import pandas as pd
from datetime import datetime

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="The NewsType Chronicle",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Label / style maps ────────────────────────────────────────────────────────
LABEL_MAP = {
    "LABEL_0": "World News",
    "LABEL_1": "Sports",
    "LABEL_2": "Business",
    "LABEL_3": "Science / Technology",
}

CATEGORY_META = {
    "World News":           {"emoji": "🌍", "color": "#1a3a5c", "light": "#e6f1fb", "badge": "#378ADD"},
    "Sports":               {"emoji": "🏆", "color": "#c05a1a", "light": "#faece7", "badge": "#D85A30"},
    "Business":             {"emoji": "💼", "color": "#1e4d2b", "light": "#eaf3de", "badge": "#639922"},
    "Science / Technology": {"emoji": "🔬", "color": "#4a1e6e", "light": "#f3eafe", "badge": "#7F77DD"},
}

SAMPLES = [
    {
        "cat": "Sports",
        "headline": "Giddy Phelps Touches Gold for First Time",
        "text": "Michael Phelps won the gold medal in the 400 individual medley and set a world record in a time of 4 minutes 8.26 seconds.",
    },
    {
        "cat": "Business",
        "headline": "Apple Launches Graphics Software, Video Bundle",
        "text": "Apple Computer Inc. began shipping a new program designed to let users create real-time motion graphics and unveiled a discount video-editing bundle featuring its flagship Final Cut Pro software.",
    },
    {
        "cat": "Science / Technology",
        "headline": "Mars Rovers Relay Images Through Mars Express",
        "text": "ESA's Mars Express has relayed pictures from one of NASA's Mars rovers for the first time, as part of a set of interplanetary networking demonstrations.",
    },
    {
        "cat": "Business",
        "headline": "Morgan Stanley Profit Falls 34 Percent",
        "text": "U.S. investment bank Morgan Stanley said quarterly profit dropped 34 percent amid reduced trading revenue, falling well short of Wall Street's expectations.",
    },
    {
        "cat": "World News",
        "headline": "Sharon Says Gaza Evacuation Set for 2005",
        "text": "Israel's evacuation of the Gaza Strip will begin next summer and will take about 12 weeks, Prime Minister Ariel Sharon said Wednesday.",
    },
]

# ── Load model (cached) ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading BERT classifier model…")
def load_model():
    from transformers import pipeline
    try:
        # Optional: login if you need gated models
        # from huggingface_hub import login
        # login(token=st.secrets["HF_TOKEN"])  # store token in .streamlit/secrets.toml
        return pipeline(
            "text-classification",
            model="fabriceyhc/bert-base-uncased-ag_news",
        )
    except Exception as e:
        st.error(f"Model load failed: {e}")
        return None

def classify_text(text: str, classifier) -> dict:
    result = classifier(text, truncation=True, max_length=512)[0]
    label  = LABEL_MAP.get(result["label"], result["label"])
    return {"label": label, "score": round(result["score"] * 100, 1)}

# ── Session state init ────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "session_count" not in st.session_state:
    st.session_state.session_count = 0
if "session_conf_total" not in st.session_state:
    st.session_state.session_conf_total = 0.0
# Holds text loaded from a sample click so textarea pre-fills
if "pending_sample" not in st.session_state:
    st.session_state.pending_sample = ""
# Holds the last classification result to re-render after rerun
if "last_result" not in st.session_state:
    st.session_state.last_result = None   # dict: {label, score, text}

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=DM+Sans:wght@300;400;500;600&display=swap');

/* Reset Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }
[data-testid="stAppViewContainer"] {
    background: #f5f0e8 !important;
    font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stSidebar"] { display: none; }

/* ── MASTHEAD ── */
.masthead {
    background: #f5f0e8;
    border-top: 4px double #1a1208;
    border-bottom: 4px double #1a1208;
    padding: 20px 40px 16px;
    text-align: center;
    position: relative;
    margin-bottom: 0;
}
.masthead::before {
    content: '';
    position: absolute;
    top: 7px; left: 7px; right: 7px; bottom: 7px;
    border: 1px solid #1a1208;
    pointer-events: none;
}
.masthead-edition {
    font-family: 'DM Sans', sans-serif;
    font-size: 10px;
    letter-spacing: 4px;
    text-transform: uppercase;
    color: #c0392b;
    font-weight: 600;
    margin-bottom: 8px;
}
.masthead-title {
    font-family: 'Playfair Display', Georgia, serif;
    font-size: clamp(32px, 5vw, 58px);
    font-weight: 900;
    letter-spacing: -1px;
    line-height: 1;
    color: #1a1208;
    margin: 0;
}
.masthead-sub {
    font-family: 'DM Sans', sans-serif;
    font-size: 11px;
    letter-spacing: 4px;
    text-transform: uppercase;
    color: #1a1208;
    opacity: 0.55;
    margin-top: 8px;
}
.masthead-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-top: 12px;
    padding-top: 10px;
    border-top: 1px solid #d4cbb8;
    font-size: 11px;
    font-family: 'DM Sans', sans-serif;
    color: #1a1208;
    opacity: 0.55;
}

/* ── TICKER ── */
.ticker-wrap {
    background: #c0392b;
    overflow: hidden;
    padding: 7px 0;
    white-space: nowrap;
}
.ticker-inner {
    display: inline-block;
    animation: marquee 32s linear infinite;
    font-size: 11px;
    font-family: 'DM Sans', sans-serif;
    letter-spacing: 1.5px;
    font-weight: 500;
    color: #fff;
}
@keyframes marquee { 0% { transform: translateX(100vw); } 100% { transform: translateX(-100%); } }

/* ── PAPER BODY ── */
.paper-body {
    max-width: 1100px;
    margin: 0 auto;
    padding: 0 36px 60px;
}

/* ── SECTION RULE ── */
.section-rule-wrap { position: relative; margin: 24px 0 18px; }
.section-rule-line {
    border: none;
    border-top: 3px solid #1a1208;
    margin: 0;
}
.section-rule-label {
    position: absolute;
    top: -11px;
    left: 16px;
    background: #f5f0e8;
    padding: 0 10px;
    font-size: 9px;
    letter-spacing: 3px;
    text-transform: uppercase;
    font-weight: 700;
    color: #c0392b;
    font-family: 'DM Sans', sans-serif;
}

/* ── STATS STRIP ── */
.stats-strip {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    border: 1px solid #d4cbb8;
    margin: 20px 0 24px;
    background: #f5f0e8;
}
.stat-cell {
    padding: 16px 14px;
    border-right: 1px solid #d4cbb8;
    text-align: center;
}
.stat-cell:last-child { border-right: none; }
.stat-label {
    font-size: 9px;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #1a1208;
    opacity: 0.5;
    font-weight: 700;
    margin-bottom: 6px;
    font-family: 'DM Sans', sans-serif;
}
.stat-val {
    font-family: 'Playfair Display', serif;
    font-size: 26px;
    font-weight: 900;
    color: #1a1208;
    line-height: 1;
}
.stat-sub {
    font-size: 10px;
    color: #1a1208;
    opacity: 0.45;
    margin-top: 3px;
    font-family: 'DM Sans', sans-serif;
}

/* ── RESULT CARD ── */
.result-card {
    display: flex;
    border: 2px solid #1a1208;
    margin: 20px 0;
    background: #fff;
}
.result-accent { width: 10px; flex-shrink: 0; }
.result-body { padding: 22px 28px; flex: 1; }
.result-meta {
    font-size: 9px;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #1a1208;
    opacity: 0.45;
    margin-bottom: 8px;
    font-family: 'DM Sans', sans-serif;
    font-weight: 700;
}
.result-category {
    font-family: 'Playfair Display', serif;
    font-size: 36px;
    font-weight: 900;
    line-height: 1.1;
    margin-bottom: 14px;
}
.result-bar-bg {
    height: 7px;
    background: #d4cbb8;
    margin: 10px 0 8px;
    border-radius: 0;
    overflow: hidden;
}
.result-bar-fill { height: 100%; border-radius: 0; transition: width 0.8s ease; }
.result-confidence {
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 1px;
    font-family: 'DM Sans', sans-serif;
}
.result-snippet {
    margin-top: 14px;
    font-size: 12px;
    color: #1a1208;
    opacity: 0.55;
    font-style: italic;
    line-height: 1.6;
    border-top: 1px dotted #d4cbb8;
    padding-top: 12px;
    font-family: 'DM Sans', sans-serif;
}

/* ── SAMPLE CARD ── */
.sample-card {
    background: #ede8db;
    border: 1px solid #d4cbb8;
    border-radius: 0;
    padding: 14px;
    margin-bottom: 8px;
    cursor: pointer;
    transition: background 0.15s, border-color 0.15s;
}
.sample-card:hover { background: #e2dccf; border-color: #1a1208; }
.sample-cat {
    font-size: 9px;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    font-weight: 700;
    margin-bottom: 4px;
    font-family: 'DM Sans', sans-serif;
}
.sample-headline {
    font-family: 'Playfair Display', serif;
    font-size: 13px;
    line-height: 1.4;
    color: #1a1208;
}

/* ── HISTORY ITEM ── */
.history-item {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    padding: 10px 0;
    border-bottom: 1px dotted #d4cbb8;
    font-family: 'DM Sans', sans-serif;
}
.hist-dot {
    width: 10px; height: 10px;
    border-radius: 50%;
    flex-shrink: 0;
    margin-top: 4px;
}
.hist-text { flex: 1; font-size: 12px; color: #1a1208; line-height: 1.5; opacity: 0.8; }
.hist-cat { font-size: 10px; font-weight: 700; letter-spacing: 1px; flex-shrink: 0; }
.hist-conf { font-size: 10px; color: #1a1208; opacity: 0.45; flex-shrink: 0; min-width: 36px; text-align: right; }

/* ── BATCH TABLE ── */
.batch-table-wrap { margin-top: 16px; }
.btable { width: 100%; border-collapse: collapse; font-size: 12px; font-family: 'DM Sans', sans-serif; }
.btable th {
    font-size: 9px; letter-spacing: 2.5px; text-transform: uppercase;
    padding: 10px 12px; border-bottom: 2px solid #1a1208;
    text-align: left; font-weight: 700; color: #1a1208; opacity: 0.55;
    background: #f5f0e8;
}
.btable td { padding: 10px 12px; border-bottom: 1px dotted #d4cbb8; color: #1a1208; }
.btable tr:hover td { background: #ede8db; }
.cat-badge {
    display: inline-block;
    padding: 3px 10px;
    font-size: 9px;
    letter-spacing: 1px;
    font-weight: 700;
    text-transform: uppercase;
    border-radius: 0;
    font-family: 'DM Sans', sans-serif;
}

/* ── UPLOAD ZONE ── */
.upload-zone {
    border: 2px dashed #d4cbb8;
    padding: 48px 24px;
    text-align: center;
    background: #fff;
    margin-bottom: 20px;
}
.upload-title {
    font-family: 'Playfair Display', serif;
    font-size: 20px;
    font-weight: 700;
    color: #1a1208;
    margin-bottom: 8px;
}
.upload-sub {
    font-size: 12px;
    color: #1a1208;
    opacity: 0.5;
    font-family: 'DM Sans', sans-serif;
}

/* ── FOOTER ── */
.paper-footer {
    border-top: 4px double #1a1208;
    margin: 40px 36px 0;
    padding: 14px 0;
    display: flex;
    justify-content: space-between;
    font-size: 10px;
    color: #1a1208;
    opacity: 0.45;
    font-family: 'DM Sans', sans-serif;
    letter-spacing: 1px;
}

/* ── Streamlit widget overrides ── */
.stTextArea textarea {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 14px !important;
    border-radius: 0 !important;
    border: 1.5px solid #d4cbb8 !important;
    background: #fff !important;
    color: #1a1208 !important;
    padding: 14px !important;
}
.stTextArea textarea:focus { border-color: #1a1208 !important; box-shadow: none !important; }
.stButton > button {
    background: #1a1208 !important;
    color: #f5f0e8 !important;
    border: none !important;
    border-radius: 0 !important;
    font-family: 'Playfair Display', serif !important;
    font-size: 15px !important;
    font-weight: 700 !important;
    letter-spacing: 1px !important;
    padding: 12px 0 !important;
    width: 100% !important;
    transition: background 0.2s !important;
}
.stButton > button:hover { background: #c0392b !important; }
.stSelectbox > div > div {
    border-radius: 0 !important;
    border: 1.5px solid #d4cbb8 !important;
    font-family: 'DM Sans', sans-serif !important;
    background: #fff !important;
}
.stTabs [data-baseweb="tab-list"] {
    border-bottom: 2px solid #1a1208 !important;
    gap: 0 !important;
    background: transparent !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 11px !important;
    letter-spacing: 2.5px !important;
    text-transform: uppercase !important;
    font-weight: 700 !important;
    padding: 10px 22px !important;
    color: #1a1208 !important;
    background: transparent !important;
    border-radius: 0 !important;
}
.stTabs [aria-selected="true"] {
    border-bottom: 3px solid #c0392b !important;
    color: #c0392b !important;
}
.stFileUploader {
    border: 2px dashed #d4cbb8 !important;
    border-radius: 0 !important;
    background: #fff !important;
    padding: 20px !important;
}
div[data-testid="stProgress"] > div > div {
    background: #c0392b !important;
}
.stSpinner { color: #c0392b !important; }
</style>
""", unsafe_allow_html=True)

# ── MASTHEAD ─────────────────────────────────────────────────────────────────
now = datetime.now()
date_str = now.strftime("%A, %B %d, %Y")
time_str = now.strftime("%I:%M %p")

st.markdown(f"""
<div class="masthead">
    <div class="masthead-edition">✦ AI-Powered News Intelligence ✦</div>
    <div class="masthead-title">The NewsType Chronicle</div>
    <div class="masthead-sub">Powered by BERT &nbsp;·&nbsp; AG News Dataset &nbsp;·&nbsp; Four Categories &nbsp;·&nbsp; Est. 2024</div>
    <div class="masthead-bar">
        <span>{date_str}</span>
        <span>📰 &nbsp; Instant Classification Intelligence</span>
        <span>{time_str}</span>
    </div>
</div>
<div class="ticker-wrap">
    <span class="ticker-inner">
        ◆ BREAKING: AI Now Reads the News So You Don't Have To &nbsp;&nbsp;&nbsp;&nbsp;
        ◆ World Markets Edge Higher on Fed Signals &nbsp;&nbsp;&nbsp;&nbsp;
        ◆ Space Agency Confirms New Mars Findings &nbsp;&nbsp;&nbsp;&nbsp;
        ◆ Championship Finals Set Record Viewership &nbsp;&nbsp;&nbsp;&nbsp;
        ◆ Tech Giants Post Q3 Results Above Expectations &nbsp;&nbsp;&nbsp;&nbsp;
        ◆ Four Categories · Instant Accuracy · Zero Effort &nbsp;&nbsp;&nbsp;&nbsp;
    </span>
</div>
""", unsafe_allow_html=True)

# ── Load classifier ───────────────────────────────────────────────────────────
classifier = load_model()

# ── PAPER BODY STARTS ─────────────────────────────────────────────────────────
st.markdown('<div class="paper-body">', unsafe_allow_html=True)

# ── STATS STRIP ───────────────────────────────────────────────────────────────
count = st.session_state.session_count
avg   = (
    f"{st.session_state.session_conf_total / count:.0f}%"
    if count > 0 else "—"
)
st.markdown(f"""
<div class="stats-strip">
    <div class="stat-cell">
        <div class="stat-label">Model</div>
        <div class="stat-val" style="font-size:16px;padding-top:4px">BERT</div>
        <div class="stat-sub">AG News fine-tune</div>
    </div>
    <div class="stat-cell">
        <div class="stat-label">Categories</div>
        <div class="stat-val">4</div>
        <div class="stat-sub">World · Sports · Biz · Tech</div>
    </div>
    <div class="stat-cell">
        <div class="stat-label">Classified</div>
        <div class="stat-val">{count}</div>
        <div class="stat-sub">this session</div>
    </div>
    <div class="stat-cell">
        <div class="stat-label">Avg. Confidence</div>
        <div class="stat-val" style="font-size:22px">{avg}</div>
        <div class="stat-sub">over session</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["  Single Article  ", "  Batch / CSV  ", "  History  "])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — SINGLE ARTICLE
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div style="height:20px"></div>', unsafe_allow_html=True)
    col_main, col_samples = st.columns([3, 1.4], gap="large")

    # ── Sample buttons — checked BEFORE widgets render ────────────────────────
    # We need to detect a sample click in this run so we can pre-fill textarea.
    # Buttons are rendered inside col_samples; their state is read here first
    # via a pre-pass with hidden placeholders.
    sample_triggered_text = ""
    with col_samples:
        st.markdown(
            '<div class="section-rule-wrap"><hr class="section-rule-line">'
            '<span class="section-rule-label">Dispatch Samples</span></div>',
            unsafe_allow_html=True,
        )
        for s in SAMPLES:
            meta    = CATEGORY_META.get(s["cat"], {})
            col_hex = meta.get("badge", "#888")
            st.markdown(f"""
            <div class="sample-card">
                <div class="sample-cat" style="color:{col_hex}">{s['cat']}</div>
                <div class="sample-headline">{s['headline']}</div>
            </div>""", unsafe_allow_html=True)
            if st.button("Try ›", key=f"sample_{s['headline'][:20]}", use_container_width=True):
                sample_triggered_text = s["headline"] + ". " + s["text"]

    # If a sample button was just clicked, store it so textarea pre-fills
    if sample_triggered_text:
        st.session_state.pending_sample = sample_triggered_text

    # ── Main input column ─────────────────────────────────────────────────────
    with col_main:
        st.markdown(
            '<div class="section-rule-wrap"><hr class="section-rule-line">'
            '<span class="section-rule-label">Enter Article</span></div>',
            unsafe_allow_html=True,
        )

        # Pre-fill textarea with sample text if one was just selected
        default_text = st.session_state.pending_sample if st.session_state.pending_sample else ""

        user_input = st.text_area(
            label="article_input",
            label_visibility="hidden",
            value=default_text,
            height=160,
            placeholder="e.g. Apple unveils M3 MacBook Pro with record-breaking performance and neural engine advancements…",
            key="single_input",
        )
        classify_clicked = st.button("Classify Article →", key="btn_classify", use_container_width=True)

        # ── Determine what to classify this run ───────────────────────────────
        text_to_classify = None

        if sample_triggered_text:
            # Sample clicked → classify that text immediately
            text_to_classify = sample_triggered_text
            st.session_state.pending_sample = ""   # clear after use
        elif classify_clicked:
            if user_input.strip():
                text_to_classify = user_input.strip()
                st.session_state.last_result = None  # clear old result on new manual classify
            else:
                st.warning("Please paste some text or pick a sample.")

        # ── Run classification ────────────────────────────────────────────────
        if text_to_classify:
            if classifier is None:
                st.error("Model not loaded. Check your environment and try again.")
            else:
                with st.spinner("Classifying article…"):
                    res   = classify_text(text_to_classify, classifier)
                    label = res["label"]
                    score = res["score"]
                    meta  = CATEGORY_META[label]

                # Persist result so it survives any subsequent reruns
                st.session_state.last_result = {
                    "label": label,
                    "score": score,
                    "text":  text_to_classify,
                }

                # Update session stats
                st.session_state.session_count      += 1
                st.session_state.session_conf_total += score
                st.session_state.history.insert(0, {
                    "cat":  label,
                    "conf": score,
                    "text": text_to_classify[:120],
                })

        # ── Render result card (from current run OR persisted state) ──────────
        lr = st.session_state.last_result
        if lr:
            label = lr["label"]
            score = lr["score"]
            text  = lr["text"]
            meta  = CATEGORY_META[label]
            bar_w = int(score)

            descriptions = {
                "World News":           "Covers geopolitics, diplomacy, international relations, conflict, and global events.",
                "Sports":               "Athletic competitions, tournaments, player performance, records, and sporting events worldwide.",
                "Business":             "Corporate earnings, market movements, trade, investments, and economic developments.",
                "Science / Technology": "Scientific discoveries, tech product launches, space exploration, research, and innovation.",
            }

            snippet = text[:200] + ("…" if len(text) > 200 else "")

            st.markdown(f"""
            <div class="section-rule-wrap">
                <hr class="section-rule-line">
                <span class="section-rule-label">Classification Result</span>
            </div>
            <div class="result-card">
                <div class="result-accent" style="background:{meta['color']}"></div>
                <div class="result-body">
                    <div class="result-meta">Category Identified</div>
                    <div class="result-category" style="color:{meta['color']}">{meta['emoji']} {label}</div>
                    <div class="result-bar-bg">
                        <div class="result-bar-fill" style="width:{bar_w}%;background:{meta['badge']}"></div>
                    </div>
                    <div class="result-confidence" style="color:{meta['color']}">Confidence: {score}%</div>
                    <div class="result-snippet">"{snippet}"</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.info(f"**{label}** — {descriptions[label]}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — BATCH / CSV
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div style="height:20px"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="upload-zone">
        <div class="upload-title">📂 Upload your CSV file</div>
        <div class="upload-sub">Must contain a <strong>Title</strong> and/or <strong>Description</strong> column · Max 50 rows processed</div>
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Choose CSV",
        type=["csv"],
        label_visibility="collapsed",
        key="csv_upload",
    )

    use_bundled = st.checkbox("Use bundled test.csv (first 50 rows) instead", value=False)

    df = None
    if use_bundled:
        try:
            df = pd.read_csv(r"c:\Users\aravi\OneDrive\Desktop\test.csv", nrows=50)
            st.success("Loaded **test.csv** — 50 rows")
        except Exception as e:
            st.error(f"Could not load test.csv: {e}")
    elif uploaded:
        df = pd.read_csv(uploaded, nrows=50)
        st.success(f"Uploaded **{uploaded.name}** — {len(df)} rows loaded")

    if df is not None:
        # Build text column
        if "Title" in df.columns and "Description" in df.columns:
            df["_text"] = df["Title"].fillna("") + ". " + df["Description"].fillna("")
        elif "Title" in df.columns:
            df["_text"] = df["Title"].fillna("")
        elif "Description" in df.columns:
            df["_text"] = df["Description"].fillna("")
        else:
            st.error("CSV must have a **Title** or **Description** column.")
            st.stop()

        class_name_map = {1: "World News", 2: "Sports", 3: "Business", 4: "Science / Technology"}
        if "Class Index" in df.columns:
            df["True Label"] = df["Class Index"].map(class_name_map)

        n = len(df)
        st.markdown(f'<p style="font-size:12px;opacity:0.6;font-family:DM Sans,sans-serif;margin-bottom:12px">{n} articles ready to classify.</p>', unsafe_allow_html=True)

        if st.button("🚀 Run Batch Classification", key="batch_run", use_container_width=True):
            if classifier is None:
                st.error("Model not loaded.")
            else:
                prog = st.progress(0, text="Starting…")
                results = []
                for i, row in df.iterrows():
                    r = classify_text(str(row["_text"])[:512], classifier)
                    results.append(r)
                    prog.progress(len(results) / n, text=f"Classified {len(results)} / {n}")
                prog.empty()

                df["Predicted Label"] = [r["label"] for r in results]
                df["Confidence (%)"]  = [r["score"] for r in results]

                # Summary metrics
                counts = df["Predicted Label"].value_counts()
                all_cats = ["World News", "Sports", "Business", "Science / Technology"]
                cols4 = st.columns(4)
                for col, cat in zip(cols4, all_cats):
                    meta = CATEGORY_META[cat]
                    cnt  = counts.get(cat, 0)
                    col.metric(f"{meta['emoji']} {cat}", cnt)

                avg_conf = df["Confidence (%)"].mean()
                if "True Label" in df.columns:
                    correct = (df["Predicted Label"] == df["True Label"]).sum()
                    acc     = round(correct / n * 100, 1)
                    st.success(f"✅ Accuracy: **{acc}%** ({correct}/{n} correct) · Avg confidence: {avg_conf:.1f}%")
                else:
                    st.info(f"Avg confidence: {avg_conf:.1f}%")

                # Results table
                st.markdown('<div class="section-rule-wrap"><hr class="section-rule-line"><span class="section-rule-label">Article Results</span></div>', unsafe_allow_html=True)

                display_cols = []
                if "Title" in df.columns: display_cols.append("Title")
                if "True Label" in df.columns: display_cols.append("True Label")
                display_cols += ["Predicted Label", "Confidence (%)"]

                # Styled table HTML
                rows_html = ""
                for _, row in df[display_cols].iterrows():
                    pred = row["Predicted Label"]
                    meta = CATEGORY_META.get(pred, {})
                    badge_style = f"background:{meta.get('light','#eee')};color:{meta.get('color','#333')}"
                    badge_html  = f'<span class="cat-badge" style="{badge_style}">{pred}</span>'

                    title_col = f'<td style="max-width:320px">{row.get("Title", row.get("Description",""))[:80]}</td>' if "Title" in display_cols else ""
                    true_col  = f'<td>{row.get("True Label","")}</td>' if "True Label" in display_cols else ""
                    rows_html += f"<tr>{title_col}{true_col}<td>{badge_html}</td><td>{row['Confidence (%)']:.1f}%</td></tr>"

                header_title = "<th>Title</th>" if "Title" in display_cols else ""
                header_true  = "<th>True Label</th>" if "True Label" in display_cols else ""

                st.markdown(f"""
                <div class="batch-table-wrap">
                <table class="btable">
                    <thead><tr>{header_title}{header_true}<th>Predicted</th><th>Confidence</th></tr></thead>
                    <tbody>{rows_html}</tbody>
                </table>
                </div>
                """, unsafe_allow_html=True)

                # Download
                csv_out = df[display_cols].to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download results as CSV",
                    data=csv_out,
                    file_name="classified_news.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — HISTORY
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div style="height:20px"></div>', unsafe_allow_html=True)
    hist = st.session_state.history
    if not hist:
        st.markdown("""
        <div style="text-align:center;padding:60px 0;opacity:0.4;font-style:italic;
                    font-size:14px;font-family:'DM Sans',sans-serif;color:#1a1208">
            No classifications yet. Start with the Single Article tab.
        </div>
        """, unsafe_allow_html=True)
    else:
        items_html = ""
        for h in hist:
            meta = CATEGORY_META.get(h["cat"], {})
            dot  = meta.get("badge", "#888")
            col  = meta.get("color", "#333")
            items_html += f"""
            <div class="history-item">
                <div class="hist-dot" style="background:{dot}"></div>
                <div class="hist-text">{h['text']}{"…" if len(h['text']) >= 120 else ""}</div>
                <div class="hist-cat" style="color:{col}">{h['cat']}</div>
                <div class="hist-conf">{h['conf']}%</div>
            </div>"""

        st.markdown(f"""
        <div class="section-rule-wrap">
            <hr class="section-rule-line">
            <span class="section-rule-label">Classification History ({len(hist)} items)</span>
        </div>
        <div style="max-height:420px;overflow-y:auto">{items_html}</div>
        """, unsafe_allow_html=True)

        if st.button("Clear History", key="clear_hist"):
            st.session_state.history = []
            st.rerun()

# ── CLOSE paper-body + FOOTER ─────────────────────────────────────────────────
st.markdown("""
</div>
<div class="paper-footer">
    <span>© 2024 NewsType Chronicle · AI Classification Engine</span>
    <span>Model: fabriceyhc/bert-base-uncased-ag_news</span>
    <span>Powered by Hugging Face Transformers</span>
</div>
""", unsafe_allow_html=True)