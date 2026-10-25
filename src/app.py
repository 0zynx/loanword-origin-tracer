import html
import re

import joblib
import pandas as pd
import streamlit as st

from config import MODELS, PROCESSED
from preprocess import script_of, translit, clean


# --------------------------------------------------
# PAGE CONFIGURATION + VISUAL STYLE
# --------------------------------------------------
st.set_page_config(
    page_title="Loanword Origin Tracer",
    page_icon="📚",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {max-width: 1180px; padding-top: 2.1rem; padding-bottom: 3rem;}
    .eyebrow {font-size: .76rem; letter-spacing: .14em; text-transform: uppercase;
               color: #a4a9b5; font-weight: 650; margin-bottom: .55rem;}
    .hero-title {font-size: clamp(2rem, 4vw, 3.1rem); line-height: 1.08;
                 letter-spacing: -.045em; font-weight: 720; margin: 0;}
    .hero-subtitle {font-size: 1.02rem; color: #a4a9b5; margin-top: .8rem; max-width: 690px;}
    .section-kicker {font-size: .76rem; text-transform: uppercase; letter-spacing: .11em;
                     color: #a4a9b5; font-weight: 650; margin-bottom: .35rem;}
    .word-title {font-size: 2.25rem; font-weight: 720; letter-spacing: -.035em; margin: 0;}
    .word-meta {color: #a4a9b5; font-size: .95rem; margin-top: .25rem;}
    .soft-card {border: 1px solid rgba(150,155,170,.28); border-radius: 14px;
                padding: 1.15rem 1.25rem; margin: .35rem 0 .8rem 0;}
    .node {border: 1px solid rgba(150,155,170,.34); border-radius: 11px;
           padding: .8rem 1rem; text-align: center; margin: 0 auto; max-width: 520px;}
    .node-name {font-weight: 650; font-size: 1rem;}
    .node-detail {font-size: .9rem; color: #a4a9b5; margin-top: .2rem;}
    .tree-arrow {text-align: center; color: #a4a9b5; font-size: 1.35rem; line-height: 1.6;}
    div[data-testid="stExpander"] {border-radius: 12px;}
    div[data-testid="stMetric"] {border: 1px solid rgba(150,155,170,.25); padding: .85rem 1rem; border-radius: 12px;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="eyebrow">Language · History · Discovery</div>', unsafe_allow_html=True)
st.markdown('<h1 class="hero-title">Loanword Origin Tracer</h1>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">Explore a word’s recorded origins, language relationships, '
    'and machine-learning analysis in one place.</div>',
    unsafe_allow_html=True,
)
st.write("")


# --------------------------------------------------
# DATA + MODEL HELPERS
# --------------------------------------------------
@st.cache_data
def load_dataset():
    path = PROCESSED / "labeled.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path).fillna("")


@st.cache_resource
def load_model(name):
    bundle = joblib.load(MODELS / f"{name}.joblib")
    return bundle["vec"], bundle["clf"], bundle["labels"]


def split_etymology_text(text):
    """Split compact notes at known historical-stage and note-section labels."""
    text = str(text or "").strip()
    if not text:
        return []

    text = re.sub(r"(?i)\betymology\s+tree\b", "", text)
    # Add boundaries before common labels even when the source string has no newlines.
    pattern = r"(?=(?:Proto-[A-Za-z-]+|Sanskrit|Tamil|Borrowed from|Inherited from|Cognates)\b)"
    text = re.sub(pattern, "\n", text)
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    return [line for line in lines if line]


def extract_tree_stages(text):
    """Extract only stages inside an explicitly labelled etymology tree."""
    raw = str(text or "").strip()
    match = re.search(r"(?i)\betymology\s+tree\b(.*)", raw)
    if not match:
        return []

    tree_text = re.split(
        r"(?i)\b(?:Borrowed from|Inherited from|Cognates)\b",
        match.group(1),
        maxsplit=1,
    )[0]
    stages = split_etymology_text(tree_text)
    return [
        stage for stage in stages
        if not re.match(r"(?i)^(?:Borrowed from|Inherited from|Cognates)\b", stage)
    ]


def show_tree(stages):
    """Render source-recorded stages as a simple connected vertical tree."""
    for index, stage in enumerate(stages):
        st.markdown(
            '<div class="node"><div class="node-name">'
            + html.escape(stage)
            + "</div></div>",
            unsafe_allow_html=True,
        )
        if index < len(stages) - 1:
            st.markdown('<div class="tree-arrow">↓</div>', unsafe_allow_html=True)


def normalized_relation(records, relation):
    if "relation" not in records.columns:
        return pd.DataFrame()
    return records[
        records["relation"].astype(str).str.strip().str.lower().eq(relation)
    ]


# --------------------------------------------------
# SEARCH + SETTINGS
# --------------------------------------------------
df = load_dataset()
with st.sidebar:
    st.markdown("### Search settings")
    view = st.selectbox("Input representation", ["translit", "native"], index=0)
    model_file = st.selectbox("Prediction model", ["baseline_translit", "baseline_native"], index=0)
    st.caption("These settings affect the ML analysis, not the recorded dataset evidence.")

search_col, hint_col = st.columns([4, 1.15], vertical_alignment="bottom")
with search_col:
    word = st.text_input("Search a word", value="அங்கம்", placeholder="Enter a word…", label_visibility="visible")
with hint_col:
    st.markdown("<div style='color:#a4a9b5;font-size:.86rem;padding-bottom:.55rem'>Try a word in its original script or transliteration.</div>", unsafe_allow_html=True)

if not word.strip():
    st.info("Enter a word to explore its recorded history.")
    st.stop()

script = script_of(word)
processed_word = clean(word) if view == "native" else clean(translit(word, script))
if not processed_word:
    st.warning("Please enter a valid word.")
    st.stop()

try:
    vec, clf, labels = load_model(model_file)
except Exception as exc:
    st.error(f"Could not load the selected model: {exc}")
    st.stop()

probabilities = clf.predict_proba(vec.transform([processed_word]))[0]
top_indices = probabilities.argsort()[::-1][:5]
predicted_language = labels[top_indices[0]]
confidence = float(probabilities[top_indices[0]])

# Dataset lookup: independent from the model prediction.
records = pd.DataFrame()
if not df.empty:
    searchable_columns = [c for c in ["word", "word_native", "word_translit"] if c in df.columns]
    normalized_input = clean(word)
    normalized_processed = clean(processed_word)
    matches = pd.Series(False, index=df.index, dtype=bool)
    for column in searchable_columns:
        normalized_column = df[column].map(clean)
        matches |= (normalized_column == normalized_input) | (normalized_column == normalized_processed)
    records = df.loc[matches].drop_duplicates()


# --------------------------------------------------
# WORD SUMMARY
# --------------------------------------------------
st.divider()
left, right = st.columns([1.55, 1], gap="large")
with left:
    st.markdown('<div class="section-kicker">Word entry</div>', unsafe_allow_html=True)
    recorded_word = word
    recorded_language = ""
    recorded_source = ""
    recorded_relation = ""
    etymology = ""
    if not records.empty:
        first_record = records.iloc[0]
        recorded_word = str(first_record.get("word_native", "")).strip() or str(first_record.get("word", word)).strip()
        recorded_language = str(first_record.get("lang_name", "")).strip()
        recorded_source = str(first_record.get("source_name", "")).strip()
        recorded_relation = str(first_record.get("relation", "")).strip()
        etymology = str(first_record.get("etymology_text", "")).strip()

    st.markdown(f'<div class="word-title">{html.escape(recorded_word)}</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="word-meta">{html.escape(recorded_language or "Language not recorded")}</div>',
        unsafe_allow_html=True,
    )
    if recorded_relation.lower() == "borrowed":
        st.write("")
        st.success(f"Recorded as borrowed{f' · Source: {recorded_source}' if recorded_source else ''}")
    elif recorded_relation.lower() == "inherited":
        st.write("")
        st.info("Recorded as inherited, not borrowed")
    elif records.empty:
        st.write("")
        st.warning("No exact dataset record found")
    else:
        st.write("")
        st.caption("The dataset does not specify a borrowing relation for this record.")

with right:
    st.markdown('<div class="section-kicker">Recorded relationship</div>', unsafe_allow_html=True)
    if recorded_source and recorded_language and recorded_relation.lower() == "borrowed":
        a, arrow, b = st.columns([2, 1, 2], vertical_alignment="center")
        with a:
            st.markdown(f'<div class="node"><div class="node-name">{html.escape(recorded_source)}</div><div class="node-detail">Recorded source</div></div>', unsafe_allow_html=True)
        with arrow:
            st.markdown('<div class="tree-arrow">→</div>', unsafe_allow_html=True)
        with b:
            st.markdown(f'<div class="node"><div class="node-name">{html.escape(recorded_language)}</div><div class="node-detail">Word language</div></div>', unsafe_allow_html=True)
        st.caption("This direct relationship is recorded in the dataset; it is separate from the ML prediction.")
    elif recorded_relation.lower() == "inherited":
        st.markdown('<div class="soft-card">The dataset marks this word as inherited. It is not displayed as a loanword pathway.</div>', unsafe_allow_html=True)
        if recorded_source:
            st.caption(f"Recorded historical source: {recorded_source}")
    else:
        st.markdown('<div class="soft-card">A borrowing relationship is not established by the available exact-match record.</div>', unsafe_allow_html=True)


# --------------------------------------------------
# INTERACTIVE HISTORICAL TREE
# --------------------------------------------------
st.divider()
st.markdown('<div class="section-kicker">Word history</div>', unsafe_allow_html=True)
st.markdown("## Explore the word’s history")
st.caption("Select a node in the tree to inspect the recorded stage. Connections are a visual guide to the order in the source notes, not independent proof of a direct borrowing route.")

if df.empty:
    st.info("The processed dataset was not found. Check data/processed/labeled.csv.")
elif records.empty:
    st.info("No exact matching word was found in the dataset. The ML analysis is still available below, but a historical origin has not been established by this lookup.")
else:
    tree_stages = extract_tree_stages(etymology)
    if tree_stages:
        graph_col, detail_col = st.columns([1.55, 0.85], gap="large")
        stage_key = f"selected_stage_{clean(recorded_word)}"
        if stage_key not in st.session_state or st.session_state[stage_key] not in tree_stages:
            st.session_state[stage_key] = tree_stages[-1]

        # Render a real SVG graph. Stages are placed as connected nodes on a
        # canvas instead of being stacked as ordinary Streamlit buttons.
        with graph_col:
            st.markdown('<div class="section-kicker">Origin graph</div>', unsafe_allow_html=True)
            node_w, node_h = 210, 76
            gap_x = 58
            pad_x, pad_y = 28, 34
            canvas_w = pad_x * 2 + len(tree_stages) * node_w + max(0, len(tree_stages) - 1) * gap_x
            canvas_h = 178
            parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {canvas_w} {canvas_h}" width="100%" role="img" aria-label="Connected historical stages graph">']
            parts.append('<rect x="1" y="1" width="99.5%" height="99%" rx="16" fill="none" stroke="currentColor" stroke-opacity=".16"/>')
            # Edges first, so they appear behind nodes.
            for i in range(len(tree_stages) - 1):
                x1 = pad_x + i * (node_w + gap_x) + node_w
                x2 = x1 + gap_x
                y = pad_y + node_h / 2
                parts.append(f'<path d="M {x1} {y} C {x1 + gap_x/2} {y}, {x2 - gap_x/2} {y}, {x2} {y}" stroke="currentColor" stroke-opacity=".52" stroke-width="2.5" fill="none"/>')
                parts.append(f'<path d="M {x2-7} {y-5} L {x2} {y} L {x2-7} {y+5}" stroke="currentColor" stroke-opacity=".65" stroke-width="2" fill="none"/>')
            for i, stage in enumerate(tree_stages):
                x = pad_x + i * (node_w + gap_x)
                y = pad_y
                selected = stage == st.session_state[stage_key]
                stroke = '#8b9b75' if selected else 'currentColor'
                fill = '#e8eddf' if selected else 'none'
                text_color = '#293323' if selected else 'currentColor'
                safe_stage = html.escape(stage)
                # Split long labels into lines to keep each node legible.
                words = stage.split()
                lines, line = [], ''
                for word_part in words:
                    candidate = (line + ' ' + word_part).strip()
                    if len(candidate) > 23 and line:
                        lines.append(line)
                        line = word_part
                    else:
                        line = candidate
                if line:
                    lines.append(line)
                lines = lines[:3]
                text_bits = ''.join(f'<text x="{x + node_w/2}" y="{y + 29 + j*17}" text-anchor="middle" font-family="Arial,sans-serif" font-size="13" font-weight="600" fill="{text_color}">{html.escape(part)}</text>' for j, part in enumerate(lines))
                parts.append(f'<rect x="{x}" y="{y}" width="{node_w}" height="{node_h}" rx="12" fill="{fill}" stroke="{stroke}" stroke-width="{2.5 if selected else 1.4}"/>')
                parts.append(text_bits)
                parts.append(f'<text x="{x + node_w/2}" y="{y + node_h + 22}" text-anchor="middle" font-family="Arial,sans-serif" font-size="11" fill="currentColor" opacity=".65">Stage {i+1}</text>')
            parts.append('</svg>')
            st.markdown(''.join(parts), unsafe_allow_html=True)
            st.caption("Graph of stages explicitly listed in the source note. Edges show their displayed order, not proof of a direct borrowing route.")

            # Actual Streamlit controls select/highlight the corresponding graph node.
            st.markdown("**Inspect a node**")
            buttons = st.columns(min(len(tree_stages), 3))
            for index, stage in enumerate(tree_stages):
                with buttons[index % len(buttons)]:
                    is_selected = st.session_state[stage_key] == stage
                    if st.button(stage, key=f"stage_{clean(recorded_word)}_{index}", use_container_width=True, type="primary" if is_selected else "secondary"):
                        st.session_state[stage_key] = stage
                        st.rerun()

        with detail_col:
            selected_stage = st.session_state[stage_key]
            st.markdown('<div class="section-kicker">Selected stage</div>', unsafe_allow_html=True)
            st.markdown('<div class="soft-card"><div class="word-title" style="font-size:1.45rem">' + html.escape(selected_stage) + '</div><div class="word-meta">From the dataset’s etymology tree</div></div>', unsafe_allow_html=True)
            st.markdown("**Current word**")
            st.write(f"{recorded_word} · {recorded_language or 'Language not recorded'}")
            st.markdown("**Recorded relationship**")
            if recorded_relation.lower() == "borrowed" and recorded_source and recorded_language:
                st.write(f"{recorded_source} → {recorded_language}")
            elif recorded_relation.lower() == "inherited":
                st.write("Marked as inherited, not borrowed.")
            else:
                st.write("No borrowing relationship established by this record.")
            st.caption("Selecting a node shows its recorded text. It does not assert that every adjacent stage was the direct source of the next.")
    elif etymology:
        notes_col, detail_col = st.columns([1.2, 0.8], gap="large")
        with notes_col:
            st.markdown('<div class="section-kicker">Recorded notes</div>', unsafe_allow_html=True)
            for line in split_etymology_text(etymology):
                if line.lower().startswith(("borrowed from", "inherited from", "cognates")):
                    st.caption(line)
                else:
                    st.write(line)
        with detail_col:
            st.markdown('<div class="section-kicker">Relationship status</div>', unsafe_allow_html=True)
            if recorded_relation.lower() == "borrowed":
                st.info(f"Recorded as borrowed{f' from {recorded_source}' if recorded_source else ''}.")
            elif recorded_relation.lower() == "inherited":
                st.info("Recorded as inherited, not borrowed.")
            else:
                st.info("The record does not establish a borrowing relationship.")
        st.caption("No explicitly labelled etymology tree was found, so no sequence of stages has been inferred.")
    else:
        st.info("No etymology notes are available for this record.")

    if len(records) > 1:
        with st.expander(f"Other matching dataset records ({len(records) - 1})"):
            for number, (_, record) in enumerate(records.iloc[1:].head(9).iterrows(), start=2):
                r_word = str(record.get("word_native", "")).strip() or str(record.get("word", "")).strip()
                r_lang = str(record.get("lang_name", "")).strip() or "Language not specified"
                st.markdown(f"**{number}. {r_word} — {r_lang}**")
                st.caption(f"Source: {str(record.get('source_name', '')).strip() or 'Not recorded'} · Relation: {str(record.get('relation', '')).strip() or 'Not specified'}")


# --------------------------------------------------
# MACHINE-LEARNING ANALYSIS (SECONDARY)
# --------------------------------------------------
st.divider()
with st.expander("Machine-learning analysis", expanded=False):
    st.caption("Model output is a prediction from learned patterns. It does not prove a word’s historical origin.")
    metric1, metric2 = st.columns(2)
    metric1.metric("Predicted language", str(predicted_language))
    metric2.metric("Top model score", f"{confidence:.1%}")

    chart_data = pd.DataFrame({
        "Language": [str(labels[i]) for i in top_indices],
        "Score": [float(probabilities[i]) for i in top_indices],
    }).set_index("Language")
    st.markdown("#### Top five language scores")
    st.bar_chart(chart_data)

    st.markdown("#### Character influence")
    base_score = float(probabilities.max())
    influences = []
    for i in range(len(processed_word)):
        shortened = processed_word[:i] + processed_word[i + 1:]
        if shortened:
            altered_probabilities = clf.predict_proba(vec.transform([shortened]))[0]
            altered_score = float(altered_probabilities.max())
        else:
            altered_score = 0.0
        influences.append(max(base_score - altered_score, 0.0))

    max_influence = max(influences, default=0.0)
    if max_influence > 0:
        highlighted = []
        for char, score in zip(processed_word, influences):
            opacity = min(score / max_influence, 1.0)
            highlighted.append(
                "<span style='background-color:"
                f"rgba(220,70,70,{opacity:.2f});padding:3px 5px;margin:2px;border-radius:4px;'>"
                f"{html.escape(char)}</span>"
            )
        st.markdown(" ".join(highlighted), unsafe_allow_html=True)
        st.caption("Highlight intensity indicates how much the model’s maximum score changed when a character was removed. This is a diagnostic, not proof of linguistic origin.")
    else:
        st.write("No individual character influence stood out.")


# --------------------------------------------------
# LIMITATIONS
# --------------------------------------------------
st.caption("Dataset records and ML predictions are separate sources of information. A complete historical route should only be shown when supported by reliable evidence for each stage.")
