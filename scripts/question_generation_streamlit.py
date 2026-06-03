import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Stanza's tokenizer DataLoader uses num_workers > 0 by default, which spawns
# subprocesses where numpy is unavailable on macOS. Patch before stanza imports.
import torch.utils.data as _tud
_orig_dl_init = _tud.DataLoader.__init__
def _single_worker_init(self, *args, **kwargs):
    kwargs["num_workers"] = 0
    _orig_dl_init(self, *args, **kwargs)
_tud.DataLoader.__init__ = _single_worker_init

import streamlit as st
from langdetect import detect, LangDetectException

from knowledge_graph import KnowledgeGraph, KnowledgeGraphExtractor, resolve_coreferences
from question_generation import QuestionGenerator

from question_generation.difficulty import LEVELS, LEVEL_ORDER

_SUPPORTED = {"en": "English 🇬🇧", "fi": "Finnish 🇫🇮"}
_DIFF_LABEL = {
    "preA1": "preA1 — Beginner", "A1": "A1 — Elementary", "A2": "A2 — Pre-intermediate",
    "B1": "B1 — Intermediate", "B2": "B2 — Upper-intermediate",
    "C1": "C1 — Advanced", "C2": "C2 — Proficiency", "C2+": "C2+ — Mastery",
}
_DIFF_COLOR = {
    "preA1": "green", "A1": "green", "A2": "green",
    "B1": "orange", "B2": "orange",
    "C1": "red", "C2": "red", "C2+": "red",
}


@st.cache_resource
def _extractor(lang: str) -> KnowledgeGraphExtractor:
    return KnowledgeGraphExtractor(lang=lang)


@st.cache_resource
def _generator(lang: str) -> QuestionGenerator:
    return QuestionGenerator(lang=lang)


def _source_html(source: str, answer: str) -> str:
    body = source
    if answer and answer in body:
        body = body.replace(
            answer,
            f'<mark style="background:#ffe066;padding:1px 4px;border-radius:3px"><b>{answer}</b></mark>',
            1,
        )
    return (
        f'<div style="background:#f8f9fa;padding:10px 14px;border-radius:6px;'
        f'border-left:4px solid #adb5bd;font-size:0.9em;line-height:1.6">{body}</div>'
    )


def _chain_html(chain_path: str) -> str:
    return (
        f'<div style="font-family:monospace;font-size:0.85em;padding:8px 12px;'
        f'background:#eef2ff;border-radius:6px;margin-bottom:8px">'
        f'🔗 {chain_path}</div>'
    )


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="QGen", page_icon="🧠", layout="wide")

st.markdown("""
<style>
/* Keep left column sticky while right scrolls with the page */
[data-testid="stHorizontalBlock"] { align-items: flex-start; }
[data-testid="column"]:first-child {
    position: sticky;
    top: 3.5rem;
}
</style>
""", unsafe_allow_html=True)

# st.title("Knowledge Graph Question Generator")

already_built = "kg" in st.session_state
is_processing = st.session_state.get("processing", False)

col_left, col_right = st.columns(2, gap="large")

# ── Left: input + controls ────────────────────────────────────────────────────

with col_left:
    raw_text = ""

    if not already_built:
        tab_paste, tab_upload = st.tabs(["✏️ Paste text", "📄 Upload .txt"])
        with tab_paste:
            pasted = st.text_area(
                "text_input",
                height=220,
                placeholder="Paste an English or Finnish passage…",
                label_visibility="collapsed",
                disabled=is_processing,
            )
            if pasted.strip():
                raw_text = pasted
        with tab_upload:
            uploaded = st.file_uploader("upload", type=["txt"], label_visibility="collapsed",
                                        disabled=is_processing)
            if uploaded:
                raw_text = uploaded.read().decode("utf-8")
                st.text_area("preview", raw_text, height=160, disabled=True, label_visibility="collapsed")
    else:
        raw_text = st.session_state.raw_text
        st.text_area("locked_text", value=raw_text, disabled=True, height=220, label_visibility="collapsed")

    st.divider()

    max_diff = st.select_slider(
        "Max difficulty",
        options=LEVELS,
        value="B2",
        format_func=_DIFF_LABEL.get,
        disabled=is_processing,
    )
    num_q = st.number_input("Number of questions", min_value=1, value=10, step=1,
                            disabled=is_processing)

    btn_left, btn_right = st.columns(2)
    with btn_left:
        generate = st.button(
            "▶ Generate",
            type="primary",
            disabled=is_processing or not (already_built or bool(raw_text.strip())),
            use_container_width=True,
        )
    with btn_right:
        if already_built and st.button("✕ Reset", use_container_width=True,
                                       disabled=is_processing):
            st.session_state.clear()
            st.rerun()

# ── Right: results ────────────────────────────────────────────────────────────

with col_right:
    if generate:
        # Save text across the rerun, flip flag, rerun to render disabled buttons
        if not already_built:
            st.session_state["_pending_text"] = raw_text
        st.session_state["processing"] = True
        st.rerun()

    if is_processing:
        if not already_built:
            pending_text = st.session_state.pop("_pending_text", "")
            try:
                detected = detect(pending_text)
            except LangDetectException:
                detected = "en"
            lang = detected if detected in _SUPPORTED else "en"

            with st.spinner("Building knowledge graph…"):
                triples = resolve_coreferences(
                    _extractor(lang).extract(pending_text), lang=lang
                )
                kg = KnowledgeGraph()
                kg.add_triples(triples)
            st.session_state.update(raw_text=pending_text, lang=lang, triples=triples, kg=kg)

        lang = st.session_state.lang
        with st.spinner("Generating questions…"):
            all_qs = _generator(lang).generate(
                st.session_state.triples, st.session_state.kg,
                num_questions=100, passage=st.session_state.raw_text,
            )
            st.session_state["questions"] = [
                q for q in all_qs
                if LEVEL_ORDER.get(q.text_difficulty, 0) <= LEVEL_ORDER.get(max_diff, 7)
            ][:int(num_q)]

        st.session_state["processing"] = False
        st.rerun()

    # ── Info bar ──────────────────────────────────────────────────────────────

    if already_built:
        lang = st.session_state.lang
        kg = st.session_state.kg
        n_nodes = kg._g.number_of_nodes()
        n_edges = kg._g.number_of_edges()
        st.caption(
            f"Language: **{_SUPPORTED[lang]}** · Graph: **{n_nodes}** nodes · **{n_edges}** edges"
        )

        with st.expander("Knowledge graph"):
            c_nodes, c_edges = st.columns(2)
            with c_nodes:
                st.markdown("**Nodes**")
                for node, data in kg.nodes:
                    etype = data.get("entity_type") or "—"
                    st.markdown(f"- `{node}` [{etype}]")
            with c_edges:
                st.markdown("**Edges**")
                for src, dst, data in kg.edges:
                    st.markdown(f"- `{src}` → **{data['relation']}** → `{dst}`")

    # ── Questions ─────────────────────────────────────────────────────────────

    if "questions" in st.session_state:
        questions = st.session_state.questions
        st.subheader(f"{len(questions)} questions")

        for q in questions:
            t_colour = _DIFF_COLOR.get(q.text_difficulty, "gray")
            q_colour = _DIFF_COLOR.get(q.question_difficulty, "gray")
            if q.answer_facts:
                answer_display = f"({len(q.answer_facts)} events)"
            elif q.answer_list:
                answer_display = " / ".join(q.answer_list)
            else:
                answer_display = q.answer
            st.markdown(
                f":{t_colour}[**T:{q.text_difficulty}**] :{q_colour}[**Q:{q.question_difficulty}**] {q.text} "
                f"<sub>&nbsp;→&nbsp;<b>{answer_display}</b> &nbsp;·&nbsp; {q.answer_type or '—'}</sub>",
                unsafe_allow_html=True,
            )

            if q.answer_facts or q.chain_path or q.source:
                with st.expander("📍 Source"):
                    if q.chain_path:
                        st.markdown(_chain_html(q.chain_path), unsafe_allow_html=True)
                    if q.answer_facts:
                        for i, fact in enumerate(q.answer_facts, 1):
                            st.markdown(f"{i}. {fact}")
                    elif q.source:
                        st.markdown(_source_html(q.source, q.answer), unsafe_allow_html=True)

            st.divider()
