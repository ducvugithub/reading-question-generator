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
from question_generation.difficulty import CefrReadability

from question_generation.difficulty import LEVELS, LEVEL_ORDER

_REPO_ROOT = Path(__file__).parent.parent
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
def _cefr_readability() -> CefrReadability:
    return CefrReadability()


@st.cache_resource
def _extractor(lang: str) -> KnowledgeGraphExtractor:
    return KnowledgeGraphExtractor(lang=lang)


@st.cache_resource
def _generator(lang: str) -> QuestionGenerator:
    return QuestionGenerator(lang=lang, cefr_readability=_cefr_readability())


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

# ── State initialization ───────────────────────────────────────────────────────

analyzed = st.session_state.get("analyzed", False)
is_analyzing = st.session_state.get("is_analyzing", False)
is_generating = st.session_state.get("is_generating", False)

col_left, col_right = st.columns(2, gap="large")

# ── Left: input + controls ────────────────────────────────────────────────────

with col_left:
    st.markdown("### 📝 Input Text")

    tab_paste, tab_upload, tab_examples = st.tabs(["✏️ Paste text", "📄 Upload .txt", "📂 Examples"])
    raw_text = st.session_state.get("app_text", "")

    with tab_paste:
        pasted = st.text_area(
            "text_input",
            height=220,
            placeholder="Paste an English or Finnish passage…",
            label_visibility="collapsed",
            disabled=is_analyzing or is_generating,
            value=raw_text,
            key="paste_textarea",
        )
        if pasted:
            st.session_state["app_text"] = pasted
            raw_text = pasted

    with tab_upload:
        uploaded = st.file_uploader("upload", type=["txt"], label_visibility="collapsed",
                                    disabled=is_analyzing or is_generating, key="upload_file")
        if uploaded:
            content = uploaded.read().decode("utf-8")
            st.session_state["app_text"] = content
            raw_text = content
            st.text_area("preview", content, height=160, disabled=True, label_visibility="collapsed")

    with tab_examples:
        example_files = {}
        for lang_code, lang_label in _SUPPORTED.items():
            lang_dir = _REPO_ROOT / "data" / lang_code
            if lang_dir.exists():
                for p in sorted(lang_dir.glob("*.txt")):
                    example_files[f"{lang_label} — {p.stem}"] = p
        if example_files:
            selected = st.selectbox(
                "Select a text file",
                options=list(example_files.keys()),
                label_visibility="collapsed",
                disabled=is_analyzing or is_generating,
            )
            if st.button("Load", disabled=is_analyzing or is_generating, use_container_width=True):
                content = example_files[selected].read_text(encoding="utf-8")
                st.session_state["app_text"] = content
                raw_text = content
                st.rerun()
            if raw_text:
                st.text_area("example_preview", raw_text, height=140, disabled=True, label_visibility="collapsed")
        else:
            st.caption("No .txt files found under data/en/ or data/fi/")

    st.divider()

    # ── Step 1: Analyze ────────────────────────────────────────────────────────

    st.markdown("### 🔍 Step 1: Analyze Text")

    # Debug info
    text_len = len(raw_text.strip()) if raw_text else 0
    st.caption(f"📊 Text ready: {text_len} characters")

    btn_col1, btn_col2 = st.columns([3, 1])
    with btn_col1:
        can_analyze = not (is_analyzing or is_generating) and text_len >= 50
        analyze_btn = st.button(
            "Analyze" if can_analyze else f"Analyze (need {max(0, 50 - text_len)} more chars)",
            type="primary" if can_analyze else "secondary",
            disabled=not can_analyze,
            use_container_width=True,
        )
    with btn_col2:
        if st.button("✕ Reset", disabled=is_analyzing or is_generating, use_container_width=True):
            st.session_state.clear()
            st.rerun()

    if analyze_btn and raw_text:
        st.session_state["_pending_text"] = raw_text
        st.session_state["is_analyzing"] = True
        st.rerun()

    # ── Step 2: Generate (only after analysis) ────────────────────────────────

    if analyzed:
        st.divider()
        st.markdown("### ⚙️ Step 2: Generate Questions")

        passage_cefr = st.session_state.get("passage_cefr", "B1")
        passage_cefr_idx = LEVEL_ORDER.get(passage_cefr, 3)
        available_levels = LEVELS[:passage_cefr_idx + 1]

        # Ensure slider value is within available levels
        slider_value = passage_cefr if passage_cefr in available_levels else available_levels[-1]

        max_diff = st.select_slider(
            "Max difficulty",
            options=available_levels,
            value=slider_value,
            format_func=_DIFF_LABEL.get,
            disabled=is_generating,
        )

        num_col, gen_col = st.columns([1.5, 1], gap="medium")

        with num_col:
            num_q = st.number_input(
                "Number of questions",
                min_value=1,
                max_value=100,
                value=10,
                step=1,
                disabled=is_generating,
            )

        with gen_col:
            st.markdown("<div style='margin-top: 0.5rem'></div>", unsafe_allow_html=True)
            gen_btn = st.button(
                "Generate Questions",
                type="primary",
                disabled=is_generating,
                use_container_width=True,
            )

        if gen_btn:
            st.session_state["max_diff"] = max_diff
            st.session_state["num_q"] = num_q
            st.session_state["is_generating"] = True
            st.rerun()


# ── Right: results ────────────────────────────────────────────────────────────

with col_right:
    # Process analysis
    if is_analyzing:
        pending_text = st.session_state.pop("_pending_text", "")

        try:
            detected = detect(pending_text)
        except LangDetectException:
            detected = "en"
        lang = detected if detected in _SUPPORTED else "en"

        with st.spinner("🔍 Analyzing text…"):
            triples = resolve_coreferences(
                _extractor(lang).extract(pending_text), lang=lang
            )
            kg = KnowledgeGraph()
            kg.add_triples(triples)
            passage_cefr = _cefr_readability().estimate(pending_text)

        st.session_state.update(
            raw_text=pending_text,
            lang=lang,
            triples=triples,
            kg=kg,
            passage_cefr=passage_cefr,
            analyzed=True,
            is_analyzing=False,
        )
        st.rerun()

    # Show analysis results
    if analyzed:
        lang = st.session_state.get("lang", "en")
        kg = st.session_state.get("kg")
        passage_cefr = st.session_state.get("passage_cefr", "B1")

        st.markdown("### 📊 Analysis Results")

        # Info card
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Language", _SUPPORTED.get(lang, "Unknown"))
        with col2:
            st.metric("Passage Level", f"**{passage_cefr}**")
        with col3:
            if kg:
                st.metric("Graph Size", f"{kg._g.number_of_nodes()} nodes")

        if kg:
            st.caption(f"**{kg._g.number_of_nodes()}** nodes · **{kg._g.number_of_edges()}** edges")

            with st.expander("📈 Show Knowledge Graph"):
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

        st.divider()

    # Process generation
    if is_generating:
        lang = st.session_state.get("lang", "en")
        triples = st.session_state.get("triples", [])
        kg = st.session_state.get("kg")
        passage = st.session_state.get("raw_text", "")
        max_diff = st.session_state.get("max_diff", "B1")
        num_q = st.session_state.get("num_q", 10)

        with st.spinner("⚙️ Generating questions…"):
            all_qs = _generator(lang).generate(
                triples, kg,
                num_questions=100,
                passage=passage,
                target_template_cefr=max_diff,
            )

        # Filter by max difficulty
        passage_cefr = st.session_state.get("passage_cefr", "B1")
        if max_diff > passage_cefr:
            st.warning(
                f"📝 **Passage is {passage_cefr} level**, but you requested {max_diff} questions. "
                f"Question difficulty is limited by passage complexity."
            )

        questions = [
            q for q in all_qs
            if LEVEL_ORDER.get(q.difficulty, 0) <= LEVEL_ORDER.get(max_diff, 7)
        ][:num_q]

        st.session_state["questions"] = questions
        st.session_state["is_generating"] = False
        st.rerun()

    # Display questions
    if "questions" in st.session_state:
        questions = st.session_state["questions"]

        st.markdown(f"### ❓ Questions ({len(questions)})")

        for i, q in enumerate(questions, 1):
            colour = _DIFF_COLOR.get(q.difficulty, "gray")
            if q.answer_facts:
                answer_display = q.answer_facts[0] if len(q.answer_facts) == 1 else f"({len(q.answer_facts)} events)"
            elif q.answer_list:
                answer_display = " / ".join(q.answer_list)
            else:
                answer_display = q.answer

            st.markdown(
                f"**{i}.** :{colour}[{q.difficulty}] {q.text}  \n"
                f"<sub>→ **{answer_display}** · {q.answer_type or '—'}</sub>",
                unsafe_allow_html=True,
            )

            if q.answer_facts or q.chain_path or q.source:
                with st.expander("📍 Details"):
                    if q.chain_path:
                        st.markdown(_chain_html(q.chain_path), unsafe_allow_html=True)
                    if q.answer_facts:
                        for j, fact in enumerate(q.answer_facts, 1):
                            st.markdown(f"{j}. {fact}")
                    elif q.source:
                        st.markdown(_source_html(q.source, q.answer), unsafe_allow_html=True)

            st.divider()
