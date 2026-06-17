from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

from data_factory import DataFactory
from data_factory.config import FactorySettings

st.set_page_config(
    page_title="Data Factory",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    html, body, [class*="css"] { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }

    .block-container { max-width: 1280px; padding: 1.5rem 2rem; }

    .app-header {
        display: flex; align-items: center; justify-content: space-between;
        padding: 0 0 1.5rem 0; border-bottom: 1px solid #e9ecef; margin-bottom: 2rem;
    }
    .app-header h1 { font-size: 1.5rem; font-weight: 600; color: #111827; margin: 0; }
    .app-header .subtitle { font-size: 0.85rem; color: #6b7280; margin-top: 0.15rem; }
    .app-header .status-badge {
        display: inline-flex; align-items: center; gap: 0.4rem;
        padding: 0.3rem 0.75rem; border-radius: 999px;
        font-size: 0.75rem; font-weight: 500;
    }
    .status-badge.active { background: #d1fae5; color: #065f46; }
    .status-badge.inactive { background: #f3f4f6; color: #6b7280; }
    .status-badge .dot { width: 6px; height: 6px; border-radius: 50%; display: inline-block; }
    .status-badge.active .dot { background: #059669; }
    .status-badge.inactive .dot { background: #9ca3af; }

    .metric-row { display: flex; gap: 1rem; margin: 1.5rem 0; }
    .metric-card {
        flex: 1; background: #ffffff; border: 1px solid #e5e7eb;
        border-radius: 10px; padding: 1.25rem 1.5rem;
        box-shadow: 0 1px 2px rgba(0,0,0,0.02);
    }
    .metric-card .label { font-size: 0.75rem; font-weight: 500; color: #6b7280; text-transform: uppercase; letter-spacing: 0.05em; }
    .metric-card .value { font-size: 1.75rem; font-weight: 700; color: #111827; margin-top: 0.25rem; line-height: 1.2; }
    .metric-card .value.pass { color: #059669; }
    .metric-card .value.fail { color: #dc2626; }
    .metric-card .value.muted { color: #9ca3af; }

    .section-title {
        font-size: 1rem; font-weight: 600; color: #111827;
        margin: 1.5rem 0 1rem 0; padding-bottom: 0.5rem;
        border-bottom: 1px solid #f3f4f6;
    }

    .stButton > button {
        font-weight: 500; border-radius: 8px; font-size: 0.875rem;
        transition: all 0.15s ease;
    }
    .stButton > button[kind="primary"] {
        background: #111827; color: white; border: none;
    }
    .stButton > button[kind="primary"]:hover {
        background: #1f2937; box-shadow: 0 2px 8px rgba(0,0,0,0.12);
    }

    .stTextInput > div > div, .stTextArea > div > div, .stSelectbox > div > div {
        border-radius: 8px; border-color: #e5e7eb; font-size: 0.875rem;
    }
    .stSlider > div > div > div { background: #e5e7eb; }
    .stSlider > div > div > div > div { background: #111827; }

    .stDataFrame { border: 1px solid #e5e7eb; border-radius: 10px; overflow: hidden; }
    .stDataFrame table { font-size: 0.8rem; }
    .stDataFrame thead tr th { background: #f9fafb; font-weight: 600; color: #374151; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.04em; padding: 0.6rem 0.8rem; }

    .stTabs [data-baseweb="tab-list"] { gap: 0; border-bottom: 1px solid #e5e7eb; }
    .stTabs [data-baseweb="tab"] {
        font-size: 0.85rem; font-weight: 500; color: #6b7280;
        padding: 0.6rem 1.2rem; border: none; border-bottom: 2px solid transparent;
        transition: all 0.15s ease;
    }
    .stTabs [aria-selected="true"] { color: #111827; border-bottom-color: #111827; }
    .stTabs [data-baseweb="tab"]:hover { color: #111827; background: #f9fafb; }

    .quality-bar {
        height: 6px; border-radius: 999px; background: #e5e7eb; margin: 0.3rem 0; overflow: hidden;
    }
    .quality-bar .fill { height: 100%; border-radius: 999px; transition: width 0.3s ease; }
    .quality-bar .fill.good { background: #059669; }
    .quality-bar .fill.ok { background: #f59e0b; }
    .quality-bar .fill.bad { background: #dc2626; }

    .source-tag {
        display: inline-block; padding: 0.15rem 0.5rem; border-radius: 4px;
        font-size: 0.7rem; font-weight: 500; background: #f3f4f6; color: #6b7280;
        margin: 0.1rem 0.2rem;
    }
    .source-tag.url { background: #dbeafe; color: #1e40af; }
    .source-tag.file { background: #fef3c7; color: #92400e; }

    hr { margin: 1.5rem 0; border-color: #e5e7eb; }
    .stAlert { border-radius: 8px; font-size: 0.85rem; }
    .stInfo, .stSuccess, .stError { border: none !important; }
</style>
""",
    unsafe_allow_html=True,
)

# ── Session State ──────────────────────────────────────────────────────

for key in ("factory", "dataset", "run_history"):
    if key not in st.session_state:
        st.session_state[key] = None if key != "run_history" else []

# ── Header ─────────────────────────────────────────────────────────────

st.markdown('<div class="app-header">', unsafe_allow_html=True)
c1, c2 = st.columns([1, 1])
with c1:
    st.markdown("<h1>Data Factory</h1><div class='subtitle'>AI Training Dataset Generation Pipeline</div>", unsafe_allow_html=True)
with c2:
    st.markdown("", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────

st.sidebar.markdown("## Settings")

use_llm = st.sidebar.checkbox(
    "Enable LLM generation", value=False,
    help="When disabled, uses fast rule-based generation (no API calls). Enable for higher quality examples.",
)
api_key = st.sidebar.text_input(
    "API Key", type="password",
    disabled=not use_llm,
    placeholder="sk-or-v1-..." if use_llm else "",
    help="OpenRouter or OpenAI API key. Required when LLM is enabled.",
)
model = st.sidebar.selectbox(
    "Model",
    options=["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
    index=0,
    disabled=not use_llm,
)
chunk_size = st.sidebar.slider(
    "Chunk size (tokens)", min_value=128, max_value=2048, value=512, step=64,
    help="Larger chunks preserve more context but increase processing time per example.",
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Quality checks")
enable_toxicity = st.sidebar.checkbox("Toxicity", value=True)
enable_bias = st.sidebar.checkbox("Bias detection", value=True)
enable_diversity = st.sidebar.checkbox("Diversity", value=True)
enable_coherence = st.sidebar.checkbox("Coherence", value=True)

# Sidebar status
st.sidebar.markdown("---")
settings_kwargs: Dict[str, Any] = dict(
    llm_model=model,
    chunk_size=chunk_size,
    enable_toxicity_check=enable_toxicity,
    enable_bias_check=enable_bias,
    enable_diversity_check=enable_diversity,
    enable_coherence_check=enable_coherence,
)
if use_llm and api_key:
    settings_kwargs["llm_api_key"] = api_key
elif not use_llm:
    settings_kwargs["llm_api_key"] = ""

settings = FactorySettings(**settings_kwargs)
factory = DataFactory(settings)
st.session_state.factory = factory

if factory.llm_client:
    st.sidebar.markdown(
        f"<div class='status-badge active'><span class='dot'></span>"
        f"LLM: {factory.settings.llm_provider} / {factory.settings.llm_model}</div>",
        unsafe_allow_html=True,
    )
else:
    st.sidebar.markdown(
        "<div class='status-badge inactive'><span class='dot'></span>"
        "Rule-based generation (LLM disabled)</div>",
        unsafe_allow_html=True,
    )

st.sidebar.markdown(f"<div style='font-size:0.75rem;color:#9ca3af;margin-top:1rem;'>Data Factory v0.1.0</div>", unsafe_allow_html=True)

# ── Main Tabs ─────────────────────────────────────────────────────────

tab_pipeline, tab_dataset, tab_quality, tab_history = st.tabs(
    ["Pipeline", "Dataset", "Quality Report", "Run History"]
)

# =====================================================================
# TAB 1: PIPELINE
# =====================================================================

with tab_pipeline:
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown('<div class="section-title">Sources</div>', unsafe_allow_html=True)
        sources_input = st.text_area(
            "Sources (one per line)",
            placeholder="document.pdf\nhttps://example.com\nnotes.txt",
            height=100,
            label_visibility="collapsed",
        )

    with col2:
        st.markdown('<div class="section-title">Tasks</div>', unsafe_allow_html=True)
        task_options = st.multiselect(
            "Tasks",
            options=["qa", "summarization", "classification", "extraction"],
            default=["qa", "summarization"],
            label_visibility="collapsed",
        )

    with st.expander("Advanced options", expanded=False):
        r1, r2, r3 = st.columns(3)
        with r1:
            run_name = st.text_input("Run name (optional)", placeholder="my-run-1")
        with r2:
            export_path = st.text_input("Export path (optional)", placeholder="output/dataset.jsonl")
        with r3:
            skip_qc = st.checkbox("Skip quality check", value=False)

        max_chunks = st.slider(
            "Max chunks to process",
            min_value=1, max_value=50, value=3,
            help="Limits the number of text chunks processed. Lower values give faster results.",
        )

    run_col1, run_col2 = st.columns([3, 1])
    with run_col1:
        status_placeholder = st.empty()
    with run_col2:
        run_button = st.button("Run Pipeline", type="primary", use_container_width=True)

    if run_button:
        if not sources_input.strip():
            st.error("Please provide at least one source.")
        else:
            sources = [s.strip() for s in sources_input.split("\n") if s.strip()]
            progress_bar = status_placeholder.progress(0, text="Initializing...")

            try:
                progress_bar.progress(10, text="Loading sources...")
                dataset = factory.run(
                    sources=sources,
                    tasks=task_options if task_options else None,
                    run_name=run_name or None,
                    export_path=export_path or None,
                    skip_quality=skip_qc,
                    max_chunks=max_chunks,
                )
                progress_bar.progress(100, text="Complete")

                st.session_state.dataset = dataset
                meta = factory.metadata_tracker
                st.session_state.run_history.append({
                    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "sources": sources,
                    "tasks": task_options,
                    "documents": len(factory.documents),
                    "chunks": len(factory.chunks),
                    "examples": dataset.size,
                    "passed_qc": len(dataset.examples),
                    "cost": factory.cost_tracker.total_cost,
                    "run_id": meta.current.run_id if meta.current else "N/A",
                })

                st.success(f"Pipeline complete — {dataset.size} examples generated.")

            except Exception as e:
                progress_bar.empty()
                st.error(f"Pipeline failed: {e}")

    # Metrics row
    ds = st.session_state.dataset
    st.markdown('<div class="metric-row">', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(
            f"<div class='metric-card'><div class='label'>Documents loaded</div>"
            f"<div class='value'>{len(factory.documents) if factory.documents else '—'}</div></div>",
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f"<div class='metric-card'><div class='label'>Text chunks</div>"
            f"<div class='value'>{len(factory.chunks) if factory.chunks else '—'}</div></div>",
            unsafe_allow_html=True,
        )
    with m3:
        val = ds.size if ds else 0
        cls = "value" if val == 0 else "value pass"
        st.markdown(
            f"<div class='metric-card'><div class='label'>Examples generated</div>"
            f"<div class='{cls}'>{val}</div></div>",
            unsafe_allow_html=True,
        )
    with m4:
        cost_val = factory.cost_tracker.total_cost
        cost_display = f"${cost_val:.4f}" if cost_val > 0 else "$0.00"
        st.markdown(
            f"<div class='metric-card'><div class='label'>Total cost</div>"
            f"<div class='value'>{cost_display}</div></div>",
            unsafe_allow_html=True,
        )
    st.markdown('</div>', unsafe_allow_html=True)

# =====================================================================
# TAB 2: DATASET
# =====================================================================

with tab_dataset:
    ds = st.session_state.dataset

    if not ds or not ds.examples:
        st.info("No dataset available. Run the pipeline first.")
    else:
        st.markdown(f'<div class="section-title">Dataset: {ds.name}</div>', unsafe_allow_html=True)
        st.markdown(f"**{ds.size}** examples | {ds.metadata.get('tasks', 'N/A')}")

        col1, col2 = st.columns(2)
        with col1:
            task_types = sorted(set(ex.task_type.value for ex in ds.examples))
            task_filter = st.multiselect("Filter by task type", options=task_types, default=[])
        with col2:
            quality_filter = st.selectbox(
                "Quality filter", options=["All", "Passed QC only", "Failed QC only"]
            )

        filtered = ds.examples
        if task_filter:
            filtered = [ex for ex in filtered if ex.task_type.value in task_filter]
        if quality_filter == "Passed QC only":
            filtered = ds.passed_quality
        elif quality_filter == "Failed QC only":
            passed_ids = {r.example_id for r in ds.quality_reports if r.passed}
            filtered = [ex for ex in filtered if ex.id not in passed_ids]

        rows = []
        for ex in filtered[:200]:
            qs = ex.quality_scores or {}
            rows.append({
                "ID": ex.id[:8],
                "Type": ex.task_type.value,
                "Input": (ex.input[:100] + "...") if len(ex.input) > 100 else ex.input,
                "Output": (ex.expected_output[:100] + "...") if len(ex.expected_output) > 100 else ex.expected_output,
                "Quality": f"{qs.get('overall', 'N/A'):.3f}" if qs.get('overall') else "N/A",
            })

        st.dataframe(rows, use_container_width=True, hide_index=True)
        st.caption(f"Showing {min(len(filtered), 200)} of {len(filtered)} examples")

        dl_col1, dl_col2 = st.columns([1, 1])
        with dl_col1:
            export_json = json.dumps([ex.to_dict() for ex in ds.examples], indent=2, ensure_ascii=False)
            st.download_button(
                "Download as JSON",
                data=export_json,
                file_name=f"{ds.name}.json",
                mime="application/json",
                use_container_width=True,
            )
        with dl_col2:
            export_jsonl = "\n".join(json.dumps(ex.to_dict(), ensure_ascii=False) for ex in ds.examples)
            st.download_button(
                "Download as JSONL",
                data=export_jsonl,
                file_name=f"{ds.name}.jsonl",
                mime="application/jsonl",
                use_container_width=True,
            )

# =====================================================================
# TAB 3: QUALITY REPORT
# =====================================================================

with tab_quality:
    ds = st.session_state.dataset

    if not ds or not ds.quality_reports:
        st.info("No quality data available. Run the pipeline first.")
    else:
        reports = ds.quality_reports

        # Aggregate scores
        all_scores: Dict[str, List[float]] = {}
        for report in reports:
            for metric, score in report.scores.items():
                all_scores.setdefault(metric, []).append(score)

        # Metric cards
        st.markdown('<div class="metric-row">', unsafe_allow_html=True)
        cols = st.columns(len(all_scores))
        for i, (metric, scores) in enumerate(sorted(all_scores.items())):
            avg = sum(scores) / len(scores)
            pct = min(avg * 100, 100)
            bar_class = "good" if avg >= 0.6 else ("ok" if avg >= 0.3 else "bad")
            with cols[i]:
                st.markdown(
                    f"<div class='metric-card'><div class='label'>{metric}</div>"
                    f"<div class='value'>{avg:.3f}</div>"
                    f"<div class='quality-bar'><div class='fill {bar_class}' style='width:{pct}%'></div></div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
        st.markdown('</div>', unsafe_allow_html=True)

        # Issues
        issues = [r.issues for r in reports if r.issues]
        if issues:
            st.markdown('<div class="section-title">Issues</div>', unsafe_allow_html=True)
            flat_issues = [item for sublist in issues[:20] for item in (sublist if isinstance(sublist, list) else [sublist])]
            for issue in flat_issues[:15]:
                st.warning(issue[:200] if isinstance(issue, str) else str(issue)[:200])

        # Quality table
        st.markdown('<div class="section-title">Per-Example Scores</div>', unsafe_allow_html=True)
        qrows = []
        for report in reports[:100]:
            row = {"Example ID": report.example_id[:10], "Passed": "Yes" if report.passed else "No"}
            for metric, score in report.scores.items():
                row[metric.capitalize()] = round(score, 3)
            qrows.append(row)
        st.dataframe(qrows, use_container_width=True, hide_index=True)

# =====================================================================
# TAB 4: RUN HISTORY
# =====================================================================

with tab_history:
    history = st.session_state.run_history

    if not history:
        st.info("No runs yet. Execute a pipeline to see history here.")
    else:
        st.dataframe(
            [
                {
                    "Time": h["time"],
                    "Run ID": h.get("run_id", "N/A")[:12],
                    "Sources": ", ".join(h["sources"]),
                    "Tasks": ", ".join(str(t) for t in h.get("tasks", [])),
                    "Docs": h.get("documents", 0),
                    "Chunks": h.get("chunks", 0),
                    "Examples": h.get("examples", 0),
                    "Cost ($)": f"{h.get('cost', 0):.4f}" if h.get("cost", 0) > 0 else "0.00",
                }
                for h in reversed(history)
            ],
            use_container_width=True,
            hide_index=True,
        )

        if st.button("Clear history"):
            st.session_state.run_history = []
            st.rerun()
