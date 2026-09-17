import pandas as pd
import plotly.express as px
import streamlit as st
from rapidfuzz import fuzz as rf_fuzz
from rapidfuzz import process as rf_process

from src import config
from src.ask_engine import build_process_context
from src.alignment import format_alignment_block
from src.data_loader import (
    get_embed_client,
    get_known_jabatan,
    get_known_processes,
    get_known_roles,
    get_level1_lookup,
    load_df,
    load_embeddings,
    load_jabatan_tugas,
    load_structured_df,
)
from src.retrieval import query_by_process

st.set_page_config(page_title="Analytics Dashboard - RACI PLN", page_icon="📊", layout="wide")

# ---------- Custom styling ----------
st.markdown(
    """
    <style>
        .main > div { padding-top: 1.4rem; }

        .dashboard-header {
            background: linear-gradient(135deg, #0F4C81 0%, #1B6FA8 50%, #2E9CCA 100%);
            padding: 1.8rem 2.2rem;
            border-radius: 16px;
            margin-bottom: 1.6rem;
            box-shadow: 0 8px 24px rgba(15, 76, 129, 0.25);
        }
        .dashboard-header h1 {
            color: white;
            margin: 0;
            font-size: 1.7rem;
            font-weight: 700;
        }
        .dashboard-header p {
            color: rgba(255,255,255,0.85);
            margin: 0.35rem 0 0 0;
            font-size: 0.95rem;
        }

        div[data-testid="stMetric"] {
            background: white;
            border: 1px solid rgba(15, 76, 129, 0.12);
            border-radius: 14px;
            padding: 1rem 1.2rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.04);
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }
        div[data-testid="stMetric"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 18px rgba(15, 76, 129, 0.12);
        }
        div[data-testid="stMetricLabel"] { font-weight: 600; color: #4a5568; }
        div[data-testid="stMetricValue"] { color: #0F4C81; }

        .section-title {
            font-size: 1.1rem;
            font-weight: 700;
            color: #0F4C81;
            margin: 0.2rem 0 0.7rem 0;
            padding-bottom: 0.35rem;
            border-bottom: 3px solid #2E9CCA;
            display: inline-block;
        }

        div[data-baseweb="tab-list"] { gap: 4px; }
        button[data-baseweb="tab"] {
            border-radius: 10px 10px 0 0;
            font-weight: 600;
        }

        div[data-testid="stExpander"] {
            border-radius: 12px;
            border: 1px solid rgba(15, 76, 129, 0.12);
        }

        .stButton>button {
            border-radius: 10px;
            font-weight: 600;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="dashboard-header">
        <h1>📊 Analytics Dashboard — RACI &amp; Tupoksi PLN</h1>
        <p>Ringkasan interaktif matriks RACI, distribusi peran, dan kesesuaian tugas pokok jabatan.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    df = load_df()
    structured_df = load_structured_df()
    embeddings, embedded_ids = load_embeddings()
    jabatan_tugas = load_jabatan_tugas()
    known_roles = get_known_roles(structured_df)
    known_processes = get_known_processes(df)
    known_jabatan_tupoksi = get_known_jabatan(jabatan_tugas)
    level1_lookup = get_level1_lookup(df)
except Exception as e:
    st.error(f"Gagal memuat data: {e}")
    st.stop()

PALETTE = ["#0F4C81", "#2E9CCA", "#57C4E5", "#F4A259", "#E76F51"]

# --- Overview metrics ---
st.markdown('<p class="section-title">🧭 Ringkasan</p>', unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Proses Bisnis", f"{len(df):,}")
col2.metric("Role Unik (RACI)", f"{len(known_roles):,}")
col3.metric("Jabatan dengan Tupoksi", f"{len(known_jabatan_tupoksi):,}")
col4.metric("Total Baris Tugas Pokok", f"{sum(len(v) for v in jabatan_tugas.values()):,}")

st.write("")

tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Distribusi RACI", "🏆 Role Teratas", "🗂️ Kategori Proses", "🔍 Cakupan Tupoksi"]
)

# --- RACI type distribution ---
with tab1:
    st.markdown('<p class="section-title">Distribusi Tipe RACI</p>', unsafe_allow_html=True)
    raci_counts = structured_df["raci_type"].value_counts().reindex(["R", "A", "C", "I"]).fillna(0)
    raci_counts.index = raci_counts.index.map(config.RACI_LABELS)
    fig1 = px.bar(
        raci_counts, text_auto=True, color=raci_counts.index,
        color_discrete_sequence=PALETTE,
    )
    fig1.update_traces(marker_line_width=0, textfont_size=13, textposition="outside")
    fig1.update_layout(
        showlegend=False, xaxis_title="", yaxis_title="Jumlah baris",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=13), margin=dict(t=10, b=10),
    )
    st.plotly_chart(fig1, use_container_width=True)

# --- Top roles by RACI type ---
with tab2:
    st.markdown('<p class="section-title">Role Paling Banyak Terlibat</p>', unsafe_allow_html=True)
    c1, c2 = st.columns([1, 3])
    with c1:
        raci_pick = st.selectbox("Tipe RACI", ["R", "A", "C", "I"], format_func=lambda x: config.RACI_LABELS[x])
        top_n = st.slider("Jumlah role ditampilkan", 5, 30, 15)
    subset = structured_df[structured_df["raci_type"] == raci_pick]
    top_roles = subset["role_normalized"].value_counts().head(top_n).sort_values()
    fig2 = px.bar(
        top_roles, orientation="h", text_auto=True,
        color_discrete_sequence=["#0F4C81"],
    )
    fig2.update_traces(marker_line_width=0, textfont_size=12)
    fig2.update_layout(
        showlegend=False, yaxis_title="", xaxis_title="Jumlah proses",
        height=max(400, top_n * 28),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=10, b=10),
    )
    with c2:
        st.plotly_chart(fig2, use_container_width=True)

# --- Process count by Level 1 category ---
with tab3:
    st.markdown('<p class="section-title">Jumlah Proses per Kategori (Level 1)</p>', unsafe_allow_html=True)
    tmp = structured_df.drop_duplicates(subset=["process_id"]).copy()
    tmp["root"] = tmp["hierarchy_id"].astype(str).str.split(".").str[0]
    cat_counts = tmp.groupby("root").size()
    cat_counts.index = cat_counts.index.map(lambda r: level1_lookup.get(r, f"Kategori {r}"))
    cat_counts = cat_counts.sort_values(ascending=False)
    fig3 = px.bar(
        cat_counts, text_auto=True,
        color_discrete_sequence=["#2E9CCA"],
    )
    fig3.update_traces(marker_line_width=0)
    fig3.update_layout(
        showlegend=False, xaxis_title="", yaxis_title="Jumlah proses", xaxis_tickangle=-40,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=10, b=10),
    )
    st.plotly_chart(fig3, use_container_width=True)

# --- Tupoksi coverage gap ---
with tab4:
    st.markdown('<p class="section-title">Cakupan Tupoksi per Role RACI</p>', unsafe_allow_html=True)
    st.caption(
        "Role yang muncul di RACI matrix tapi TIDAK memiliki data Tugas Pokok yang cocok "
        "(misalnya level Direktur/GM, yang memang belum tersedia di katalog Tupoksi ini)."
    )

    @st.cache_data(show_spinner="Menghitung cakupan tupoksi...")
    def compute_tupoksi_coverage(known_roles_tuple, known_jabatan_tuple, threshold):
        rows = []
        for role in known_roles_tuple:
            match = rf_process.extractOne(role, known_jabatan_tuple, scorer=rf_fuzz.token_sort_ratio)
            found = bool(match and match[1] >= threshold)
            rows.append({
                "role": role,
                "tupoksi_ditemukan": found,
                "jabatan_cocok": match[0] if found else None,
                "skor_kecocokan": round(match[1], 1) if match else 0.0,
            })
        return pd.DataFrame(rows)

    coverage_df = compute_tupoksi_coverage(
        tuple(known_roles), tuple(known_jabatan_tupoksi), config.JABATAN_MATCH_THRESHOLD
    )
    gap_df = coverage_df[~coverage_df["tupoksi_ditemukan"]].sort_values("role")

    covered_count = int(coverage_df["tupoksi_ditemukan"].sum())
    gap_count = int((~coverage_df["tupoksi_ditemukan"]).sum())

    gc1, gc2, gc3 = st.columns([1, 1, 1.4])
    gc1.metric("✅ Role dengan data Tupoksi", covered_count)
    gc2.metric("⚠️ Role tanpa data Tupoksi", gap_count)

    with gc3:
        fig4 = px.pie(
            values=[covered_count, gap_count],
            names=["Ada Tupoksi", "Tanpa Tupoksi"],
            hole=0.6,
            color=["Ada Tupoksi", "Tanpa Tupoksi"],
            color_discrete_map={"Ada Tupoksi": "#2E9CCA", "Tanpa Tupoksi": "#E76F51"},
        )
        fig4.update_traces(textinfo="percent", textfont_size=13)
        fig4.update_layout(
            showlegend=True, legend=dict(orientation="h", y=-0.15),
            margin=dict(t=0, b=0, l=0, r=0), height=220,
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig4, use_container_width=True)

    with st.expander(f"Lihat {len(gap_df)} role tanpa data Tupoksi"):
        st.dataframe(gap_df[["role"]].reset_index(drop=True), use_container_width=True)
        st.download_button(
            "⬇️ Unduh daftar gap (CSV)",
            gap_df.to_csv(index=False).encode("utf-8"),
            file_name="tupoksi_coverage_gap.csv",
            mime="text/csv",
        )

st.divider()

# --- On-demand alignment checker ---
st.markdown(
    '<p class="section-title">🔎 Cek Kesesuaian Tugas Pokok (per Proses)</p>', unsafe_allow_html=True
)
st.caption("Memanggil API embedding hanya saat tombol ditekan — tidak dihitung otomatis untuk seluruh data.")

process_pick = st.selectbox(
    "Pilih proses bisnis", options=known_processes, index=None, placeholder="Ketik untuk mencari proses..."
)

if process_pick and st.button("🚀 Cek kesesuaian tugas pokok", type="primary"):
    with st.spinner("Menghitung kesesuaian..."):
        embed_client = get_embed_client()
        row = query_by_process(df, process_pick).iloc[0]
        context, role_candidates, proc_vec = build_process_context(row, embeddings, embedded_ids)

    st.markdown(f"**Proses:** {row['PROSES BISNIS']} (ID: {row['HIERARCHY ID']})")
    if proc_vec is None:
        st.warning("Embedding untuk proses ini tidak ditemukan di cache — coba proses lain.")
    else:
        seen = set()
        for role_name, code in role_candidates:
            if role_name in seen:
                continue
            seen.add(role_name)
            with st.container(border=True):
                st.markdown(f"**[{config.RACI_LABELS[code]}] {role_name}**")
                block = format_alignment_block(role_name, proc_vec, embed_client, jabatan_tugas, known_jabatan_tupoksi)
                st.text(block)
