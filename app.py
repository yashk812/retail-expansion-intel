import streamlit as st
import pandas as pd
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
import json
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Retail Expansion Intel", layout="wide", page_icon="🛍️")

@st.cache_data
def load_data():
    df = pd.read_csv("master_stores.csv")
    df["lat"]      = pd.to_numeric(df["lat"], errors="coerce")
    df["lng"]      = pd.to_numeric(df["lng"], errors="coerce")
    df = df.dropna(subset=["lat", "lng"])
    df["state"]    = df["state"].astype(str).str.strip()
    df["city"]     = df["city"].astype(str).str.strip()
    df["district"] = df["district"].astype(str).str.strip()
    df["pincode"]  = df["pincode"].astype(str).str.strip()
    return df

df = load_data()

COMPANY_COLORS = {
    "Baazar Kolkata": "#E63946",
    "CityKart":       "#2196F3",
    "Yousta":         "#FF9800",
    "StyleBaazar":    "#9C27B0",
    "V2 Retail":      "#00BCD4",
    "Zudio":          "#4CAF50",
    "mBaazar":        "#FF5722",
    "Vmart":          "#607D8B",
}

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/fluency/96/shop.png", width=60)
st.sidebar.title("Retail Expansion Intel")
st.sidebar.caption("Baazar Kolkata Competitive Intelligence")

page = st.sidebar.radio(
    "Navigate",
    ["🗺️ Store Map", "📊 Stats by Company", "💡 Expansion Insights"],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Total stores tracked:** {len(df):,}")
for co, cnt in df["company"].value_counts().items():
    color = COMPANY_COLORS.get(co, "#888")
    st.sidebar.markdown(
        f"<span style='color:{color}'>■</span> **{co}**: {cnt}",
        unsafe_allow_html=True,
    )

# ── Shared opportunity score helper ───────────────────────────────────────────
def compute_scores(df, group_cols):
    """Compute opportunity scores grouped by group_cols (e.g. ['district','state'] or ['city','state'])."""
    bk_df   = df[df["company"] == "Baazar Kolkata"]
    comp_df = df[df["company"] != "Baazar Kolkata"]
    bk_states = set(bk_df["state"].unique())

    comp_by = comp_df.groupby(group_cols).size().reset_index(name="competitor_stores")
    bk_by   = bk_df.groupby(group_cols).size().reset_index(name="bk_stores")

    scores = comp_by.merge(bk_by, on=group_cols, how="left")
    scores["bk_stores"]         = scores["bk_stores"].fillna(0).astype(int)
    scores["adjacency_bonus"]   = scores["state"].apply(lambda s: 3 if s in bk_states else 0)
    scores["opportunity_score"] = (
        scores["competitor_stores"] - scores["bk_stores"] * 3 + scores["adjacency_bonus"]
    )
    scores = scores.sort_values("opportunity_score", ascending=False).reset_index(drop=True)

    area_col = group_cols[0]
    comp_cos = comp_df.groupby(group_cols)["company"].apply(
        lambda x: ", ".join(sorted(x.unique()))
    ).reset_index(name="competitors_present")
    top_pins = comp_df.groupby(group_cols)["pincode"].apply(
        lambda x: ", ".join(x.value_counts().head(3).index.tolist())
    ).reset_index(name="top_pincodes")
    comp_breakdown = (
        comp_df.groupby(group_cols + ["company"]).size()
        .reset_index(name="n")
        .sort_values("n", ascending=False)
        .groupby(group_cols)
        .apply(lambda x: "<br>".join(f"&nbsp;&nbsp;{r.company}: {r.n}" for r in x.itertuples()))
        .reset_index(name="comp_breakdown")
    )

    scores = scores.merge(comp_cos, on=group_cols, how="left")
    scores = scores.merge(top_pins, on=group_cols, how="left")
    scores = scores.merge(comp_breakdown, on=group_cols, how="left")
    return scores, bk_df, comp_df, bk_states


def render_opportunity_ui(scores, bk_df, df, area_col, key_prefix, default_min=3):
    """Render filters, bar chart, table and map for opportunity scores."""
    bk_core_states = sorted(bk_df["state"].unique().tolist())
    all_states     = sorted(scores["state"].dropna().unique().tolist())

    col1, col2, col3 = st.columns(3)
    with col1:
        filter_states = st.multiselect(
            "Filter by State", options=all_states, default=bk_core_states,
            key=f"{key_prefix}_state",
        )
    with col2:
        filter_bk = st.selectbox(
            "BK Presence",
            ["All", "No BK stores (pure gaps)", "Has some BK stores"],
            key=f"{key_prefix}_bk",
        )
    with col3:
        min_comp = st.slider("Min competitor stores", 1, 50, default_min, key=f"{key_prefix}_min")

    idf = scores[scores["competitor_stores"] >= min_comp].copy()
    if filter_states:
        idf = idf[idf["state"].isin(filter_states)]
    if filter_bk == "No BK stores (pure gaps)":
        idf = idf[idf["bk_stores"] == 0]
    elif filter_bk == "Has some BK stores":
        idf = idf[idf["bk_stores"] > 0]

    idf = idf.head(50).reset_index(drop=True)
    idf.index += 1

    # Bar chart
    top10 = idf.head(10)
    label = area_col.capitalize()
    fig_op = px.bar(
        top10[::-1], x="opportunity_score", y=area_col, orientation="h",
        color="state",
        title=f"Top 10 Opportunity {label}s",
        labels={"opportunity_score": "Opportunity Score", area_col: label},
    )
    fig_op.update_layout(height=380, margin=dict(l=0, r=20, t=40, b=20))
    st.plotly_chart(fig_op, use_container_width=True)

    # Table
    st.subheader(f"📋 Top Opportunity {label}s ({len(idf)} shown)")
    display_cols = {
        area_col:              label,
        "state":               "State",
        "competitor_stores":   "Competitor Stores",
        "bk_stores":           "BK Stores",
        "adjacency_bonus":     "Adj. Bonus",
        "opportunity_score":   "Opportunity Score",
        "competitors_present": "Competitors Present",
        "top_pincodes":        "Top PIN Codes",
    }
    show_df = idf[[c for c in display_cols if c in idf.columns]].rename(columns=display_cols)

    def highlight_score(val):
        if isinstance(val, (int, float)):
            if val >= 20: return "background-color: #ffe0e0; color: #333"
            elif val >= 10: return "background-color: #fff8cc; color: #333"
        return ""

    try:
        styled = show_df.style.map(highlight_score, subset=["Opportunity Score"])
    except AttributeError:
        styled = show_df.style.applymap(highlight_score, subset=["Opportunity Score"])
    st.dataframe(styled, use_container_width=True, height=500)

    # Map
    st.subheader("🗺️ Opportunity Map")
    st.caption("Bubble size = Opportunity Score. Click for details.")

    centers = df.groupby([area_col, "state"])[["lat", "lng"]].mean().reset_index()
    map_data = idf.merge(centers, on=[area_col, "state"], how="left").dropna(subset=["lat", "lng"])

    if not map_data.empty:
        m2 = folium.Map(
            location=[map_data["lat"].mean(), map_data["lng"].mean()],
            zoom_start=5, tiles="CartoDB positron"
        )
        max_score = map_data["opportunity_score"].max()

        for _, row in map_data.iterrows():
            r_size  = max(6, int(row["opportunity_score"] / max_score * 30))
            opacity = 0.4 + 0.5 * (row["opportunity_score"] / max_score)
            bk_note = f"BK stores: {int(row['bk_stores'])}" if row["bk_stores"] > 0 else "⚠️ No BK presence"
            breakdown = row.get("comp_breakdown", "")
            popup_html = f"""
            <div style='font-family:sans-serif;min-width:220px'>
              <b>📍 {row[area_col]}, {row['state']}</b><br>
              <b style='color:#E63946'>Score: {int(row['opportunity_score'])}</b><br>
              <b>Competitor stores: {int(row['competitor_stores'])}</b><br>
              {breakdown}<br>
              {bk_note}<br>
              Top PINs: {row.get('top_pincodes', 'N/A')}
            </div>"""
            folium.CircleMarker(
                location=[row["lat"], row["lng"]],
                radius=r_size, color="#E63946", fill=True,
                fill_color="#E63946", fill_opacity=opacity,
                popup=folium.Popup(popup_html, max_width=250),
                tooltip=f"{row[area_col]} | Score: {int(row['opportunity_score'])}",
            ).add_to(m2)

        bk_cluster = MarkerCluster(name="BK Existing Stores").add_to(m2)
        for _, row in bk_df.dropna(subset=["lat", "lng"]).iterrows():
            folium.CircleMarker(
                location=[row["lat"], row["lng"]],
                radius=6, color="#2E7D32", fill=True,
                fill_color="#2E7D32", fill_opacity=1.0,
                tooltip=f"BK: {row['store_name']}",
            ).add_to(bk_cluster)

        folium.LayerControl().add_to(m2)
        st_folium(m2, width="100%", height=580, returned_objects=[])
    else:
        st.info("No map data for current filters.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 – STORE MAP
# ══════════════════════════════════════════════════════════════════════════════
if page == "🗺️ Store Map":
    st.title("🗺️ Store Map")
    st.caption("Visualise all stores with clustering. Zoom in to see hotbeds.")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        states = ["All States"] + sorted(df["state"].dropna().unique().tolist())
        sel_state = st.selectbox("State", states)
    with col2:
        fdf = df if sel_state == "All States" else df[df["state"] == sel_state]
        districts = ["All Districts"] + sorted(fdf["district"].dropna().unique().tolist())
        sel_district = st.selectbox("District", districts)
    with col3:
        fdf2 = fdf if sel_district == "All Districts" else fdf[fdf["district"] == sel_district]
        cities = ["All Cities"] + sorted(fdf2["city"].dropna().unique().tolist())
        sel_city = st.selectbox("City", cities)
    with col4:
        companies = sorted(df["company"].unique().tolist())
        sel_companies = st.multiselect("Companies", companies, default=companies)

    mdf = df.copy()
    if sel_state    != "All States":    mdf = mdf[mdf["state"]    == sel_state]
    if sel_district != "All Districts": mdf = mdf[mdf["district"] == sel_district]
    if sel_city     != "All Cities":    mdf = mdf[mdf["city"]     == sel_city]
    mdf = mdf[mdf["company"].isin(sel_companies)]

    st.caption(f"Showing **{len(mdf):,}** stores")

    if len(mdf):
        clat, clng = mdf["lat"].mean(), mdf["lng"].mean()
        zoom = 5 if sel_state == "All States" else (8 if sel_district == "All Districts" else 11)
    else:
        clat, clng, zoom = 22.5, 83.0, 5

    m = folium.Map(location=[clat, clng], zoom_start=zoom, tiles="CartoDB positron")

    for company in sel_companies:
        cdf = mdf[mdf["company"] == company]
        if cdf.empty:
            continue
        color   = COMPANY_COLORS.get(company, "#888")
        cluster = MarkerCluster(name=company, show=True).add_to(m)
        for _, row in cdf.iterrows():
            popup_html = f"""
            <div style='font-family:sans-serif;min-width:180px'>
              <b style='color:{color}'>{row['store_name']}</b><br>
              <small>{row['city']}, {row['district']}</small><br>
              <small>{row['state']} – {row['pincode']}</small>
            </div>"""
            folium.CircleMarker(
                location=[row["lat"], row["lng"]],
                radius=7, color=color, fill=True,
                fill_color=color, fill_opacity=0.85,
                popup=folium.Popup(popup_html, max_width=220),
                tooltip=f"{company} | {row['city']}",
            ).add_to(cluster)

    folium.LayerControl(collapsed=False).add_to(m)
    st_folium(m, width="100%", height=620, returned_objects=[])

    st.markdown("**Legend**")
    cols = st.columns(len(sel_companies))
    for i, co in enumerate(sel_companies):
        with cols[i]:
            st.markdown(
                f"<span style='background:{COMPANY_COLORS.get(co,'#888')};padding:2px 8px;"
                f"border-radius:4px;color:white;font-size:12px'>{co}</span>",
                unsafe_allow_html=True,
            )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 – STATS BY COMPANY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Stats by Company":
    st.title("📊 Stats by Company")
    st.caption("Compare store counts and market share across companies, states, districts and cities.")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        states2 = ["All States"] + sorted(df["state"].dropna().unique().tolist())
        sel_state2 = st.selectbox("State", states2, key="s2_state")
    with col2:
        fdf_s = df if sel_state2 == "All States" else df[df["state"] == sel_state2]
        districts2 = ["All Districts"] + sorted(fdf_s["district"].dropna().unique().tolist())
        sel_district2 = st.selectbox("District", districts2, key="s2_dist")
    with col3:
        fdf_d = fdf_s if sel_district2 == "All Districts" else fdf_s[fdf_s["district"] == sel_district2]
        cities2 = ["All Cities"] + sorted(fdf_d["city"].dropna().unique().tolist())
        sel_city2 = st.selectbox("City", cities2, key="s2_city")
    with col4:
        companies2 = sorted(df["company"].unique().tolist())
        sel_cos2 = st.multiselect("Companies", companies2, default=companies2, key="s2_cos")

    sdf = df.copy()
    if sel_state2    != "All States":    sdf = sdf[sdf["state"]    == sel_state2]
    if sel_district2 != "All Districts": sdf = sdf[sdf["district"] == sel_district2]
    if sel_city2     != "All Cities":    sdf = sdf[sdf["city"]     == sel_city2]
    sdf = sdf[sdf["company"].isin(sel_cos2)]

    bk   = sdf[sdf["company"] == "Baazar Kolkata"]
    comp = sdf[sdf["company"] != "Baazar Kolkata"]
    st.markdown("---")
    kc = st.columns(4)
    kc[0].metric("Total Stores (filtered)", len(sdf))
    kc[1].metric("Baazar Kolkata", len(bk))
    kc[2].metric("All Competitors", len(comp))
    kc[3].metric("States covered", sdf["state"].nunique())
    st.markdown("---")

    # ── Overall bar chart ────────────────────────────────────────────────────
    co_counts = sdf.groupby("company").size().reset_index(name="stores").sort_values("stores", ascending=True)
    fig_bar = px.bar(
        co_counts, x="stores", y="company", orientation="h",
        color="company", color_discrete_map=COMPANY_COLORS,
        title="Store Count by Company",
        labels={"stores": "Number of Stores", "company": ""},
    )
    fig_bar.update_layout(showlegend=False, height=320, margin=dict(l=0, r=20, t=40, b=20))
    st.plotly_chart(fig_bar, use_container_width=True)

    # ── State-level market share ─────────────────────────────────────────────
    st.subheader("🏪 Market Share by State")
    st.caption("Stacked bars show each company's share of total stores in each state.")

    # Only show states with BK presence by default, user can toggle
    bk_states_only = st.checkbox("Show only states where BK operates", value=True, key="ms_bk_only")
    bk_active_states = set(df[df["company"] == "Baazar Kolkata"]["state"].unique())

    ms_df = sdf.copy()
    if bk_states_only:
        ms_df = ms_df[ms_df["state"].isin(bk_active_states)]

    state_co = ms_df.groupby(["state", "company"]).size().reset_index(name="stores")
    # Sort states by total stores desc
    state_order = state_co.groupby("state")["stores"].sum().sort_values(ascending=True).index.tolist()

    fig_ms = px.bar(
        state_co,
        x="stores", y="state", color="company", orientation="h",
        color_discrete_map=COMPANY_COLORS,
        category_orders={"state": state_order},
        title="Store Count by State & Company",
        labels={"stores": "Stores", "state": "", "company": "Company"},
        barmode="stack",
    )
    fig_ms.update_layout(
        height=max(400, len(state_order) * 28),
        margin=dict(l=0, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_ms, use_container_width=True)

    # ── Head-to-head heatmap ─────────────────────────────────────────────────
    st.subheader("Head-to-Head: Area Breakdown")
    group_by = st.radio("Group by", ["State", "District", "City"], horizontal=True)
    gb_col   = group_by.lower()

    pivot = (
        sdf.groupby([gb_col, "company"])
        .size()
        .reset_index(name="stores")
        .pivot(index=gb_col, columns="company", values="stores")
        .fillna(0).astype(int)
    )
    if "Baazar Kolkata" in pivot.columns:
        pivot = pivot.sort_values("Baazar Kolkata", ascending=False)
    else:
        pivot["_t"] = pivot.sum(axis=1)
        pivot = pivot.sort_values("_t", ascending=False).drop(columns="_t")

    top_n = st.slider("Show top N areas", 5, 50, 20)
    pivot_show = pivot.head(top_n)

    fig_heat = go.Figure(data=go.Heatmap(
        z=pivot_show.values,
        x=pivot_show.columns.tolist(),
        y=pivot_show.index.tolist(),
        colorscale="YlOrRd",
        text=pivot_show.values,
        texttemplate="%{text}",
        showscale=True,
    ))
    fig_heat.update_layout(
        title=f"Stores per {group_by} × Company",
        height=max(350, top_n * 22),
        margin=dict(l=140, r=20, t=50, b=60),
        xaxis_tickangle=-30,
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    with st.expander("📋 Raw table"):
        st.dataframe(pivot_show, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 – EXPANSION INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💡 Expansion Insights":
    st.title("💡 Expansion Insights for Baazar Kolkata")

    with st.expander("🧮 How the Opportunity Score works", expanded=False):
        st.markdown("""
        For each area, we calculate:

        | Component | Formula |
        |-----------|---------|
        | **Competitor Presence** | Count of all non-BK stores in that area |
        | **BK Presence** | Count of BK stores in that area |
        | **Adjacency Bonus** | +3 if the area's state already has ≥1 BK store |
        | **Opportunity Score** | `Competitor Presence − (BK Presence × 3) + Adjacency Bonus` |

        High score = **lots of competitor activity, little/no BK footprint**.
        Adjacency bonus surfaces areas in states BK already operates — logical next steps.
        """)

    tab1, tab2 = st.tabs(["📍 District Opportunities", "🏙️ City Opportunities"])

    # ── Tab 1: District ───────────────────────────────────────────────────────
    with tab1:
        st.caption("Strategic view — which districts to prioritise at a macro level.")
        scores_d, bk_df, _, _ = compute_scores(df, ["district", "state"])
        render_opportunity_ui(scores_d, bk_df, df, "district", key_prefix="dist", default_min=3)

    # ── Tab 2: City ───────────────────────────────────────────────────────────
    with tab2:
        st.caption("Tactical view — which specific cities within target districts have competitor presence but no BK.")
        scores_c, bk_df, _, _ = compute_scores(df, ["city", "state"])
        render_opportunity_ui(scores_c, bk_df, df, "city", key_prefix="city", default_min=2)
