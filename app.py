import streamlit as st
import os
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
    "Baazar Kolkata": "#C62828",
    "CityKart":       "#2196F3",
    "Yousta":         "#FF9800",
    "StyleBaazar":    "#9C27B0",
    "V2 Retail":      "#00BCD4",
    "Zudio":          "#4CAF50",
    "mBaazar":        "#FF6F00",
    "Vmart":          "#607D8B",
}

# ── States BK operates in or is targeting ─────────────────────────────────────
FOCUS_STATES = sorted([
    # BK current states
    "West Bengal", "Bihar", "Odisha", "Tripura", "Sikkim",
    # Adjacent & target states
    "Jharkhand", "Uttar Pradesh", "Chhattisgarh", "Assam",
    # North East 7 sisters
    "Meghalaya", "Manipur", "Mizoram", "Nagaland",
    "Arunachal Pradesh",
])

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/fluency/96/shop.png", width=60)
st.sidebar.title("Retail Expansion Intel")
st.sidebar.caption("Baazar Kolkata Competitive Intelligence")

page = st.sidebar.radio(
    "Navigate",
    ["🗺️ Store Map", "📊 Stats by Company", "💡 Expansion Insights", "📋 Master Data"],
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

# ── Haversine distance ────────────────────────────────────────────────────────
import numpy as np

def haversine_km(lat1, lng1, lat2, lng2):
    """Vectorised Haversine — lat2/lng2 can be arrays."""
    R = 6371
    lat1, lng1, lat2, lng2 = map(np.radians, [lat1, lng1, lat2, lng2])
    dlat = lat2 - lat1
    dlng = lng2 - lng1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlng/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))

# ── Shared opportunity score helper ───────────────────────────────────────────
def compute_scores(df, group_cols, radius_km=200, adj_cap=10):
    """Compute opportunity scores grouped by group_cols.
    Adjacency bonus = BK stores within radius_km but outside the target area (capped at 10).
    """
    bk_df   = df[df["company"] == "Baazar Kolkata"]
    comp_df = df[df["company"] != "Baazar Kolkata"]

    comp_by = comp_df.groupby(group_cols).size().reset_index(name="competitor_stores")
    bk_by   = bk_df.groupby(group_cols).size().reset_index(name="bk_stores")

    scores = comp_by.merge(bk_by, on=group_cols, how="left")
    scores["bk_stores"] = scores["bk_stores"].fillna(0).astype(int)

    # Centroid per area
    area_centers = df.groupby(group_cols)[["lat","lng"]].mean().reset_index()
    scores = scores.merge(area_centers, on=group_cols, how="left")

    # BK store locations
    bk_lats = bk_df["lat"].values
    bk_lngs = bk_df["lng"].values
    bk_areas = bk_df[group_cols[0]].values  # area name for exclusion

    def adjacency_bonus(row):
        if pd.isna(row["lat"]) or pd.isna(row["lng"]):
            return 0
        dists = haversine_km(row["lat"], row["lng"], bk_lats, bk_lngs)
        outside = bk_areas != row[group_cols[0]]
        nearby = (dists <= radius_km) & outside
        return min(int(nearby.sum()), adj_cap)

    scores["adjacency_bonus"] = scores.apply(adjacency_bonus, axis=1)
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
    return scores, bk_df, comp_df


def render_city_ui(scores, bk_df, df, key_prefix, default_min=2):
    """City-specific opportunity UI with PS ratio metrics."""
    bk_core_states = sorted(bk_df["state"].unique().tolist())
    all_states     = [s for s in FOCUS_STATES if s in scores["state"].values]

    col1, col2, col3 = st.columns(3)
    with col1:
        filter_states = st.multiselect("Filter by State", options=all_states, default=bk_core_states, key=f"{key_prefix}_state")
    with col2:
        filter_bk = st.selectbox("BK Presence", ["All", "No BK stores (pure gaps)", "Has some BK stores"], key=f"{key_prefix}_bk")
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

    # ── Bar chart — top 10 by new stores needed (or opp score if no PS data) ──
    top10 = idf.head(10).copy()
    chart_col = "new_stores_needed" if top10["new_stores_needed"].notna().any() else "opportunity_score"
    chart_label = "New BK Stores Needed" if chart_col == "new_stores_needed" else "Opportunity Score"
    fig_op = px.bar(
        top10[::-1], x=chart_col, y="city", orientation="h", color="state",
        title=f"Top 10 Cities — {chart_label}",
        labels={chart_col: chart_label, "city": "City"},
    )
    fig_op.update_layout(height=380, margin=dict(l=0, r=20, t=40, b=20))
    st.plotly_chart(fig_op, use_container_width=True)

    # ── Table ─────────────────────────────────────────────────────────────────
    st.subheader(f"📋 City Opportunities ({len(idf)} shown)")
    display_cols = {
        "city":               "City",
        "state":              "State",
        "competitor_stores":  "Competitor Stores",
        "bk_stores":          "BK Stores (current)",
        "new_stores_needed":  "New BK Stores Needed",
        "recommended_bk":     "Recommended BK Total",
        "pop_2026":           "Pop 2026 (est.)",
        "bk_ps_ratio":        "BK PS Ratio",
        "adjacency_bonus":    "Adj. Bonus",
        "ps_score":           "PS Score",
        "opportunity_score":  "Opportunity Score",
        "competitors_present":"Competitors Present",
        "top_pincodes":       "Top PIN Codes",
    }
    # Sort by new_stores_needed numerically before string formatting
    idf = idf.sort_values("new_stores_needed", ascending=False, na_position="last").reset_index(drop=True)
    show_df = idf[[c for c in display_cols if c in idf.columns]].rename(columns=display_cols)

    # Keep New BK Stores Needed as numeric for proper sorting
    # Format Pop and PS Ratio as strings (display only), but NOT new_stores_needed
    if "Pop 2026 (est.)" in show_df.columns:
        show_df["Pop 2026 (est.)"] = show_df["Pop 2026 (est.)"].apply(
            lambda x: int(x) if pd.notna(x) else None
        )
    if "BK PS Ratio" in show_df.columns:
        show_df["BK PS Ratio"] = show_df["BK PS Ratio"].apply(
            lambda x: int(x) if pd.notna(x) else None
        )
    if "New BK Stores Needed" in show_df.columns:
        show_df["New BK Stores Needed"] = show_df["New BK Stores Needed"].apply(
            lambda x: int(x) if pd.notna(x) else None
        )
    if "Recommended BK Total" in show_df.columns:
        show_df["Recommended BK Total"] = show_df["Recommended BK Total"].apply(
            lambda x: int(x) if pd.notna(x) else None
        )

    def highlight_new_stores(val):
        if val is None or pd.isna(val): return ""
        n = int(val)
        if n >= 5:   return "background-color: #ff9999; color: #000; font-weight: bold"
        elif n >= 2: return "background-color: #ffe066; color: #000; font-weight: bold"
        elif n == 1: return "background-color: #d4f0a0; color: #000"
        return ""

    try:
        styled = show_df.style.map(highlight_new_stores, subset=["New BK Stores Needed"])
    except Exception:
        try:
            styled = show_df.style.applymap(highlight_new_stores, subset=["New BK Stores Needed"])
        except Exception:
            styled = show_df

    st.dataframe(
        styled,
        use_container_width=True,
        height=520,
        column_config={
            "Pop 2026 (est.)":      st.column_config.NumberColumn(format="%d"),
            "BK PS Ratio":          st.column_config.NumberColumn(format="%d"),
            "New BK Stores Needed": st.column_config.NumberColumn(format="%d"),
            "Recommended BK Total": st.column_config.NumberColumn(format="%d"),
        }
    )


    # ── Map ───────────────────────────────────────────────────────────────────
    st.subheader("🗺️ Opportunity Map")
    st.caption("Bubble size = New BK Stores Needed (or Opportunity Score if no PS data). Click for details.")

    centers = df.groupby(["city", "state"])[["lat", "lng"]].mean().reset_index()
    idf_map = idf.drop(columns=["lat","lng"], errors="ignore")
    map_data = idf_map.merge(centers, on=["city", "state"], how="left").dropna(subset=["lat", "lng"])

    if not map_data.empty:
        m2 = folium.Map(location=[map_data["lat"].mean(), map_data["lng"].mean()], zoom_start=5, tiles="CartoDB positron")

        size_col = "new_stores_needed" if map_data["new_stores_needed"].notna().any() else "opportunity_score"
        max_val  = map_data[size_col].max() or 1

        for _, row in map_data.iterrows():
            val     = row[size_col] if pd.notna(row.get(size_col)) else 0
            r_size  = max(5, int(val / max_val * 30))
            opacity = 0.4 + 0.5 * (val / max_val)

            bk_note   = f"BK stores: {int(row['bk_stores'])}" if row["bk_stores"] > 0 else "⚠️ No BK presence"
            pop_str   = f"{int(row['pop_2026']):,}" if pd.notna(row.get("pop_2026")) else "N/A"
            ps_str    = f"{int(row['bk_ps_ratio']):,}" if pd.notna(row.get("bk_ps_ratio")) else "N/A"
            new_str   = f"{int(row['new_stores_needed'])}" if pd.notna(row.get("new_stores_needed")) else "N/A"
            rec_str   = f"{int(row['recommended_bk'])}" if pd.notna(row.get("recommended_bk")) else "N/A"
            breakdown = row.get("comp_breakdown", "")

            popup_html = f"""
            <div style='font-family:sans-serif;min-width:240px'>
              <b>🏙️ {row["city"]}, {row["state"]}</b><br>
              <b style='color:#E63946'>Opp Score: {int(row["opportunity_score"])}</b><br>
              <hr style='margin:4px 0'>
              🆕 <b>New BK stores needed: {new_str}</b><br>
              📊 Recommended BK total: {rec_str}<br>
              {bk_note}<br>
              👥 Pop 2026 est.: {pop_str}<br>
              📐 BK PS ratio: {ps_str}<br>
              <hr style='margin:4px 0'>
              🏪 Competitor stores: {int(row["competitor_stores"])}<br>
              {breakdown}<br>
              📌 Top PINs: {row.get("top_pincodes", "N/A")}
            </div>"""
            folium.CircleMarker(
                location=[row["lat"], row["lng"]],
                radius=r_size, color="#E63946", fill=True,
                fill_color="#E63946", fill_opacity=opacity,
                popup=folium.Popup(popup_html, max_width=260),
                tooltip=f"{row['city']}: {new_str} new stores needed",
            ).add_to(m2)

        # BK existing stores
        bk_cluster = MarkerCluster(name="BK Existing Stores").add_to(m2)
        for _, row in bk_df.dropna(subset=["lat", "lng"]).iterrows():
            bk_popup = f"""
            <div style='font-family:sans-serif;min-width:180px'>
              <b style='color:#2E7D32'>{row["store_name"]}</b><br>
              <small>{row["city"]}, {row["district"]}</small><br>
              <small>{row["state"]} – {row["pincode"]}</small>
            </div>"""
            folium.CircleMarker(
                location=[row["lat"], row["lng"]],
                radius=6, color="#2E7D32", fill=True,
                fill_color="#2E7D32", fill_opacity=1.0,
                tooltip=f"BK: {row['store_name']}",
                popup=folium.Popup(bk_popup, max_width=220),
            ).add_to(bk_cluster)

        # Warehouse
        WH_LAT, WH_LNG = 22.5958, 88.2676
        folium.Marker(
            location=[WH_LAT, WH_LNG],
            popup=folium.Popup("<b>🏭 BK Central Warehouse</b><br>493 B, GT Road, Shibpur, Howrah", max_width=220),
            tooltip="BK Central Warehouse (Howrah)",
            icon=folium.Icon(color="darkblue", icon="home", prefix="fa"),
        ).add_to(m2)

        folium.LayerControl().add_to(m2)
        st_folium(m2, width="100%", height=580, returned_objects=[])
    else:
        st.info("No cities match current filters.")


def render_opportunity_ui(scores, bk_df, df, area_col, key_prefix, default_min=3):
    """Render filters, bar chart, table and map for opportunity scores."""
    bk_core_states = sorted(bk_df["state"].unique().tolist())
    all_states     = [s for s in FOCUS_STATES if s in scores["state"].values]

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
    idf_map = idf.drop(columns=["lat","lng"], errors="ignore")
    map_data = idf_map.merge(centers, on=[area_col, "state"], how="left").dropna(subset=["lat", "lng"])

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
            bk_popup = f"""<div style='font-family:sans-serif;min-width:180px'>
              <b style='color:#2E7D32'>{row['store_name']}</b><br>
              <small>{row['city']}, {row['district']}</small><br>
              <small>{row['state']} – {row['pincode']}</small>
            </div>"""
            folium.CircleMarker(
                location=[row["lat"], row["lng"]],
                radius=6, color="#2E7D32", fill=True,
                fill_color="#2E7D32", fill_opacity=1.0,
                tooltip=f"BK: {row['store_name']}",
                popup=folium.Popup(bk_popup, max_width=220),
            ).add_to(bk_cluster)

        # Warehouse marker
        WH_LAT, WH_LNG = 22.5958, 88.2676
        folium.Marker(
            location=[WH_LAT, WH_LNG],
            popup=folium.Popup("<b>🏭 BK Central Warehouse</b><br>493 B, GT Road, Shibpur, Howrah", max_width=220),
            tooltip="BK Central Warehouse (Howrah)",
            icon=folium.Icon(color="darkblue", icon="home", prefix="fa"),
        ).add_to(m2)

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
        sel_states = st.multiselect("State", FOCUS_STATES, default=[], key="map_state",
                                     placeholder="All focus states")
    with col2:
        fdf = df[df["state"].isin(sel_states)] if sel_states else df[df["state"].isin(FOCUS_STATES)]
        districts = ["All Districts"] + sorted(fdf["district"].dropna().unique().tolist())
        sel_district = st.selectbox("District", districts)
    with col3:
        fdf2 = fdf if sel_district == "All Districts" else fdf[fdf["district"] == sel_district]
        cities = ["All Cities"] + sorted(fdf2["city"].dropna().unique().tolist())
        sel_city = st.selectbox("City", cities)
    with col4:
        companies = sorted(df["company"].unique().tolist())
        sel_companies = st.multiselect("Companies", companies, default=companies)

    mdf = df[df["state"].isin(sel_states)] if sel_states else df[df["state"].isin(FOCUS_STATES)]
    if sel_district != "All Districts": mdf = mdf[mdf["district"] == sel_district]
    if sel_city     != "All Cities":    mdf = mdf[mdf["city"]     == sel_city]
    mdf = mdf[mdf["company"].isin(sel_companies)]

    # Zoom logic
    has_state_filter = bool(sel_states)
    has_district_filter = sel_district != "All Districts"

    st.caption(f"Showing **{len(mdf):,}** stores")

    if len(mdf):
        clat, clng = mdf["lat"].mean(), mdf["lng"].mean()
        zoom = 5 if not has_state_filter else (8 if not has_district_filter else 11)
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
                tooltip=f"{row['store_name']}",
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
        sel_states2 = st.multiselect("State", FOCUS_STATES, default=[], key="s2_state",
                                      placeholder="All focus states")
    with col2:
        fdf_s = df[df["state"].isin(sel_states2)] if sel_states2 else df[df["state"].isin(FOCUS_STATES)]
        districts2 = ["All Districts"] + sorted(fdf_s["district"].dropna().unique().tolist())
        sel_district2 = st.selectbox("District", districts2, key="s2_dist")
    with col3:
        fdf_d = fdf_s if sel_district2 == "All Districts" else fdf_s[fdf_s["district"] == sel_district2]
        cities2 = ["All Cities"] + sorted(fdf_d["city"].dropna().unique().tolist())
        sel_city2 = st.selectbox("City", cities2, key="s2_city")
    with col4:
        companies2 = sorted(df["company"].unique().tolist())
        sel_cos2 = st.multiselect("Companies", companies2, default=companies2, key="s2_cos")

    sdf = df[df["state"].isin(sel_states2)] if sel_states2 else df[df["state"].isin(FOCUS_STATES)]
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
    ms_df = sdf.copy()
    if bk_states_only:
        ms_df = ms_df[ms_df["state"].isin(set(df[df["company"]=="Baazar Kolkata"]["state"].unique()))]

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

    available_cos = sorted(sdf["company"].unique().tolist())
    hth_col1, hth_col2, hth_col3 = st.columns([2, 3, 2])
    with hth_col1:
        default_primary = "Baazar Kolkata" if "Baazar Kolkata" in available_cos else available_cos[0]
        primary_co = st.selectbox("Primary company", available_cos,
                                   index=available_cos.index(default_primary), key="hth_primary")
    with hth_col2:
        compare_cos = st.multiselect("Compare against", 
                                      [c for c in available_cos if c != primary_co],
                                      default=[c for c in available_cos if c != primary_co],
                                      key="hth_compare")
    with hth_col3:
        top_n = st.slider("Show top N areas", 5, 50, 20)

    selected_cos = [primary_co] + compare_cos
    sdf_hth = sdf[sdf["company"].isin(selected_cos)]

    pivot = (
        sdf_hth.groupby([gb_col, "company"])
        .size()
        .reset_index(name="stores")
        .pivot(index=gb_col, columns="company", values="stores")
        .fillna(0).astype(int)
    )
    # Reorder columns: primary first, then rest
    col_order = [primary_co] + [c for c in compare_cos if c in pivot.columns]
    pivot = pivot[[c for c in col_order if c in pivot.columns]]
    pivot = pivot.sort_values(primary_co, ascending=False)
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
        title=f"Stores per {group_by} × Company (sorted by {primary_co})",
        height=max(350, top_n * 22),
        margin=dict(l=140, r=20, t=50, b=60),
        xaxis_tickangle=-30,
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    with st.expander("📋 Raw table"):
        st.dataframe(pivot_show, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 – EXPANSION INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💡 Expansion Insights":
    st.title("💡 Expansion Insights for Baazar Kolkata")

    tab_city, tab_state, tab_dist = st.tabs(["🏙️ City Opportunities", "🗺️ State Overview", "📍 District Rollup"])

    # ── Shared computation ─────────────────────────────────────────────────────
    focus_df = df[df["state"].isin(FOCUS_STATES)].copy()

    city_agg = focus_df.groupby(["city","state","district"]).agg(
        total_stores   = ("store_name","count"),
        bk_stores      = ("company", lambda x: (x=="Baazar Kolkata").sum()),
        pop_2026       = ("pop_2026","first"),
        lat            = ("lat","mean"),
        lng            = ("lng","mean"),
    ).reset_index()

    comp_stores = focus_df[focus_df["company"]!="Baazar Kolkata"].groupby(["city","state"]).size().reset_index(name="competitor_stores")
    comp_names  = focus_df[focus_df["company"]!="Baazar Kolkata"].groupby(["city","state"])["company"].apply(
        lambda x: ", ".join(sorted(x.unique()))
    ).reset_index(name="competitors_present")
    top_pins = focus_df.groupby(["city","state"])["pincode"].apply(
        lambda x: ", ".join(x.value_counts().head(3).index.tolist())
    ).reset_index(name="top_pincodes")

    city_agg = city_agg.merge(comp_stores, on=["city","state"], how="left")
    city_agg = city_agg.merge(comp_names,  on=["city","state"], how="left")
    city_agg = city_agg.merge(top_pins,    on=["city","state"], how="left")
    city_agg["competitor_stores"] = city_agg["competitor_stores"].fillna(0).astype(int)

    # Deduplicate - keep highest store count per city+state
    city_agg = city_agg.sort_values("total_stores", ascending=False)
    city_agg = city_agg.drop_duplicates(subset=["city","state"], keep="first").reset_index(drop=True)

    # ── State PS ratios ───────────────────────────────────────────────────────
    # State PS = total state urban pop (census) / total stores in that state
    towns_df = pd.read_csv("census_towns_2026.csv") if os.path.exists("census_towns_2026.csv") else pd.DataFrame()

    state_urban_pop = pd.DataFrame()
    if not towns_df.empty:
        state_urban_pop = towns_df[towns_df["state_std"].isin(FOCUS_STATES)].groupby("state_std").agg(
            state_urban_pop_2026=("pop_2026","sum")
        ).reset_index().rename(columns={"state_std":"state"})

    state_total_stores = focus_df.groupby("state").size().reset_index(name="state_total_stores")
    state_ps_df = state_urban_pop.merge(state_total_stores, on="state", how="left")
    state_ps_df["state_ps"] = state_ps_df["state_urban_pop_2026"] / state_ps_df["state_total_stores"]
    state_ps_map = dict(zip(state_ps_df["state"], state_ps_df["state_ps"]))

    # Fallback: focus-wide avg for any state not found
    has_pop = city_agg[city_agg["pop_2026"].notna()]
    focus_avg_ps = float(has_pop["pop_2026"].sum() / has_pop["total_stores"].sum()) if len(has_pop) > 0 else 100000.0

    city_agg["state_ps"]  = city_agg["state"].map(state_ps_map).fillna(focus_avg_ps)
    city_agg["city_ps"]   = (city_agg["pop_2026"] / city_agg["total_stores"]).round(0)

    city_agg["stores_needed"]     = (city_agg["pop_2026"] / city_agg["state_ps"]).round(0).fillna(0).clip(lower=1)
    city_agg["gap_stores"]        = (city_agg["stores_needed"] - city_agg["total_stores"]).clip(lower=0)
    city_agg["bk_stores_to_open"] = city_agg["gap_stores"].fillna(0).astype(int)
    city_agg["stores_needed"]     = city_agg["stores_needed"].astype(int)

    # Adjacency tier
    bk_locs  = df[df["company"]=="Baazar Kolkata"].dropna(subset=["lat","lng"])
    bk_lats  = bk_locs["lat"].values
    bk_lngs  = bk_locs["lng"].values
    bk_names = bk_locs["city"].values

    def get_tier_nearest(row):
        if row["bk_stores"] > 0: return "P0 (BK present)", row["city"], 0
        if pd.isna(row["lat"]) or pd.isna(row["lng"]): return "P3 (>200km)", None, None
        dists = haversine_km(row["lat"], row["lng"], bk_lats, bk_lngs)
        idx = dists.argmin(); d = round(float(dists[idx]))
        tier = "P1 (<100km)" if d < 100 else ("P2 (100-200km)" if d < 200 else "P3 (>200km)")
        return tier, bk_names[idx], d

    city_agg[["tier","nearest_bk","nearest_bk_km"]] = city_agg.apply(
        lambda r: pd.Series(get_tier_nearest(r)), axis=1
    )

    # ── Tab 1: City Opportunities ──────────────────────────────────────────────
    with tab_city:
        st.caption(f"Each city benchmarked against its own **state PS ratio** (state urban pop ÷ state total stores). Focus avg: **{int(focus_avg_ps):,}** people/store.")

        with st.expander("🧮 How it works", expanded=False):
            st.markdown(f"""
            **Focus States Avg PS Ratio** = Total urban pop (focus states) ÷ Total stores = **{int(focus_avg_ps):,} people/store**

            | Metric | Formula |
            |--------|---------|
            | **City PS ratio** | City pop ÷ all stores in city (BK + competitors) |
            | **Stores city should have** | `round(city pop ÷ state PS ratio)` |
            | **Gap stores** | `max(0, should have − existing)` → BK fills this gap |
            | **Tier** | P0 = BK present · P1 = <100km · P2 = 100-200km · P3 = >200km from nearest BK |

            Population: Census of India 2011, extrapolated to 2026 using state-level urban CAGRs.
            """)

        c1, c2, c3 = st.columns(3)
        with c1:
            filter_states = st.multiselect("State", FOCUS_STATES, default=[], key="ins_state",
                                            placeholder="All focus states")
        with c2:
            filter_tier = st.multiselect("Tier", ["P0 (BK present)","P1 (<100km)","P2 (100-200km)","P3 (>200km)"],
                                          default=["P1 (<100km)"], key="ins_tier")
        with c3:
            min_gap = st.slider("Min BK stores to open", 0, 10, 0, key="ins_mingap")

        idf = city_agg.copy()
        if filter_states: idf = idf[idf["state"].isin(filter_states)]
        if filter_tier:   idf = idf[idf["tier"].isin(filter_tier)]
        idf = idf[idf["bk_stores_to_open"] >= min_gap]
        idf = idf.sort_values("bk_stores_to_open", ascending=False).reset_index(drop=True)
        idf.index += 1

        # ── State PS reference cards ───────────────────────────────────────────
        active_states = filter_states if filter_states else FOCUS_STATES
        ps_cards = state_ps_df[state_ps_df["state"].isin(active_states)].sort_values("state_ps")
        if not ps_cards.empty:
            st.markdown("**State PS Ratios being used as benchmarks:**")
            cols = st.columns(min(len(ps_cards), 7))
            for i, (_, row) in enumerate(ps_cards.iterrows()):
                with cols[i % len(cols)]:
                    st.metric(
                        label=row["state"].replace(" Bengal","_B.").replace("Arunachal Pradesh","Arunachal"),
                        value=f"{int(row['state_ps']):,}",
                        help=f"State urban pop: {int(row['state_urban_pop_2026']):,} ÷ {int(row['state_total_stores'])} stores"
                    )

        # ── State summary cards ────────────────────────────────────────────
        active_states_filter = filter_states if filter_states else FOCUS_STATES
        state_summary_data = idf.groupby("state").agg(
            proposed_stores=("bk_stores_to_open","sum"),
            num_cities=("city","count"),
        ).reset_index()
        state_summary_data = state_summary_data.merge(
            state_ps_df[["state","state_ps"]], on="state", how="left"
        )
        if not state_summary_data.empty:
            cols_ss = st.columns(min(len(state_summary_data), 4))
            for i, (_, sr) in enumerate(state_summary_data.iterrows()):
                with cols_ss[i % len(cols_ss)]:
                    st.metric(
                        label=f"🏙️ {sr['state']}",
                        value=f"{int(sr['proposed_stores'])} new BK stores",
                        delta=f"{int(sr['num_cities'])} cities | PS {int(sr['state_ps']):,}",
                        delta_color="off",
                    )

        # Bar chart top 10

        top10 = idf.head(10)
        if not top10.empty:
            fig = px.bar(top10[::-1], x="bk_stores_to_open", y="city",
                         orientation="h", color="state",
                         title="Top 10 Cities — BK Stores to Open",
                         labels={"bk_stores_to_open":"BK Stores to Open","city":"City"})
            fig.update_layout(height=380, margin=dict(l=0,r=20,t=40,b=20))
            st.plotly_chart(fig, use_container_width=True)

        # Table
        st.subheader(f"📋 City Opportunities ({len(idf)} shown)")
        display_cols = {
            "city":"City", "state":"State", "district":"District", "tier":"Tier",
            "bk_stores_to_open":"BK Stores to Open", "bk_stores":"BK Stores (now)",
            "total_stores":"Total Stores", "stores_needed":"Stores Should Have",
            "pop_2026":"Pop 2026 (est.)", "city_ps":"City PS Ratio",
            "competitor_stores":"Competitor Stores", "competitors_present":"Competitors",
            "nearest_bk":"Nearest BK City", "nearest_bk_km":"Distance (km)",
            "top_pincodes":"Top PINs",
        }
        show = idf[[c for c in display_cols if c in idf.columns]].rename(columns=display_cols).copy()
        for col in ["Pop 2026 (est.)","City PS Ratio"]:
            if col in show.columns:
                show[col] = show[col].apply(lambda x: int(x) if pd.notna(x) else None)

        def hl_stores(val):
            if val is None: return ""
            try:
                n = int(val)
                if n >= 5:  return "background-color: #ff9999; color: #000; font-weight: bold"
                if n >= 2:  return "background-color: #ffe066; color: #000; font-weight: bold"
                if n == 1:  return "background-color: #d4f0a0; color: #000"
            except: pass
            return ""

        try:
            styled = show.style.map(hl_stores, subset=["BK Stores to Open"])
        except Exception:
            styled = show.style.applymap(hl_stores, subset=["BK Stores to Open"])

        st.dataframe(styled, use_container_width=True, height=520,
                     column_config={
                         "Pop 2026 (est.)":    st.column_config.NumberColumn(format="%d"),
                         "City PS Ratio":      st.column_config.NumberColumn(format="%d"),
                         "State PS Ratio":     st.column_config.NumberColumn(format="%d"),
                         "BK Stores to Open":  st.column_config.NumberColumn(format="%d"),
                         "Stores Should Have": st.column_config.NumberColumn(format="%d"),
                         "Distance (km)":      st.column_config.NumberColumn(format="%d"),
                     })

        # Map
        st.subheader("🗺️ Opportunity Map")
        st.caption("Bubble size = BK stores to open. Red=P1, Orange=P2, Grey=P3. Green = existing BK.")
        map_df = idf[idf["bk_stores_to_open"] > 0].dropna(subset=["lat","lng"]).copy()
        if not map_df.empty:
            m2 = folium.Map(location=[map_df["lat"].mean(), map_df["lng"].mean()],
                            zoom_start=5, tiles="CartoDB positron")
            max_val = max(map_df["bk_stores_to_open"].max(), 1)
            tier_color = {"P0 (BK present)":"#2E7D32","P1 (<100km)":"#E63946",
                          "P2 (100-200km)":"#FF9800","P3 (>200km)":"#9E9E9E"}

            for _, row in map_df.iterrows():
                val   = int(row["bk_stores_to_open"]) if pd.notna(row["bk_stores_to_open"]) else 0
                r_sz  = max(5, int(val / max_val * 30))
                color = tier_color.get(row["tier"], "#E63946")
                pop_s = f"{int(row['pop_2026']):,}" if pd.notna(row.get("pop_2026")) else "N/A"
                ps_s  = f"{int(row['city_ps']):,}" if pd.notna(row.get("city_ps")) else "N/A"
                popup_html = f"""
                <div style='font-family:sans-serif;min-width:220px'>
                  <b>🏙️ {row["city"]}, {row["state"]}</b><br>
                  <b style='color:{color}'>{row["tier"]}</b><br>
                  <hr style='margin:4px 0'>
                  🆕 <b>BK stores to open: {val}</b><br>
                  🏪 Existing: {int(row["total_stores"])} total (BK: {int(row["bk_stores"])})<br>
                  👥 Pop 2026: {pop_s}<br>
                  📐 City PS: {ps_s} | State PS: {int(row["state_ps"]):,}<br>
                  🏪 Competitors: {int(row.get("competitor_stores",0))}<br>
                  📌 Top PINs: {row.get("top_pincodes","N/A")}<br>
                  📍 Nearest BK: {row.get("nearest_bk","N/A")} ({row.get("nearest_bk_km","?")} km)
                </div>"""
                folium.CircleMarker(
                    location=[row["lat"], row["lng"]],
                    radius=r_sz, color=color, fill=True,
                    fill_color=color, fill_opacity=0.75,
                    popup=folium.Popup(popup_html, max_width=260),
                    tooltip=f"{row['city']}: {val} stores to open ({row['tier']})",
                ).add_to(m2)

            bk_cluster = MarkerCluster(name="BK Existing Stores").add_to(m2)
            for _, row in bk_locs.iterrows():
                folium.CircleMarker(
                    location=[row["lat"], row["lng"]],
                    radius=6, color="#2E7D32", fill=True, fill_color="#2E7D32", fill_opacity=1.0,
                    tooltip=f"BK: {row['store_name']}",
                    popup=folium.Popup(
                        f"<b style='color:#2E7D32'>{row['store_name']}</b><br>{row['city']}, {row['state']}",
                        max_width=200),
                ).add_to(bk_cluster)

            folium.Marker([22.5958, 88.2676],
                popup=folium.Popup("<b>🏭 BK Central Warehouse</b><br>493 B, GT Road, Howrah", max_width=220),
                tooltip="BK Central Warehouse",
                icon=folium.Icon(color="darkblue", icon="home", prefix="fa"),
            ).add_to(m2)
            folium.LayerControl().add_to(m2)
            st_folium(m2, width="100%", height=580, returned_objects=[])

    # ── Tab 2: State Overview ────────────────────────────────────────────────
    with tab_state:
        st.caption("State-level summary — where BK operates today vs where we're proposing to enter.")

        COMPETITORS = ["CityKart","Yousta","StyleBaazar","V2 Retail","Zudio","mBaazar","Vmart"]

        # BK existing stores per state
        bk_by_state  = focus_df[focus_df["company"]=="Baazar Kolkata"].groupby("state").size().reset_index(name="bk_stores_current")
        # Proposed new BK stores from city_agg
        prop_by_state = city_agg.groupby("state")["bk_stores_to_open"].sum().reset_index(name="bk_stores_proposed")
        # Competitor counts per company per state
        comp_counts = {}
        for comp in COMPETITORS:
            comp_counts[comp] = focus_df[focus_df["company"]==comp].groupby("state").size().reset_index(name=comp)

        state_tab_df = state_ps_df[["state","state_urban_pop_2026","state_ps"]].copy()
        state_tab_df = state_tab_df.merge(bk_by_state,  on="state", how="left")
        state_tab_df = state_tab_df.merge(prop_by_state, on="state", how="left")
        state_tab_df["bk_stores_current"]  = state_tab_df["bk_stores_current"].fillna(0).astype(int)
        state_tab_df["bk_stores_proposed"] = state_tab_df["bk_stores_proposed"].fillna(0).astype(int)
        state_tab_df["bk_total_proposed"]  = state_tab_df["bk_stores_current"] + state_tab_df["bk_stores_proposed"]
        state_tab_df["status"] = state_tab_df["bk_stores_current"].apply(
            lambda x: "🟢 Existing Market" if x > 0 else "🔵 New Market"
        )

        for comp in COMPETITORS:
            state_tab_df = state_tab_df.merge(comp_counts[comp], on="state", how="left")
            state_tab_df[comp] = state_tab_df[comp].fillna(0).astype(int)

        state_tab_df["total_competitor_stores"] = state_tab_df[COMPETITORS].sum(axis=1).astype(int)
        state_tab_df = state_tab_df.sort_values("bk_stores_proposed", ascending=False).reset_index(drop=True)
        state_tab_df.index += 1

        # KPI summary
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Focus States", len(state_tab_df))
        k2.metric("Existing BK Markets", int((state_tab_df["bk_stores_current"]>0).sum()))
        k3.metric("New Markets", int((state_tab_df["bk_stores_current"]==0).sum()))
        k4.metric("Total New BK Stores Proposed", int(state_tab_df["bk_stores_proposed"].sum()))

        st.markdown("---")

        # Bar chart
        fig_s = px.bar(
            state_tab_df.sort_values("bk_stores_proposed"),
            x="bk_stores_proposed", y="state", orientation="h",
            color="status",
            color_discrete_map={"🟢 Existing Market":"#2E7D32","🔵 New Market":"#1565C0"},
            title="Proposed New BK Stores by State",
            labels={"bk_stores_proposed":"New BK Stores","state":"State"}
        )
        fig_s.update_layout(height=420, margin=dict(l=0,r=20,t=40,b=20))
        st.plotly_chart(fig_s, use_container_width=True)

        # Table
        st.subheader("📋 State Overview")
        display_state = {
            "state":                  "State",
            "status":                 "Status",
            "bk_stores_current":      "BK Stores (now)",
            "bk_stores_proposed":     "New BK Stores Proposed",
            "bk_total_proposed":      "BK Total (after)",
            "state_urban_pop_2026":   "Urban Pop 2026 (est.)",
            "state_ps":               "State PS Ratio",
            "total_competitor_stores":"Total Competitor Stores",
        }
        for comp in COMPETITORS:
            display_state[comp] = comp

        show_s = state_tab_df[[c for c in display_state if c in state_tab_df.columns]].rename(columns=display_state)

        def hl_status(val):
            if "Existing" in str(val): return "background-color: #e8f5e9; color: #1B5E20"
            if "New" in str(val):      return "background-color: #e3f2fd; color: #0D47A1"
            return ""

        def hl_proposed(val):
            if val is None: return ""
            try:
                n = int(val)
                if n >= 20: return "background-color: #ff9999; color: #000; font-weight: bold"
                if n >= 5:  return "background-color: #ffe066; color: #000; font-weight: bold"
            except: pass
            return ""

        try:
            styled_s = show_s.style.map(hl_status, subset=["Status"]).map(hl_proposed, subset=["New BK Stores Proposed"])
        except Exception:
            styled_s = show_s.style.applymap(hl_status, subset=["Status"]).applymap(hl_proposed, subset=["New BK Stores Proposed"])

        st.dataframe(styled_s, use_container_width=True, height=560,
                     column_config={
                         "Urban Pop 2026 (est.)":    st.column_config.NumberColumn(format="%d"),
                         "State PS Ratio":           st.column_config.NumberColumn(format="%d"),
                         "BK Stores (now)":          st.column_config.NumberColumn(format="%d"),
                         "New BK Stores Proposed":   st.column_config.NumberColumn(format="%d"),
                         "BK Total (after)":         st.column_config.NumberColumn(format="%d"),
                         "Total Competitor Stores":  st.column_config.NumberColumn(format="%d"),
                         **{comp: st.column_config.NumberColumn(format="%d") for comp in COMPETITORS},
                     })

    # ── Tab 3: District Rollup ─────────────────────────────────────────────────
    with tab_dist:
        st.caption("Which districts should BK prioritise? Ranked by P1 cities (within 100km of existing BK).")

        with st.expander("🧮 How it works", expanded=False):
            st.markdown("""
            Districts ranked by number of P1 cities — cities already within reach of BK's supply chain.

            | Column | Meaning |
            |--------|---------|
            | **P1 Cities** | Cities <100km from nearest BK — most actionable |
            | **P1 BK to Open** | Total new BK stores across P1 cities in this district |
            | **Total BK to Open** | Across all city tiers in the district |
            """)

        dist_state_filter = st.multiselect("Filter States", FOCUS_STATES, default=[], key="dist_state",
                                            placeholder="All focus states")

        dist_df = city_agg.copy()
        if dist_state_filter: dist_df = dist_df[dist_df["state"].isin(dist_state_filter)]

        dist_df["is_p1"] = dist_df["tier"] == "P1 (<100km)"
        dist_df["is_p2"] = dist_df["tier"] == "P2 (100-200km)"
        dist_df["is_p3"] = dist_df["tier"] == "P3 (>200km)"

        dist_sum = dist_df.groupby(["district","state"]).agg(
            p1_cities        = ("is_p1","sum"),
            p2_cities        = ("is_p2","sum"),
            p3_cities        = ("is_p3","sum"),
            total_bk_to_open = ("bk_stores_to_open","sum"),
            p1_bk_to_open    = ("bk_stores_to_open", lambda x: int(x[dist_df.loc[x.index,"is_p1"]].sum())),
            total_stores     = ("total_stores","sum"),
            bk_stores        = ("bk_stores","sum"),
            num_cities       = ("city","count"),
        ).reset_index()
        dist_sum = dist_sum.sort_values(["p1_bk_to_open","p1_cities"], ascending=False).reset_index(drop=True)
        dist_sum.index += 1

        # Bar chart
        top_dist = dist_sum.head(15)
        fig_d = px.bar(top_dist[::-1], x="p1_cities", y="district", orientation="h",
                       color="state", title="Top Districts by P1 Cities",
                       labels={"p1_cities":"P1 Cities","district":"District"})
        fig_d.update_layout(height=420, margin=dict(l=0,r=20,t=40,b=20))
        st.plotly_chart(fig_d, use_container_width=True)

        st.subheader(f"📋 District Rollup ({len(dist_sum)} shown)")
        st.dataframe(
            dist_sum.rename(columns={
                "district":"District","state":"State",
                "p1_cities":"P1 Cities","p2_cities":"P2 Cities","p3_cities":"P3 Cities",
                "total_bk_to_open":"Total BK to Open","p1_bk_to_open":"P1 BK to Open",
                "total_stores":"Total Stores","bk_stores":"BK Stores","num_cities":"Cities"
            }),
            use_container_width=True, height=520,
            column_config={
                "P1 Cities":      st.column_config.NumberColumn(format="%d"),
                "P2 Cities":      st.column_config.NumberColumn(format="%d"),
                "P3 Cities":      st.column_config.NumberColumn(format="%d"),
                "Total BK to Open": st.column_config.NumberColumn(format="%d"),
                "P1 BK to Open":  st.column_config.NumberColumn(format="%d"),
            }
        )

# ══════════════════════════════════════════════════════════════════════════════
elif page == "📋 Master Data":
    st.title("📋 Master Data")
    st.caption(f"Full store dataset — {len(df):,} rows. Filter and inspect.")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        md_companies = st.multiselect("Company", sorted(df["company"].unique()), default=[], key="md_co",
                                       placeholder="All companies")
    with col2:
        md_states = st.multiselect("State", FOCUS_STATES, default=[], key="md_state",
                                    placeholder="All focus states")
    with col3:
        fdf_md = df.copy()
        if md_states: fdf_md = fdf_md[fdf_md["state"].isin(md_states)]
        md_districts = st.multiselect("District", sorted(fdf_md["district"].dropna().unique()), default=[], key="md_dist",
                                       placeholder="All districts")
    with col4:
        search = st.text_input("Search store name", placeholder="e.g. Patna, Zudio...", key="md_search")

    mdf = df.copy()
    if md_companies: mdf = mdf[mdf["company"].isin(md_companies)]
    if md_states:    mdf = mdf[mdf["state"].isin(md_states)]
    if md_districts: mdf = mdf[mdf["district"].isin(md_districts)]
    if search:       mdf = mdf[mdf["store_name"].str.contains(search, case=False, na=False)]

    st.caption(f"Showing **{len(mdf):,}** of {len(df):,} rows")

    display = mdf[["store_name","company","pincode","city","district","state","lat","lng"]].reset_index(drop=True)
    display.index += 1
    display.columns = ["Store Name","Company","Pincode","City","District","State","Lat","Lng"]

    st.dataframe(
        display,
        use_container_width=True,
        height=620,
        column_config={
            "Lat": st.column_config.NumberColumn(format="%.4f"),
            "Lng": st.column_config.NumberColumn(format="%.4f"),
            "Pincode": st.column_config.TextColumn(),
        }
    )

    # Download button
    csv = display.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download filtered data as CSV", csv, "filtered_stores.csv", "text/csv")
