"""
Streamlit Frontend — Global M4.5+ Earthquake Spatio-Temporal Analysis
=====================================================================
Six tabs: Data Verification | Catalog Browser | Global Map |
          Window Comparison | Machine Learning | Export

Startup:
1. uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
2. streamlit run frontend/app.py --server.port 8501
"""
from datetime import date
from pathlib import Path

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

OUT_DIR = Path(__file__).resolve().parents[1] / "outputs"
for sub in ["csv", "png", "html"]:
    (OUT_DIR / sub).mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="Global M4.5+ Earthquake Analysis", layout="wide")

import os
API_BASE = os.getenv(
    "EARTHQUAKE_API_BASE",
    "http://127.0.0.1:8000/api/v1",
)
# Allow override via sidebar in dev mode
_dev_override = st.sidebar.text_input(
    "FastAPI Address (dev override)",
    value=API_BASE,
    disabled=(os.getenv("EARTHQUAKE_API_BASE", "") == ""),
)
if _dev_override != API_BASE:
    API_BASE = _dev_override


NO_PROXY = {"http": None, "https": None}  # bypass system proxy for localhost

def api_get(path, params=None):
    try:
        r = requests.get(f"{API_BASE}{path}", params=params, timeout=120, proxies=NO_PROXY)
    except requests.ConnectionError:
        raise RuntimeError(
            "Connection refused — is the FastAPI backend running? "
            "Check: uvicorn backend.app.main:app --host 127.0.0.1 --port 8000"
        )
    if not r.ok:
        try:
            detail = r.json().get("detail", r.text)
        except Exception:
            detail = r.text
        raise RuntimeError(detail)
    return r.json()


def api_post(path, json_data=None, files=None):
    try:
        r = requests.post(f"{API_BASE}{path}", json=json_data, files=files, timeout=120, proxies=NO_PROXY)
    except requests.ConnectionError:
        raise RuntimeError(
            "Connection refused — is the FastAPI backend running? "
            "Check: uvicorn backend.app.main:app --host 127.0.0.1 --port 8000"
        )
    if not r.ok:
        try:
            detail = r.json().get("detail", r.text)
        except Exception:
            detail = r.text
        raise RuntimeError(detail)
    return r.json()


st.title("Global M4.5+ Earthquake Spatio-Temporal Analysis")
st.markdown("USGS ANSS Comprehensive Earthquake Catalog - 2024-2025")

tabs = st.tabs([
    "1. Data Verification", "2. Catalog Browser", "3. Global Map",
    "4. Window Comparison", "5. Machine Learning", "6. Export",
])

# ================================================================
# Tab 1: Data Verification
# ================================================================
with tabs[0]:
    st.subheader("Upload and Verify Earthquake Catalog")
    up = st.file_uploader("Upload USGS ComCat CSV", type=["csv"])
    if up and st.button("Upload and Verify", type="primary"):
        try:
            r = api_post("/catalog/upload", files={"file": (up.name, up.getvalue(), "text/csv")})
            st.success(r["message"])
            v = r["verification"]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Records", v["record_count"])
            c2.metric("ID Duplicates", v["id_duplicate_count"])
            c3.metric("Lat Invalid", v["latitude_invalid_count"])
            c4.metric("Lon Invalid", v["longitude_invalid_count"])
            if v["verification_passed"]:
                st.success("Verification PASSED")
            else:
                st.error("Verification FAILED")
            st.json(v["missing_quality_fields"])
            st.json(v)
        except RuntimeError as e:
            st.error(str(e))

# ================================================================
# Tab 2: Catalog Browser
# ================================================================
with tabs[1]:
    st.subheader("Earthquake Catalog Browser")
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        sd = st.date_input("Start Date", value=date(2024, 1, 1), key="cat_sd")
        ed = st.date_input("End Date", value=date(2025, 12, 31), key="cat_ed")
    with fc2:
        min_m = st.number_input("Min Mag", 4.5, value=4.5, step=0.1, key="cat_minm")
        max_m = st.number_input("Max Mag", 4.5, value=9.5, step=0.1, key="cat_maxm")
    with fc3:
        min_d = st.number_input("Min Depth (km)", value=-10.0, key="cat_mind")
        max_d = st.number_input("Max Depth (km)", value=800.0, key="cat_maxd")
    kw = st.text_input("Place keyword (not for country stats)", key="cat_kw")
    ps = st.selectbox("Page Size", [50, 100, 200, 500], index=1, key="cat_ps")

    # Build filter params (shared between Execute Filter and page changes)
    filter_params = {
        "start_time": f"{sd.isoformat()}T00:00:00Z",
        "end_time": f"{ed.isoformat()}T23:59:59.999Z",
        "min_mag": min_m, "max_mag": max_m,
        "min_depth": min_d, "max_depth": max_d,
        "place_keyword": kw or None,
        "page_size": ps,
    }

    if st.button("Execute Filter", key="cat_exec"):
        st.session_state["cat_filter_params"] = filter_params
        st.session_state["cat_total"] = None
        # cat_page is a widget key (number_input below) — must be reset
        # before that widget is instantiated on the next run, not after.
        st.session_state["cat_page"] = 1
        st.rerun()

    # If we have stored filter params, use them for pagination
    active_params = st.session_state.get("cat_filter_params", filter_params)

    if st.session_state.get("cat_filter_params") is not None or st.session_state.get("cat_total") is not None:
        # Determine total on first load
        if st.session_state.get("cat_total") is None:
            try:
                r_first = api_get("/catalog/events", params={**active_params, "page": 1})
                st.session_state["cat_total"] = r_first["total"]
            except RuntimeError as e:
                st.error(str(e))
                st.session_state["cat_total"] = 0

    total = st.session_state.get("cat_total", 0)
    if total > 0:
        total_pages = max(1, (total + ps - 1) // ps)

        # Pagination controls.
        # cat_page must be seeded BEFORE st.number_input is instantiated
        # (with `key=` set, Streamlit reads the initial value from
        # session_state; setting the key afterwards raises
        # StreamlitAPIException). Prev/next therefore use on_click
        # callbacks, which run before the next script run and may modify
        # widget keys.
        if "cat_page" not in st.session_state:
            st.session_state["cat_page"] = 1
        # Clamp in case page size changed and shrank total_pages
        st.session_state["cat_page"] = min(st.session_state["cat_page"], total_pages)
        cur_page = st.session_state["cat_page"]

        def _prev_page():
            st.session_state["cat_page"] = max(1, st.session_state["cat_page"] - 1)

        def _next_page():
            st.session_state["cat_page"] = min(total_pages, st.session_state["cat_page"] + 1)

        pc1, pc2, pc3, pc4 = st.columns([1, 2, 1, 1])
        with pc1:
            st.number_input(
                "Page", min_value=1, max_value=total_pages,
                step=1, key="cat_page",
            )
        with pc2:
            st.caption(f"共 {total} 条记录，{total_pages} 页")
        with pc3:
            st.button("上一页", on_click=_prev_page, disabled=(cur_page <= 1), key="cat_prev")
        with pc4:
            st.button("下一页", on_click=_next_page, disabled=(cur_page >= total_pages), key="cat_next")

        try:
            r = api_get("/catalog/events", params={**active_params, "page": cur_page})
            df = pd.DataFrame(r["items"])
            if not df.empty:
                cols = [c for c in ["time", "latitude", "longitude", "depth", "mag", "magType", "place"] if c in df.columns]
                st.dataframe(df[cols], use_container_width=True, hide_index=True)
                st.download_button("Download CSV", df.to_csv(index=False, encoding="utf-8-sig"),
                                   "filtered_events.csv", "text/csv")
        except RuntimeError as e:
            st.error(str(e))

# ================================================================
# Tab 3: Global Map
# ================================================================
with tabs[2]:
    st.subheader("Global Earthquake Spatial Distribution")
    st.caption("14,953 points — loading may take 10-15 seconds.")
    map_sample = st.slider("Sample size (fewer points = faster)", 1000, 15000, 5000, 1000,
                           help="Reduce if the map is slow to load.")
    if st.button("Load Global Map"):
        try:
            with st.spinner("Fetching earthquake data from backend..."):
                r = api_get("/catalog/events", params={"page": 1, "page_size": 20000})
            mdf = pd.DataFrame(r["items"])
            if mdf.empty:
                st.warning("No data.")
            else:
                mdf["mag"] = pd.to_numeric(mdf["mag"], errors="coerce")
                mdf["depth"] = pd.to_numeric(mdf["depth"], errors="coerce")
                if len(mdf) > map_sample:
                    mdf = mdf.sample(map_sample, random_state=42)
                with st.spinner(f"Rendering {len(mdf)} points on map..."):
                    fig = px.scatter_geo(
                        mdf, lat="latitude", lon="longitude", size="mag", color="depth",
                        hover_name="place", hover_data=["time", "mag", "magType", "depth", "id"],
                        projection="natural earth",
                        title=f"2024-2025 Global M4.5+ Earthquake Epicenters (n={len(mdf)})",
                        color_continuous_scale="Viridis_r",
                    )
                    fig.update_layout(height=650, margin=dict(l=0, r=0, t=50, b=0))
                st.plotly_chart(fig, use_container_width=True)
                fig.write_html(str(OUT_DIR / "html/global_map.html"), include_plotlyjs="cdn")
        except RuntimeError as e:
            st.error(str(e))

# ================================================================
# Tab 4: Window Comparison
# ================================================================
with tabs[3]:
    st.subheader("Spatial Grid & Coordinate Window Comparison")
    if st.button("Load 10x10 Grid"):
        try:
            grid = api_get("/statistics/grid", params={"grid_size": 10})
            gdf = pd.DataFrame(grid)
            st.metric("Non-empty Cells", f"{len(gdf)} / 648")
            st.dataframe(gdf.nlargest(20, "count"), use_container_width=True, hide_index=True)
            fig = px.density_heatmap(gdf, x="lon_min", y="lat_min", z="count",
                                     nbinsx=36, nbinsy=18, color_continuous_scale="YlOrRd",
                                     title="10x10 Spatial Grid Event Count")
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            fig.write_html(str(OUT_DIR / "html/grid_heatmap.html"), include_plotlyjs="cdn")
        except RuntimeError as e:
            st.error(str(e))

    st.subheader("Three Fixed 10x10 Regions")
    FW = [
        {"name": "Window A", "lat_min": 25, "lat_max": 50, "lon_min": 125, "lon_max": 150},
        {"name": "Window B", "lat_min": -60, "lat_max": 15, "lon_min": -85, "lon_max": -65},
        {"name": "Window C", "lat_min": -15, "lat_max": 15, "lon_min": 90, "lon_max": 145},
    ]
    if st.button("Compare Three Regions"):
        try:
            wdata = [api_post("/statistics/window", json_data=w) for w in FW]
            rows = [{
                "Region": w["name"], "Events": w["event_count"],
                "Mean Mag": round(w["mean_magnitude"], 2) if w["mean_magnitude"] else None,
                "Max Mag": w["max_magnitude"],
                "Mean Depth": round(w["mean_depth"], 1) if w["mean_depth"] else None,
                "Max Depth": w["max_depth"],
                "Shallow": w["shallow_count"], "Intermediate": w["intermediate_count"], "Deep": w["deep_count"],
            } for w in wdata]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            fig = px.bar(pd.DataFrame([{"Region": w["name"], "Count": w["event_count"]} for w in wdata]),
                         x="Region", y="Count", color="Region", title="Event Count Comparison")
            st.plotly_chart(fig, use_container_width=True)
        except RuntimeError as e:
            st.error(str(e))

# ================================================================
# Tab 5: Machine Learning
# ================================================================
with tabs[4]:
    st.subheader("K-Means Global Earthquake Spatial Clustering")
    st.markdown("**Method**: Spherical coordinate transform (lat/lon to 3D unit sphere) + K-Means.")

    c1, c2 = st.columns(2)
    with c1:
        inc_d = st.checkbox("Include depth", value=False)
    with c2:
        inc_m = st.checkbox("Include magnitude", value=False)
    st.caption("Default: spatial-only (sphere_x, sphere_y, sphere_z).")

    if st.button("Evaluate K=2 to K=10"):
        try:
            ev = api_get("/ml/kmeans/evaluate", params={
                "k_min": 2, "k_max": 10, "include_depth": inc_d, "include_magnitude": inc_m,
            })
            edf = pd.DataFrame(ev["results"])
            st.session_state["kmeans_eval"] = edf
            st.session_state["recommended_k"] = ev["recommended_k"]
            st.metric("Recommended K", ev["recommended_k"])
            st.metric("Best Silhouette", f"{ev['best_silhouette_score']:.4f}")
            fig1 = px.line(edf, x="k", y="silhouette_score", markers=True, title="Silhouette Score vs K")
            st.plotly_chart(fig1, use_container_width=True)
            fig2 = px.line(edf, x="k", y="inertia", markers=True, title="Inertia (Elbow) vs K")
            st.plotly_chart(fig2, use_container_width=True)
            st.dataframe(edf, use_container_width=True, hide_index=True)
            fig1.write_image(str(OUT_DIR / "png/silhouette_curve.png"))
            edf.to_csv(str(OUT_DIR / "csv/kmeans_evaluation.csv"), index=False, encoding="utf-8-sig")
        except RuntimeError as e:
            st.error(str(e))

    dk = st.session_state.get("recommended_k", 6)
    sk = st.number_input("Select K", 2, 20, int(dk), 1)
    if st.button("Train K-Means", type="primary"):
        try:
            r = api_post("/ml/kmeans/train", json_data={
                "n_clusters": sk, "include_depth": inc_d, "include_magnitude": inc_m, "random_state": 42,
            })
            st.success(f"Trained! Silhouette = {r['silhouette_score']:.4f}")
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Clusters", r["n_clusters"])
            mc2.metric("Training Records", r["training_record_count"])
            mc3.metric("Excluded", r["excluded_record_count"])
            sdf = pd.DataFrame(r["cluster_summary"])
            st.subheader("Cluster Summary")
            st.dataframe(sdf, use_container_width=True, hide_index=True)

            cl = api_get("/ml/kmeans/results")
            cdf = pd.DataFrame(cl["items"])
            cdf["cluster"] = cdf["cluster"].astype(str)
            fig = px.scatter_geo(
                cdf, lat="latitude", lon="longitude", color="cluster", size="mag",
                hover_name="place", hover_data=["time", "mag", "depth", "magType", "cluster"],
                projection="natural earth", title=f"K-Means Clustering (K={sk})",
            )
            fig.update_layout(height=650, margin=dict(l=0, r=0, t=50, b=0))
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("ML Results Analysis")
            st.markdown("#### 1. Cluster Size Distribution")
            st.plotly_chart(px.bar(sdf, x="cluster", y="event_count", color="cluster",
                                   title="Events per Cluster"), use_container_width=True)
            st.markdown("#### 2. Cluster Size Proportion")
            sdf["pct"] = (sdf["event_count"] / sdf["event_count"].sum() * 100).round(1)
            st.dataframe(sdf[["cluster", "event_count", "pct", "mean_magnitude", "max_magnitude", "mean_depth"]],
                         use_container_width=True, hide_index=True)
            st.markdown("#### 3. Temporal Stability")
            cdf["month"] = pd.to_datetime(cdf["time"], utc=True).dt.to_period("M").astype(str)
            cm = cdf.groupby(["month", "cluster"]).size().reset_index(name="count")
            st.plotly_chart(px.line(cm, x="month", y="count", color="cluster",
                                    title="Monthly Events per Cluster"), use_container_width=True)
            st.markdown("""
            #### 4. Limitations
            - Cluster labels have no hazard ranking.
            - K-Means prefers spherical clusters; real seismic belts are elongated.
            - Only catalog positions are used - no fault mechanisms or plate boundaries.
            - Results describe historical data only; **not** for prediction or risk assessment.
            """)
            cdf.to_csv(str(OUT_DIR / "csv/clustered_events.csv"), index=False, encoding="utf-8-sig")
            sdf.to_csv(str(OUT_DIR / "csv/cluster_summary.csv"), index=False, encoding="utf-8-sig")
            fig.write_html(str(OUT_DIR / "html/cluster_map.html"), include_plotlyjs="cdn")
            st.success("Outputs saved.")
        except RuntimeError as e:
            st.error(str(e))

# ================================================================
# Tab 6: Export
# ================================================================
with tabs[5]:
    st.subheader("Export Results")
    if st.button("Monthly Statistics CSV"):
        try:
            m = pd.DataFrame(api_get("/statistics/monthly"))
            st.dataframe(m, use_container_width=True, hide_index=True)
            m.to_csv(str(OUT_DIR / "csv/monthly_statistics.csv"), index=False, encoding="utf-8-sig")
            st.plotly_chart(px.bar(m, x="year_month", y="count", title="Monthly Count"), use_container_width=True)
            st.success("Saved.")
        except RuntimeError as e:
            st.error(str(e))
    if st.button("Top 10 Events"):
        try:
            t = pd.DataFrame(api_get("/statistics/top-events", params={"n": 10}))
            st.dataframe(t, use_container_width=True, hide_index=True)
            t.to_csv(str(OUT_DIR / "csv/top10_events.csv"), index=False, encoding="utf-8-sig")
            st.success("Saved.")
        except RuntimeError as e:
            st.error(str(e))
    if st.button("Monthly Energy Index"):
        try:
            en = pd.DataFrame(api_get("/statistics/monthly-energy"))
            st.dataframe(en, use_container_width=True, hide_index=True)
            en.to_csv(str(OUT_DIR / "csv/monthly_energy.csv"), index=False, encoding="utf-8-sig")
            st.plotly_chart(px.bar(en, x="year_month", y="energy_pct", title="Energy Index %"), use_container_width=True)
            st.success("Saved.")
        except RuntimeError as e:
            st.error(str(e))
    if st.button("Overview JSON"):
        try:
            st.json(api_get("/statistics/overview"))
        except RuntimeError as e:
            st.error(str(e))
