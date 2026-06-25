import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    KFold,
    RepeatedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

warnings.filterwarnings("ignore")

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Prediksi Kemacetan – Citimall Dumai",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Constants ───────────────────────────────────────────────────────────────
C = 2265.48  # Kapasitas jalan (SMP/jam)
VB = 37.8  # Kecepatan arus bebas (km/jam)
LOS_MAP = {
    "A": "≤0.20",
    "B": "0.20–0.44",
    "C": "0.44–0.64",
    "D": "0.64–0.80",
    "E": "0.80–1.00",
    "F": ">1.00",
}
LOS_COLOR = {
    "A": "#2ecc71",
    "B": "#27ae60",
    "C": "#f39c12",
    "D": "#e67e22",
    "E": "#e74c3c",
    "F": "#8e44ad",
}


# ─── Load Data ───────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df_kend = pd.read_csv("data_kendaraan.csv")
    df_kin = pd.read_csv("data_kinerja.csv")
    df_pddk = pd.read_csv("data_penduduk.csv")
    df_jam = pd.read_csv("data_perjam.csv")

    df_kend["Total_Kendaraan"] = (
        df_kend["Mobil_Penumpang"]
        + df_kend["Truk"]
        + df_kend["Bus"]
        + df_kend["Sepeda_Motor"]
    )

    Q_puncak_H1 = df_kin[df_kin["Hari_Ke"] == 1]["Q_smp_jam"].values[0]
    df_jam["Faktor_Waktu"] = df_jam["Q_total_H1"] / Q_puncak_H1

    Q_avg_2019 = df_kin[df_kin["Tahun"] == 2019]["Q_smp_jam"].mean()
    df_kin_2019 = df_kin[df_kin["Tahun"] == 2019].copy()
    df_kin_2019["Faktor_Hari"] = df_kin_2019["Q_smp_jam"] / Q_avg_2019

    df_full = df_kin.merge(
        df_kend[
            [
                "Tahun",
                "Total_Kendaraan",
                "Sepeda_Motor",
                "Mobil_Penumpang",
                "Truk",
                "Bus",
            ]
        ],
        on="Tahun",
        how="left",
    ).merge(df_pddk, on="Tahun", how="left")

    df_full["X_tahun"] = df_full["Tahun"] - 2015
    df_full["Rasio_Kend_Pddk"] = df_full["Total_Kendaraan"] / df_full["Penduduk"]

    return df_kend, df_kin, df_pddk, df_jam, df_full


try:
    df_kend, df_kin, df_pddk, df_jam, df_full = load_data()
    data_ok = True
except Exception as e:
    st.error(
        f"❌ Gagal load data: {e}\n\nPastikan file CSV ada di folder yang sama dengan app.py"
    )
    data_ok = False

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/8/82/Traffic_light_icon.svg",
        width=60,
    )
    st.title("🚦 Citimall Dumai")
    st.caption("Prediksi Kemacetan Lalu Lintas")
    st.divider()

    page = st.radio(
        "Navigasi",
        [
            "🏠 Overview",
            "📊 EDA & Analisis",
            "⏱️ Pola Per Jam",
            "🛣️ Kinerja Jalan",
            "🤖 Model & Prediksi",
            "📈 Before vs After Tuning",
        ],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown("**Dataset Info**")
    if data_ok:
        st.metric(
            "Periode Historis", f"{df_kend['Tahun'].min()}–{df_kend['Tahun'].max()}"
        )
        st.metric("Kapasitas Jalan", f"{C:,.0f} SMP/jam")
        st.metric("Tipe Jalan", "2/2-TT")
    st.caption("UTS Machine Learning\nUniversitas Amikom Yogyakarta")

if not data_ok:
    st.stop()


# ─── Helper: hitung metrik ───────────────────────────────────────────────────
def hitung_metrik(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {"R2": r2, "MAE": mae, "MSE": mse, "RMSE": rmse, "MAPE": mape}


# ─── Helper: LoS badge ────────────────────────────────────────────────────────
def los_badge(los):
    color = LOS_COLOR.get(los, "#95a5a6")
    return f'<span style="background:{color};color:white;padding:2px 8px;border-radius:4px;font-weight:bold">{los}</span>'


# ════════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ════════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.title("🚦 Prediksi Kemacetan Lalu Lintas")
    st.subheader("Citimall Dumai – Metode Regresi Linear & Polynomial")
    st.caption("Universitas Amikom Yogyakarta | UTS Machine Learning")
    st.divider()

    # KPI row
    col1, col2, col3, col4 = st.columns(4)
    latest_kin = df_kin[df_kin["Tahun"] == df_kin["Tahun"].max()]
    avg_dj = latest_kin["Dj"].mean()
    avg_q = latest_kin["Q_smp_jam"].mean()
    latest_kend = df_kend[df_kend["Tahun"] == df_kend["Tahun"].max()]
    total_kend = latest_kend["Total_Kendaraan"].values[0]
    latest_pddk = df_pddk[df_pddk["Tahun"] == df_pddk["Tahun"].max()]
    total_pddk = latest_pddk["Penduduk"].values[0]

    with col1:
        st.metric("📅 Tahun Data Terbaru", df_kin["Tahun"].max())
    with col2:
        st.metric("🚗 Total Kendaraan Terdaftar", f"{total_kend:,}")
    with col3:
        st.metric("👥 Populasi Dumai", f"{total_pddk:,}")
    with col4:
        los_val = latest_kin["LoS"].mode()[0]
        st.metric("🛣️ Level of Service", los_val, delta=f"Dj avg: {avg_dj:.4f}")

    st.divider()

    col_l, col_r = st.columns(2)

    # Tren Total Kendaraan
    with col_l:
        st.subheader("📈 Tren Total Kendaraan (2015–2024)")
        fig = px.line(
            df_kend,
            x="Tahun",
            y="Total_Kendaraan",
            markers=True,
            color_discrete_sequence=["#3498db"],
        )
        fig.add_scatter(
            x=df_kend["Tahun"],
            y=df_kend["Sepeda_Motor"],
            name="Sepeda Motor",
            line=dict(dash="dot", color="#e74c3c"),
        )
        fig.add_scatter(
            x=df_kend["Tahun"],
            y=df_kend["Mobil_Penumpang"],
            name="Mobil",
            line=dict(dash="dot", color="#2ecc71"),
        )
        fig.update_layout(
            height=320,
            legend=dict(orientation="h", y=-0.25),
            yaxis_title="Jumlah Kendaraan",
            xaxis_title="Tahun",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Tren Derajat Jenuh
    with col_r:
        st.subheader("📊 Tren Derajat Jenuh (Dj) per Survei")
        fig2 = px.bar(
            df_kin,
            x="Hari_Ke",
            y="Dj",
            color="Tahun",
            barmode="group",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig2.add_hline(
            y=0.75,
            line_dash="dash",
            line_color="red",
            annotation_text="Batas Kritis (0.75)",
        )
        fig2.update_layout(
            height=320, yaxis_title="Derajat Jenuh (Dj)", xaxis_title="Hari Ke-"
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Info cards
    st.divider()
    c1, c2, c3 = st.columns(3)
    with c1:
        st.info(
            "**Tentang Proyek**\n\nMemprediksi tingkat kemacetan jalan di sekitar Citimall Dumai menggunakan data lalu lintas historis 2019–2024 dengan metode *Linear Regression* dan *Polynomial Regression*."
        )
    with c2:
        st.success(
            "**Level of Service (LoS)**\n\n- **A–B**: Arus bebas 🟢\n- **C**: Stabil ⚠️\n- **D**: Hampir kritis 🟠\n- **E–F**: Kemacetan 🔴"
        )
    with c3:
        st.warning(
            "**Dataset yang Digunakan**\n\n- `data_kendaraan.csv` — Jumlah kendaraan\n- `data_penduduk.csv` — Populasi\n- `data_kinerja.csv` — Volume & Dj jalan\n- `data_perjam.csv` — Pola jam puncak"
        )

# ════════════════════════════════════════════════════════════════════════════════
# PAGE: EDA
# ════════════════════════════════════════════════════════════════════════════════
elif page == "📊 EDA & Analisis":
    st.title("📊 Exploratory Data Analysis")
    st.divider()

    tab1, tab2, tab3 = st.tabs(["🚗 Komposisi Kendaraan", "👥 Populasi", "🔗 Korelasi"])

    with tab1:
        st.subheader("Komposisi Kendaraan Terdaftar 2015–2024")
        col_l, col_r = st.columns(2)

        with col_l:
            # Stacked area chart
            fig = go.Figure()
            types = ["Sepeda_Motor", "Mobil_Penumpang", "Truk", "Bus"]
            colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]
            labels = ["Sepeda Motor", "Mobil Penumpang", "Truk", "Bus"]
            for t, c, l in zip(types, colors, labels):
                fig.add_trace(
                    go.Scatter(
                        x=df_kend["Tahun"],
                        y=df_kend[t],
                        mode="lines",
                        name=l,
                        fill="tonexty",
                        line=dict(color=c),
                    )
                )
            fig.update_layout(
                height=350,
                title="Tren per Jenis Kendaraan",
                yaxis_title="Jumlah",
                xaxis_title="Tahun",
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_r:
            # Latest year pie chart
            latest_y = df_kend["Tahun"].max()
            row = df_kend[df_kend["Tahun"] == latest_y].iloc[0]
            vals = [
                row["Sepeda_Motor"],
                row["Mobil_Penumpang"],
                row["Truk"],
                row["Bus"],
            ]
            fig2 = px.pie(
                names=labels,
                values=vals,
                color_discrete_sequence=colors,
                title=f"Komposisi Kendaraan {latest_y}",
            )
            fig2.update_layout(height=350)
            st.plotly_chart(fig2, use_container_width=True)

        st.dataframe(
            df_kend.style.highlight_max(axis=0, color="#d5f5e3").highlight_min(
                axis=0, color="#fadbd8"
            ),
            use_container_width=True,
        )

    with tab2:
        st.subheader("Pertumbuhan Populasi Kota Dumai")
        df_merged = df_kend.merge(df_pddk, on="Tahun")
        df_merged["Rasio"] = df_merged["Total_Kendaraan"] / df_merged["Penduduk"]

        col_l, col_r = st.columns(2)
        with col_l:
            fig = px.bar(
                df_pddk,
                x="Tahun",
                y="Penduduk",
                color="Penduduk",
                color_continuous_scale="Blues",
                title="Jumlah Penduduk per Tahun",
            )
            fig.update_layout(height=320)
            st.plotly_chart(fig, use_container_width=True)

        with col_r:
            fig2 = px.line(
                df_merged,
                x="Tahun",
                y="Rasio",
                markers=True,
                title="Rasio Kendaraan / Penduduk",
                color_discrete_sequence=["#e74c3c"],
            )
            fig2.update_layout(height=320, yaxis_title="Kend / Jiwa")
            st.plotly_chart(fig2, use_container_width=True)

        growth = (df_pddk["Penduduk"].iloc[-1] / df_pddk["Penduduk"].iloc[0] - 1) * 100
        st.metric(
            "Pertumbuhan Penduduk (Total)",
            f"+{growth:.1f}%",
            delta=f"{df_pddk['Tahun'].min()} → {df_pddk['Tahun'].max()}",
        )

    with tab3:
        st.subheader("Analisis Korelasi Antar Variabel")
        df_corr = df_kend.merge(df_pddk, on="Tahun").drop("Tahun", axis=1)
        corr_m = df_corr.corr()

        fig = px.imshow(
            corr_m,
            text_auto=".2f",
            color_continuous_scale="RdYlGn",
            title="Heatmap Korelasi",
            aspect="auto",
            zmin=-1,
            zmax=1,
        )
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

        st.info(
            "💡 Korelasi mendekati **+1** berarti hubungan positif kuat (naik bersama). Korelasi **-1** berarti hubungan negatif kuat."
        )

        # Scatter matrix
        st.subheader("Scatter Matrix (Kendaraan vs Penduduk)")
        df_scatter = df_corr.copy()
        fig2 = px.scatter_matrix(
            df_scatter, dimensions=df_scatter.columns, color_continuous_scale="Viridis"
        )
        fig2.update_layout(height=500)
        st.plotly_chart(fig2, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════════
# PAGE: POLA PER JAM
# ════════════════════════════════════════════════════════════════════════════════
elif page == "⏱️ Pola Per Jam":
    st.title("⏱️ Pola Lalu Lintas Per Jam")
    st.caption("Data Survei: Kamis, 21 November 2019")
    st.divider()

    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Volume Lalu Lintas per Periode")
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df_jam["Jam_Mulai"],
                y=df_jam["Q_total_H1"],
                name="Total 2 Arah",
                mode="lines+markers",
                line=dict(color="#2c3e50", width=2.5),
                marker=dict(size=8),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df_jam["Jam_Mulai"],
                y=df_jam["Q_utara_H1"],
                name="Arah Utara",
                mode="lines+markers",
                line=dict(color="#3498db", dash="dash"),
                marker=dict(size=6),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df_jam["Jam_Mulai"],
                y=df_jam["Q_barat_H1"],
                name="Arah Barat",
                mode="lines+markers",
                line=dict(color="#e74c3c", dash="dash"),
                marker=dict(size=6),
            )
        )
        jam_labels = [f"{int(j)}:{int((j % 1) * 60):02d}" for j in df_jam["Jam_Mulai"]]
        fig.update_layout(
            height=380,
            yaxis_title="Q (SMP/jam)",
            xaxis=dict(tickvals=df_jam["Jam_Mulai"], ticktext=jam_labels, tickangle=45),
            legend=dict(orientation="h", y=-0.3),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.subheader("Faktor Waktu per Periode")
        colors_bar = [
            "#e74c3c" if f == df_jam["Faktor_Waktu"].max() else "#3498db"
            for f in df_jam["Faktor_Waktu"]
        ]
        fig2 = go.Figure(
            go.Bar(
                x=df_jam["Jam_Mulai"],
                y=df_jam["Faktor_Waktu"],
                marker_color=colors_bar,
                opacity=0.85,
                text=[f"{f:.3f}" for f in df_jam["Faktor_Waktu"]],
                textposition="outside",
            )
        )
        fig2.update_layout(
            height=380,
            yaxis_title="Faktor Waktu",
            xaxis=dict(tickvals=df_jam["Jam_Mulai"], ticktext=jam_labels, tickangle=45),
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    # Peak hour summary
    idx_peak = df_jam["Q_total_H1"].idxmax()
    peak_row = df_jam.iloc[idx_peak]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("🔴 Jam Puncak", peak_row["Periode"])
    with c2:
        st.metric("📈 Q Puncak", f"{peak_row['Q_total_H1']:.1f} SMP/jam")
    with c3:
        st.metric("➡️ Arah Utara Peak", f"{df_jam.loc[idx_peak, 'Q_utara_H1']:.1f}")
    with c4:
        st.metric("⬅️ Arah Barat Peak", f"{df_jam.loc[idx_peak, 'Q_barat_H1']:.1f}")

    st.subheader("📋 Tabel Data Per Jam")
    st.dataframe(
        df_jam.style.highlight_max(
            subset=["Q_total_H1", "Faktor_Waktu"], color="#fadbd8"
        ),
        use_container_width=True,
    )

# ════════════════════════════════════════════════════════════════════════════════
# PAGE: KINERJA JALAN
# ════════════════════════════════════════════════════════════════════════════════
elif page == "🛣️ Kinerja Jalan":
    st.title("🛣️ Kinerja Jalan – Volume & Derajat Jenuh")
    st.divider()

    # Filter tahun
    tahun_list = sorted(df_kin["Tahun"].unique())
    sel_tahun = st.multiselect("Pilih Tahun", tahun_list, default=tahun_list)
    df_filt = df_kin[df_kin["Tahun"].isin(sel_tahun)]

    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Volume Q per Hari Survei")
        fig = px.bar(
            df_filt,
            x="Hari",
            y="Q_smp_jam",
            color="Tahun",
            barmode="group",
            text_auto=".0f",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.add_hline(
            y=C,
            line_dash="dash",
            line_color="black",
            annotation_text=f"Kapasitas C={C}",
        )
        fig.update_layout(height=380, yaxis_title="Q (SMP/jam)", xaxis_title="Hari")
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.subheader("Derajat Jenuh (Dj) per Hari Survei")
        fig2 = px.bar(
            df_filt,
            x="Hari",
            y="Dj",
            color="Tahun",
            barmode="group",
            text_auto=".3f",
            color_discrete_sequence=px.colors.qualitative.Set1,
        )
        fig2.add_hline(
            y=0.75, line_dash="dash", line_color="red", annotation_text="Kritis 0.75"
        )
        fig2.add_hline(
            y=0.45, line_dash="dot", line_color="orange", annotation_text="LoS C 0.45"
        )
        fig2.update_layout(height=380, yaxis_title="Dj", xaxis_title="Hari")
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    st.subheader("📋 Level of Service (LoS) Summary")

    los_counts = df_filt["LoS"].value_counts().reset_index()
    los_counts.columns = ["LoS", "Frekuensi"]

    col_a, col_b = st.columns([1, 2])
    with col_a:
        for _, row in df_filt.iterrows():
            color = LOS_COLOR.get(row["LoS"], "#95a5a6")
            st.markdown(
                f"**{row['Tahun']} – Hari {row['Hari_Ke']} ({row['Hari']})** → "
                f"<span style='background:{color};color:white;padding:2px 8px;"
                f"border-radius:4px'>{row['LoS']}</span> "
                f"(Dj={row['Dj']:.4f})",
                unsafe_allow_html=True,
            )
    with col_b:
        fig3 = px.pie(
            los_counts,
            names="LoS",
            values="Frekuensi",
            color="LoS",
            color_discrete_map=LOS_COLOR,
            title="Distribusi Level of Service",
        )
        fig3.update_layout(height=320)
        st.plotly_chart(fig3, use_container_width=True)

    st.divider()
    st.subheader("📋 Tabel Kinerja Jalan")
    st.dataframe(df_filt, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL & PREDIKSI
# ════════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Model & Prediksi":
    st.title("🤖 Model Regresi & Prediksi Kemacetan")
    st.divider()

    # ── Build Models ────────────────────────────────────────────────────────
    feature_cols = ["X_tahun", "Total_Kendaraan", "Penduduk", "Q_smp_jam"]
    target_col = "Dj"

    X = df_full[feature_cols]
    y = df_full[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)
    X_all_sc = scaler.transform(X)

    model_lr = LinearRegression()
    model_lr.fit(X_train_sc, y_train)

    poly_pipe = Pipeline(
        [
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("scaler", StandardScaler()),
            ("lr", LinearRegression()),
        ]
    )
    poly_pipe.fit(X_train, y_train)

    tab_eval, tab_pred, tab_future = st.tabs(
        ["📐 Evaluasi Model", "🔍 Visualisasi Prediksi", "🔮 Prediksi Masa Depan"]
    )

    with tab_eval:
        st.subheader("Metrik Evaluasi Model")

        m_lr_train = hitung_metrik(y_train, model_lr.predict(X_train_sc))
        m_lr_test = hitung_metrik(y_test, model_lr.predict(X_test_sc))
        m_poly_test = hitung_metrik(y_test, poly_pipe.predict(X_test))

        kf5 = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_lr = cross_val_score(
            Pipeline([("sc", StandardScaler()), ("lr", LinearRegression())]),
            X,
            y,
            cv=kf5,
            scoring="r2",
        )
        cv_poly = cross_val_score(
            Pipeline(
                [
                    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                    ("sc", StandardScaler()),
                    ("lr", LinearRegression()),
                ]
            ),
            X,
            y,
            cv=kf5,
            scoring="r2",
        )

        df_metrics = pd.DataFrame(
            {
                "Model": ["Linear Regression", "Polynomial Regression (deg=2)"],
                "R² Train": [m_lr_train["R2"], "—"],
                "R² Test": [f"{m_lr_test['R2']:.6f}", f"{m_poly_test['R2']:.6f}"],
                "MAE": [f"{m_lr_test['MAE']:.6f}", f"{m_poly_test['MAE']:.6f}"],
                "MAPE (%)": [f"{m_lr_test['MAPE']:.4f}", f"{m_poly_test['MAPE']:.4f}"],
                "RMSE": [f"{m_lr_test['RMSE']:.6f}", f"{m_poly_test['RMSE']:.6f}"],
                "CV R² (5-Fold)": [
                    f"{cv_lr.mean():.6f} ± {cv_lr.std():.4f}",
                    f"{cv_poly.mean():.6f} ± {cv_poly.std():.4f}",
                ],
            }
        )
        st.dataframe(df_metrics, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Linear R² Test", f"{m_lr_test['R2']:.4f}")
            st.metric("Linear CV R² (mean)", f"{cv_lr.mean():.4f}")
        with c2:
            st.metric("Polynomial R² Test", f"{m_poly_test['R2']:.4f}")
            st.metric("Polynomial CV R² (mean)", f"{cv_poly.mean():.4f}")

        # Koefisien regresi linear
        st.divider()
        st.subheader("Koefisien Linear Regression")
        coef_df = pd.DataFrame(
            {
                "Fitur": feature_cols,
                "Koefisien": model_lr.coef_,
                "Arah": ["(+)" if c > 0 else "(-)" for c in model_lr.coef_],
            }
        )
        st.dataframe(coef_df, use_container_width=True)
        st.info(f"Intercept = {model_lr.intercept_:.6f}")

    with tab_pred:
        st.subheader("Actual vs Predicted – Derajat Jenuh (Dj)")

        y_pred_lr = model_lr.predict(X_all_sc)
        y_pred_poly = poly_pipe.predict(X)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=list(range(len(y))),
                y=y.values,
                name="Actual Dj",
                mode="lines+markers",
                line=dict(color="#2c3e50", width=2.5),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=list(range(len(y))),
                y=y_pred_lr,
                name="Linear Regression",
                mode="lines+markers",
                line=dict(color="#3498db", dash="dash"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=list(range(len(y))),
                y=y_pred_poly,
                name="Polynomial (deg=2)",
                mode="lines+markers",
                line=dict(color="#e74c3c", dash="dot"),
            )
        )
        fig.update_layout(
            height=400,
            yaxis_title="Derajat Jenuh (Dj)",
            xaxis_title="Indeks Sampel",
            legend=dict(orientation="h", y=-0.2),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Residual
        st.subheader("Residual Linear Regression")
        residuals = y.values - y_pred_lr
        fig2 = go.Figure(
            go.Bar(
                x=list(range(len(residuals))),
                y=residuals,
                marker_color=["#e74c3c" if r < 0 else "#2ecc71" for r in residuals],
            )
        )
        fig2.add_hline(y=0, line_color="black", line_width=1.5)
        fig2.update_layout(height=280, yaxis_title="Residual", xaxis_title="Indeks")
        st.plotly_chart(fig2, use_container_width=True)

    with tab_future:
        # ── Build df_prediksi (annual peak Dj & Q for 2019–2027) ────────────
        all_years = list(range(2019, 2028))
        hist_years_a = df_kend["Tahun"].values

        lr_kend2 = LinearRegression()
        lr_kend2.fit(hist_years_a.reshape(-1, 1), df_kend["Total_Kendaraan"].values)
        lr_pddk2 = LinearRegression()
        lr_pddk2.fit(df_pddk["Tahun"].values.reshape(-1, 1), df_pddk["Penduduk"].values)
        lr_q2 = LinearRegression()
        q_by_yr = df_kin.groupby("Tahun")["Q_smp_jam"].mean()
        lr_q2.fit(q_by_yr.index.values.reshape(-1, 1), q_by_yr.values)

        prediksi_rows = []
        for yr in all_years:
            total_k = lr_kend2.predict([[yr]])[0]
            pddk_v = lr_pddk2.predict([[yr]])[0]
            q_v = lr_q2.predict([[yr]])[0]
            X_inp = scaler.transform([[yr - 2015, total_k, pddk_v, q_v]])
            dj_pred = float(model_lr.predict(X_inp)[0])
            prediksi_rows.append(
                {
                    "Tahun": yr,
                    "Dj_Prediksi": dj_pred,
                    "Q_SMP_Jam": q_v,
                    "Total_Kendaraan": int(total_k),
                    "Penduduk": int(pddk_v),
                }
            )
        df_prediksi = pd.DataFrame(prediksi_rows)

        # ── Cell 9 lookup tables ─────────────────────────────────────────────
        HARI_FAKTOR = {
            "Senin": 1.129,
            "Selasa": 1.050,
            "Rabu": 0.960,
            "Kamis": 0.926,
            "Jumat": 0.985,
            "Sabtu": 1.062,
            "Minggu": 0.898,
        }
        WAKTU_FAKTOR = {
            6.0: 0.40,
            6.5: 0.687,
            7.0: 0.676,
            7.5: 0.628,
            8.0: 0.565,
            8.5: 0.539,
            9.0: 0.510,
            9.5: 0.490,
            10.0: 0.500,
            10.5: 0.530,
            11.0: 0.614,
            11.5: 0.580,
            12.0: 0.560,
            12.5: 0.530,
            13.0: 0.510,
            13.5: 0.503,
            14.0: 0.452,
            14.5: 0.431,
            15.0: 0.450,
            15.5: 0.472,
            16.0: 0.534,
            16.5: 0.579,
            17.0: 0.618,
            17.5: 0.686,
            18.0: 0.641,
            18.5: 0.586,
            19.0: 0.552,
            19.5: 0.466,
            20.0: 0.430,
            20.5: 0.380,
            21.0: 0.300,
        }
        ARAH_FAKTOR = {
            "Total 2 Arah": 1.000,
            "Menuju Arah Utara": 0.433,
            "Menuju Arah Barat": 0.567,
        }
        KEND_PROPORSI = {
            "Semua Kendaraan (SMP)": 1.000,
            "SM – Sepeda Motor": 400.5 / 604.1,
            "KR – Kendaraan Ringan": 162.0 / 604.1,
            "KB – Kendaraan Berat": 41.6 / 604.1,
        }

        def get_los(dj):
            if dj <= 0.20:
                return "A"
            elif dj <= 0.44:
                return "B"
            elif dj <= 0.64:
                return "C"
            elif dj <= 0.80:
                return "D"
            elif dj <= 1.00:
                return "E"
            else:
                return "F"

        def get_status(dj):
            if dj <= 0.20:
                return "✅ Arus sangat bebas"
            elif dj <= 0.44:
                return "✅ Arus stabil"
            elif dj <= 0.64:
                return "⚠️ Arus mulai padat"
            elif dj <= 0.80:
                return "🟠 Hampir kritis – perlu perhatian"
            elif dj <= 1.00:
                return "🔴 Kritis – kemacetan terjadi"
            else:
                return "🚨 Macet total – kapasitas terlampaui"

        def get_rekomendasi(dj):
            if dj <= 0.44:
                return "Kondisi normal. Tidak ada intervensi diperlukan."
            elif dj <= 0.64:
                return "Mulai padat. Siapkan pengatur lalu lintas saat jam puncak."
            elif dj <= 0.80:
                return "Hampir kritis. Pertimbangkan manajemen parkir & rekayasa lalu lintas."
            elif dj <= 1.00:
                return "KRITIS. Butuh pengalihan arus, penambahan kapasitas, atau pembatasan kendaraan."
            else:
                return "DARURAT. Segera lakukan pengalihan arus & koordinasi dengan Dishub."

        # ── Inputs ───────────────────────────────────────────────────────────
        st.subheader("🎮 Alat Prediksi Interaktif – Cell 9")
        st.caption("Ubah parameter di bawah → hasil langsung update otomatis.")
        st.divider()

        col_inp1, col_inp2 = st.columns([1, 1])
        with col_inp1:
            inp_tahun = st.slider("📅 Tahun", 2019, 2027, 2024)
            inp_hari = st.selectbox("📆 Hari", list(HARI_FAKTOR.keys()), index=4)
            jam_options = {
                f"{int(j):02d}:{int((j % 1) * 60):02d}": j
                for j in sorted(WAKTU_FAKTOR.keys())
            }
            inp_jam_label = st.select_slider(
                "🕐 Jam Survei", options=list(jam_options.keys()), value="17:30"
            )
            inp_jam = jam_options[inp_jam_label]

        with col_inp2:
            inp_arah = st.radio("🧭 Arah Lalu Lintas", list(ARAH_FAKTOR.keys()))
            inp_kend = st.selectbox("🚗 Tipe Kendaraan", list(KEND_PROPORSI.keys()))
            show_chart = st.checkbox("📊 Tampilkan grafik fluktuasi harian", value=True)

        st.divider()

        # ── Kalkulasi ────────────────────────────────────────────────────────
        row_pred = df_prediksi[df_prediksi["Tahun"] == inp_tahun].iloc[0]
        Dj_puncak = row_pred["Dj_Prediksi"]
        Q_puncak = row_pred["Q_SMP_Jam"]

        f_hari = HARI_FAKTOR[inp_hari]
        jam_keys = sorted(WAKTU_FAKTOR.keys())
        f_waktu = float(
            np.interp(inp_jam, jam_keys, [WAKTU_FAKTOR[k] for k in jam_keys])
        )
        f_arah = ARAH_FAKTOR[inp_arah]
        f_kend = KEND_PROPORSI[inp_kend]

        Q_pred = Q_puncak * f_hari * f_waktu * f_arah * f_kend
        Dj_pred = min(Dj_puncak * f_hari * f_waktu * f_arah, 1.5)
        los_val = get_los(Dj_pred)
        sisa_kap = max(0, C - Q_pred)
        los_color = LOS_COLOR.get(los_val, "#95a5a6")

        # ── Output cards ─────────────────────────────────────────────────────
        col_r1, col_r2, col_r3, col_r4 = st.columns(4)
        with col_r1:
            st.metric(
                "🚦 Derajat Jenuh (Dj)",
                f"{Dj_pred:.4f}",
                delta=f"Puncak: {Dj_puncak:.4f}",
            )
        with col_r2:
            st.metric(
                "📊 Volume Q", f"{Q_pred:.1f} SMP/jam", delta=f"Puncak: {Q_puncak:.1f}"
            )
        with col_r3:
            st.metric(
                "🏁 Sisa Kapasitas",
                f"{sisa_kap:.1f} SMP/jam",
                delta=f"{sisa_kap / C * 100:.1f}% tersisa",
            )
        with col_r4:
            st.markdown(
                f"""
            <div style='background:{los_color};border-radius:12px;padding:16px;text-align:center'>
                <div style='color:white;font-size:2.5rem;font-weight:bold'>{los_val}</div>
                <div style='color:white;font-size:0.8rem'>Level of Service</div>
            </div>""",
                unsafe_allow_html=True,
            )

        st.markdown(f"**Status:** {get_status(Dj_pred)}")
        st.info(f"💡 **Rekomendasi:** {get_rekomendasi(Dj_pred)}")

        # ── Faktor breakdown ──────────────────────────────────────────────────
        st.divider()
        st.subheader("🔬 Breakdown Faktor")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric(
                "Faktor Hari",
                f"{f_hari:.3f}",
                delta=f"{(f_hari - 1) * 100:+.1f}% dari avg",
            )
        with c2:
            st.metric("Faktor Waktu", f"{f_waktu:.3f}")
        with c3:
            st.metric("Faktor Arah", f"{f_arah:.3f}")

        # ── Gauge ─────────────────────────────────────────────────────────────
        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number+delta",
                value=Dj_pred,
                delta={"reference": Dj_puncak, "valueformat": ".4f"},
                title={
                    "text": f"Dj — {inp_hari}, {inp_jam_label}, {inp_tahun}",
                    "font": {"size": 14},
                },
                number={"valueformat": ".4f"},
                gauge={
                    "axis": {"range": [0, 1.2], "tickformat": ".2f"},
                    "bar": {"color": los_color, "thickness": 0.3},
                    "steps": [
                        {"range": [0, 0.20], "color": "#d5f5e3"},
                        {"range": [0.20, 0.44], "color": "#a9dfbf"},
                        {"range": [0.44, 0.64], "color": "#fdebd0"},
                        {"range": [0.64, 0.80], "color": "#fad7a0"},
                        {"range": [0.80, 1.00], "color": "#f5b7b1"},
                        {"range": [1.00, 1.20], "color": "#e8daef"},
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 3},
                        "thickness": 0.8,
                        "value": 0.75,
                    },
                },
            )
        )
        fig_gauge.update_layout(height=320)
        st.plotly_chart(fig_gauge, use_container_width=True)

        # ── Hourly fluctuation chart ──────────────────────────────────────────
        if show_chart:
            st.divider()
            st.subheader(f"📈 Fluktuasi Dj Sepanjang Hari — {inp_hari}, {inp_tahun}")
            jam_list = sorted(WAKTU_FAKTOR.keys())
            dj_harian = [
                min(Dj_puncak * f_hari * WAKTU_FAKTOR[j] * f_arah, 1.5)
                for j in jam_list
            ]
            q_harian = [
                Q_puncak * f_hari * WAKTU_FAKTOR[j] * f_arah * f_kend for j in jam_list
            ]
            jam_labels_h = [f"{int(j):02d}:{int((j % 1) * 60):02d}" for j in jam_list]

            fig_h = make_subplots(specs=[[{"secondary_y": True}]])
            fig_h.add_trace(
                go.Scatter(
                    x=jam_labels_h,
                    y=dj_harian,
                    name="Dj",
                    mode="lines+markers",
                    line=dict(color="#e74c3c", width=2.5),
                    fill="tozeroy",
                    fillcolor="rgba(231,76,60,0.1)",
                ),
                secondary_y=False,
            )
            fig_h.add_trace(
                go.Bar(
                    x=jam_labels_h,
                    y=q_harian,
                    name="Q (SMP/jam)",
                    opacity=0.4,
                    marker_color="#3498db",
                ),
                secondary_y=True,
            )
            vline_idx = (
                jam_labels_h.index(inp_jam_label)
                if inp_jam_label in jam_labels_h
                else 0
            )
            fig_h.add_vline(
                x=vline_idx,
                line_dash="dash",
                line_color="purple",
                annotation_text=f"▶ {inp_jam_label}",
            )
            fig_h.add_hline(
                y=0.75,
                line_dash="dot",
                line_color="orange",
                annotation_text="Kritis 0.75",
                secondary_y=False,
            )
            for lo, hi, clr, lbl in [
                (0, 0.20, "#2ecc71", "A"),
                (0.20, 0.44, "#27ae60", "B"),
                (0.44, 0.64, "#f39c12", "C"),
                (0.64, 0.80, "#e67e22", "D"),
                (0.80, 1.00, "#e74c3c", "E"),
            ]:
                fig_h.add_hrect(
                    y0=lo,
                    y1=hi,
                    fillcolor=clr,
                    opacity=0.06,
                    annotation_text=f"LoS {lbl}",
                    annotation_position="right",
                    line_width=0,
                    secondary_y=False,
                )
            fig_h.update_yaxes(title_text="Derajat Jenuh (Dj)", secondary_y=False)
            fig_h.update_yaxes(title_text="Q (SMP/jam)", secondary_y=True)
            fig_h.update_xaxes(tickangle=45)
            fig_h.update_layout(height=380, legend=dict(orientation="h", y=-0.2))
            st.plotly_chart(fig_h, use_container_width=True)

        # ── Annual trend ──────────────────────────────────────────────────────
        st.divider()
        st.subheader("📅 Tren Dj Tahunan (2019–2027)")
        dj_hist_avg = df_kin.groupby("Tahun")["Dj"].mean().reset_index()
        df_prediksi["LoS"] = df_prediksi["Dj_Prediksi"].apply(get_los)

        fig_trend = go.Figure()
        fig_trend.add_trace(
            go.Scatter(
                x=dj_hist_avg["Tahun"],
                y=dj_hist_avg["Dj"],
                name="Historis (avg)",
                mode="lines+markers",
                line=dict(color="#2c3e50", width=2.5),
            )
        )
        fig_trend.add_trace(
            go.Scatter(
                x=df_prediksi["Tahun"],
                y=df_prediksi["Dj_Prediksi"],
                name="Prediksi Model",
                mode="lines+markers",
                line=dict(color="#e74c3c", dash="dash"),
                marker=dict(size=9, symbol="diamond"),
            )
        )
        sel_dj = df_prediksi[df_prediksi["Tahun"] == inp_tahun]["Dj_Prediksi"].values[0]
        fig_trend.add_trace(
            go.Scatter(
                x=[inp_tahun],
                y=[sel_dj],
                name=f"Dipilih ({inp_tahun})",
                mode="markers",
                marker=dict(size=14, color="purple", symbol="star"),
            )
        )
        fig_trend.add_hline(
            y=0.75,
            line_dash="dash",
            line_color="orange",
            annotation_text="Kritis (0.75)",
        )
        fig_trend.add_hline(
            y=1.0,
            line_dash="dash",
            line_color="red",
            annotation_text="Kapasitas Penuh (1.0)",
        )
        fig_trend.update_layout(
            height=350,
            yaxis_title="Dj (jam puncak)",
            xaxis_title="Tahun",
            legend=dict(orientation="h", y=-0.2),
        )
        st.plotly_chart(fig_trend, use_container_width=True)

        # ── Table ─────────────────────────────────────────────────────────────
        st.subheader("📋 Tabel Prediksi Tahunan (2019–2027)")

        def color_los_tbl(val):
            return f"background-color:{LOS_COLOR.get(val, '#fff')};color:white"

        st.dataframe(
            df_prediksi.style.map(color_los_tbl, subset=["LoS"]).format(
                {
                    "Dj_Prediksi": "{:.4f}",
                    "Q_SMP_Jam": "{:.2f}",
                    "Total_Kendaraan": "{:,}",
                    "Penduduk": "{:,}",
                }
            ),
            use_container_width=True,
        )

# ════════════════════════════════════════════════════════════════════════════════
# PAGE: BEFORE vs AFTER TUNING
# ════════════════════════════════════════════════════════════════════════════════
elif page == "📈 Before vs After Tuning":
    st.title("📈 Before vs After Tuning – Perbandingan Model")
    st.divider()

    feature_cols = ["X_tahun", "Total_Kendaraan", "Penduduk", "Q_smp_jam"]
    X_all = df_full[feature_cols]
    y_all = df_full["Dj"]

    def hm(y_true, y_pred):
        return hitung_metrik(np.array(y_true), np.array(y_pred))

    kf5 = KFold(n_splits=5, shuffle=True, random_state=42)

    with st.spinner("⚙️ Melatih semua model..."):
        # BEFORE
        X_tr_b, X_te_b, y_tr_b, y_te_b = train_test_split(
            X_all, y_all, test_size=0.25, random_state=42
        )
        sc_b = StandardScaler()
        pipe_b = Pipeline([("sc", StandardScaler()), ("lr", LinearRegression())])
        pipe_b.fit(X_tr_b, y_tr_b)
        m_b_train = hm(y_tr_b, pipe_b.predict(X_tr_b))
        m_b_test = hm(y_te_b, pipe_b.predict(X_te_b))
        cv5_before = cross_val_score(pipe_b, X_all, y_all, cv=5, scoring="r2")
        bfr = {
            **m_b_test,
            "R2_train": m_b_train["R2"],
            "CV_R2": cv5_before.mean(),
            "Gap": abs(m_b_train["R2"] - m_b_test["R2"]),
        }

        # AFTER
        from sklearn.linear_model import Lasso, Ridge
        from sklearn.model_selection import GridSearchCV

        X_tr_a, X_te_a, y_tr_a, y_te_a = train_test_split(
            X_all, y_all, test_size=0.2, random_state=42, shuffle=False
        )

        after_models = {}

        # Linear
        pipe_lr = Pipeline([("sc", StandardScaler()), ("lr", LinearRegression())])
        pipe_lr.fit(X_tr_a, y_tr_a)
        m_lr = hm(y_te_a, pipe_lr.predict(X_te_a))
        cv_lr = cross_val_score(pipe_lr, X_all, y_all, cv=kf5, scoring="r2").mean()
        after_models["Linear Regression"] = {
            **m_lr,
            "R2_train": hm(y_tr_a, pipe_lr.predict(X_tr_a))["R2"],
            "CV_R2": cv_lr,
            "Gap": abs(hm(y_tr_a, pipe_lr.predict(X_tr_a))["R2"] - m_lr["R2"]),
        }

        # Ridge
        ridge_gs = GridSearchCV(
            Pipeline([("sc", StandardScaler()), ("ridge", Ridge())]),
            {"ridge__alpha": [0.001, 0.01, 0.1, 1, 10, 100]},
            cv=kf5,
            scoring="r2",
        )
        ridge_gs.fit(X_tr_a, y_tr_a)
        m_ri = hm(y_te_a, ridge_gs.predict(X_te_a))
        best_a_r = ridge_gs.best_params_["ridge__alpha"]
        after_models[f"Ridge (α={best_a_r})"] = {
            **m_ri,
            "R2_train": hm(y_tr_a, ridge_gs.predict(X_tr_a))["R2"],
            "CV_R2": ridge_gs.best_score_,
            "Gap": abs(hm(y_tr_a, ridge_gs.predict(X_tr_a))["R2"] - m_ri["R2"]),
        }

        # Lasso
        lasso_gs = GridSearchCV(
            Pipeline([("sc", StandardScaler()), ("lasso", Lasso(max_iter=10000))]),
            {"lasso__alpha": [0.0001, 0.001, 0.01, 0.1]},
            cv=kf5,
            scoring="r2",
        )
        lasso_gs.fit(X_tr_a, y_tr_a)
        m_la = hm(y_te_a, lasso_gs.predict(X_te_a))
        best_a_l = lasso_gs.best_params_["lasso__alpha"]
        after_models[f"Lasso (α={best_a_l})"] = {
            **m_la,
            "R2_train": hm(y_tr_a, lasso_gs.predict(X_tr_a))["R2"],
            "CV_R2": lasso_gs.best_score_,
            "Gap": abs(hm(y_tr_a, lasso_gs.predict(X_tr_a))["R2"] - m_la["R2"]),
        }

        # Poly + Ridge
        poly_gs = GridSearchCV(
            Pipeline(
                [
                    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                    ("sc", StandardScaler()),
                    ("ridge", Ridge()),
                ]
            ),
            {"ridge__alpha": [0.001, 0.01, 0.1, 1, 10, 100]},
            cv=kf5,
            scoring="r2",
        )
        poly_gs.fit(X_tr_a, y_tr_a)
        m_po = hm(y_te_a, poly_gs.predict(X_te_a))
        best_a_p = poly_gs.best_params_["ridge__alpha"]
        after_models[f"Poly2+Ridge (α={best_a_p})"] = {
            **m_po,
            "R2_train": hm(y_tr_a, poly_gs.predict(X_tr_a))["R2"],
            "CV_R2": poly_gs.best_score_,
            "Gap": abs(hm(y_tr_a, poly_gs.predict(X_tr_a))["R2"] - m_po["R2"]),
        }

    best_name = max(after_models, key=lambda k: after_models[k]["CV_R2"])
    best = after_models[best_name]

    # ── Summary Cards ─────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    with col1:
        st.error(
            f"**BEFORE** – Linear Regression\n\nR² Test: `{bfr['R2']:.6f}`\nCV R²: `{bfr['CV_R2']:.6f}`\nMAE: `{bfr['MAE']:.6f}`"
        )
    with col2:
        st.success(
            f"**BEST AFTER** – {best_name}\n\nR² Test: `{best['R2']:.6f}`\nCV R²: `{best['CV_R2']:.6f}`\nMAE: `{best['MAE']:.6f}`"
        )
    with col3:
        delta_r2 = best["R2"] - bfr["R2"]
        delta_mae = best["MAE"] - bfr["MAE"]
        delta_cv = best["CV_R2"] - bfr["CV_R2"]
        sign_r2 = "+" if delta_r2 >= 0 else ""
        sign_mae = "+" if delta_mae >= 0 else ""
        sign_cv = "+" if delta_cv >= 0 else ""
        st.info(
            f"**Perubahan**\n\nΔ R² Test: `{sign_r2}{delta_r2:.6f}`\nΔ CV R²: `{sign_cv}{delta_cv:.6f}`\nΔ MAE: `{sign_mae}{delta_mae:.6f}`"
        )

    st.divider()

    # ── Charts ────────────────────────────────────────────────────────────
    all_labels = ["BEFORE\nLinear Reg"] + [f"AFTER\n{n}" for n in after_models.keys()]
    all_r2 = [bfr["R2"]] + [v["R2"] for v in after_models.values()]
    all_cv = [bfr["CV_R2"]] + [v["CV_R2"] for v in after_models.values()]
    all_mae = [bfr["MAE"]] + [v["MAE"] for v in after_models.values()]
    all_rmse = [bfr["RMSE"]] + [v["RMSE"] for v in after_models.values()]
    all_gap = [bfr["Gap"]] + [v["Gap"] for v in after_models.values()]
    colors = ["#e74c3c"] + ["#3498db"] * len(after_models)

    col_l, col_r = st.columns(2)
    with col_l:
        fig = go.Figure(
            go.Bar(
                y=all_labels,
                x=all_r2,
                orientation="h",
                marker_color=colors,
                opacity=0.85,
                text=[f"{v:.4f}" for v in all_r2],
                textposition="outside",
            )
        )
        fig.add_vline(x=bfr["R2"], line_dash="dash", line_color="red")
        fig.update_layout(
            height=320, title="R² Score Test Set", xaxis_title="R² (↑ lebih baik)"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        fig2 = go.Figure(
            go.Bar(
                y=all_labels,
                x=all_cv,
                orientation="h",
                marker_color=colors,
                opacity=0.85,
                text=[f"{v:.4f}" for v in all_cv],
                textposition="outside",
            )
        )
        fig2.add_vline(x=bfr["CV_R2"], line_dash="dash", line_color="red")
        fig2.update_layout(
            height=320, title="CV R² (5-Fold)", xaxis_title="CV R² (↑ lebih baik)"
        )
        st.plotly_chart(fig2, use_container_width=True)

    col_l2, col_r2 = st.columns(2)
    with col_l2:
        fig3 = go.Figure(
            go.Bar(
                y=all_labels,
                x=all_mae,
                orientation="h",
                marker_color=colors,
                opacity=0.85,
                text=[f"{v:.6f}" for v in all_mae],
                textposition="outside",
            )
        )
        fig3.add_vline(x=bfr["MAE"], line_dash="dash", line_color="red")
        fig3.update_layout(height=320, title="MAE (↓ lebih baik)")
        st.plotly_chart(fig3, use_container_width=True)

    with col_r2:
        gap_cols = [
            "#e74c3c" if i == 0 else ("#2ecc71" if g < 0.01 else "#3498db")
            for i, g in enumerate(all_gap)
        ]
        fig4 = go.Figure(
            go.Bar(
                y=all_labels,
                x=all_gap,
                orientation="h",
                marker_color=gap_cols,
                opacity=0.85,
                text=[f"{v:.6f}" for v in all_gap],
                textposition="outside",
            )
        )
        fig4.add_vline(
            x=0.05, line_dash="dot", line_color="orange", annotation_text="0.05"
        )
        fig4.update_layout(
            height=320,
            title="Gap Train-Test (Overfitting Indicator)",
            xaxis_title="|R²Train − R²Test| (↓ mendekati 0 = tidak overfit)",
        )
        st.plotly_chart(fig4, use_container_width=True)

    # ── Full comparison table ──────────────────────────────────────────────
    st.divider()
    st.subheader("📋 Tabel Perbandingan Lengkap")
    rows = [
        {
            "Status": "BEFORE",
            "Model": "Linear Regression (75:25)",
            "R² Test": bfr["R2"],
            "MAE": bfr["MAE"],
            "MAPE (%)": bfr["MAPE"],
            "RMSE": bfr["RMSE"],
            "CV R²": bfr["CV_R2"],
            "Gap": bfr["Gap"],
        }
    ]
    for name, res in after_models.items():
        tag = "BEST" if name == best_name else "AFTER"
        rows.append(
            {
                "Status": tag,
                "Model": name,
                "R² Test": res["R2"],
                "MAE": res["MAE"],
                "MAPE (%)": res["MAPE"],
                "RMSE": res["RMSE"],
                "CV R²": res["CV_R2"],
                "Gap": res["Gap"],
            }
        )
    df_cmp = pd.DataFrame(rows)

    def highlight_row(row):
        if row["Status"] == "BEFORE":
            return ["background-color: #fadbd8"] * len(row)
        elif row["Status"] == "BEST":
            return ["background-color: #d5f5e3"] * len(row)
        return [""] * len(row)

    st.dataframe(
        df_cmp.style.apply(highlight_row, axis=1).format(
            {
                c: "{:.6f}"
                for c in ["R² Test", "MAE", "MAPE (%)", "RMSE", "CV R²", "Gap"]
            }
        ),
        use_container_width=True,
    )

    st.success(f"🏆 **Best Model: {best_name}** — CV R² = {best['CV_R2']:.6f}")
