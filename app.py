import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Verilumen AI Test Intelligence",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM DASHBOARD DESIGN
# ============================================================

st.markdown("""
<style>

.stApp {
    background-color: #f5f8fc;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}

.hero {
    background: linear-gradient(
        110deg,
        #102a43,
        #176b87,
        #16a085
    );
    padding: 30px;
    border-radius: 18px;
    color: white;
    margin-bottom: 25px;
}

.hero h1 {
    color: white;
    font-size: 35px;
}

.hero p {
    color: #e2f3f5;
    font-size: 16px;
}

div[data-testid="stMetric"] {
    background: white;
    padding: 18px;
    border-radius: 13px;
    border: 1px solid #e1e8f0;
    box-shadow: 0 3px 10px rgba(0,0,0,0.04);
}

section[data-testid="stSidebar"] {
    background-color: #edf3f8;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# COLOR SETTINGS
# ============================================================

COLORS = {
    "PASS": "#16a085",
    "FAIL": "#e05d5d",
    "Other": "#64748b"
}

TEMPLATE = "plotly_white"


def style_chart(fig, height=400):

    fig.update_layout(
        template=TEMPLATE,
        height=height,
        margin=dict(l=20, r=20, t=60, b=20),
        title_font=dict(
            size=18,
            color="#18324b"
        ),
        legend_title_text="",
        hovermode="closest"
    )

    return fig


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def find_column(df, candidates):

    lookup = {
        str(c).strip().lower(): c
        for c in df.columns
    }

    for candidate in candidates:

        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    return None


def clean_result(df):

    result_col = find_column(
        df,
        [
            "Result",
            "result",
            "Pass_Fail",
            "PASS_FAIL",
            "Status",
            "Test_Result"
        ]
    )

    if result_col:

        df[result_col] = (
            df[result_col]
            .astype("string")
            .str.strip()
            .str.upper()
        )

        df[result_col] = df[result_col].replace({
            "1": "PASS",
            "0": "FAIL",
            "GOOD": "PASS",
            "BAD": "FAIL",
            "TRUE": "PASS",
            "FALSE": "FAIL"
        })

    return df, result_col


def yield_summary(data, group_col, result_col):

    valid = data[
        data[result_col].isin(["PASS", "FAIL"])
    ]

    summary = (
        valid.groupby(group_col)[result_col]
        .apply(
            lambda x: (x == "PASS").mean() * 100
        )
        .reset_index(name="Yield")
    )

    return summary.sort_values("Yield")


def download_csv(data, filename, label):

    st.download_button(
        label=label,
        data=data.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
        use_container_width=True
    )


# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="hero">

<h1>🔬 Verilumen AI Test Intelligence</h1>

<p>
Semiconductor ATE Analytics |
Yield Monitoring |
Failure Prediction |
Anomaly Detection |
Machine Learning
</p>

</div>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("📁 Data Management")

uploaded_file = st.sidebar.file_uploader(
    "Upload Semiconductor ATE CSV",
    type=["csv"]
)

st.sidebar.divider()

st.sidebar.markdown("### 📊 Dashboard Modules")

st.sidebar.markdown("""
- Executive Overview
- Yield Analysis
- Failure Investigation
- Lot Analysis
- Wafer Analysis
- Parameter Analysis
- Correlation Analysis
- Anomaly Detection
- Machine Learning
- Data Export
""")

st.sidebar.divider()

st.sidebar.caption(
    "Verilumen AI Test Intelligence"
)


# ============================================================
# WAITING FOR CSV
# ============================================================

if uploaded_file is None:

    st.info(
        "👈 Upload your ATE CSV file using the sidebar."
    )

    st.markdown("## 🚀 Dashboard Features")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("### 📊 Analytics")
        st.write(
            "Interactive bar charts, pie charts, "
            "histograms and box plots."
        )

    with c2:
        st.markdown("### 🤖 Machine Learning")
        st.write(
            "PASS/FAIL prediction and model comparison."
        )

    with c3:
        st.markdown("### 🚨 Anomaly Detection")
        st.write(
            "Identify unusual test measurements "
            "using Isolation Forest."
        )

    st.stop()


# ============================================================
# LOAD DATA
# ============================================================

try:

    df = pd.read_csv(uploaded_file)

except Exception as error:

    st.error(f"CSV loading error: {error}")

    st.stop()


original_rows = len(df)

duplicate_count = int(
    df.duplicated().sum()
)

df = df.drop_duplicates().copy()

df, result_col = clean_result(df)

numeric_columns = (
    df.select_dtypes(include=np.number)
    .columns.tolist()
)


# ============================================================
# SIDEBAR FILTERS
# ============================================================

st.sidebar.markdown("### 🔎 Data Filters")

filtered = df.copy()

lot_col = find_column(
    df,
    ["Lot", "Lot_ID", "lot_id"]
)

wafer_col = find_column(
    df,
    ["Wafer", "Wafer_ID", "WaferID"]
)

test_col = find_column(
    df,
    ["Test_Name", "TestName", "Test"]
)

failure_col = find_column(
    df,
    ["Failure_Mode", "FailureMode", "Failure"]
)

if lot_col:

    lot_options = sorted(
        df[lot_col].dropna().astype(str).unique()
    )

    selected_lots = st.sidebar.multiselect(
        "Select Lots",
        lot_options,
        default=lot_options
    )

    filtered = filtered[
        filtered[lot_col].astype(str).isin(selected_lots)
    ]


if wafer_col:

    wafer_options = sorted(
        df[wafer_col].dropna().astype(str).unique()
    )

    selected_wafers = st.sidebar.multiselect(
        "Select Wafers",
        wafer_options,
        default=wafer_options
    )

    filtered = filtered[
        filtered[wafer_col].astype(str).isin(selected_wafers)
    ]


if test_col:

    test_options = sorted(
        df[test_col].dropna().astype(str).unique()
    )

    selected_tests = st.sidebar.multiselect(
        "Select Tests",
        test_options,
        default=test_options
    )

    filtered = filtered[
        filtered[test_col].astype(str).isin(selected_tests)
    ]


if result_col:

    result_options = [
        x for x in ["PASS", "FAIL"]
        if x in df[result_col].dropna().unique()
    ]

    selected_results = st.sidebar.multiselect(
        "Select Result",
        result_options,
        default=result_options
    )

    filtered = filtered[
        filtered[result_col].isin(selected_results)
    ]


if filtered.empty:

    st.warning(
        "No records match your filters. "
        "Please change the sidebar selections."
    )

    st.stop()


# ============================================================
# EXECUTIVE OVERVIEW
# ============================================================

st.divider()

st.subheader("📊 Executive Overview")

pass_count = 0
fail_count = 0

if result_col:

    pass_count = int(
        (filtered[result_col] == "PASS").sum()
    )

    fail_count = int(
        (filtered[result_col] == "FAIL").sum()
    )

total_results = pass_count + fail_count

yield_rate = (
    pass_count / total_results * 100
    if total_results > 0
    else 0
)

missing_values = int(
    filtered.isna().sum().sum()
)

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric(
    "Total Records",
    f"{len(filtered):,}"
)

c2.metric(
    "PASS Records",
    f"{pass_count:,}"
)

c3.metric(
    "FAIL Records",
    f"{fail_count:,}"
)

c4.metric(
    "Test Yield",
    f"{yield_rate:.2f}%"
)

c5.metric(
    "Missing Values",
    f"{missing_values:,}"
)

st.caption(
    f"Original records: {original_rows:,} | "
    f"Duplicates removed: {duplicate_count:,}"
)


# ============================================================
# PASS FAIL ANALYSIS
# ============================================================

st.divider()

st.subheader("🎯 PASS / FAIL Analysis")

if result_col and total_results > 0:

    result_counts = (
        filtered[result_col]
        .value_counts()
        .rename_axis("Result")
        .reset_index(name="Records")
    )

    col1, col2 = st.columns(2)

    with col1:

        fig = px.bar(
            result_counts,
            x="Result",
            y="Records",
            color="Result",
            color_discrete_map=COLORS,
            text="Records",
            title="PASS vs FAIL Distribution"
        )

        fig.update_traces(
            textposition="outside"
        )

        st.plotly_chart(
            style_chart(fig),
            use_container_width=True
        )

    with col2:

        fig = px.pie(
            result_counts,
            names="Result",
            values="Records",
            hole=0.55,
            color="Result",
            color_discrete_map=COLORS,
            title="PASS / FAIL Percentage"
        )

        fig.update_traces(
            textinfo="percent+label"
        )

        st.plotly_chart(
            style_chart(fig),
            use_container_width=True
        )

else:

    st.info("PASS/FAIL data is not available.")


# ============================================================
# LOT ANALYSIS
# ============================================================

st.divider()

st.subheader("🏭 Lot Yield Analysis")

if result_col and lot_col:

    lot_data = yield_summary(
        filtered,
        lot_col,
        result_col
    )

    fig = px.bar(
        lot_data,
        x="Yield",
        y=lot_col,
        orientation="h",
        color="Yield",
        color_continuous_scale="Tealgrn",
        title="Yield Percentage by Lot",
        text=lot_data["Yield"].round(2)
    )

    fig.update_layout(
        yaxis={"categoryorder": "total ascending"},
        xaxis_title="Yield (%)"
    )

    st.plotly_chart(
        style_chart(fig),
        use_container_width=True
    )

    with st.expander("View Lot Yield Table"):

        st.dataframe(
            lot_data,
            use_container_width=True,
            hide_index=True
        )

else:

    st.info("Lot ID or PASS/FAIL column not detected.")


# ============================================================
# WAFER ANALYSIS
# ============================================================

st.divider()

st.subheader("💿 Wafer Yield Analysis")

if result_col and wafer_col:

    wafer_data = yield_summary(
        filtered,
        wafer_col,
        result_col
    )

    fig = px.bar(
        wafer_data,
        x=wafer_col,
        y="Yield",
        color="Yield",
        color_continuous_scale="Viridis",
        title="Yield by Wafer"
    )

    fig.update_layout(
        xaxis_title="Wafer ID",
        yaxis_title="Yield (%)"
    )

    st.plotly_chart(
        style_chart(fig),
        use_container_width=True
    )

    with st.expander("View Wafer Yield Table"):

        st.dataframe(
            wafer_data,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# TEST ANALYSIS
# ============================================================

st.divider()

st.subheader("🧪 Test Analysis")

if test_col:

    test_counts = (
        filtered[test_col]
        .value_counts()
        .rename_axis("Test")
        .reset_index(name="Count")
    )

    fig = px.bar(
        test_counts,
        x="Count",
        y="Test",
        orientation="h",
        text="Count",
        color="Count",
        color_continuous_scale="Blues",
        title="Test Execution Distribution"
    )

    fig.update_layout(
        yaxis={"categoryorder": "total ascending"}
    )

    st.plotly_chart(
        style_chart(fig),
        use_container_width=True
    )

    if result_col:

        test_failure = (
            filtered.assign(
                _fail=(filtered[result_col] == "FAIL")
            )
            .groupby(test_col)["_fail"]
            .mean()
            .mul(100)
            .reset_index(name="Failure_Rate")
        )

        fig = px.bar(
            test_failure,
            x=test_col,
            y="Failure_Rate",
            color="Failure_Rate",
            color_continuous_scale="Reds",
            title="Failure Rate by Test"
        )

        fig.update_layout(
            yaxis_title="Failure Rate (%)"
        )

        st.plotly_chart(
            style_chart(fig),
            use_container_width=True
        )


# ============================================================
# FAILURE MODE ANALYSIS
# ============================================================

st.divider()

st.subheader("❌ Failure Mode Analysis")

if result_col and failure_col:

    failures = filtered[
        filtered[result_col] == "FAIL"
    ]

    if not failures.empty:

        modes = (
            failures[failure_col]
            .fillna("Unspecified")
            .value_counts()
            .rename_axis("Failure Mode")
            .reset_index(name="Count")
        )

        col1, col2 = st.columns(2)

        with col1:

            fig = px.bar(
                modes,
                x="Count",
                y="Failure Mode",
                orientation="h",
                color="Count",
                color_continuous_scale="Reds",
                title="Top Failure Modes"
            )

            fig.update_layout(
                yaxis={"categoryorder": "total ascending"}
            )

            st.plotly_chart(
                style_chart(fig),
                use_container_width=True
            )

        with col2:

            fig = px.pie(
                modes,
                names="Failure Mode",
                values="Count",
                hole=0.45,
                title="Failure Mode Composition"
            )

            st.plotly_chart(
                style_chart(fig),
                use_container_width=True
            )

    else:

        st.success("No failures in the current filtered data.")

else:

    st.info("Failure mode column was not detected.")


# ============================================================
# NUMERIC PARAMETER ANALYSIS
# ============================================================

st.divider()

st.subheader("📈 Numeric Parameter Analysis")

available_numeric = [
    col for col in numeric_columns
    if col != result_col
]

if available_numeric:

    selected_parameter = st.selectbox(
        "Select Test Parameter",
        available_numeric
    )

    col1, col2 = st.columns(2)

    with col1:

        fig = px.histogram(
            filtered,
            x=selected_parameter,
            color=result_col if result_col else None,
            marginal="box",
            nbins=40,
            barmode="overlay",
            opacity=0.75,
            title=f"{selected_parameter} Distribution"
        )

        st.plotly_chart(
            style_chart(fig),
            use_container_width=True
        )

    with col2:

        if result_col:

            fig = px.box(
                filtered,
                x=result_col,
                y=selected_parameter,
                color=result_col,
                color_discrete_map=COLORS,
                points="outliers",
                title=f"{selected_parameter} by Result"
            )

        else:

            fig = px.box(
                filtered,
                y=selected_parameter,
                points="outliers",
                title=f"{selected_parameter} Box Plot"
            )

        st.plotly_chart(
            style_chart(fig),
            use_container_width=True
        )


# ============================================================
# SCATTER PLOT
# ============================================================

if len(available_numeric) >= 2:

    st.subheader("🔬 Parameter Relationship")

    col1, col2 = st.columns(2)

    with col1:

        x_parameter = st.selectbox(
            "X Axis",
            available_numeric,
            index=0
        )

    with col2:

        y_parameter = st.selectbox(
            "Y Axis",
            available_numeric,
            index=1
        )

    fig = px.scatter(
        filtered,
        x=x_parameter,
        y=y_parameter,
        color=result_col if result_col else None,
        color_discrete_map=COLORS,
        opacity=0.75,
        title=f"{y_parameter} vs {x_parameter}"
    )

    st.plotly_chart(
        style_chart(fig),
        use_container_width=True
    )


# ============================================================
# CORRELATION HEATMAP
# ============================================================

st.divider()

st.subheader("🌡️ Correlation Heatmap")

if len(available_numeric) >= 2:

    correlation = filtered[
        available_numeric
    ].corr(numeric_only=True)

    fig = px.imshow(
        correlation,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Test Parameter Correlation"
    )

    st.plotly_chart(
        style_chart(fig, 500),
        use_container_width=True
    )


# ============================================================
# TIME TREND
# ============================================================

timestamp_col = find_column(
    filtered,
    ["Timestamp", "DateTime", "Date", "Test_Time"]
)

if timestamp_col and result_col:

    st.divider()

    st.subheader("🕒 Yield Trend Analysis")

    trend = filtered.copy()

    trend[timestamp_col] = pd.to_datetime(
        trend[timestamp_col],
        errors="coerce"
    )

    trend = trend.dropna(
        subset=[timestamp_col]
    )

    if not trend.empty:

        trend["Period"] = (
            trend[timestamp_col].dt.floor("D")
        )

        daily = (
            trend.assign(
                _pass=(trend[result_col] == "PASS")
            )
            .groupby("Period")["_pass"]
            .mean()
            .mul(100)
            .reset_index(name="Yield")
        )

        fig = px.line(
            daily,
            x="Period",
            y="Yield",
            markers=True,
            title="Daily Test Yield Trend"
        )

        fig.update_layout(
            yaxis_title="Yield (%)",
            yaxis_range=[0, 100]
        )

        st.plotly_chart(
            style_chart(fig),
            use_container_width=True
        )


# ============================================================
# DATA QUALITY
# ============================================================

st.divider()

st.subheader("🧹 Data Quality Analysis")

quality = pd.DataFrame({

    "Column": filtered.columns,

    "Data Type": [
        str(filtered[c].dtype)
        for c in filtered.columns
    ],

    "Missing Values": [
        int(filtered[c].isna().sum())
        for c in filtered.columns
    ],

    "Missing Percentage": [
        round(
            filtered[c].isna().mean() * 100,
            2
        )
        for c in filtered.columns
    ],

    "Unique Values": [
        int(filtered[c].nunique())
        for c in filtered.columns
    ]

})

st.dataframe(
    quality,
    use_container_width=True,
    hide_index=True
)

fig = px.bar(
    quality.sort_values(
        "Missing Percentage",
        ascending=False
    ),
    x="Missing Percentage",
    y="Column",
    orientation="h",
    color="Missing Percentage",
    color_continuous_scale="OrRd",
    title="Missing Values by Column"
)

fig.update_layout(
    yaxis={"categoryorder": "total ascending"}
)

st.plotly_chart(
    style_chart(fig),
    use_container_width=True
)


# ============================================================
# ANOMALY DETECTION
# ============================================================

st.divider()

st.subheader("🚨 AI Anomaly Detection")

if len(available_numeric) >= 2 and len(filtered) >= 10:

    contamination = st.slider(
        "Anomaly Detection Sensitivity",
        min_value=0.01,
        max_value=0.20,
        value=0.05,
        step=0.01
    )

    anomaly_data = (
        filtered[available_numeric]
        .replace([np.inf, -np.inf], np.nan)
    )

    imputer = SimpleImputer(
        strategy="median"
    )

    anomaly_matrix = imputer.fit_transform(
        anomaly_data
    )

    detector = IsolationForest(
        contamination=contamination,
        random_state=42
    )

    flags = detector.fit_predict(
        anomaly_matrix
    )

    anomaly_view = filtered.copy()

    anomaly_view["Anomaly_Status"] = np.where(
        flags == -1,
        "Review",
        "Typical"
    )

    anomaly_view["Anomaly_Score"] = (
        -detector.decision_function(
            anomaly_matrix
        )
    )

    anomaly_count = int(
        (flags == -1).sum()
    )

    c1, c2 = st.columns(2)

    c1.metric(
        "Potential Anomalies",
        f"{anomaly_count:,}"
    )

    c2.metric(
        "Typical Records",
        f"{len(filtered) - anomaly_count:,}"
    )

    fig = px.scatter(
        anomaly_view,
        x=available_numeric[0],
        y=available_numeric[1],
        color="Anomaly_Status",
        title="Anomaly Detection Map"
    )

    st.plotly_chart(
        style_chart(fig),
        use_container_width=True
    )

    st.markdown("### Records Requiring Review")

    suspicious = (
        anomaly_view[
            anomaly_view["Anomaly_Status"] == "Review"
        ]
        .sort_values(
            "Anomaly_Score",
            ascending=False
        )
    )

    st.dataframe(
        suspicious.head(100),
        use_container_width=True
    )

    download_csv(
        anomaly_view,
        "ate_anomaly_results.csv",
        "⬇️ Download Anomaly Results"
    )

else:

    st.info(
        "At least 10 records and 2 numeric columns "
        "are required for anomaly detection."
    )


# ============================================================
# MACHINE LEARNING
# ============================================================

st.divider()

st.subheader("🤖 Machine Learning Failure Prediction")

if result_col and len(available_numeric) >= 2:

    ml_df = filtered[
        filtered[result_col].isin(
            ["PASS", "FAIL"]
        )
    ].copy()

    features = [
        col for col in available_numeric
        if col in ml_df.columns
    ]

    if (
        len(features) >= 2
        and ml_df[result_col].nunique() == 2
        and len(ml_df) >= 30
    ):

        X = ml_df[features].replace(
            [np.inf, -np.inf],
            np.nan
        )

        y = (
            ml_df[result_col] == "FAIL"
        ).astype(int)

        if y.value_counts().min() >= 2:

            X_train, X_test, y_train, y_test = (
                train_test_split(
                    X,
                    y,
                    test_size=0.20,
                    random_state=42,
                    stratify=y
                )
            )

            models = {

                "Logistic Regression": Pipeline([

                    (
                        "imputer",
                        SimpleImputer(
                            strategy="median"
                        )
                    ),

                    (
                        "scaler",
                        StandardScaler()
                    ),

                    (
                        "model",
                        LogisticRegression(
                            max_iter=1500,
                            class_weight="balanced"
                        )
                    )

                ]),

                "Random Forest": Pipeline([

                    (
                        "imputer",
                        SimpleImputer(
                            strategy="median"
                        )
                    ),

                    (
                        "model",
                        RandomForestClassifier(
                            n_estimators=250,
                            random_state=42,
                            class_weight="balanced",
                            min_samples_leaf=2
                        )
                    )

                ])

            }

            metrics = []
            predictions = {}

            for name, model in models.items():

                model.fit(
                    X_train,
                    y_train
                )

                pred = model.predict(
                    X_test
                )

                probability = (
                    model.predict_proba(X_test)[:, 1]
                )

                predictions[name] = pred

                metrics.append({

                    "Model": name,

                    "Accuracy": accuracy_score(
                        y_test,
                        pred
                    ),

                    "Precision": precision_score(
                        y_test,
                        pred,
                        zero_division=0
                    ),

                    "Recall": recall_score(
                        y_test,
                        pred,
                        zero_division=0
                    ),

                    "F1 Score": f1_score(
                        y_test,
                        pred,
                        zero_division=0
                    ),

                    "ROC AUC": roc_auc_score(
                        y_test,
                        probability
                    )

                })

            metrics_df = pd.DataFrame(
                metrics
            )

            st.markdown(
                "### Model Performance Comparison"
            )

            st.dataframe(
                metrics_df.style.format({
                    "Accuracy": "{:.3f}",
                    "Precision": "{:.3f}",
                    "Recall": "{:.3f}",
                    "F1 Score": "{:.3f}",
                    "ROC AUC": "{:.3f}"
                }),
                use_container_width=True,
                hide_index=True
            )

            fig = px.bar(
                metrics_df.melt(
                    id_vars="Model",
                    var_name="Metric",
                    value_name="Score"
                ),
                x="Metric",
                y="Score",
                color="Model",
                barmode="group",
                range_y=[0, 1],
                title="Machine Learning Model Comparison"
            )

            st.plotly_chart(
                style_chart(fig),
                use_container_width=True
            )

            # Confusion matrix

            selected_model = st.selectbox(
                "Select Model for Confusion Matrix",
                list(models.keys())
            )

            cm = confusion_matrix(
                y_test,
                predictions[selected_model],
                labels=[0, 1]
            )

            fig = px.imshow(
                cm,
                text_auto=True,
                x=["Predicted PASS", "Predicted FAIL"],
                y=["Actual PASS", "Actual FAIL"],
                color_continuous_scale="Blues",
                title="Confusion Matrix"
            )

            st.plotly_chart(
                style_chart(fig, 350),
                use_container_width=True
            )

            # Random Forest Feature Importance

            if selected_model == "Random Forest":

                importance = (
                    models[selected_model]
                    .named_steps["model"]
                    .feature_importances_
                )

                importance_df = pd.DataFrame({

                    "Feature": features,

                    "Importance": importance

                }).sort_values(
                    "Importance"
                )

                fig = px.bar(
                    importance_df,
                    x="Importance",
                    y="Feature",
                    orientation="h",
                    title="Feature Importance"
                )

                st.plotly_chart(
                    style_chart(fig),
                    use_container_width=True
                )

            st.caption(
                "Model metrics are calculated on a held-out "
                "test split. These are baseline results, "
                "not production qualification."
            )

        else:

            st.warning(
                "Not enough examples in both PASS and FAIL classes."
            )

    else:

        st.warning(
            "At least 30 records, two numeric features, "
            "and both PASS and FAIL classes are required."
        )

else:

    st.info(
        "A PASS/FAIL result column and numeric test "
        "parameters are required for ML."
    )


# ============================================================
# DATA EXPORT
# ============================================================

st.divider()

st.subheader("📥 Download Your Analysis")

col1, col2 = st.columns(2)

with col1:

    download_csv(
        filtered,
        "verilumen_filtered_data.csv",
        "⬇️ Download Filtered Dataset"
    )

with col2:

    download_csv(
        df,
        "verilumen_cleaned_data.csv",
        "⬇️ Download Full Cleaned Dataset"
    )


# ============================================================
# DATA PREVIEW
# ============================================================

st.divider()

st.subheader("🗂️ Dataset Explorer")

preview_rows = st.slider(
    "Number of rows to display",
    min_value=10,
    max_value=min(500, len(filtered)),
    value=min(100, len(filtered)),
    step=10
)

st.dataframe(
    filtered.head(preview_rows),
    use_container_width=True,
    hide_index=True
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown("""
<center>

### 🔬 Verilumen AI Test Intelligence

Semiconductor ATE Analytics and Machine Learning

<small>
Yield Monitoring | Failure Analysis | Anomaly Detection |
Predictive Analytics
</small>

</center>
""", unsafe_allow_html=True)
