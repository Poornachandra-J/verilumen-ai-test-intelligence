import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

import plotly.express as px


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Verilumen AI Test Intelligence",
    page_icon="🔬",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🔬 Verilumen AI Test Intelligence")

st.markdown(
    """
### Semiconductor ATE Test Analysis, Yield Monitoring,
Failure Prediction & Anomaly Detection
"""
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("📁 Data")

uploaded_file = st.sidebar.file_uploader(
    "Upload ATE CSV",
    type=["csv"]
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def find_column(df, possible_names):

    for name in possible_names:

        if name in df.columns:
            return name

    return None


def clean_result_column(df):

    result_col = find_column(
        df,
        [
            "Result",
            "result",
            "Pass_Fail",
            "PASS_FAIL",
            "Status",
            "status"
        ]
    )

    if result_col is None:
        return df, None

    df[result_col] = (
        df[result_col]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    return df, result_col


def numeric_columns(df):

    return df.select_dtypes(
        include=np.number
    ).columns.tolist()


# ============================================================
# NO FILE YET
# ============================================================

if uploaded_file is None:

    st.info(
        "👈 Upload the assessment CSV from the sidebar to begin."
    )

    st.markdown(
        """
### What this application will do

1. Validate the CSV
2. Analyze data quality
3. Calculate test yield
4. Investigate failures
5. Analyze lots / wafers when available
6. Detect anomalies
7. Train ML models
8. Compare model performance
9. Predict PASS / FAIL
10. Provide engineering observations
"""
    )

    st.stop()


# ============================================================
# LOAD CSV
# ============================================================

try:

    df = pd.read_csv(uploaded_file)

except Exception as error:

    st.error(
        f"Could not read the CSV: {error}"
    )

    st.stop()


# ============================================================
# BASIC CLEANING
# ============================================================

original_rows = len(df)

duplicate_count = int(
    df.duplicated().sum()
)

df = df.drop_duplicates().copy()

missing_values = int(
    df.isna().sum().sum()
)

df, result_col = clean_result_column(df)


# ============================================================
# HEADER INFORMATION
# ============================================================

st.success(
    f"CSV loaded successfully: {len(df):,} records"
)


# ============================================================
# KPI SECTION
# ============================================================

st.subheader("📊 Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Records",
        f"{len(df):,}"
    )

with col2:

    st.metric(
        "Columns",
        f"{len(df.columns):,}"
    )

with col3:

    st.metric(
        "Duplicates Removed",
        f"{duplicate_count:,}"
    )

with col4:

    st.metric(
        "Missing Values",
        f"{missing_values:,}"
    )


# ============================================================
# RESULT ANALYSIS
# ============================================================

if result_col is not None:

    pass_count = int(
        (df[result_col] == "PASS").sum()
    )

    fail_count = int(
        (df[result_col] == "FAIL").sum()
    )

    total_results = pass_count + fail_count

    if total_results > 0:

        yield_rate = (
            pass_count /
            total_results *
            100
        )

        fail_rate = (
            fail_count /
            total_results *
            100
        )

    else:

        yield_rate = 0
        fail_rate = 0

    st.divider()

    st.subheader("🎯 Test Yield")

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "PASS",
            f"{pass_count:,}"
        )

    with c2:

        st.metric(
            "FAIL",
            f"{fail_count:,}"
        )

    with c3:

        st.metric(
            "Yield",
            f"{yield_rate:.2f}%"
        )

    # --------------------------------------------------------
    # PASS / FAIL CHART
    # --------------------------------------------------------

    result_counts = (
        df[result_col]
        .value_counts()
        .reset_index()
    )

    result_counts.columns = [
        "Result",
        "Count"
    ]

    fig = px.bar(
        result_counts,
        x="Result",
        y="Count",
        title="PASS / FAIL Distribution",
        text="Count"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# DATA PREVIEW
# ============================================================

with st.expander("🔍 View Dataset"):

    st.dataframe(
        df.head(100),
        use_container_width=True
    )


# ============================================================
# FAILURE ANALYSIS
# ============================================================

st.divider()

st.subheader("❌ Failure Analysis")

if result_col is not None:

    failures = df[
        df[result_col] == "FAIL"
    ].copy()

    if len(failures) == 0:

        st.success(
            "No FAIL records were found."
        )

    else:

        # ----------------------------------------------------
        # TEST NAME
        # ----------------------------------------------------

        test_col = find_column(
            df,
            [
                "Test_Name",
                "TestName",
                "Test",
                "test_name"
            ]
        )

        if test_col:

            top_tests = (
                failures[test_col]
                .value_counts()
                .head(10)
                .reset_index()
            )

            top_tests.columns = [
                test_col,
                "Failures"
            ]

            fig = px.bar(
                top_tests,
                x="Failures",
                y=test_col,
                orientation="h",
                title="Top Failing Tests"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        # ----------------------------------------------------
        # FAILURE MODE
        # ----------------------------------------------------

        failure_mode_col = find_column(
            df,
            [
                "Failure_Mode",
                "FailureMode",
                "Failure",
                "failure_mode"
            ]
        )

        if failure_mode_col:

            modes = (
                failures[failure_mode_col]
                .value_counts()
                .head(10)
                .reset_index()
            )

            modes.columns = [
                failure_mode_col,
                "Failures"
            ]

            fig = px.bar(
                modes,
                x=failure_mode_col,
                y="Failures",
                title="Common Failure Modes"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


# ============================================================
# LOT ANALYSIS
# ============================================================

st.divider()

st.subheader("🏭 Lot / Wafer Analysis")

lot_col = find_column(
    df,
    [
        "Lot",
        "Lot_ID",
        "Lot_ID",
        "lot",
        "lot_id"
    ]
)

wafer_col = find_column(
    df,
    [
        "Wafer",
        "Wafer_ID",
        "WaferID",
        "wafer",
        "wafer_id"
    ]
)


if result_col is not None and lot_col:

    lot_summary = (
        df.groupby(lot_col)[result_col]
        .apply(
            lambda x:
            (x == "PASS").mean() * 100
        )
        .reset_index(
            name="Yield"
        )
        .sort_values(
            "Yield"
        )
    )

    st.markdown(
        "### Yield by Lot"
    )

    fig = px.bar(
        lot_summary.head(20),
        x=lot_col,
        y="Yield",
        title="Lowest-Yield Lots"
    )

    fig.update_yaxes(
        title="Yield (%)"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


if result_col is not None and wafer_col:

    wafer_summary = (
        df.groupby(wafer_col)[result_col]
        .apply(
            lambda x:
            (x == "PASS").mean() * 100
        )
        .reset_index(
            name="Yield"
        )
        .sort_values(
            "Yield"
        )
    )

    st.markdown(
        "### Yield by Wafer"
    )

    fig = px.bar(
        wafer_summary.head(20),
        x=wafer_col,
        y="Yield",
        title="Lowest-Yield Wafers"
    )

    fig.update_yaxes(
        title="Yield (%)"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# NUMERIC DATA DISTRIBUTIONS
# ============================================================

st.divider()

st.subheader("📈 Numeric Test Parameters")

numbers = numeric_columns(df)

if len(numbers) > 0:

    selected_numeric = st.selectbox(
        "Select a numeric parameter",
        numbers
    )

    fig = px.histogram(
        df,
        x=selected_numeric,
        nbins=40,
        title=f"Distribution of {selected_numeric}"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

else:

    st.info(
        "No numeric columns were detected."
    )


# ============================================================
# ANOMALY DETECTION
# ============================================================

st.divider()

st.subheader("🚨 Anomaly Detection")

anomaly_features = numbers.copy()

if len(anomaly_features) >= 2:

    anomaly_data = df[
        anomaly_features
    ].copy()

    anomaly_data = anomaly_data.apply(
        pd.to_numeric,
        errors="coerce"
    )

    anomaly_data = anomaly_data.replace(
        [np.inf, -np.inf],
        np.nan
    )

    anomaly_data = anomaly_data.fillna(
        anomaly_data.median()
    )

    anomaly_data = anomaly_data.fillna(0)

    detector = IsolationForest(
        contamination="auto",
        random_state=42
    )

    detector.fit(
        anomaly_data
    )

    df["Anomaly_Score"] = (
        -detector.decision_function(
            anomaly_data
        )
    )

    df["Anomaly"] = (
        detector.predict(
            anomaly_data
        ) == -1
    )

    anomaly_count = int(
        df["Anomaly"].sum()
    )

    st.metric(
        "Potential Anomalies",
        anomaly_count
    )

    suspicious = (
        df.sort_values(
            "Anomaly_Score",
            ascending=False
        )
        .head(20)
    )

    st.dataframe(
        suspicious,
        use_container_width=True
    )

else:

    st.info(
        "At least two numeric columns are needed for anomaly detection."
    )


# ============================================================
# MACHINE LEARNING
# ============================================================

st.divider()

st.subheader("🤖 Failure Prediction")

if result_col is None:

    st.warning(
        "A PASS/FAIL result column was not detected, so ML training cannot start."
    )

else:

    ml_df = df[
        df[result_col].isin(
            ["PASS", "FAIL"]
        )
    ].copy()

    # --------------------------------------------------------
    # Select numeric features
    # --------------------------------------------------------

    ml_features = [
        col
        for col in numbers
        if col != result_col
    ]

    # Remove generated anomaly columns
    ml_features = [
        col
        for col in ml_features
        if col not in [
            "Anomaly",
            "Anomaly_Score"
        ]
    ]

    if len(ml_features) < 2:

        st.warning(
            "Not enough numeric features for machine learning."
        )

    elif ml_df[result_col].nunique() < 2:

        st.warning(
            "Both PASS and FAIL records are required."
        )

    else:

        X = ml_df[
            ml_features
        ]

        y = (
            ml_df[result_col] == "FAIL"
        ).astype(int)

        # ----------------------------------------------------
        # Train / Test
        # ----------------------------------------------------

        X_train, X_test, y_train, y_test = (
            train_test_split(
                X,
                y,
                test_size=0.20,
                random_state=42,
                stratify=y
            )
        )

        # ----------------------------------------------------
        # Logistic Regression
        # ----------------------------------------------------

        logistic_model = Pipeline(
            [
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
                        max_iter=1000,
                        class_weight="balanced"
                    )
                )
            ]
        )

        logistic_model.fit(
            X_train,
            y_train
        )

        logistic_pred = (
            logistic_model.predict(
                X_test
            )
        )

        logistic_prob = (
            logistic_model.predict_proba(
                X_test
            )[:, 1]
        )

        # ----------------------------------------------------
        # Random Forest
        # ----------------------------------------------------

        forest_model = Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    )
                ),

                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=200,
                        random_state=42,
                        class_weight="balanced"
                    )
                )
            ]
        )

        forest_model.fit(
            X_train,
            y_train
        )

        forest_pred = (
            forest_model.predict(
                X_test
            )
        )

        forest_prob = (
            forest_model.predict_proba(
                X_test
            )[:, 1]
        )

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        def calculate_metrics(
            name,
            prediction,
            probability
        ):

            return {
                "Model": name,

                "Accuracy":
                    accuracy_score(
                        y_test,
                        prediction
                    ),

                "Precision":
                    precision_score(
                        y_test,
                        prediction,
                        zero_division=0
                    ),

                "Recall":
                    recall_score(
                        y_test,
                        prediction,
                        zero_division=0
                    ),

                "F1":
                    f1_score(
                        y_test,
                        prediction,
                        zero_division=0
                    ),

                "ROC-AUC":
                    roc_auc_score(
                        y_test,
                        probability
                    )
            }

        metrics = pd.DataFrame(
            [
                calculate_metrics(
                    "Logistic Regression",
                    logistic_pred,
                    logistic_prob
                ),

                calculate_metrics(
                    "Random Forest",
                    forest_pred,
                    forest_prob
                )
            ]
        )

        metrics_display = metrics.copy()

        for column in [
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "ROC-AUC"
        ]:

            metrics_display[column] = (
                metrics_display[column]
                .round(3)
            )

        st.dataframe(
            metrics_display,
            use_container_width=True
        )

        # ----------------------------------------------------
        # Best Model
        # ----------------------------------------------------

        best_index = (
            metrics["F1"]
            .idxmax()
        )

        best_model_name = (
            metrics.loc[
                best_index,
                "Model"
            ]
        )

        if best_model_name == "Random Forest":

            best_model = forest_model

        else:

            best_model = logistic_model

        st.success(
            f"Best baseline model by F1: {best_model_name}"
        )

        # ----------------------------------------------------
        # Confusion Matrix
        # ----------------------------------------------------

        if best_model_name == "Random Forest":

            best_predictions = forest_pred

        else:

            best_predictions = logistic_pred

        matrix = confusion_matrix(
            y_test,
            best_predictions
        )

        matrix_df = pd.DataFrame(
            matrix,
            index=["Actual PASS", "Actual FAIL"],
            columns=["Predicted PASS", "Predicted FAIL"]
        )

        st.markdown(
            "### Confusion Matrix"
        )

        st.dataframe(
            matrix_df,
            use_container_width=True
        )

        # ----------------------------------------------------
        # Feature Importance
        # ----------------------------------------------------

        if best_model_name == "Random Forest":

            importance = (
                best_model
                .named_steps["model"]
                .feature_importances_
            )

            importance_df = pd.DataFrame(
                {
                    "Feature": ml_features,
                    "Importance": importance
                }
            ).sort_values(
                "Importance",
                ascending=False
            )

            st.markdown(
                "### Feature Influence"
            )

            fig = px.bar(
                importance_df,
                x="Importance",
                y="Feature",
                orientation="h",
                title="Random Forest Feature Importance"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


# ============================================================
# ENGINEERING OBSERVATIONS
# ============================================================

st.divider()

st.subheader("🧠 Engineering Observations")

observations = []

if result_col is not None:

    observations.append(
        f"Detected result column: `{result_col}`."
    )

    if fail_count > 0:

        observations.append(
            f"{fail_count:,} FAIL records were identified."
        )

    if yield_rate < 95:

        observations.append(
            f"Overall yield is {yield_rate:.2f}%, which indicates "
            "a relatively high failure rate and deserves investigation."
        )

    else:

        observations.append(
            f"Overall yield is {yield_rate:.2f}%."
        )

if duplicate_count > 0:

    observations.append(
        f"{duplicate_count:,} exact duplicate rows were removed "
        "during basic data cleaning."
    )

if missing_values > 0:

    observations.append(
        f"The dataset contains {missing_values:,} missing values."
    )

for observation in observations:

    st.write(
        "• " + observation
    )


st.divider()

st.caption(
    "Verilumen AI Test Intelligence | "
    "Starter engineering application"
)
