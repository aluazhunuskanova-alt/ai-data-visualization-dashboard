import pandas as pd
import plotly.express as px
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error


# ------------------ PLOTS ------------------
def generate_plots(df):
    plots = {}

    df.columns = df.columns.str.strip()

    # Scatter
    if "GDP per capita" in df.columns and "Score" in df.columns:
        clean_df = df[["GDP per capita", "Score"]].dropna()

        fig = px.scatter(
            clean_df,
            x="GDP per capita",
            y="Score",
            color="Score",
            title="GDP vs Happiness Score",
            trendline="ols"
        )

        fig.update_layout(template="plotly_white")
        plots["scatter"] = fig.to_html(full_html=False)

    # Histogram
    if "Score" in df.columns:
        fig = px.histogram(
            df,
            x="Score",
            nbins=20,
            title="Happiness Score Distribution"
        )

        fig.update_layout(template="plotly_white")
        plots["histogram"] = fig.to_html(full_html=False)

    # Bar chart (TOP 10)
    if "Country or region" in df.columns and "Score" in df.columns:
        top = df.sort_values("Score", ascending=False).head(10)

        fig = px.bar(
            top,
            x="Country or region",
            y="Score",
            color="Score",
            title="Top 10 Happiest Countries"
        )

        fig.update_layout(template="plotly_white")
        plots["bar"] = fig.to_html(full_html=False)

    # Box plot
    if "Score" in df.columns:
        fig = px.box(
            df,
            y="Score",
            points="all",
            title="Score Distribution"
        )

        fig.update_layout(template="plotly_white")
        plots["box"] = fig.to_html(full_html=False)

    return plots


# ------------------ HEATMAP ------------------
def generate_correlation_heatmap(df):
    numeric_df = df.select_dtypes(include=["number"])

    if numeric_df.shape[1] < 2:
        return None

    corr = numeric_df.corr()

    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        title="Correlation Heatmap"
    )

    fig.update_layout(template="plotly_white")

    return fig.to_html(full_html=False)


# ------------------ INSIGHTS ------------------
def generate_insights(df):
    insights = []

    numeric_df = df.select_dtypes(include=["number"])

    if numeric_df.shape[1] >= 2:
        corr = numeric_df.corr()
        max_corr = corr.abs().unstack().sort_values(ascending=False)
        max_corr = max_corr[max_corr < 1].head(1)

        for (c1, c2), v in max_corr.items():
            insights.append(f"Strong relationship between {c1} and {c2} (r={v:.2f})")

    for col in numeric_df.columns:
        insights.append(f"Average {col}: {df[col].mean():.2f}")

    return insights


# ------------------ ML MODEL ------------------
def train_model(df):
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import r2_score, mean_squared_error
    df = df.copy()

    # 🔥 FEATURE ENGINEERING (CORRECT PLACE)
    if "GDP per capita" in df.columns:
        df["log_gdp"] = np.log1p(df["GDP per capita"])

    if "Social support" in df.columns and "Healthy life expectancy" in df.columns:
        df["support_x_health"] = df["Social support"] * df["Healthy life expectancy"]

    if "Freedom to make life choices" in df.columns and "Perceptions of corruption" in df.columns:
        df["freedom_x_corruption"] = (
            df["Freedom to make life choices"] * df["Perceptions of corruption"]
        )

    # Features
    features = [
        "log_gdp",
        "Social support",
        "Healthy life expectancy",
        "Freedom to make life choices",
        "Generosity",
        "Perceptions of corruption",
        "support_x_health",
        "freedom_x_corruption"
    ]

    features = [f for f in features if f in df.columns]

    if "Score" not in df.columns or len(features) < 2:
        return {"error": "Not enough data"}

    model_df = df[features + ["Score"]].dropna()

    X = model_df[features]
    y = model_df["Score"]

    from sklearn.ensemble import RandomForestRegressor
    

    model = RandomForestRegressor(
    n_estimators=1000,
    max_depth=None,
    random_state=42
)


    model.fit(X, y)

    y_pred = model.predict(X)
    
    model.fit(X, y)
    y_pred = model.predict(X)
    

    return {
        "r2_score": round(r2_score(y, y_pred), 3),
        "mse": round(mean_squared_error(y, y_pred), 3),
        "coefficients": dict(zip(features, model.feature_importances_))
    }


# ------------------ MAIN ------------------
def analyze_data(df: pd.DataFrame):
    df.columns = df.columns.str.strip()

    summary = {
        "shape": df.shape,
        "columns": list(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "describe": df.select_dtypes(include=["number"]).describe().to_dict()
    }

    plots = generate_plots(df)
    heatmap = generate_correlation_heatmap(df)
    insights = generate_insights(df)
    model_results = train_model(df)

    return {
        "summary": summary,
        "preview": df.head(20).to_dict(orient="records"),
        "plots": plots,
        "correlation_heatmap": heatmap,
        "insights": insights,
        "model": model_results
    }