import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.config import DATA_PATH
import re

def clean_price_eda(price_str):
    if pd.isna(price_str): return None
    if "/aana" in str(price_str): return None
    match = re.search(r'Rs\.\s*([\d\.]+)\s*(Cr|Lakh)?', str(price_str))
    if not match: return None
    try:
        val = float(match.group(1))
        unit = match.group(2)
        if unit == 'Cr': return val * 10000000
        if unit == 'Lakh': return val * 100000
        return val
    except: return None

def get_eda_plots(ml_metrics=None):
    try:
        df = pd.read_csv(DATA_PATH)
        df['Price_Val'] = df['PRICE'].apply(clean_price_eda)
        df = df.dropna(subset=['Price_Val'])
        
        # 1. Price Distribution
        fig_dist = px.histogram(
            df, x="Price_Val", 
            title="House Price Distribution",
            labels={'Price_Val': 'Price (NPR)'},
            color_discrete_sequence=['#3b82f6'],
            nbins=30
        )
        fig_dist.update_layout(template="plotly_white")

        # 2. Top Locations by Volume
        loc_counts = df['LOCATION'].value_counts().nlargest(10).reset_index()
        loc_counts.columns = ['Location', 'Count']
        fig_loc = px.bar(
            loc_counts, x="Location", y="Count",
            title="Top 10 Locations by Number of Listings",
            color="Count",
            color_continuous_scale="Blues"
        )
        fig_loc.update_layout(template="plotly_white")

        # 3. Average Price by Location
        avg_price_loc = df.groupby('LOCATION')['Price_Val'].mean().nlargest(10).reset_index()
        avg_price_loc['Price_Cr'] = avg_price_loc['Price_Val'] / 10000000
        fig_avg_price = px.bar(
            avg_price_loc, x="LOCATION", y="Price_Cr",
            title="Top 10 Most Expensive Locations (Avg Price in Cr)",
            labels={'Price_Cr': 'Avg Price (Cr)'},
            color="Price_Cr",
            color_continuous_scale="Viridis"
        )
        fig_avg_price.update_layout(template="plotly_white")

        # 4. Correlation Heatmap (Selected Features)
        df_corr = df[['Price_Val', 'FLOOR', 'BEDROOM', 'BATHROOM']].copy()
        for col in df_corr.columns:
            df_corr[col] = pd.to_numeric(df_corr[col], errors='coerce')
        corr = df_corr.corr()
        fig_corr = px.imshow(
            corr, text_auto=True, 
            title="Feature Correlation Heatmap",
            color_continuous_scale="RdBu_r"
        )

        # 5. ML Model Comparison
        fig_model = None
        if ml_metrics:
            metric_data = []
            # Find the champion for highlighting
            best_model = None
            if hasattr(ml_metrics, 'best_model_name'): # If passed as object
                 best_model = ml_metrics.best_model_name
            
            for name, score in ml_metrics.items():
                is_champion = " (Champion)" if (best_model and name == best_model) else ""
                metric_data.append({"Model": name + is_champion, "R2 Score": score, "Is Champion": bool(best_model and name == best_model)})
            
            metric_df = pd.DataFrame(metric_data)
            fig_model = px.bar(
                metric_df, x="Model", y="R2 Score",
                title="AI Model Performance Comparison (R² Accuracy)",
                labels={'R2 Score': 'R² Accuracy Score'},
                color="Is Champion",
                color_discrete_map={True: '#10b981', False: '#3b82f6'}, # Green for champion, Blue for others
                hover_data=["Model", "R2 Score"]
            )
            fig_model.update_layout(template="plotly_white", yaxis_range=[0, 1], showlegend=False)

        return {
            "dist": fig_dist,
            "loc": fig_loc,
            "avg_price": fig_avg_price,
            "corr": fig_corr,
            "model_comp": fig_model,
            "stats": {
                "total": len(df),
                "avg": df['Price_Val'].mean() / 10000000,
                "max": df['Price_Val'].max() / 10000000,
                "min": df['Price_Val'].min() / 10000000
            }
        }
    except Exception as e:
        print(f"EDA Error: {e}")
        return None
