"""
Streamlit Dashboard for NGS RAG Observability

Visualizes RAG pipeline health metrics, retrieval accuracy, and drift detection.

Usage:
    streamlit run src/observability/dashboard.py
"""

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


DB_PATH = "observability/metrics.db"


def load_retrieval_metrics(days: int = 30) -> pd.DataFrame:
    """Load retrieval metrics from SQLite."""
    if not Path(DB_PATH).exists():
        return pd.DataFrame()

    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query("""
            SELECT 
                timestamp,
                strategy,
                hybrid,
                exact_match,
                in_top_k,
                distance,
                latency_ms,
                num_results
            FROM retrieval_metrics
            WHERE timestamp > datetime('now', ? || ' days')
            ORDER BY timestamp
        """, conn, params=(f"-{days}",))

        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['date'] = df['timestamp'].dt.date

        return df

    return pd.DataFrame()


def load_strategy_comparison() -> pd.DataFrame:
    """Load strategy comparison data."""
    if not Path(DB_PATH).exists():
        return pd.DataFrame()

    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query("""
            SELECT 
                strategy,
                hybrid,
                COUNT(*) as total_queries,
                SUM(CASE WHEN exact_match THEN 1 ELSE 0 END) as exact_matches,
                SUM(CASE WHEN in_top_k THEN 1 ELSE 0 END) as top_k_matches,
                AVG(distance) as avg_distance,
                AVG(latency_ms) as avg_latency_ms
            FROM retrieval_metrics
            GROUP BY strategy, hybrid
        """, conn)

        if not df.empty:
            df['exact_accuracy'] = df['exact_matches'] / df['total_queries']
            df['tolerance_accuracy'] = df['top_k_matches'] / df['total_queries']
            df['hybrid_label'] = df['hybrid'].apply(lambda x: 'Hybrid' if x else 'Vector')

        return df

    return pd.DataFrame()


def main():
    st.set_page_config(
        page_title="NGS RAG Observability",
        page_icon="🧬",
        layout="wide",
    )

    st.title("🧬 NGS RAG Pipeline Observability")
    st.markdown("### Monitoring retrieval quality, latency, and drift")

    # Sidebar
    st.sidebar.header("Settings")
    days = st.sidebar.slider("History (days)", 7, 90, 30)

    # Load data
    df = load_retrieval_metrics(days)
    strategy_df = load_strategy_comparison()

    if df.empty:
        st.warning("⚠️ No metrics found. Run some tests or queries to generate data.")
        st.info("Tip: Run `python scripts/drift_monitor.py` to generate metrics.")
        return

    # Calculate summary metrics
    total_queries = len(df)
    exact_accuracy = df['exact_match'].mean() if 'exact_match' in df.columns else 0
    tolerance_accuracy = df['in_top_k'].mean() if 'in_top_k' in df.columns else 0
    avg_latency = df['latency_ms'].mean() if 'latency_ms' in df.columns else 0
    avg_distance = df['distance'].mean() if 'distance' in df.columns else 0

    # Top row - KPI cards
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Total Queries", f"{total_queries:,}")
    with col2:
        st.metric("Exact Accuracy", f"{exact_accuracy*100:.1f}%")
    with col3:
        st.metric("Tolerance Accuracy", f"{tolerance_accuracy*100:.1f}%")
    with col4:
        st.metric("Avg Latency", f"{avg_latency:.0f} ms")
    with col5:
        st.metric("Avg Distance", f"{avg_distance:.3f}")

    st.divider()

    # Row 2 - Charts
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 Retrieval Accuracy Over Time")

        if 'date' in df.columns and not df.empty:
            daily_acc = df.groupby('date').agg({
                'exact_match': 'mean',
                'in_top_k': 'mean',
            }).reset_index()

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=daily_acc['date'],
                y=daily_acc['exact_match'] * 100,
                mode='lines+markers',
                name='Exact Match %',
                line=dict(color='green')
            ))
            fig.add_trace(go.Scatter(
                x=daily_acc['date'],
                y=daily_acc['in_top_k'] * 100,
                mode='lines+markers',
                name='Tolerance Match %',
                line=dict(color='orange')
            ))
            fig.update_layout(
                yaxis_title="Accuracy %",
                xaxis_title="Date",
                height=300,
                margin=dict(l=0, r=0, t=0, b=0),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No daily data available yet.")

    with col2:
        st.subheader("⏱️ Latency Distribution")

        if 'latency_ms' in df.columns and not df.empty:
            fig = px.histogram(
                df,
                x='latency_ms',
                nbins=30,
                labels={'latency_ms': 'Latency (ms)'},
                height=300,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No latency data available yet.")

    st.divider()

    # Row 3 - Strategy Comparison
    st.subheader("🔍 Strategy Comparison")

    if not strategy_df.empty:
        # Pivot for grouped bar chart
        fig = go.Figure()

        for _, row in strategy_df.iterrows():
            label = f"{row['strategy']} ({row['hybrid_label']})"
            fig.add_trace(go.Bar(
                x=[label],
                y=[row['exact_accuracy'] * 100],
                name='Exact Accuracy',
                showlegend=False,
            ))

        fig.update_layout(
            title="Exact Accuracy by Strategy",
            yaxis_title="Accuracy %",
            xaxis_title="Strategy",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Detailed table
        st.subheader("📊 Detailed Metrics")
        display_df = strategy_df[[
            'strategy', 'hybrid_label', 'total_queries',
            'exact_accuracy', 'tolerance_accuracy',
            'avg_distance', 'avg_latency_ms'
        ]].copy()
        display_df['exact_accuracy'] = display_df['exact_accuracy'].apply(lambda x: f"{x*100:.1f}%")
        display_df['tolerance_accuracy'] = display_df['tolerance_accuracy'].apply(lambda x: f"{x*100:.1f}%")
        display_df['avg_distance'] = display_df['avg_distance'].apply(lambda x: f"{x:.3f}")
        display_df['avg_latency_ms'] = display_df['avg_latency_ms'].apply(lambda x: f"{x:.0f} ms")

        display_df.columns = [
            'Strategy', 'Mode', 'Queries',
            'Exact Acc', 'Tolerance Acc',
            'Avg Distance', 'Avg Latency'
        ]

        st.dataframe(display_df, use_container_width=True)
    else:
        st.info("No strategy comparison data available yet.")

    st.divider()

    # Row 4 - Raw Data
    with st.expander("📋 Raw Data (Last 100 Queries)"):
        st.dataframe(df.tail(100), use_container_width=True)


if __name__ == "__main__":
    main()
