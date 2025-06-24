


import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

@st.cache_data
def load_data(stock):
    if stock == "NVDA":
        train_df = pd.read_csv("nvda_train_predictions.csv", parse_dates=['hour_bin'])
        test_df = pd.read_csv("nvda_test_predictions.csv", parse_dates=['hour_bin'])
    elif stock == "TSLA":
        train_df = pd.read_csv("tsla_train_predictions.csv", parse_dates=['hour_bin'])
        test_df = pd.read_csv("tsla_test_predictions.csv", parse_dates=['hour_bin'])
    else:
        raise ValueError("Invalid stock selected.")

    train_df['set'] = 'train'
    test_df['set'] = 'test'
    return pd.concat([train_df, test_df], axis=0).sort_values('hour_bin'), test_df

def plot_with_test_animation(df):
    df = df.copy().sort_values('hour_bin')
    train_df = df[df['set'] == 'train']
    test_df = df[df['set'] == 'test']

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=train_df['hour_bin'],
        y=train_df['price_at_news'],
        mode='lines+markers',
        name='Train Stock Price',
        line=dict(color='lightblue')
    ))

    def add_markers(df_subset, name, symbol, color):
        fig.add_trace(go.Scatter(
            x=df_subset['hour_bin'],
            y=df_subset['price_at_news'],
            mode='markers',
            name=name,
            marker=dict(symbol=symbol, size=10, color=color)
        ))

    add_markers(train_df[(train_df['predicted_up'] == 1) & (train_df['price_moved_up'] == 1)],
                'Train Correct Up', 'triangle-up', 'green')
    add_markers(train_df[(train_df['predicted_up'] == 0) & (train_df['price_moved_up'] == 0)],
                'Train Correct Down', 'triangle-down', 'red')
    add_markers(train_df[(train_df['predicted_up'] != train_df['price_moved_up'])],
                'Train Wrong', 'x', 'orange')

    test_start = test_df['hour_bin'].min().to_pydatetime()

    fig.add_shape(
        type="line",
        x0=test_start,
        x1=test_start,
        y0=0,
        y1=1,
        xref="x",
        yref="paper",
        line=dict(color="gray", width=2, dash="dash"),
    )

    fig.add_annotation(
        x=test_start,
        y=1.05,
        xref="x",
        yref="paper",
        showarrow=False,
        text="Test Start",
        font=dict(color="gray", size=12)
    )

    frames = []
    for i in range(1, len(test_df) + 1):
        current = test_df.iloc[:i]
        frame_data = [
            go.Scatter(
                x=current['hour_bin'],
                y=current['price_at_news'],
                mode='lines+markers',
                name='Test Stock Price',
                line=dict(color='blue')
            ),
            go.Scatter(
                x=current[(current['predicted_up'] == 1) & (current['price_moved_up'] == 1)]['hour_bin'],
                y=current[(current['predicted_up'] == 1) & (current['price_moved_up'] == 1)]['price_at_news'],
                mode='markers',
                name='Test Correct Up',
                marker=dict(symbol='triangle-up', size=10, color='green')
            ),
            go.Scatter(
                x=current[(current['predicted_up'] == 0) & (current['price_moved_up'] == 0)]['hour_bin'],
                y=current[(current['predicted_up'] == 0) & (current['price_moved_up'] == 0)]['price_at_news'],
                mode='markers',
                name='Test Correct Down',
                marker=dict(symbol='triangle-down', size=10, color='red')
            ),
            go.Scatter(
                x=current[(current['predicted_up'] != current['price_moved_up'])]['hour_bin'],
                y=current[(current['predicted_up'] != current['price_moved_up'])]['price_at_news'],
                mode='markers',
                name='Test Wrong',
                marker=dict(symbol='x', size=10, color='orange')
            )
        ]
        frames.append(go.Frame(data=frame_data, name=str(i)))

    fig.add_trace(go.Scatter(
        x=[], y=[], mode='markers', name='Test Animation',
        marker=dict(size=1, color='rgba(0,0,0,0)')
    ))

    fig.frames = frames

    fig.update_layout(
        title="Stock Price with Predictions and Evaluations",
        xaxis_title="Datetime",
        yaxis_title="Stock Price ($)",
        height=650,
        xaxis=dict(
            tickformat="%b %d %H:%M",
            tickangle=45,
            nticks=80,
            showgrid=True
        ),
        yaxis=dict(
            tickformat=".2f",
            showgrid=True,
            nticks=50
        ),
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            y=1.05,
            x=1,
            xanchor="right",
            yanchor="top",
            buttons=[dict(label="▶ Play Test Reveal",
                          method="animate",
                          args=[None, {"frame": {"duration": 150, "redraw": True},
                                       "fromcurrent": True, "mode": "immediate"}])]
        )],
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig

def main():
    st.set_page_config(layout="wide")
    st.title("📈 Stock Price Prediction - News Headlines")

    st.sidebar.header("📊 Select Stock")
    stock_choice = st.sidebar.selectbox("Choose a stock", ["NVDA", "TSLA"])

    df, test_df = load_data(stock_choice)

    st.sidebar.header("📅 Date Filter")
    date_range = st.sidebar.date_input("Select date range",
        [df['hour_bin'].min(), df['hour_bin'].max()])
    if len(date_range) == 2:
        start = pd.to_datetime(date_range[0])
        end = pd.to_datetime(date_range[1])
        df = df[(df['hour_bin'] >= start) & (df['hour_bin'] <= end)]

    st.plotly_chart(plot_with_test_animation(df), use_container_width=True)

    st.subheader("📋 Prediction Table")
    st.dataframe(df[['hour_bin', 'title', 'price_at_news', 'price_moved_up', 'predicted_up', 'prob_up', 'set']])

    st.subheader("❌ Incorrect Predictions")
    wrong_preds = df[(df['predicted_up'] != df['price_moved_up']) & (df['set'] == 'test')]
    st.dataframe(wrong_preds[['hour_bin', 'title', 'price_at_news', 'price_moved_up', 'predicted_up', 'prob_up']])

    st.subheader("📊 Classification Report on Test Data")
    y_true = test_df['price_moved_up']
    y_pred = test_df['predicted_up']

    acc = accuracy_score(y_true, y_pred)
    st.markdown(f"**Accuracy:** {acc:.4f}")

    report = classification_report(y_true, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.format("{:.2f}"))

    st.subheader("🧩 Confusion Matrix")
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=["Actual Down", "Actual Up"], columns=["Predicted Down", "Predicted Up"])
    st.dataframe(cm_df)

if __name__ == "__main__":
    main()





















