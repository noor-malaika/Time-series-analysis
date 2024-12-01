import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

# Configure Streamlit page
st.set_page_config(
    page_title="Multan Weather Forecast",
    page_icon="🌤️",
    layout="wide",
)

# Title and description
st.markdown(
    """
    <h1 style="text-align: center; color: #4CAF50;">🌤️ Multan Weather Forecast 🌤️</h1>
    <p style="text-align: center; color: #6C757D;">
    Explore weather forecasts and trends for Multan with interactive and visually engaging plots.
    </p>
    """,
    unsafe_allow_html=True,
)

# Sidebar settings
st.sidebar.title("Settings")
forecast_period = st.sidebar.slider("Forecast Period (days)", min_value=30, max_value=730, value=365, step=30)
show_components = st.sidebar.checkbox("Show Components (Trend, Seasonality)", value=True)
show_radar_plot = st.sidebar.checkbox("Show Seasonal Variability", value=False)
show_stats = st.sidebar.checkbox("Show Summary Statistics", value=True)

# Load the pre-trained NeuralProphet model
fileName = 'weather_prediction/neuralProphet_weather.joblib'
loaded_model = joblib.load(fileName)

# Generate future dataframe and make predictions
future = loaded_model.make_future_dataframe(periods=forecast_period)
prediction = loaded_model.predict(future)

# --- Plotly Interactive Forecast Plot ---
st.subheader("Interactive Weather Forecast")
fig = go.Figure()

# Actual data points (if present)
data = pd.read_csv("weather_prediction/archive.csv")
fig.add_trace(go.Scatter(
    x=prediction['ds'],
    y=data['apparent_temperature_max (Â°C)'],
    mode='markers',
    name='Actual Data',
    marker=dict(color='green', size=5, opacity=0.7)
))

# Forecasted data points
fig.add_trace(go.Scatter(
    x=prediction['ds'],
    y=prediction['yhat'],
    mode='lines',
    name='Forecasted Data',
    line=dict(color='blue', width=2)
))

# Confidence Interval (upper and lower bounds)
fig.add_trace(go.Scatter(
    x=prediction['ds'],
    y=prediction['yhat_lower'],
    mode='lines',
    name='Lower Bound',
    line=dict(color='rgba(0, 0, 255, 0.2)', dash='dot'),
    showlegend=False
))
fig.add_trace(go.Scatter(
    x=prediction['ds'],
    y=prediction['yhat_upper'],
    mode='lines',
    name='Confidence interval', # Actually upper bound (set ci for better understanding of viz)
    line=dict(color='rgba(0, 0, 255, 0.2)', dash='dot'),
    fill='tonexty',
    fillcolor='rgba(0, 0, 255, 0.2)',
    showlegend=True
))

fig.update_layout(
    title=f"Temperature Forecast for Multan (Next {forecast_period} Days)",
    xaxis_title="Date",
    yaxis_title="Temperature (°C)",
    template="plotly_dark",
    hovermode="closest"
)

# Display plot
st.plotly_chart(fig)

# Function to convert day of year to a date string
def day_of_year_to_date(day_of_year):
    # Use datetime to create a date object for the first day of the year, then add the day_of_year
    start_date = datetime(datetime.now().year, 1, 1)
    day_date = start_date.replace(year=datetime.now().year) + pd.to_timedelta(day_of_year - 1, unit='D')
    return day_date.strftime('%b %d')  # Format as 'Month Day' (e.g., 'Jan 01')

# --- Forecast Components: Trend, Seasonality, etc ---
if show_components:
    st.subheader("Forecast Components (Trend, Seasonality, etc.)")
    
    # Trend plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=prediction['ds'],
        y=prediction['trend'],
        mode='lines',
        name='Trend',
        line=dict(color='orange', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=prediction['ds'],
        y=prediction['trend_lower'],
        mode='lines',
        name='Trend Lower Bound',
        line=dict(color='rgba(255, 165, 0, 0.2)', dash='dot'),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=prediction['ds'],
        y=prediction['trend_upper'],
        mode='lines',
        name='Trend Upper Bound',
        line=dict(color='rgba(255, 165, 0, 0.2)', dash='dot'),
        fill='tonexty',
        fillcolor='rgba(255, 165, 0, 0.2)',
        showlegend=False
    ))
    fig.update_layout(
        title="Trend Component",
        xaxis_title="Date",
        yaxis_title="Temperature (°C)",
        template="plotly_dark"
    )
    st.plotly_chart(fig)

    # --- Plotly Interactive Average Yearly Trend ---
    st.subheader("Average Yearly Temperature Trend (Seasonality)")

    fig = go.Figure()
    # Extract the day of year and the corresponding predicted temperature
    prediction['day_of_year'] = pd.to_datetime(prediction['ds']).dt.dayofyear
    prediction['month'] = pd.to_datetime(prediction['ds']).dt.month
    
    # Aggregate by day of the year
    daily_avg_temp = prediction.groupby('day_of_year')['yhat'].mean().reset_index()
    
    # Map day_of_year to date strings
    monthly_labels = [day_of_year_to_date(day) for day in daily_avg_temp['day_of_year']]
    
    # Add the average yearly temperature trend (one per day)
    fig.add_trace(go.Scatter(
        x=daily_avg_temp['day_of_year'],
        y=daily_avg_temp['yhat'],
        mode='lines',
        name='Average Yearly Trend',
        line=dict(color='blue', width=2),
        hovertemplate='<b>%{customdata}</b><br>' + 'Temperature: %{y:.2f}°C<br>' +
                    'Day of Year: %{x}<extra></extra>',  # Hover text with custom month and day mapping
        customdata=monthly_labels,  # Month labels for hover
    ))

    # Update layout to show month ticks on the x-axis
    fig.update_layout(
        title="Average Yearly Temperature Trend (Seasonality)",
        xaxis_title="Day of the Year",
        yaxis_title="Temperature (°C)",
        xaxis=dict(
            tickmode='array',
            tickvals=[1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335],  # Day numbers for each month
            ticktext=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],  # Month labels
        ),
        template="plotly_dark"
    )

    # Show the plot in Streamlit
    st.plotly_chart(fig)

# --- Radar Plot for Seasonal Variability ---
if show_radar_plot:
    st.subheader("Seasonal Variability - Radar Plot")

    # Aggregating data by month
    monthly_avg_temp = prediction.groupby('month')['yhat'].mean().reset_index()

    # Create radar plot
    categories = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    values = monthly_avg_temp['yhat'].values.tolist()

    # Make the radar plot
    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Average Monthly Temperature',
        line=dict(color='blue')
    ))

    # Update layout for better visibility
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[min(values) - 5, max(values) + 5]  # Adjust the range for visibility
            ),
        ),
        title="Monthly Temperature Distribution",
        template="plotly_dark"
    )

    st.plotly_chart(fig)


# --- Summary Statistics ---
if show_stats:
    st.sidebar.subheader("Summary Statistics")
    max_temp = prediction['yhat_upper'].max()
    min_temp = prediction['yhat_lower'].min()
    avg_temp = prediction['yhat'].mean()

    st.sidebar.metric("Max Temp", f"{max_temp:.2f} °C")
    st.sidebar.metric("Min Temp", f"{min_temp:.2f} °C")
    st.sidebar.metric("Average Temp", f"{avg_temp:.2f} °C")


# Footer
st.markdown(
    """
    <hr>
    <footer style="text-align: center;">
    Built using Streamlit by Malaika | © 2024
    </footer>
    """,
    unsafe_allow_html=True,
)
