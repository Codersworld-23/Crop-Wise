import altair as alt
import requests
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
from scripts.weather import get_weather
from scripts.geolocation import get_coordinates
from scripts.statistics import plot_distribution, get_confidence_interval, hypothesis_test, regression_plot, multiple_regression, anova_test
import plotly.express as px

# ----- App Title -----
st.set_page_config(page_title="Crop Disease Analytics", layout="wide")
st.title("🌾 Crop Disease Analytics Dashboard")

# ----- Image Upload Section -----
uploaded_image = st.file_uploader("📷 Upload an image of the diseased crop", type=["jpg", "png"])

if uploaded_image:
    st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)
    st.session_state["uploaded_image"] = uploaded_image

# ML placeholders
predicted_crop = st.session_state.get("predicted_crop")
predicted_disease = st.session_state.get("predicted_disease")
confidence = st.session_state.get("confidence")

if uploaded_image:

    if predicted_crop and predicted_disease and confidence is not None:
        st.markdown("### 🧠 ML Prediction")
        st.success(f"🟢 Crop: **{predicted_crop.title()}**")
        st.error(f"🔴 Disease: **{predicted_disease.title()}**")
        st.info(f"📊 Confidence: **{round(confidence * 100, 2)}%**")

        # ----- Tailored Advisory -----
        st.markdown("### 📋 Tailored Advisory")
        try:
            advisory_df = pd.read_csv("data/advisory.csv")
            match = advisory_df[
                (advisory_df["crop"].str.lower() == predicted_crop.lower()) &
                (advisory_df["disease"].str.lower() == predicted_disease.lower())
            ]
            if not match.empty:
                row = match.iloc[0]
                st.success(f"🧴 Recommended Fungicide: {row['fungicide']}")
                st.warning(f"🛠️ Method: {row['method']}")
                st.info(f"💡 Extra Tip: {row['tip']}")
            else:
                st.info("ℹ️ No advisory data available for this prediction.")
        except FileNotFoundError:
            st.error("📂 advisory.csv not found. Please add it to the `data/` folder.")
    else:
        st.info("🧠 Awaiting ML model integration to generate predictions.")

# ----- Location Input -----
st.markdown("---")
st.subheader("📍 Enter your Region to Fetch Weather & Location Insights")
location = st.text_input("Enter a city/town name")

if location:
    lat, lng = get_coordinates(location)
    weather = get_weather(location)

    if lat and lng and weather:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Latitude", lat)
            st.metric("lnggitude", lng)
        with col2:
            st.metric("Temperature (°C)", weather["main"]["temp"])
            st.metric("Humidity (%)", weather["main"]["humidity"])

        st.success(f"🌤️ Weather Summary: {weather['weather'][0]['description'].title()}")

        # ----- Dynamic Weather-Based Disease Insights -----
        with st.expander("🌿 Smart Weather-Based Disease Insights", expanded=True):
            condition = weather["weather"][0]["main"].lower()
            description = weather["weather"][0]["description"].lower()
            temp = weather["main"]["temp"]
            humidity = weather["main"]["humidity"]
            wind_speed = weather["wind"]["speed"]

            # 🌤️ Visual weather summary with icons
            condition_emoji = {
                "clear": "☀️", "clouds": "☁️", "rain": "🌧️", "drizzle": "🌦️", "thunderstorm": "⛈️",
                "snow": "❄️", "mist": "🌫️", "fog": "🌫️", "haze": "🌫️"
            }
            emoji = condition_emoji.get(condition, "🌡️")

            st.subheader(f"{emoji} Current Weather Conditions")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🌡 Temperature", f"{temp} °C")
            with col2:
                st.metric("💧 Humidity", f"{humidity} %")
            with col3:
                st.metric("🌬 Wind Speed", f"{wind_speed} m/s")

            st.caption(f"**Condition:** {condition.title()} | **Description:** {description.title()}")

            # 💡 Weather-Based Insights
            st.markdown("### 🧠 Disease Risk Insights")

            insight_shown = False

            if humidity > 80:
                st.warning("⚠️ High humidity detected. Fungal diseases like leaf rust and mildew may be more prevalent.")
                insight_shown = True
            if humidity > 80 and ("rain" in condition or "drizzle" in condition):
                st.error("🚨 Prolngged humidity and rainfall can accelerate the spread of fungal infections in crops.")
                insight_shown = True
            if temp < 15 and humidity > 70:
                st.info("🧊 Cold and damp conditions may increase risk of soil-borne pathogens like root rot.")
                insight_shown = True
            if wind_speed > 10:
                st.warning("🌬️ High wind speeds may promote the spread of airborne diseases like rust or blight.")
                insight_shown = True
            if condition in ["fog", "mist", "haze"]:
                st.warning("🌫️ Foggy or misty conditions detected. Watch for moisture-loving diseases like powdery mildew.")
                insight_shown = True
            if "light rain" in description:
                st.info("☔ Light rain can lead to dew formation, creating favorable conditions for early-stage fungal growth.")
                insight_shown = True
            if temp > 35 and humidity < 40:
                st.warning("🔥 Hot and dry weather may attract pests like aphids and mites.")
                insight_shown = True
            if not insight_shown:
                st.info("✅ No concerning weather-based disease risks detected under current conditions.")
    else:
        st.error("❌ Could not fetch weather or geolocation data.")

# ----- Visual Dashboard Section -----
st.markdown("---")
st.subheader("📊 Analytics Dashboard")

tab1, tab2, tab3 = st.tabs(["Disease vs Region", "Disease vs Weather", "Statistical Analysis"])

with tab1:
    st.markdown("### 🌍 Region-wise Disease Risk (Live Prediction Map)")
    import time
    from streamlit import cache_data

    try:
        region_df = pd.read_csv("data/IndianCities.csv")
        st.success("🗺️ Loaded city dataset successfully.")

        refresh = st.button("🔄 Refresh Live Weather Data")

        @st.cache_data(ttl=3600)
        def get_cached_weather(city):
            return get_weather(city)

        st.info("📡 Fetching weather data and assigning disease risk levels...")
        progress = st.progress(0)

        results = []
        for i, (_, row) in enumerate(region_df.iterrows()):
            city = row["City"]
            state = row["State"]
            lat = row["Latitude"]
            lng = row["Longitude"]

            weather = get_weather(city) if refresh else get_cached_weather(city)

            if weather:
                temp = weather["main"]["temp"]
                humidity = weather["main"]["humidity"]
                wind = weather["wind"]["speed"]
                desc = weather["weather"][0]["description"].title()

                if humidity > 80 and temp > 25:
                    risk = "High"
                elif humidity > 60:
                    risk = "Medium"
                else:
                    risk = "Low"

                results.append({
                    "region": city,
                    "state": state,
                    "lat": lat,
                    "lng": lng,
                    "temp": temp,
                    "humidity": humidity,
                    "wind": wind,
                    "description": desc,
                    "risk": risk
                })
            else:
                results.append({
                    "region": city,
                    "state": state,
                    "lat": lat,
                    "lng": lng,
                    "temp": None,
                    "humidity": None,
                    "wind": None,
                    "description": "Unavailable",
                    "risk": "Unknown"
                })

            progress.progress((i + 1) / len(region_df))

        progress.empty()

        weather_df = pd.DataFrame(results)

        color_map = {
            "High": "red",
            "Medium": "orange",
            "Low": "green",
            "Unknown": "gray"
        }

        fig = px.scatter_map(
            weather_df,
            lat="lat",
            lon="lng",
            color="risk",
            color_discrete_map=color_map,
            hover_name="region",
            hover_data={
                "state": True,
                "temp": ":.1f°C",
                "humidity": ":.1f%",
                "wind": ":.1f km/h",
                "description": True,
                "risk": True,
                "lat": False,
                "lng": False
            },
            size=[10] * len(weather_df),  # Uniform size for all markers
            size_max=10,  # Maximum size
            zoom=5,
            height=600,
            title="<b>🌱 Real-Time Crop Disease Risk by Region</b>",
        )

        fig.update_traces(
            marker=dict(
                opacity=0.8,
                sizemode = 'diameter',
                size = 16
            ),
            selector=dict(mode='markers')
        )

        # Custom hover template
        fig.update_traces(
            hovertemplate=(
                "<b>%{hovertext}</b><br>"
                "<b>%{customdata[0]}</b><br>"
                "<b>Temp:</b> %{customdata[1]}<br>"
                "<b>Humidity:</b> %{customdata[2]}<br>"
                "<b>Wind:</b> %{customdata[3]}<br>"
                "<b>Conditions:</b> %{customdata[4]}<br>"
                "<b>Risk:</b> %{customdata[5]}<extra></extra>"
            )
        )

        fig.update_layout(
            mapbox_style="stamen-terrain",
            margin={"r": 0, "t": 60, "l": 0, "b": 0},
            hovermode="closest",
            legend_title_text="<b>Disease Risk</b>",
            plot_bgcolor="rgba(0,0,0,0)",
            mapbox=dict(
                bearing=0,
                pitch=0,
                zoom=4,
                center=dict(lat=20.5937, lon=78.9629)  # Center on India
            )
        )

        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Failed to load or process map: {str(e)}")

with tab2:
    st.markdown("### 📉 Weather Trends Over Time")
    #insights coming soon.....

with tab3:
    st.markdown("### 📊 Advanced Statistical Analysis")

    try:
        df = pd.read_csv("data/stats_data.csv")

        concept = st.selectbox("📚 Select a Concept", [
            "📈 Normal Distribution", "🎯 Central Limit Theorem", "📏 Confidence Interval",
            "🔬 Hypothesis Testing", "📊 Linear Regression", "🧠 Multiple Regression", "🧪 ANOVA"
        ])

        st.markdown("---")

        if concept == "Normal Distribution":
            st.markdown("#### 📈 Temperature Distribution")
            chart = alt.Chart(df).mark_bar().encode(
                alt.X("temperature", bin=alt.Bin(maxbins=30)),
                y='count()'
            ).properties(title='Temperature Distribution')
            st.altair_chart(chart, use_container_width=True)

        elif concept == "Central Limit Theorem":
            st.markdown("#### 🧠 Central Limit Theorem Demonstration")
            means = [df["temperature"].sample(30).mean() for _ in range(500)]
            sample_df = pd.DataFrame({"sample_means": means})
            fig = px.histogram(sample_df, x="sample_means", nbins=30, title="CLT - Sampling Distribution")
            fig.update_layout(bargap=0.1)
            st.plotly_chart(fig)

        elif concept == "Confidence Interval":
            st.markdown("#### 🎯 95% Confidence Interval for Humidity")
            ci = get_confidence_interval(df, "humidity")
            fig = px.histogram(df, x="humidity", nbins=30, title="Humidity Distribution with Confidence Interval")
            fig.add_vline(x=ci[0], line_dash="dash", line_color="red", annotation_text="Lower 95% CI")
            fig.add_vline(x=ci[1], line_dash="dash", line_color="green", annotation_text="Upper 95% CI")
            st.plotly_chart(fig)
            st.success(f"CI: ({ci[0]:.2f}, {ci[1]:.2f})")

        elif concept == "Hypothesis Testing":
            st.markdown("#### 🔍 Hypothesis Testing with Two Regions")
            t, p, g1, g2 = hypothesis_test(df, "disease_severity", "region")
            if t:
                st.write(f"H₀: Disease severity is the same in {g1} and {g2}")
                st.warning(f"T-Statistic = {t:.2f}, P-Value = {p:.4f}")
            fig = px.box(df, x="region", y="disease_severity", color="region", points="all")
            st.plotly_chart(fig)

        elif concept == "Linear Regression":
            st.markdown("#### 🔗 Linear Regression")
            fig = px.scatter(df, x="temperature", y="disease_severity", trendline="ols", color="region")
            st.plotly_chart(fig)

        elif concept == "Multiple Regression":
            st.markdown("#### 🧠 Multiple Regression Analysis")
            coef, intercept = multiple_regression(df, ["temperature", "humidity", "wind_speed"], "disease_severity")
            coef_df = pd.DataFrame({"Feature": ["Temperature", "Humidity", "Wind Speed"], "Coefficient": coef})
            fig = px.bar(coef_df, x="Feature", y="Coefficient", color="Feature", title="Feature Importance")
            st.plotly_chart(fig)
            st.json({"Intercept": intercept})

        elif concept == "ANOVA":
            st.markdown("#### 🧪 ANOVA Across Regions")
            f, p = anova_test(df, "disease_severity", "region")
            st.info(f"F-Statistic = {f:.2f}, P-Value = {p:.4f}")
            fig = px.box(df, x="region", y="disease_severity", color="region", points="all")
            st.plotly_chart(fig)

    #Intentionally getting an error like the following as of now because stats_data is being generated in the backend
    except FileNotFoundError:
        st.error("❌ stats_data.csv not found. Please ensure it exists in the /data folder.")

# ----- Footer -----
st.markdown("---")
st.caption("© 2025 Crop Disease Analytics | Built with Streamlit")