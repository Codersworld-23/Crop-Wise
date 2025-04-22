import altair as alt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
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
import plotly.express as px
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# ----- App Title -----
st.set_page_config(page_title="CropWise", layout="wide")
st.title("🌾 Welcome To CropWise")

# ----- Image Upload Section -----
uploaded_image = st.file_uploader("📷 Upload an image of the diseased crop", type=["jpg", "png"])

if uploaded_image:
    st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)

    # --- ML Output Placeholder and Fallback ---
    predicted_crop = st.session_state.get("predicted_crop", "wheat")
    predicted_disease = st.session_state.get("predicted_disease", "black rust")

    st.markdown("### 🧠 ML Prediction")
    st.success(f"🟢 Crop: **{predicted_crop.title()}**")
    st.error(f"🔴 Disease: **{predicted_disease.title()}**")

    if "predicted_crop" not in st.session_state or "predicted_disease" not in st.session_state:
        st.caption("ℹ️ Using default values until ML model is integrated.")

    # ----- AI-Based Treatment Advisory -----
    st.markdown("### 💊 AI-Based Treatment Advisory")

    def fetch_treatment_advisory(crop, disease):
        prompt = f"""Provide concise, bullet-point treatment advice for {disease} in {crop} crops.
        Include:
        - Recommended pesticides/fungicides
        - Application dosage
        - Optimal timing
        - Cultural control methods
        - Resistance management tips"""

        try:
            model = genai.GenerativeModel('gemini-1.5-pro')
            response = model.generate_content(
                prompt,
                generation_config={
                    "max_output_tokens": 500,
                    "temperature": 0.5
                }
            )
            return response.text
        except Exception as e:
            st.error(f"⚠️ Error: {str(e)}. Please try again later.")
            return None


    if st.button("🔍 Get AI Treatment Advisory"):
        with st.spinner("Fetching treatment advice..."):
            advice = fetch_treatment_advisory(predicted_crop, predicted_disease)
            if advice:
                st.write(advice)
            else:
                st.warning("No treatment advice could be retrieved.")


# ----- Location Input -----
st.markdown("---")
st.subheader("📍 Enter your location to get relevant insights")
location = st.text_input("Enter a city/town name")

if location:
    lat, lng = get_coordinates(location)
    weather = get_weather(location)

    if lat and lng and weather:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Latitude", lat)
        with col2:
            st.metric("Longitude", lng)

        st.success(f"🌤️ Weather Summary: {weather['weather'][0]['description'].title()}")

        # ----- Dynamic Weather-Based Disease Insights -----
        with st.expander(" ☁ Smart Weather Insights", expanded=True):
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
            st.markdown("### 🧠 General Weather Based Disease Risk Insights")

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

selected_tab = st.sidebar.radio("📂 Choose a section", ["Live Disease Risk Map", "Weather Analytics", "Crop Market vs Geostatistics"])

if selected_tab == "Live Disease Risk Map":
    st.markdown("### 🌍 Live Disease Risk Map")

    from streamlit import cache_data

    if st.button("📡 Generate Real-Time Disease Risk Map"):
        try:
            region_df = pd.read_csv("data/IndianCities.csv")
            st.success("🗺️ Loaded city dataset successfully.")

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

                weather = get_cached_weather(city)

                if weather:
                    temp = weather["main"]["temp"]
                    humidity = weather["main"]["humidity"]
                    wind = weather["wind"]["speed"]
                    desc = weather["weather"][0]["description"].title()

                    # --- Enhanced logic ---
                    if humidity > 85 and temp > 30 and wind < 5:
                        risk = "Very High"
                    elif humidity > 80 and temp > 25:
                        risk = "High"
                    elif 60 < humidity <= 80 and 20 < temp <= 30:
                        if wind < 7:
                            risk = "Medium"
                        else:
                            risk = "Low"
                    elif humidity < 40 and temp > 35:
                        risk = "Dry Heat Risk"
                    elif wind >= 12 and humidity < 50:
                        risk = "Wind-Pest Risk"
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
                "Very High": "darkred",
                "High": "red",
                "Medium": "orange",
                "Low": "green",
                "Dry Heat Risk": "purple",
                "Wind-Pest Risk": "blue",
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
                size=[10] * len(weather_df),
                size_max=10,
                zoom=5,
                height=600,
                title="<b>🌱 Real-Time Crop Disease Risk by Region</b>",
            )

            fig.update_traces(
                marker=dict(opacity=0.85, sizemode='diameter', size=16),
                selector=dict(mode='markers')
            )

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
                    center=dict(lat=20.5937, lon=78.9629)
                )
            )

            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"❌ Failed to load or process map: {str(e)}")
    else:
        st.info("👆 Click the button above to generate the disease risk map.")

##########################################################################################################3333

elif selected_tab == "Weather Analytics":
    st.markdown("### 🌤️ Weather Trends & Insights")

    # Load city data
    try:
        city_coords = pd.read_csv("data/IndianCities.csv")
        city = st.selectbox("Select a Region to Analyze", 
                            options=sorted(city_coords["City"].unique()),
                            index=None,
                            placeholder="Choose a city...")

        if not city:
            st.info("Please select a city out of the agri prone cities available here to begin analysis")
            st.stop()

    except Exception as e:
        st.error("⚠️ Could not load city database. Please check the data file.")
        st.stop()

    # Get coordinates
    try:
        row = city_coords[city_coords["City"] == city].iloc[0]
        lat, lon = row["Latitude"], row["Longitude"]
        st.success(f"📍 Analyzing **{city}** (Lat: {lat:.4f}, Lon: {lon:.4f})")
    except Exception as e:
        st.error(f"🚨 Could not find coordinates for {city}")
        st.stop()

    # ===== WEATHER ANALYSIS =====
    st.subheader("⛅ Weather Conditions")

    def get_safe_weather(lat, lon, days=30):
        base_params = {
            "latitude": lat,
            "longitude": lon,
            "daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_sum"],
            "past_days": days,
            "timezone": "auto"
        }

        try:
            response = requests.get("https://api.open-meteo.com/v1/forecast", params=base_params, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Weather API error: {str(e)}")
            return None

    try:
        with st.spinner("Loading weather data..."):
            weather_data = get_safe_weather(lat, lon)

            if weather_data and "daily" in weather_data:
                weather_df = pd.DataFrame({
                    "date": pd.to_datetime(weather_data["daily"]["time"]),
                    "temp_max": weather_data["daily"]["temperature_2m_max"],
                    "temp_min": weather_data["daily"]["temperature_2m_min"],
                    "precipitation": weather_data["daily"].get("precipitation_sum", [0]*len(weather_data["daily"]["time"]))
                })

                # Temperature visualization
                st.subheader("🌡️ Temperature Trends")
                fig_temp = px.line(weather_df, x="date", y=["temp_max", "temp_min"],
                                   title=f"Temperature Trend in {city}",
                                   labels={"value": "Temperature (°C)", "date": "Date"})
                st.plotly_chart(fig_temp, use_container_width=True)

                # Precipitation visualization
                st.subheader("🌧️ Rainfall Patterns")
                fig_rain = px.bar(weather_df, x="date", y="precipitation",
                                  title=f"Daily Rainfall in {city} (mm)")
                st.plotly_chart(fig_rain, use_container_width=True)

                # Heatmap
                st.subheader("🌡️ Temperature vs Rainfall Heatmap")
                heatmap_df = weather_df.copy()
                heatmap_df["temp_avg"] = heatmap_df[["temp_max", "temp_min"]].mean(axis=1)
                fig_heatmap = px.density_heatmap(heatmap_df, x="temp_avg", y="precipitation",
                                                 nbinsx=30, nbinsy=30,
                                                 title="Heatmap: Avg Temp vs Precipitation",
                                                 labels={"temp_avg": "Average Temp (°C)", "precipitation": "Rainfall (mm)"})
                st.plotly_chart(fig_heatmap, use_container_width=True)

                # Summary Metrics
                st.subheader("📊 Summary Insights")
                cols = st.columns(3)
                metrics = [
                    ("🌡️ Avg Temp", f"{weather_df[['temp_max','temp_min']].mean().mean():.1f}°C"),
                    ("🔥 Max Temp", f"{weather_df['temp_max'].max():.1f}°C"),
                    ("🌧️ Total Rain", f"{weather_df['precipitation'].sum():.1f} mm")
                ]

                for col, (label, value) in zip(cols, metrics):
                    col.metric(label, value)

            else:
                st.warning("Weather data unavailable. Try again later.")

    except Exception as e:
        st.error(f"Weather data error: {str(e)}")


# ===== CROP MARKET ANALYTICS =====
elif selected_tab == "Crop Market vs Geostatistics":
    import streamlit as st
    import pandas as pd
    import requests
    import json
    from datetime import datetime, timedelta
    import plotly.express as px
    import plotly.graph_objects as go
    import seaborn as sns
    import matplotlib.pyplot as plt
    import folium
    from streamlit_folium import st_folium
    from folium.plugins import HeatMap
    from scipy import stats
    import statsmodels.api as sm
    import numpy as np

    # Base URL and AJAX endpoints
    base_url = "https://enam.gov.in/web/"
    trade_data_url = base_url + "Ajax_ctrl/trade_data_list"
    commodity_url = base_url + "Ajax_ctrl/commodity_list"

    # Headers for HTTP requests
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
        "X-Requested-With": "XMLHttpRequest"
    }

    # Load IndianCities.csv
    @st.cache_data
    def load_cities():
        cities_df = pd.read_csv("data/IndianCities.csv")
        # Standardize City names to title case and strip spaces
        cities_df["City"] = cities_df["City"].str.strip().str.title()
        return cities_df

    # Fetch states from CSV
    def get_states():
        cities_df = load_cities()
        return sorted(cities_df["State"].unique())

    # Fetch cities for a given state
    def get_cities(state):
        cities_df = load_cities()
        return sorted(cities_df[cities_df["State"] == state]["City"].tolist())

    # Fetch commodity list based on filters
    def fetch_commodity_list(state_name, apmc_name, from_date, to_date):
        payload = {
            "language": "en",
            "stateName": state_name,
            "apmcName": apmc_name,
            "fromDate": from_date,
            "toDate": to_date
        }
        try:
            response = requests.post(commodity_url, headers=headers, data=payload)
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == 200:
                    return [item["commodity"] for item in data.get("data", [])]
            return []
        except Exception as e:
            st.error(f"Error fetching commodities: {e}")
            return []

    # Fetch trade data
    def fetch_trade_data(state_name, apmc_name, commodity_name, from_date, to_date):
        payload = {
            "language": "en",
            "stateName": state_name,
            "apmcName": apmc_name,
            "commodityName": commodity_name,
            "fromDate": from_date,
            "toDate": to_date
        }
        try:
            response = requests.post(trade_data_url, headers=headers, data=payload)
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == 200:
                    return pd.DataFrame(data.get("data", []))
            return pd.DataFrame()
        except Exception as e:
            st.error(f"Error fetching trade data: {e}")
            return pd.DataFrame()

    # Format number with commas
    def number_format(num):
        try:
            num = str(int(float(num)))
            if len(num) == 4:
                return f"{num[:1]},{num[1:]}"
            elif len(num) == 5:
                return f"{num[:2]},{num[2:]}"
            elif len(num) == 6:
                return f"{num[:1]},{num[1:4]},{num[4:]}"
            elif len(num) == 7:
                return f"{num[:2]},{num[2:5]},{num[5:]}"
            elif len(num) == 8:
                return f"{num[:1]},{num[1:4]},{num[4:7]},{num[7:]}"
            return num
        except:
            return str(num)

    # Calculate confidence interval
    def calculate_confidence_interval(data, confidence=0.95):
        n = len(data)
        mean = np.mean(data)
        std_err = stats.sem(data)
        interval = std_err * stats.t.ppf((1 + confidence) / 2, n - 1)
        return mean, mean - interval, mean + interval

    # Clean APMC names by removing common suffixes
    def clean_apmc_name(apmc):
        if pd.isna(apmc):
            return apmc
        # Remove common suffixes (case-insensitive) and normalize
        cleaned = apmc.strip()
        cleaned = cleaned.replace(" APMC", "").replace(" Market", "").replace(" Agri Market", "")
        # Convert to title case to match IndianCities.csv
        return cleaned.title().strip()

    # Function for the Crop Market vs Geostatistics tab
    def crop_market_geostatistics_tab():
        st.header("💰 Crop Market vs Geostatistics")
        
        # Load cities data
        cities_df = load_cities()
        
        # User inputs within the tab
        col1, col2 = st.columns(2)
        with col1:
            state = st.selectbox("Select State", ["All"] + get_states())
        with col2:
            if state != "All":
                cities = get_cities(state)
            else:
                cities = []
            city = st.selectbox("Select City (APMC)", ["All"] + cities)
        
        col3, col4 = st.columns(2)
        with col3:
            today = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)  # Current date
            min_from_date = today - timedelta(days=30)  # Allow 30 days back
            from_date = st.date_input("From Date", min_value=min_from_date, max_value=today, 
                                    value=today - timedelta(days=7))  # Default to 7 days ago
        with col4:
            to_date = st.date_input("To Date", min_value=min_from_date, max_value=today, 
                                    value=today)  # Default to today
        
        # Fetch commodities
        commodities = fetch_commodity_list(state if state != "All" else "", 
                                        city if city != "All" else "", 
                                        from_date.strftime("%Y-%m-%d"), 
                                        to_date.strftime("%Y-%m-%d"))
        commodity = st.selectbox("Select Commodity", ["All"] + commodities)
        
        # Fetch data button
        if st.button("Fetch Data"):
            trade_df = fetch_trade_data(state if state != "All" else "", 
                                    city if city != "All" else "", 
                                    commodity if commodity != "All" else "", 
                                    from_date.strftime("%Y-%m-%d"), 
                                    to_date.strftime("%Y-%m-%d"))
            
            if not trade_df.empty:
                st.session_state["trade_df"] = trade_df
                st.success("Data fetched successfully!")
            else:
                st.error("No data found for the selected filters.")
                st.session_state["trade_df"] = pd.DataFrame()
        
        # Display data if available
        if "trade_df" in st.session_state and not st.session_state["trade_df"].empty:
            trade_df = st.session_state["trade_df"]
            
            # Convert price columns to numeric
            for col in ["min_price", "modal_price", "max_price", "commodity_arrivals", "commodity_traded"]:
                trade_df[col] = pd.to_numeric(trade_df[col], errors="coerce")
            
            # Clean APMC names
            trade_df["cleaned_apmc"] = trade_df["apmc"].apply(clean_apmc_name)
            
            # Warn about cleaned APMC names not in IndianCities.csv
            unmatched_apmcs = trade_df[~trade_df["cleaned_apmc"].isin(cities_df["City"])]["cleaned_apmc"].unique()
            if len(unmatched_apmcs) > 0:
                st.warning(f"Cleaned APMC names not found in IndianCities.csv: {', '.join(unmatched_apmcs)}. These will not appear in geospatial visualizations. Consider adding them to the CSV.")
            
            
            # Display raw data
            st.subheader("Trade Data")
            st.dataframe(trade_df[["state", "apmc", "cleaned_apmc", "commodity", "min_price", "modal_price", 
                                "max_price", "commodity_arrivals", "commodity_traded", "Commodity_Uom", "created_at"]])
            
            # Visualizations
            st.subheader("Visualizations")
            
            # Line Chart: Price trends over time
            if "created_at" in trade_df.columns and "modal_price" in trade_df.columns:
                trade_df["created_at"] = pd.to_datetime(trade_df["created_at"])
                fig = px.line(trade_df, x="created_at", y=["min_price", "modal_price", "max_price"], 
                            color="commodity", title="Price Trends Over Time",
                            labels={"created_at": "Date", "value": "Price (Rs.)", "variable": "Price Type"})
                st.plotly_chart(fig)
            
            # Bar Graph: Modal prices by APMC
            if "apmc" in trade_df.columns and "modal_price" in trade_df.columns:
                fig = px.bar(trade_df, x="apmc", y="modal_price", color="commodity", 
                            title="Modal Prices by APMC", labels={"modal_price": "Modal Price (Rs.)", "apmc": "APMC"})
                st.plotly_chart(fig)
            
            # Heatmap: Correlation between numerical variables
            st.subheader("Correlation Heatmap")
            numeric_cols = ["min_price", "modal_price", "max_price", "commodity_arrivals", "commodity_traded"]
            corr_matrix = trade_df[numeric_cols].corr()
            fig, ax = plt.subplots()
            sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
            ax.set_title("Correlation Heatmap of Trade Metrics")
            st.pyplot(fig)
            
            # Box Plot: Modal price distribution
            if "modal_price" in trade_df.columns:
                st.subheader("Box Plot: Modal Price Distribution")
                if commodity == "All" and "commodity" in trade_df.columns:
                    fig = px.box(trade_df, x="commodity", y="modal_price", 
                                title="Modal Price Distribution by Commodity",
                                labels={"modal_price": "Modal Price (Rs.)", "commodity": "Commodity"})
                else:
                    fig = px.box(trade_df, x="created_at", y="modal_price", 
                                title="Modal Price Distribution by Date",
                                labels={"modal_price": "Modal Price (Rs.)", "created_at": "Date"})
                st.plotly_chart(fig)
            
            # 3D Scatter Plot: Modal price, arrivals, and traded
            if all(col in trade_df.columns for col in ["modal_price", "commodity_arrivals", "commodity_traded", "commodity"]):
                st.subheader("3D Scatter Plot: Price, Arrivals, and Traded Volume")
                fig = px.scatter_3d(trade_df, x="commodity_arrivals", y="commodity_traded", z="modal_price", 
                                    color="commodity", title="3D Scatter: Arrivals, Traded, and Modal Price",
                                    labels={"commodity_arrivals": "Arrivals", "commodity_traded": "Traded", 
                                            "modal_price": "Modal Price (Rs.)"})
                st.plotly_chart(fig)
            
            # Bubble Map: Price visualization
            st.subheader("Price Bubble Map")
            map_df = trade_df.merge(cities_df, left_on="cleaned_apmc", right_on="City", how="inner")
            if not map_df.empty:
                m = folium.Map(location=[20.5937, 78.9629], zoom_start=5)  # Center of India
                for _, row in map_df.iterrows():
                    folium.CircleMarker(
                        location=[row["Latitude"], row["Longitude"]],
                        radius=row["modal_price"] / 1000,  # Scale radius by price
                        popup=f"{row['apmc']}: {number_format(row['modal_price'])} Rs.",
                        color="blue",
                        fill=True,
                        fill_color="blue"
                    ).add_to(m)
                st_folium(m, width=700, height=500)
            else:
                st.warning("No matching cities found for bubble map. Ensure cleaned APMC names exist in IndianCities.csv.")
            
            # Geospatial Heatmap: Price intensity
            st.subheader("Geospatial Heatmap: Price Intensity")
            if not map_df.empty:
                m = folium.Map(location=[20.5937, 78.9629], zoom_start=5)
                heat_data = [[row["Latitude"], row["Longitude"], row["modal_price"]] for _, row in map_df.iterrows()]
                HeatMap(heat_data, radius=15).add_to(m)
                st_folium(m, width=700, height=500)
            else:
                st.warning("No matching cities found for geospatial heatmap. Ensure cleaned APMC names exist in IndianCities.csv.")
            
            # Parallel Coordinates Plot: Multivariate analysis
            if all(col in trade_df.columns for col in ["min_price", "modal_price", "max_price", "commodity_arrivals", "commodity_traded"]):
                st.subheader("Parallel Coordinates Plot (Multivariate Analysis)")
                plot_df = trade_df[["min_price", "modal_price", "max_price", "commodity_arrivals", "commodity_traded", "commodity"]].dropna()
                for col in numeric_cols:
                    plot_df[col] = (plot_df[col] - plot_df[col].min()) / (plot_df[col].max() - plot_df[col].min())
                fig = px.parallel_coordinates(plot_df, color="modal_price", 
                                            labels={"min_price": "Min Price", "modal_price": "Modal Price", 
                                                    "max_price": "Max Price", "commodity_arrivals": "Arrivals", 
                                                    "commodity_traded": "Traded"},
                                            title="Parallel Coordinates: Trade Metrics")
                st.plotly_chart(fig)
            
            # Statistical Analysis
            st.subheader("Statistical Analysis")
            
            # Simple Linear Regression: Commodity Arrivals vs. Modal Price
            if "commodity_arrivals" in trade_df.columns and "modal_price" in trade_df.columns:
                X = trade_df["commodity_arrivals"].dropna()
                y = trade_df["modal_price"].dropna()
                if len(X) > 1 and len(y) > 1:
                    X = sm.add_constant(X)  # Add intercept
                    model = sm.OLS(y, X).fit()
                    st.write("### Linear Regression: Commodity Arrivals vs. Modal Price")
                    st.write(model.summary())
                    
                    # Plot regression
                    fig, ax = plt.subplots()
                    sns.regplot(x="commodity_arrivals", y="modal_price", data=trade_df, ax=ax)
                    ax.set_title("Regression: Arrivals vs. Modal Price")
                    ax.set_xlabel("Commodity Arrivals")
                    ax.set_ylabel("Modal Price (Rs.)")
                    st.pyplot(fig)
            
            # T-Test: Compare modal prices between time periods
            if "created_at" in trade_df.columns and "modal_price" in trade_df.columns:
                trade_df["created_at"] = pd.to_datetime(trade_df["created_at"])
                mid_point = pd.to_datetime("2025-04-15")  # Split at mid-week
                group1 = trade_df[trade_df["created_at"] <= mid_point]["modal_price"].dropna()
                group2 = trade_df[trade_df["created_at"] > mid_point]["modal_price"].dropna()
                if len(group1) > 1 and len(group2) > 1:
                    t_stat, p_value = stats.ttest_ind(group1, group2)
                    st.write("### T-Test: Modal Prices Between Time Periods")
                    st.write("Comparing April 13–15 vs. April 16–19")
                    st.write(f"T-Statistic: {t_stat:.4f}, P-Value: {p_value:.4f}")
                    if p_value < 0.05:
                        st.write("Significant difference in modal prices between periods (p < 0.05).")
                    else:
                        st.write("No significant difference in modal prices between periods (p >= 0.05).")
                else:
                    st.write("### T-Test: Not enough data for time period comparison.")
                    
            
            # Confidence Interval: Modal price by commodity or APMC
            if "modal_price" in trade_df.columns:
                st.write("### Confidence Intervals: Modal Price")
                if commodity == "All" and "commodity" in trade_df.columns:
                    for comm in trade_df["commodity"].unique():
                        prices = trade_df[trade_df["commodity"] == comm]["modal_price"].dropna()
                        if len(prices) > 1:
                            mean, ci_lower, ci_upper = calculate_confidence_interval(prices)
                            st.write(f"{comm}: Mean = {mean:.2f} Rs., 95% CI = [{ci_lower:.2f}, {ci_upper:.2f}] Rs.")
                elif city == "All" and "apmc" in trade_df.columns:
                    for apmc in trade_df["apmc"].unique():
                        prices = trade_df[trade_df["apmc"] == apmc]["modal_price"].dropna()
                        if len(prices) > 1:
                            mean, ci_lower, ci_upper = calculate_confidence_interval(prices)
                            st.write(f"{apmc}: Mean = {mean:.2f} Rs., 95% CI = [{ci_lower:.2f}, {ci_upper:.2f}] Rs.")
                else:
                    prices = trade_df["modal_price"].dropna()
                    if len(prices) > 1:
                        mean, ci_lower, ci_upper = calculate_confidence_interval(prices)
                        st.write(f"Selected Data: Mean = {mean:.2f} Rs., 95% CI = [{ci_lower:.2f}, {ci_upper:.2f}] Rs.")

    crop_market_geostatistics_tab()

# ----- Footer -----
st.markdown("---")
st.caption("© 2025 Crop Disease Analytics | Built with Streamlit")
