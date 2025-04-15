import streamlit as st
from scripts.weather import get_weather
from scripts.geolocation import get_coordinates

# ----- App Title -----
st.set_page_config(page_title="Crop Disease Analytics", layout="wide")
st.title("🌾 Crop Disease Analytics Dashboard")

# ----- Image Upload Section -----
uploaded_image = st.file_uploader("📷 Upload an image of the diseased crop", type=["jpg", "png"])

if uploaded_image:
    st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)
    
    # --- Placeholder: ML Output ---
    st.markdown("### 🧠 ML Prediction (Simulated)")
    st.success("🟢 Crop: **Wheat**")
    st.error("🔴 Disease: **Leaf Rust**")
    st.info("📊 Confidence: **92.4%**")

# ----- Location Input -----
st.markdown("---")
st.subheader("📍 Enter your Region to Fetch Weather & Location Insights")
location = st.text_input("Enter a city/town name", value="Chandigarh")

if location:
    lat, lon = get_coordinates(location)
    weather = get_weather(location)

    if lat and lon and weather:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Latitude", lat)
            st.metric("Longitude", lon)
        with col2:
            st.metric("Temperature (°C)", weather["main"]["temp"])
            st.metric("Humidity (%)", weather["main"]["humidity"])
        
        st.success(f"🌤️ Weather Summary: {weather['weather'][0]['description'].title()}")
        
        # 🔔 Weather Alert Based on Humidity (Logic-Based Insight)
        if weather["main"]["humidity"] > 80:
            st.warning("⚠️ High humidity detected. Fungal diseases may be more prevalent.")
    else:
        st.error("❌ Could not fetch weather or geolocation data.")

# ----- Visual Dashboard Section -----
st.markdown("---")
st.subheader("📊 Analytics Dashboard")

tab1, tab2, tab3 = st.tabs(["Disease vs Region", "Weather vs Disease", "Statistical Analysis"])

with tab1:
    st.markdown("#### 🗺️ Disease Occurrence Map (Placeholder)")
    st.info("Visualize the frequency of diseases by region using geospatial data.")

with tab2:
    st.markdown("#### 📈 Weather Conditions vs Disease Trends")
    st.info("Visualizing trends with temperature, humidity, and disease severity.")

with tab3:
    st.markdown("#### 🧮 Statistical Summary")
    st.markdown("""
    - ✅ Mean Temperature: **27.4°C**
    - ✅ Mean Humidity: **68%**
    - ✅ Hypothesis Test: Is disease rate higher in high humidity? → *p-value: 0.04*
    - ✅ Confidence Interval (95%) for temp: **[25.6°C, 29.2°C]**
    """)
    st.caption("These are dummy values. Real stats will be generated from processed data.")

# ----- Footer -----
st.markdown("---")
st.caption("© 2025 Crop Disease Analytics | Built with Streamlit")