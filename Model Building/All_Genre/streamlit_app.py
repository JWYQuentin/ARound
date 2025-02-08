import streamlit as st
import joblib
import pandas as pd

# Load the trained model (Replace with actual model path)
model = joblib.load("/Users/zhangyuchuan/Downloads/random_forest_model.pkl")

# Page Title
st.set_page_config(page_title="🎟️ A Round Entertainment Ticket Revenue Prediction", page_icon="💰", layout="wide")

# Custom CSS for styling
st.markdown(
    """
    <style>
    .stApp { background-color: #f7f7f7; }
    .stButton>button { background-color: #ff4b4b; color: white; font-size: 18px; }
    .css-18e3th9 { padding: 2rem; }
    .stNumberInput>div>input { font-size: 16px; }
    </style>
    """,
    unsafe_allow_html=True
)

# Main Title
st.title("🎟️ Ticket Revenue Prediction App")
st.subheader("💰 Estimate revenue for your event based on key factors")

# Create two columns for layout
col1, col2 = st.columns(2)

# Sidebar for user inputs
with st.sidebar:
    st.header("🔧 Enter Event Details")
    capacity = st.number_input("🎭 Avg. Event Capacity", min_value=100, max_value=100000, value=5000)
    ticket_max = st.number_input("💵 Max Ticket Price (USD)", min_value=1, max_value=1000, value=100)
    ticket_min = st.number_input("💵 Min Ticket Price (USD)", min_value=1, max_value=1000, value=30)
    listeners = st.number_input("🎧 Headliner Monthly Listeners", min_value=1000, max_value=100000000, value=1000000)
    yt_videos = st.number_input("📺 YouTube Video Count", min_value=0, max_value=10000, value=50)
    yt_views = st.number_input("📈 YouTube View Count", min_value=1000, max_value=1000000000, value=100000000)
    sp_followers = st.number_input("🎵 Spotify Followers", min_value=1000, max_value=1000000000, value=1000000)

# Predict button
if st.sidebar.button("🚀 Predict Revenue"):
    input_data = pd.DataFrame([{
        "Avg._Event_Capacity": capacity,
        "Ticket_Price_Max_USD": ticket_max,
        "Ticket_Price_Min_USD": ticket_min,
        "headliner_monthly_listeners": listeners,
        "yt_Video_Count": yt_videos,
        "yt_View_Count": yt_views,
        "sp_followers": sp_followers
    }])

    # Make Prediction
    prediction = model.predict(input_data)[0]

    # Display prediction with styling
    col1.success(f"💰 Estimated Revenue: **${prediction:,.2f}**")

# Footer
st.markdown("---")
st.markdown("💡 *Developed for accurate revenue forecasting using machine learning.*")

