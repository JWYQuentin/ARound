import numpy as np
import pandas as pd
import streamlit as st
import joblib
import matplotlib.pyplot as plt

# Load trained model
model = joblib.load("/Users/zhangyuchuan/Downloads/random_forest_model.pkl")

st.image("/Users/zhangyuchuan/Desktop/ARE.logo.png", width=200)  # Adjust width as needed
st.title("Ticket Pricing & Revenue Prediction")

# User Inputs
capacity = st.number_input("🎭 Event Capacity", min_value=0, max_value=100000, value=5000)
listeners = st.number_input("🎧 Artist Monthly Listeners", min_value=0, max_value=1000000000000, value=1000000)
yt_videos = st.number_input("📺 Artisit YouTube Video Count", min_value=0, max_value=10000000, value=50)
yt_views = st.number_input("📈 Artist YouTube View Count", min_value=0, max_value=10000000000000, value=100000000)
sp_followers = st.number_input("🎵 Artist Spotify Followers", min_value=0, max_value=10000000000000, value=1000000)

# Define ticket price range
ticket_min_range = st.slider("💵 Estimated Min Ticket Price Range (USD)", min_value=10, max_value=200, value=(20, 22))
ticket_max_range = st.slider("💵 Estimated Max Ticket Price Range (USD)", min_value=50, max_value=500, value=(100, 102))

# Start Prediction Button
if st.button("🚀 Predict Optimal Pricing range and Revenue"):
    # Generate ticket price combinations
    ticket_min_values = np.arange(ticket_min_range[0], ticket_min_range[1] + 1)
    ticket_max_values = np.arange(ticket_max_range[0], ticket_max_range[1] + 1)

    # Store predicted revenue
    revenue_data = []

    # Iterate through combinations of min/max ticket prices
    for min_price in ticket_min_values:
        for max_price in ticket_max_values:
            input_data = pd.DataFrame([{
                "Avg._Event_Capacity": capacity,
                "Ticket_Price_Max_USD": max_price,
                "Ticket_Price_Min_USD": min_price,
                "headliner_monthly_listeners": listeners,
                "yt_Video_Count": yt_videos,
                "yt_View_Count": yt_views,
                "sp_followers": sp_followers
            }])
            
            predicted_revenue = model.predict(input_data)[0]
            revenue_data.append(((min_price, max_price), predicted_revenue))

    # Convert to DataFrame
    df_revenue = pd.DataFrame(revenue_data, columns=["Ticket Price Combo", "Predicted Revenue"])

    # Find the optimal ticket price combination
    optimal_row = df_revenue.loc[df_revenue["Predicted Revenue"].idxmax()]
    optimal_prices = optimal_row["Ticket Price Combo"]
    optimal_revenue = optimal_row["Predicted Revenue"]

    # Display histogram
    st.subheader("📊 Revenue Prediction Histogram")
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.bar([str(combo) for combo in df_revenue["Ticket Price Combo"]], df_revenue["Predicted Revenue"], color='skyblue')
    plt.xlabel("(Min Price, Max Price)")
    plt.ylabel("Predicted Revenue")
    plt.title("Predicted Revenue Across Ticket Price Combinations")
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig)

    # Display optimal ticket pricing strategy
    st.subheader("🏆 Optimal Ticket Pricing")
    st.success(f"💰 Highest Predicted Revenue: **${optimal_revenue:,.2f}**")
    st.info(f"🎟 Optimal Min Ticket Price: **${optimal_prices[0]}**")
    st.info(f"🎟 Optimal Max Ticket Price: **${optimal_prices[1]}**")
