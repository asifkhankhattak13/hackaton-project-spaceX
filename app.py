# spacex_dashboard.py
import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import joblib

# Load data
df = pd.read_csv("spacex_cleaned.csv")
# get year
df['year'] = pd.to_datetime(df['launch_date']).dt.year

# Load trained model
model = joblib.load("random_forest_model.pkl")

# Page setup
st.set_page_config(layout="wide")
st.title("SpaceX Launch Analysis & Prediction Dashboard")

# Historical Launch Data Viewer 
st.header("Launch History Viewer")

years = sorted(df['year'].unique())
launchpads = sorted(df['site_name'].unique())

selected_year = st.selectbox("Filter by Year", options=["All"] + years)
selected_pad = st.selectbox("Filter by Launchpad", options=["All"] + launchpads)

filtered_df = df.copy()
if selected_year != "All":
    filtered_df = filtered_df[filtered_df['year'] == selected_year]
if selected_pad != "All":
    filtered_df = filtered_df[filtered_df['site_name'] == selected_pad]

st.dataframe(filtered_df)

#  Geospatial Map
st.header("Launch Sites Map")

map_df = filtered_df.copy()
map_df['color'] = map_df['success'].apply(lambda x: 'green' if x == 1 else 'red')

launch_map = folium.Map(location=[28.5, -80.6], zoom_start=3)

for _, row in map_df.iterrows():
    location_name = row.get('location', 'Unknown')
    lat_lon = location_name.split(',')[-2:] if ',' in location_name else ["28.5", "-80.6"]
    try:
        lat, lon = float(lat_lon[0]), float(lat_lon[1])
    except:
        lat, lon = 28.5, -80.6  # fallback

    folium.CircleMarker(
        location=[lat, lon],
        radius=6,
        color=row['color'],
        fill=True,
        fill_opacity=0.7,
        popup=f"{row['mission_name']} | {row['rocket_name']} | Success: {row['success']}"
    ).add_to(launch_map)

st_folium(launch_map, width=700)

# Prediction Tool
st.header("Launch Success Predictor")

st.markdown("Enter launch details to estimate success probability:")

# Feature Inputs (match features used in training)
payload_mass = st.slider("Payload Mass (kg)", min_value=0, max_value=10000, value=5000)
year = st.slider("Launch Year", min_value=2006, max_value=2025, value=2020)

rocket_name = st.selectbox("Rocket Name", sorted(df['rocket_name'].unique()))
launchpad_name = st.selectbox("Launchpad Name", sorted(df['site_name'].unique()))

# Prepare input in same format as training set
input_dict = {
    'payload_mass': payload_mass,
    'year': year,
    f'rocket_name_{rocket_name}': 1,
    f'site_name_{launchpad_name}': 1
}


# All feature columns from training
model_features = model.feature_names_in_
input_data = pd.DataFrame([0] * len(model_features), index=model_features).T

# Set matching features
for col in input_dict:
    if col in input_data.columns:
        input_data[col] = input_dict[col]

if st.button("Predict Launch Success"):
    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]
    st.success(f"Predicted Success: {'Yes' if pred == 1 else 'No'} with probability {prob:.2f}")