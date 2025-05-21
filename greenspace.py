import streamlit as st
import pandas as pd
import numpy as np
import osmnx as ox
import folium
from sklearn.preprocessing import MinMaxScaler
from streamlit_folium import st_folium
import pickle
import hashlib
import os

# Set page configuration
st.set_page_config(layout="wide")
st.title("🌱 Urban Green Space Visualizer")

# City coordinates dictionary
city_coords = {
    'Amaravati': (16.5096679, 80.5184535),
    'Visakhapatnam': (17.6935526, 83.2921297),
    'Guwahati': (26.1805978, 91.753943),
    'Patna': (25.6093239, 85.1235252),
    'Chandigarh': (30.7334421, 76.7797143),
    'Delhi': (28.6273928, 77.1716954),
    'Ahmedabad': (23.0215374, 72.5800568),
    'Gurugram': (28.4646148, 77.0299194),
    'Jorapokhar': (23.7167069, 86.4110166),
    'Bengaluru': (12.98815675, 77.62260003796),
    'Thiruvananthapuram': (8.4882267, 76.947551),
    'Mumbai': (19.054999, 72.8692035),
    'Shillong': (25.5760446, 91.8825282),
    'Bhopal': (23.2584857, 77.401989),
    'Aizawl': (23.7433532, 92.7382756),
    'Brajrajnagar': (21.8498594, 83.9254698),
    'Talcher': (20.9322302, 85.2005822),
    'Amritsar': (31.6356659, 74.8787496),
    'Jaipur': (26.9154576, 75.8189817),
    'Hyderabad': (17.360589, 78.4740613),
    'Chennai': (13.0836939, 80.270186),
    'Coimbatore': (11.0018115, 76.9628425),
    'Lucknow': (26.8381, 80.9346001),
    'Kolkata': (22.5726459, 88.3638953)
}

# Load pollution dataset
@st.cache_data
def load_pollution_data():
    return pd.read_csv("merged_local_pollution.csv")

# Get parks for a city
@st.cache_data
def get_parks(city_name):
    # Use a filename safe for the city name
    city_hash = hashlib.sha1(city_name.encode('utf-8')).hexdigest()
    cache_dir = "cache"
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    cache_path = f"{cache_dir}/{city_hash}_parks.pkl"
    try:
        # Try to load from cache
        with open(cache_path, "rb") as f:
            parks = pickle.load(f)
        return parks
    except Exception:
        # If not cached, fetch and cache
        parks = ox.features_from_place(city_name + ", India", tags={'leisure': 'park'})
        with open(cache_path, "wb") as f:
            pickle.dump(parks, f)
        return parks

# Compute prioritization scores
@st.cache_data(show_spinner=False)
def prioritize(df, selected_city=None):
    scores = []
    for _, row in df.iterrows():
        city = row["City"]
        lat, lon = city_coords.get(city, (None, None))
        if not lat or not lon:
            continue
        pollution_score = row["Average_AQI"]
        if city == selected_city:
            try:
                parks = get_parks(city)
                park_density = len(parks)
            except:
                park_density = 0
        else:
            park_density = np.nan
        scores.append({
            "City": city,
            "Latitude": lat,
            "Longitude": lon,
            "PollutionScore": pollution_score,
            "ParkDensity": park_density
        })
    df_score = pd.DataFrame(scores)
    # Only scale if at least one city has a valid park_density
    if df_score["ParkDensity"].notna().sum() == 0:
        df_score["PriorityScore"] = np.nan
        return df_score.sort_values("City")
    scaler = MinMaxScaler()
    # Only scale non-nan rows
    mask = df_score["ParkDensity"].notna()
    df_score.loc[mask, ["PollutionScore", "ParkDensity"]] = scaler.fit_transform(df_score.loc[mask, ["PollutionScore", "ParkDensity"]])
    df_score["PriorityScore"] = df_score["PollutionScore"] * 0.7 + df_score["ParkDensity"] * 0.3
    return df_score.sort_values("PriorityScore", ascending=False)

# Recommend monitoring stations
# Recommend monitoring stations based on high AQI and no nearby parks

def recommend_stations(df_score, df_pollution):
    recommendations = {}
    for _, row in df_score.iterrows():
        city = row['City']
        lat, lon = row['Latitude'], row['Longitude']
        # Filter pollution data for the given city
        city_df = df_pollution[df_pollution['City'] == city]
        # Filter non-empty stations and pick top 5 stations with the highest AQI (high pollution)
        best_stations = city_df[city_df['StationName'].notna()][['StationName', 'Average_AQI', 'Latitude', 'Longitude']]
        best_stations = best_stations.sort_values(by='Average_AQI', ascending=False)
        # Recommend stations with AQI > 100 and no park within 2km
        parks = get_parks(city)
        recommended = []
        for _, st in best_stations.iterrows():
            aqi = st['Average_AQI']
            if pd.isna(aqi) or aqi < 100:
                continue
            st_lat, st_lon = st['Latitude'], st['Longitude']
            # Check if any park is within 2km
            has_nearby_park = False
            for _, park in parks.iterrows():
                park_lat, park_lon = park.geometry.centroid.y, park.geometry.centroid.x
                dist = np.sqrt((float(st_lat) - float(park_lat))**2 + (float(st_lon) - float(park_lon))**2) * 111  # rough km
                if dist < 2.0:
                    has_nearby_park = True
                    break
            if not has_nearby_park:
                recommended.append([st['StationName'], aqi])
            if len(recommended) >= 5:
                break
        recommendations[city] = recommended
    # Convert recommendations to a list of dictionaries for easy presentation
    result = [{"City": city, "Stations": stations} for city, stations in recommendations.items()]
    return result




# Show static-style folium map
def show_map(df_score, selected_city=None):
    m = folium.Map(location=[20.5937, 78.9629], zoom_start=5, control_scale=False, zoom_control=False)

    for _, row in df_score.iterrows():
        color = "green" if row["City"] == selected_city else "red"
        folium.CircleMarker(
            location=[row["Latitude"], row["Longitude"]],
            radius=7,
            color=color,
            fill=True,
            fill_opacity=0.6,
            popup=str(f"{row['City']}<br>Priority Score: {row['PriorityScore']:.2f}")
        ).add_to(m)

    if selected_city:
        try:
            gdf_parks = get_parks(selected_city)
            for _, row in gdf_parks.iterrows():
                if row.geometry.centroid.is_valid:
                    folium.Marker(
                        location=[row.geometry.centroid.y, row.geometry.centroid.x],
                        icon=folium.Icon(color='green', icon='tree-conifer'),
                        popup=str("Park")
                    ).add_to(m)
        except:
            st.warning(f"Parks data for {selected_city} could not be loaded.")

    return m

# Main app logic
def main():
    df_pollution = load_pollution_data()
    all_cities = sorted(df_pollution['City'].drop_duplicates())
    selected_city = st.selectbox("🔍 Select a city to visualize park distribution:", all_cities, key="city_select")
    # Only recompute prioritize when selected_city changes
    df_score = prioritize(df_pollution, selected_city)

    st.subheader("🗺️ Green Space & Pollution Map")
    map_object = show_map(df_score, selected_city)
    st_folium(map_object, width=1000, height=550, key="map")

    st.subheader("📍 Recommended Green Space Locations")
    stations = recommend_stations(df_score, df_pollution)
    for item in stations:
        if item['City'] == selected_city:
            st.markdown(f"**{item['City']}**")
            for station in item['Stations']:
                st.write(f"Station: {station[0]}, AQI: {station[1]}")

# Run the app
if __name__ == "__main__":
    main()
