
import geopandas as gpd
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import pandas as pd
import dotenv
import json
import pandas as pd
import os



# Optionally: export to CSV
# df_all_aps.to_csv("all_access_points.csv", index=False)

dotenv.load_dotenv()

# Load API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY



# ---------------------
# Load preprocessed data
# ---------------------
# Convert floor names like "1st Floor", "2nd Floor" to integers
def extract_floor_number(floor_str):
    try:
        return str(floor_str.split()[0][0])  # Gets the first digit (works for 1st, 2nd, etc.)
    except:
        return None



# @st.cache_data
def geojson_load_data():
    # Replace with paths to your processed files if needed
    df = pd.read_pickle("Data/monthly_utilisation_when_occupied_vs_occupancy.pkl")
    gdf = gpd.read_file( "Data/SFO12.geojson")

    # Define the months
    # months = pd.date_range("2025-04-01", "2025-05-01", freq="MS")
    # month_names = months.strftime("%B %Y")  # E.g., "January 2025"

    # Repeat the dataframe for each month
    # repeated_gdf = pd.concat(
    #     [gdf.assign(Month=month, Month_Name=name) for month, name in zip(months, month_names)],
    #     ignore_index=True
    # )
    # Ensure CRS is correct
    # repeated_gdf = repeated_gdf.to_crs(epsg=4326)
    df['lvl'] = df['lvl'].astype(str)

    df_agg = df.groupby(['Floor Name','lvl','Workspace Name', 'Workspace Type',
                          'Space_Capacity_label', 'Workspace_Capacity_Category']).agg(
                          {'Monthly_Utilisation_when_Occupied_mean': 'median',
                          'Monthly_Occupancy%': 'median',
                          'Avg_Dwell_Time': 'median'}
                      ).reset_index()
    # round off and add labels
    df_agg['Monthly_Utilisation'] = df_agg['Monthly_Utilisation_when_Occupied_mean'].round(0).astype(str) + '%'
    df_agg['Monthly_Occupancy'] = df_agg['Monthly_Occupancy%'].round(0).astype(str) + '%'
    df_agg['Monthly_Dwell_Time'] = df_agg['Avg_Dwell_Time'].round(0).astype(str) + ' min'

    gdf['lvl'] = gdf['lvl'].astype(str)
    # clean names
    df_agg['Workspace Name'] = (df_agg['Workspace Name'].astype(str).str.strip().str.lower().str.replace(r'\s+', ' ', regex=True)) # collapse multiple spaces
    gdf['name'] = (gdf['name'].astype(str).str.strip().str.lower().str.replace(r'\s+', ' ', regex=True) ) # collapse multiple spaces

    # repeated_gdf['lvl'] = repeated_gdf['lvl'].astype(str)
    gdf_mod = pd.merge(gdf, df_agg[['Floor Name','lvl','Workspace Name', 'Workspace Type',
                                    'Space_Capacity_label', 'Workspace_Capacity_Category',
                                    'Monthly_Utilisation','Monthly_Occupancy','Monthly_Dwell_Time']], left_on=['lvl','name'],
                right_on=['lvl','Workspace Name'], how='left')

    gdf_mod = gdf_mod.loc[gdf_mod['sType'] != 'Floor Outline',].reset_index(drop=True)
    return gdf_mod


# Set your folder path here
json_folder_path = "Data/access_point/"

# @st.cache_data
def load_json_files(folder_path):
    # List to collect access point records from all files
    all_ap_records = []
    # Loop through all text files in the folder
    for filename in os.listdir(json_folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, "r") as file:
                    data = json.load(file)

                # Extract floor name
                floor_name = None
                for loc in data.get("locationHierarchy", []):
                    if loc.get("type") == "floor":
                        floor_name = loc.get("name")
                        break

                # Extract access point details
                access_points = data.get("accessPoints", [])
                for ap in access_points:
                    all_ap_records.append({
                        "Source_File": filename,
                        "Floor_Name": floor_name,
                        "Device_Name": ap.get("name"),
                        "MAC_Address": ap.get("macAddress"),
                        "IP_Address": ap.get("ipAddress"),
                        "Model": ap.get("model"),
                        "Serial": ap.get("serial"),
                        "Geo_Position": ap.get("geoPosition"),
                        "X(Feet)": ap.get("x"),
                        "Y(Feet)": ap.get("y")
                    })

            except Exception as e:
                print(f"Error processing file {filename}: {e}")

    # Create a combined DataFrame
    df_all_aps = pd.DataFrame(all_ap_records)
    df_all_aps["lvl"] = df_all_aps["Floor_Name"].apply(extract_floor_number)
    return df_all_aps

# Show the DataFrame
df_all_aps = load_json_files(json_folder_path)
gdf_mod = geojson_load_data()




###################################################################
st.set_page_config(layout="wide")  # <– Set wide layout here

# Sidebar filters
st.sidebar.title("Filter Options")

# Month filter
# month_options = sorted(gdf_mod["Month_Name"].dropna().unique())
# selected_month = st.sidebar.selectbox("Select Month:", month_options)

# Floor filter
floor_options = sorted(gdf_mod["lvl"].dropna().unique())
selected_floor = st.sidebar.selectbox("Select Floor:", floor_options)

# Room capacity filter
# capacity_options = sorted(gdf_mod["Workspace_Capacity_Category"].dropna().unique())
# selected_capacity = st.sidebar.selectbox("Select Room Capacity:", capacity_options)

# Remove the marker

###  Filter and Clean Data
st.title("Spatial Analysis: Room Utilisation & Occupancy (Jan - May 2025)")

# Utilisation and Occupancy filter sliders
util_threshold = st.slider("Utilisation %", 0, 100, 40)
occ_threshold = st.slider("Occupancy %", 0, 100, 40)


# Clean and filter data
gdf_filtered = gdf_mod.copy()

# Filter by floor and Month
gdf_filtered = gdf_filtered[gdf_filtered['lvl'] == selected_floor].reset_index(drop=True)

# Remove rooms with anomalously high utilization or occupancy
def parse_percentage(val):
    try:
        return float(str(val).replace("%", "").strip())
    except:
        return np.nan

gdf_filtered["util_pct"] = gdf_filtered["Monthly_Utilisation"].apply(parse_percentage)
gdf_filtered["occ_pct"] = gdf_filtered["Monthly_Occupancy"].apply(parse_percentage)

# Filter out anomalies
# gdf_filtered = gdf_filtered[
#     (gdf_filtered["util_pct"] <= util_threshold) &
#     (gdf_filtered["occ_pct"] <= occ_threshold)
# ]

## shortlist flag
# gdf_filtered['color_legend'] = np.where(
#     np.logical_or(gdf_filtered["util_pct"] >= util_threshold,
#                        gdf_filtered["occ_pct"] >= occ_threshold),
#     'Anomalous Rooms',
#     np.where(
#         gdf_filtered['Workspace Name'].isna(),
#         'Other Workspaces','Other Meeting Rooms'
#     )
# )

gdf_filtered['color_legend'] = np.where(
    (gdf_filtered["util_pct"] >= util_threshold) & (gdf_filtered["occ_pct"] >= occ_threshold),
    'High Utilisation & Occupancy',
    np.where(
        (gdf_filtered["util_pct"] >= util_threshold) & (gdf_filtered["occ_pct"] < occ_threshold),
        'High Utilisation But Low Occupancy',
        np.where(
            (gdf_filtered["util_pct"] < util_threshold) & (gdf_filtered["occ_pct"] < occ_threshold),
            'Low Utilisation & Occupancy',
            np.where(
                (gdf_filtered["util_pct"] < util_threshold) & (gdf_filtered["occ_pct"] >= occ_threshold),
                'Low Utilisation But High Occupancy',
                'Other Workspaces'
            )
        )
    )
)


########### Animate Map Through Months (Plotly)

# # Add color for visual differentiation
# gdf_filtered['color'] = gdf_filtered['name'].apply(
#     lambda x: 'green' if str(x).strip().lower() == 'inside out' else 'blue'
# )

# Drop NA months and format
# gdf_filtered = gdf_filtered.dropna(subset=['Month Name'])
# gdf_filtered['Month_Name'] = gdf_filtered['Month_Name'].astype(str)

# STEP 1: Parse latitude and longitude from geoPosition
# Filter by floor and Month
df_all_aps[['lat', 'lon']] = df_all_aps['Geo_Position'].str.split(',', expand=True).astype(float)
df_all_aps = df_all_aps.loc[df_all_aps["lvl"] == selected_floor,]

# STEP 2: Create base choropleth map
fig = px.choropleth_mapbox(
    gdf_filtered,
    geojson=gdf_filtered.geometry,
    locations=gdf_filtered.index,
    hover_name="name",
    hover_data=["lvl", 'Space_Capacity_label', 'Workspace_Capacity_Category',
                'Monthly_Occupancy', 'Monthly_Utilisation', 'Monthly_Dwell_Time'],
    color="color_legend",
    color_discrete_map={
        'High Utilisation & Occupancy': '#d62728',
        'High Utilisation But Low Occupancy': '#87CEEB',
        'Low Utilisation & Occupancy': '#2ca02c',
        'Low Utilisation But High Occupancy': '#1f77b4',
        'Other Workspaces': "grey"
    },
    mapbox_style="carto-positron",
    center={
        "lat": gdf_filtered.geometry.centroid.y.mean(),
        "lon": gdf_filtered.geometry.centroid.x.mean()
    },
    zoom=19,
    opacity=0.6,
    title=f"Animated Spatial Analysis - Floor {selected_floor}"
)

# STEP 3: Overlay access points using scatter_mapbox
fig.add_trace(go.Scattermapbox(
    lat=df_all_aps['lat'],
    lon=df_all_aps['lon'],
    mode='markers+text',
    marker=go.scattermapbox.Marker(
        size=10,
        color='black',
        symbol='circle'
    ),
    # text=df_all_aps['Device_Name'],
    # textposition="top right",
    name='Access Points',
    hoverinfo='text',
    hovertext=df_all_aps.apply(
        lambda row: f"{row['Device_Name']}<br>{row['IP_Address']}<br>{row['Model']}", axis=1
    )
))

# STEP 4: Update layout (Mapbox token not needed for 'carto-positron')
fig.update_layout(
    margin={"r":0,"t":40,"l":0,"b":0},
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
)


# fig.update_traces(marker_line_width=1, marker_line_color="black")
fig.update_layout(margin={"r": 0, "t": 30, "l": 0, "b": 0}, height=800)
st.plotly_chart(fig, use_container_width=True)
