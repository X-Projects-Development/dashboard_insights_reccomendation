
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
import dotenv
import json



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

def geojson_load_data(selected_weekday,selected_hour):
    # Replace with paths to your processed files if needed
    df = pd.read_pickle("Data/hourly_utilisation_occupancy_environmental_metirc.pkl")
    gdf = gpd.read_file( "Data/SFO12.geojson")
    # df_co2_humidity_temp_agg = pd.read_pickle("df_co2_humidity_temp_agg.pkl")
    df_area = pd.read_csv("Data/workspace_area_sfo12.csv")

    df['lvl'] = df['lvl'].astype(str)

    df_agg = df.loc[np.logical_and(df['Day Name'].isin(selected_weekday),
                                   df['Hour'].isin(selected_hour))].groupby(['Floor Name', 'lvl', 'Workspace Name', 'Workspace Type',
                        'Space Capacity', 'Workspace_Capacity_Category'
                    ]).agg({
                        'Occupancy%': 'mean',
                        'Utilisation_mean': 'mean',
                        'Average_CO2': 'median',
                        'Average_Temp': 'median',
                        'Average_Humidity': 'median',
                        'Average_ApparentTemp': 'median',
                        'CO2_90percentile': lambda x: x.quantile(0.9),
                        'Temp_90percentile': lambda x: x.quantile(0.9),
                        'Humidity_90percentile': lambda x: x.quantile(0.9),
                        'ApparentTemp_90percentile': lambda x: x.quantile(0.9)
                    }).rename(columns={
                        'Occupancy%': 'Monthly_Occupancy',
                        'Utilisation_mean': 'Monthly_Utilisation',
                        'Average_CO2': 'Co2(ppm)',
                        'Average_Temp': 'Temp(C)',
                        'Average_Humidity': 'Humidity(%)',
                        'Average_ApparentTemp': 'ApparentTemp(C)',
                        'CO2_90percentile': 'Co2(ppm)_90th_percentitle',
                        'Temp_90percentile': 'Temp(C)_90th_percentitle',
                        'Humidity_90percentile': 'Humidity(%)_90th_percentitle',
                        'ApparentTemp_90percentile': 'ApparentTemp(C)_90th_percentitle'
                    }).reset_index()
    # round off and add labels
    df_agg['Monthly_Utilisation'] = df_agg['Monthly_Utilisation'].round(0).astype(str) + '%'
    df_agg['Monthly_Occupancy'] = df_agg['Monthly_Occupancy'].round(0).astype(str) + '%'
    df_agg['Space_Capacity_label'] = df_agg['Space Capacity'].round(0).astype(str) + ' seats'
    # df_agg['Monthly_Dwell_Time'] = df_agg['Avg_Dwell_Time'].round(0).astype(str) + ' min'

    gdf['lvl'] = gdf['lvl'].astype(str)
    # clean names
    df_agg['Workspace Name'] = (df_agg['Workspace Name'].astype(str).str.strip().str.lower().str.replace(r'\s+', ' ', regex=True)) # collapse multiple spaces
    gdf['name'] = (gdf['name'].astype(str).str.strip().str.lower().str.replace(r'\s+', ' ', regex=True) ) # collapse multiple spaces
    df_area['Workspace Name'] = (df_area['Workspace Name'].astype(str).str.strip().str.lower().str.replace(r'\s+', ' ', regex=True)) # collapse multiple spaces
    # df_co2_humidity_temp_agg['Workspace Name'] = (df_co2_humidity_temp_agg['Workspace Name'].astype(str).str.strip().str.lower().str.replace(r'\s+', ' ', regex=True)) # collapse multiple spaces


    df_area["lvl"] = df_area["Floor Name"].apply(extract_floor_number)
  

    # round off numbers (dont add unit if nan)
    # Apparent Temp
    df_agg['ApparentTemp(C)'] = df_agg['ApparentTemp(C)'].apply(
        lambda x: f"{x:.1f} C" if pd.notnull(x) else x
    )
    df_agg['ApparentTemp(C)_90th_percentile'] = df_agg['ApparentTemp(C)_90th_percentitle'].apply(
        lambda x: f"{x:.1f} C" if pd.notnull(x) else x
    )

    # Humidity
    df_agg['Humidity(%)'] = df_agg['Humidity(%)'].apply(
        lambda x: f"{x:.1f}%" if pd.notnull(x) else x
    )
    df_agg['Humidity(%)_90th_percentile'] = df_agg['Humidity(%)_90th_percentitle'].apply(
        lambda x: f"{x:.1f}%" if pd.notnull(x) else x
    )

    # Temperature
    df_agg['Temp(C)'] = df_agg['Temp(C)'].apply(
        lambda x: f"{x:.1f} C" if pd.notnull(x) else x
    )
    df_agg['Temp(C)_90th_percentile'] = df_agg['Temp(C)_90th_percentitle'].apply(
        lambda x: f"{x:.1f} C" if pd.notnull(x) else x
    )

    # CO2
    df_agg['Co2(ppm)'] = df_agg['Co2(ppm)'].apply(
        lambda x: f"{x:.0f} ppm" if pd.notnull(x) else x
    )
    df_agg['Co2(ppm)_90th_percentile'] = df_agg['Co2(ppm)_90th_percentitle'].apply(
        lambda x: f"{x:.0f} ppm" if pd.notnull(x) else x
    )

    df_area['Area_Sqft'] = df_area['Area_Sqft'].apply(
        lambda x: f"{round(x):.0f} sqft" if pd.notnull(x) else x
    )
    # repeated_gdf['lvl'] = repeated_gdf['lvl'].astype(str)
    gdf_mod = pd.merge(gdf, df_agg[['Floor Name','lvl','Workspace Name', 'Workspace Type',
                                    'Space_Capacity_label', 'Workspace_Capacity_Category',
                                    'Monthly_Utilisation','Monthly_Occupancy','ApparentTemp(C)',
                                    'ApparentTemp(C)_90th_percentile','Humidity(%)','Humidity(%)_90th_percentitle',
                                    'Temp(C)','Temp(C)_90th_percentitle','Co2(ppm)','Co2(ppm)_90th_percentitle']], left_on=['lvl','name'],
                right_on=['lvl','Workspace Name'], how='left')

    gdf_mod = pd.merge(gdf_mod, df_area[['lvl','Workspace Name','Area_Sqft']], left_on=['lvl','name'],
                right_on=['lvl','Workspace Name'], how='left')

    # gdf_mod = pd.merge(gdf_mod, df_co2_humidity_temp_agg, left_on=['lvl','name'],
    #             right_on=['lvl','Workspace Name'], how='left')

    gdf_mod = gdf_mod.loc[~np.logical_or(gdf_mod['sType'] == 'Floor Outline',
                                        gdf_mod['type'] == 'marker'),].reset_index(drop=True)
    # drop columns
    del gdf_mod['Workspace Name_y']
    del gdf_mod['Workspace Name_x']

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




###################################################################
st.set_page_config(layout="wide")  # <– Set wide layout here

# Sidebar filters
st.sidebar.title("Filter Options")

with st.sidebar.expander("Filter Options (Day & Time)", expanded=False):
  # Filter for Day of week (Multi select)
  weekday_options = ['Monday','Tuesday', 'Wednesday', 'Thursday', 'Friday']
  default_weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
  selected_weekday = st.pills(
      "Select Day of Week",
      options=weekday_options,
      default=default_weekdays,
      selection_mode="multi"
  )


  # Filter for Hour of day (Multi select)
  hours_options = list(range(7, 19))
  default_hours = list(range(7, 19))  # Office hours: 9 AM to 5 PM
  selected_hour = st.multiselect(
      "Select Hour of Day (24Hrs)",
      options=hours_options,
      default=default_hours
  )

################################### Load data ################################### 

# Show the DataFrame
df_all_aps = load_json_files(json_folder_path)
gdf_mod = geojson_load_data(selected_weekday,selected_hour)

# Floor filter
floor_options = sorted(gdf_mod["lvl"].dropna().unique())
selected_floor = st.sidebar.selectbox("Select Floor:", floor_options)

# Main area: Metric Definitions
with st.sidebar.expander("Metric Definitions", expanded=False):
    st.markdown("""
    **Occupancy (%):**  
    The average percentage of time a workspace is occupied during the selected Days & Hours.

    **Utilisation (%):**  
    The average proportion of seat usage during hours when the space is occupied. This reflects how intensively the space is used **when** it's in use, not just whether it's occupied.

    **Humidity (%):**  
    The monthly median of relative humidity values, indicating the moisture content in the air when workspace is occupied

    **Temperature (°C):**  
    The monthly median of ambient temperature values. Captures general thermal conditions in the workspace when occupied

    **Apparent Temperature (°C):**  
    The perceived temperature that combines temperature, humidity, and airflow effects. Monthly medain of apparent temperature readings when workspace is occupied.
    For details: https://en.wikipedia.org/wiki/Apparent_temperature

    **CO₂ (ppm):**  
    The monthly median of carbon dioxide concentration when workspace is occupied. Elevated levels (especially above 1000 ppm) may suggest poor ventilation or high occupant density.

    **90th Percentile Metrics (for CO₂, Temp, Humidity, Apparent Temp):**  
    These represent the upper range of environmental exposure by taking the 90th percentile of hourly values across the month. They help identify peak conditions that could affect occupant comfort, performance, or air quality. Computed when workspace is occupied
    """)

############################################################################

###  Filter and Clean Data
st.title("Spatial Analysis (Jan - May 2025)")

# Clean and filter data
gdf_filtered = gdf_mod.copy()

# Filter by floor and Month
gdf_filtered = gdf_filtered[gdf_filtered['lvl'] == selected_floor].reset_index(drop=True)
# Filter by floor and Month
df_all_aps[['lat', 'lon']] = df_all_aps['Geo_Position'].str.split(',', expand=True).astype(float)
df_all_aps = df_all_aps.loc[df_all_aps["lvl"] == selected_floor,]

# Remove rooms with anomalously high utilization or occupancy
def parse_percentage(val):
    try:
        return float(str(val).replace("%", "").strip())
    except:
        return np.nan

def parse_temp(val):
    try:
        return float(str(val).replace("C", "").strip())
    except:
        return np.nan

def parse_co2(val):
    try:
        return float(str(val).replace("ppm", "").strip())
    except:
        return np.nan


gdf_filtered['humidity_pct'] = gdf_filtered['Humidity(%)_90th_percentitle'].apply(parse_percentage)
gdf_filtered['temp_pct'] = gdf_filtered['Temp(C)_90th_percentitle'].apply(parse_temp)
gdf_filtered['apparent_temp_pct'] = gdf_filtered['ApparentTemp(C)_90th_percentile'].apply(parse_temp)
gdf_filtered['co2_pct'] = gdf_filtered['Co2(ppm)_90th_percentitle'].apply(parse_co2)



# Filter out anomalies


# For Humidity
gdf_filtered['Humidity(%)_Legend'] = np.where(
    np.logical_and(gdf_filtered["humidity_pct"] >= 30,
                    gdf_filtered["humidity_pct"] <= 60),
    'Ideal Humidity(30%-60%)',
    np.where(
        (gdf_filtered["humidity_pct"] < 30),
        'Low Humidity(<30%)',
    np.where(
        (gdf_filtered["humidity_pct"] > 60),
        'High Humidity(>60%)',
        'Other Workspaces'
    ) ) )

# for Co2
gdf_filtered['Co2(ppm)_Legend'] = np.where(
    np.logical_and(gdf_filtered["co2_pct"] >= 400,
                    gdf_filtered["co2_pct"] <= 800),
    'Ideal Co2(400ppm-800ppm)',
    np.where(
        (gdf_filtered["co2_pct"] < 400),
        'Low Co2(<400ppm)',
        np.where(
        (gdf_filtered["co2_pct"] > 800),
        'High Co2(>800)',
        'Other Workspaces'
    ) ) )


# for Temperature
gdf_filtered['Temp(C)_Legend'] = np.where(
    np.logical_and(gdf_filtered["temp_pct"] >= 20,
                    gdf_filtered["temp_pct"] <= 26),
    'Ideal Temperature(20C-26C)',
    np.where(
        (gdf_filtered["temp_pct"] < 20),
        'Low Temperature(<20C)',
        np.where(
            (gdf_filtered["temp_pct"] > 26),
            'High Temperature(>26C)',
            'Other Workspaces'
        ) ) )

# For Apperant Temperature
gdf_filtered['ApparentTemp(C)_Legend'] = np.where(
    np.logical_and(gdf_filtered["apparent_temp_pct"] >= 20,
                    gdf_filtered["apparent_temp_pct"] <= 26),
    'Ideal Apparent Temperature(20C-26C)',
    np.where(
        (gdf_filtered["apparent_temp_pct"] < 20),
        'Low Apparent Temperature(<20C)',
        np.where(
            (gdf_filtered["apparent_temp_pct"] > 26),
            'High Apparent Temperature(>26C)',
            'Other Workspaces'
        ) ) )

#################### Charts ##############################
# add tab to streamtlit app
tabA, tabB, tabC, tabD, tabE = st.tabs(["Occupancy & Utilisation", 'Humidity', "CO2", 'Temperature', 'Apparent Temperature'])

################## Occupancy & Utilisation
with tabA:
  # Utilisation and Occupancy filter sliders
  util_threshold = st.slider("Utilisation %", 0, 100, 40)
  occ_threshold = st.slider("Occupancy %", 0, 100, 40)

  gdf_filtered["util_pct"] = gdf_filtered["Monthly_Utilisation"].apply(parse_percentage)
  gdf_filtered["occ_pct"] = gdf_filtered["Monthly_Occupancy"].apply(parse_percentage)

  gdf_filtered['Occupancy_Utilisation_Legend'] = np.where(
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
  # STEP 2: Create base choropleth map
  fig = px.choropleth_mapbox(
      gdf_filtered,
      geojson=gdf_filtered.geometry,
      locations=gdf_filtered.index,
      hover_name="name",
      hover_data=["lvl",'Area_Sqft','Space_Capacity_label', 'Workspace_Capacity_Category',
                  'Monthly_Occupancy', 'Monthly_Utilisation'],
      color="Occupancy_Utilisation_Legend",
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
      title=f"Occupancy & Utilisation Analysis - Floor {selected_floor}"
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
      ),
       visible='legendonly'  # 👈 this makes it hidden by default
  ))

  # STEP 4: Update layout (Mapbox token not needed for 'carto-positron')
  fig.update_layout(
      margin={"r":0,"t":40,"l":0,"b":0},
      legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
  )


  # fig.update_traces(marker_line_width=1, marker_line_color="black")
  fig.update_layout(margin={"r": 0, "t": 30, "l": 0, "b": 0}, height=800)
  st.plotly_chart(fig, use_container_width=True)

###################### Humidity
with tabB:
  # STEP 2: Create base choropleth map
  fig = px.choropleth_mapbox(
      gdf_filtered,
      geojson=gdf_filtered.geometry,
      locations=gdf_filtered.index,
      hover_name="name",
      hover_data=["lvl", 'Space_Capacity_label',
                  # 'Workspace_Capacity_Category',
                  # 'Monthly_Occupancy', 'Monthly_Utilisation', 'Monthly_Dwell_Time',
                  'Humidity(%)','Humidity(%)_90th_percentitle'],
      color="Humidity(%)_Legend",
      mapbox_style="carto-positron",
      color_discrete_map={
           'Low Humidity(<30%)': '#87CEEB',   # Light Blue
          'Ideal Humidity(30%-60%)': '#1f77b4', # Blue
         'High Humidity(>60%)': '#d62728' ,  # Red
          'Other Workspaces': "grey"
      },
      center={
          "lat": gdf_filtered.geometry.centroid.y.mean(),
          "lon": gdf_filtered.geometry.centroid.x.mean()
      },
      zoom=19,
      opacity=0.6,
      title=f"Humidity(%) Analysis - Floor {selected_floor}"
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
      name='Access Points',
      hoverinfo='text',
      hovertext=df_all_aps.apply(
          lambda row: f"{row['Device_Name']}<br>{row['IP_Address']}<br>{row['Model']}", axis=1
      ),
       visible='legendonly'  # 👈 this makes it hidden by default
  ))

  # STEP 4: Update layout (Mapbox token not needed for 'carto-positron')
  fig.update_layout(
      margin={"r":0,"t":40,"l":0,"b":0},
      legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
  )


  # fig.update_traces(marker_line_width=1, marker_line_color="black")
  fig.update_layout(margin={"r": 0, "t": 30, "l": 0, "b": 0}, height=800)
  st.plotly_chart(fig, use_container_width=True)


###################### CO2
with tabC:
  # STEP 2: Create base choropleth map
  fig = px.choropleth_mapbox(
      gdf_filtered,
      geojson=gdf_filtered.geometry,
      locations=gdf_filtered.index,
      hover_name="name",
      hover_data=["lvl", 'Space_Capacity_label',
                  # 'Workspace_Capacity_Category',
                  # 'Monthly_Occupancy', 'Monthly_Utilisation', 'Monthly_Dwell_Time',
                  'Co2(ppm)','Co2(ppm)_90th_percentitle'],
      color="Co2(ppm)_Legend",
      mapbox_style="carto-positron",
      color_discrete_map={
           'Low Co2(<400ppm)': '#87CEEB',   # Light Blue
          'Ideal Co2(400ppm-800ppm)': '#1f77b4', # Blue
         'High Co2(>800)': '#d62728' ,  # Red
          'Other Workspaces': "grey"
      },
      center={
          "lat": gdf_filtered.geometry.centroid.y.mean(),
          "lon": gdf_filtered.geometry.centroid.x.mean()
      },
      zoom=19,
      opacity=0.6,
      title=f"CO2(ppm) Analysis - Floor {selected_floor}"
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
      name='Access Points',
      hoverinfo='text',
      hovertext=df_all_aps.apply(
          lambda row: f"{row['Device_Name']}<br>{row['IP_Address']}<br>{row['Model']}", axis=1
      ),
       visible='legendonly'  # 👈 this makes it hidden by default
  ))

  # STEP 4: Update layout (Mapbox token not needed for 'carto-positron')
  fig.update_layout(
      margin={"r":0,"t":40,"l":0,"b":0},
      legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
  )


  # fig.update_traces(marker_line_width=1, marker_line_color="black")
  fig.update_layout(margin={"r": 0, "t": 30, "l": 0, "b": 0}, height=800)
  st.plotly_chart(fig, use_container_width=True)


###################### Temperature(C)
with tabD:
  # STEP 2: Create base choropleth map
  fig = px.choropleth_mapbox(
      gdf_filtered,
      geojson=gdf_filtered.geometry,
      locations=gdf_filtered.index,
      hover_name="name",
      hover_data=["lvl", 'Space_Capacity_label',
                  # 'Workspace_Capacity_Category',
                  # 'Monthly_Occupancy', 'Monthly_Utilisation', 'Monthly_Dwell_Time',
                  'Temp(C)','Temp(C)_90th_percentitle'],
      color='Temp(C)_Legend',
      mapbox_style="carto-positron",
      color_discrete_map={
           'Low Temperature(<20C)': '#87CEEB',   # Light Blue
         'Ideal Temperature(20C-26C)': '#1f77b4', # Blue
        'High Temperature(>26C)': '#d62728' ,  # Red
          'Other Workspaces': "grey"
      },
      center={
          "lat": gdf_filtered.geometry.centroid.y.mean(),
          "lon": gdf_filtered.geometry.centroid.x.mean()
      },
      zoom=19,
      opacity=0.6,
      title=f"Temp(C) Analysis - Floor {selected_floor}"
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
      name='Access Points',
      hoverinfo='text',
      hovertext=df_all_aps.apply(
          lambda row: f"{row['Device_Name']}<br>{row['IP_Address']}<br>{row['Model']}", axis=1
      ),
       visible='legendonly'  # 👈 this makes it hidden by default
  ))

  # STEP 4: Update layout (Mapbox token not needed for 'carto-positron')
  fig.update_layout(
      margin={"r":0,"t":40,"l":0,"b":0},
      legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
  )


  # fig.update_traces(marker_line_width=1, marker_line_color="black")
  fig.update_layout(margin={"r": 0, "t": 30, "l": 0, "b": 0}, height=800)
  st.plotly_chart(fig, use_container_width=True)

###################### 'ApparentTemp(C)'
with tabE:
  # STEP 2: Create base choropleth map
  fig = px.choropleth_mapbox(
      gdf_filtered,
      geojson=gdf_filtered.geometry,
      locations=gdf_filtered.index,
      hover_name="name",
      hover_data=["lvl", 'Space_Capacity_label',
                  # 'Workspace_Capacity_Category',
                  # 'Monthly_Occupancy', 'Monthly_Utilisation', 'Monthly_Dwell_Time',
                  'ApparentTemp(C)','ApparentTemp(C)_90th_percentile'],
      color='ApparentTemp(C)_Legend',
      mapbox_style="carto-positron",
      color_discrete_map={
           'Low Apparent Temperature(<20C)': '#87CEEB',   # Light Blue
         'Ideal Apparent Temperature(20C-26C)': '#1f77b4', # Blue
        'High Apparent Temperature(>26C)': '#d62728' ,  # Red
          'Other Workspaces': "grey"
      },
      center={
          "lat": gdf_filtered.geometry.centroid.y.mean(),
          "lon": gdf_filtered.geometry.centroid.x.mean()
      },
      zoom=19,
      opacity=0.6,
      title=f"Temp(C) Analysis - Floor {selected_floor}"
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
      name='Access Points',
      hoverinfo='text',
      hovertext=df_all_aps.apply(
          lambda row: f"{row['Device_Name']}<br>{row['IP_Address']}<br>{row['Model']}", axis=1
      ),
       visible='legendonly'  # 👈 this makes it hidden by default
  ))

  # STEP 4: Update layout (Mapbox token not needed for 'carto-positron')
  fig.update_layout(
      margin={"r":0,"t":40,"l":0,"b":0},
      legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
  )


  # fig.update_traces(marker_line_width=1, marker_line_color="black")
  fig.update_layout(margin={"r": 0, "t": 30, "l": 0, "b": 0}, height=800)
  st.plotly_chart(fig, use_container_width=True)
