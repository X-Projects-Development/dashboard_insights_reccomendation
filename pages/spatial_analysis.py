
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
import matplotlib.pyplot as plt
import geopandas as gpd
import contextily as ctx  # for basemaps
from shapely.geometry import Point
from adjustText import adjust_text
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
from shapely.geometry import Point

dotenv.load_dotenv()

# Load API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


# from langchain.chat_models import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import Runnable
from PIL import Image
import base64
from io import BytesIO


################################################## GENAI SUMMARY BOT #############################################


def encode_image_to_base64(image_path):
    """Encodes an image to base64 for OpenAI Vision."""
    with Image.open(image_path) as img:
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()


def analyze_floor_plan(floor_table,
                       image_path: str, floor_number: int,
                       utilisation_threshold: float = 30.0,
                       occupancy_threshold: float = 30.0,
                       metric: str = 'occ_&_util') -> dict:
    """
    Analyze a floor plan image using OpenAI Vision model with JSON output.

    Returns:
        dict: A dictionary with "insight" and "recommendation"
    """

    # Step 1: Encode image
    base64_image = encode_image_to_base64(image_path)

    # Step 2: Define output parser
    parser = JsonOutputParser()
    floor_table = floor_table.loc[floor_table['name']!='none',]
    # Step 3: Construct the prompt
    if metric == 'occ_&_util':
      floor_table = floor_table[['name', 'Space_Capacity_label', 'Workspace_Capacity_Category',
                                  'Monthly_Occupancy','Monthly_Utilisation','zone']].reset_index(drop=True)
      prompt = ChatPromptTemplate.from_messages([
          ("system", "You are a workplace analytics expert. Respond ONLY in JSON."),
          ("human", """ Analyze the attached floor plan chart (image) for Floor {floor}.

Context:
- Workspaces are color-coded based on the following thresholds:
  - Utilization: {utilisation_threshold}% 
  - Occupancy: {occupancy_threshold}%
- Definitions:
  - **Utilization**: Ratio of number of people occupying a workspace to its seating capacity.
  - **Occupancy**: Ratio of the number of hours a workspace is occupied during standard office hours (7am–7pm).
  - **Capacity categories**:
    - Small: 1–5 seats
    - Medium: 6–10 seats
    - Large: 11–20 seats
    - Extra Large: 21+ seats

Floor Details:
{floor_table}

Instructions:
Review the floor plan image and identify patterns based on the color-coded utilization and occupancy. Return your answer as a JSON object with:

- **insight**: Describe which specific areas or zones in the floor plan image (e.g., "northwest corner", "central cluster", "eastern row", etc.) show consistent patterns in utilization and occupancy. Refer to visible layout groupings or color clusters. Be as specific and quantifiable as possible. Do not hallucinate or guess any rooms that aren't visible. Ignore "Other Workspaces".

- **recommendation**: Based on these patterns, suggest improvements in space allocation or efficiency (e.g., room resizing, repurposing underutilized areas, or relocating heavily used zones).

Respond in the following format:
{{
"insight": "...",
"recommendation": "..."
}}
 """)
      ])
# """Analyze this floor plan chart (image) for Floor {floor}.
#   Context:
#   - Workspaces have been color coded based on Utilisation threshold: {utilisation_threshold}%  and  Occupancy threshold: {occupancy_threshold}%
#   - Utilisation of a workspace is ratio of number of people occupying the room to number of seats in the workspace
#   - Occupancy of a workspace is ratio of number times the room is occupied during usual office hours (7am to 7pm)
#   - Capacity of a workspace is the number of seats in the workspace. Small Rooms : 1-5 seats, Medium Rooms : 6-10 seats, Large Rooms : 11-20 seats and Extra Large Rooms : 21+ seats

#   Floor Details:
#   {floor_table}

#   ONLY provide:
#   - insights (Based on the Floor plan chart IDENTIFY different areas/zones in office showing any pattern in occupancy and utilisation.Ignore any observation on 'Other Workspaces'. Try to quantify insights as much as possible. Ensure insights are completely grounded in the Floor plan chart. Dont halucinate rooms.)
#   - recommendation (Suggest any improvements for space allocation or efficiency.)

#   Return your answer as a JSON object like:
#       {{
#         "insight": "...",
#         "recommendation": "..."
#       }}
#   """
    elif metric == 'Temp':
      floor_table = floor_table[['name', 'Space_Capacity_label', 'Workspace_Capacity_Category',
                                  'Temp(C)','Temp(C)_90th_percentile','zone']].reset_index(drop=True)
      prompt = ChatPromptTemplate.from_messages([
          ("system", "You are a workplace analytics expert. Respond ONLY in JSON."),
          ("human", """Analyze the attached floor plan chart (image) for Floor {floor}.

Context:
- Workspaces are color-coded based on temperature thresholds defined in the chart's legend.
- **Avg Temperature**: Average temperature of the workspace when occupied.
- **Temperature_90th_percentile**: The 90th percentile temperature reading when occupied — highlights temperature under peak or extreme conditions.
- **Capacity categories**:
  - Small: 1–5 seats
  - Medium: 6–10 seats
  - Large: 11–20 seats
  - Extra Large: 21+ seats

Floor Plan Details:
{floor_table}

**Instructions:**
Provide a JSON object with the following fields:

- `insight`: Identify and describe specific **areas or zones** in the floor plan image (e.g., "north corridor", "eastern cluster", "central spine") where temperature patterns (average or 90th percentile) stand out. Focus on **visible zones** and color-coded groupings in the image. Quantify differences or thresholds where possible. **Do not reference 'Other Workspaces'** and **do not hallucinate or assume rooms not visible in the chart**.

- `recommendation`: Suggest actionable improvements for better temperature management based on the observed patterns. (e.g., HVAC adjustments, room repurposing, targeted monitoring.)

Respond strictly in the following JSON format:
      {{
        "insight": "...",
        "recommendation": "..."
      }}
  """)
      ])

    elif metric == 'Apparent_Temp':
      floor_table = floor_table[['name', 'Space_Capacity_label', 'Workspace_Capacity_Category',
                                  'ApparentTemp(C)','ApparentTemp(C)_90th_percentile','zone']].reset_index(drop=True)
      prompt = ChatPromptTemplate.from_messages([
          ("system", "You are a workplace analytics expert. Respond ONLY in JSON."),
          ("human", """Analyze the attached floor plan chart (image) for Floor {floor}.

Context:
- Workspaces are color-coded based on temperature thresholds defined in the legend.
- **Apparent Temperature** (also known as “feels like” temperature) is calculated using air temperature and humidity.
  - **Avg Apparent Temperature**: Average apparent temperature of the room when occupied.
  - **Apparent_Temperature_90th_percentile**: The 90th percentile of apparent temperature when occupied, indicating conditions during more extreme scenarios.
- **Capacity categories**:
  - Small: 1–5 seats
  - Medium: 6–10 seats
  - Large: 11–20 seats
  - Extra Large: 21+ seats

Floor Plan Details:
{floor_table}

Instructions:
Provide a response as a JSON object with the following fields:

- `insight`: Identify **specific areas or zones** in the floor plan image (e.g., “southwest wing”, “central block”, “eastern row”) that show noticeable patterns in **average or 90th percentile apparent temperature**. Quantify differences where possible (e.g., “2°C higher than other zones”). Base all insights solely on what is visible in the floor plan image — do **not reference 'Other Workspaces'** or assume the presence of any rooms not shown.

- `recommendation`: Suggest practical improvements for temperature management based on the observed patterns. These might include HVAC rebalancing, zoning changes, occupancy adjustments, or sensor placement.

Return your response in the following format:
      {{
        "insight": "...",
        "recommendation": "..."
      }}
  """)
      ])
    elif metric == 'Humidity':
      floor_table = floor_table[['name', 'Space_Capacity_label', 'Workspace_Capacity_Category',
                                 'Humidity(%)','Humidity(%)_90th_percentile','zone']].reset_index(drop=True)
      prompt = ChatPromptTemplate.from_messages([
          ("system", "You are a workplace analytics expert. Respond ONLY in JSON."),
          ("human", """Analyze the attached floor plan chart (image) for Floor {floor}.

Context:
- Workspaces are color-coded based on humidity thresholds indicated in the legend.
- **Avg Humidity**: Average humidity in the room when it is occupied.
- **Humidity_90th_percentile**: The 90th percentile humidity reading when occupied, representing more extreme conditions.
- **Capacity categories**:
  - Small: 1–5 seats
  - Medium: 6–10 seats
  - Large: 11–20 seats
  - Extra Large: 21+ seats

Floor Plan Details:
{floor_table}

Instructions:
Provide your response as a JSON object containing:

- `insight`: Identify **specific areas or zones** in the floor plan image (e.g., "northeast section", "central corridor", "cluster of large rooms in the west") that show noticeable patterns in humidity—either average or 90th percentile. Quantify patterns where possible (e.g., “above 60% in all small rooms on the eastern side”). Base all insights **strictly on what is visible in the floor plan image**. Do **not** reference 'Other Workspaces', and **do not invent rooms**.

- `recommendation`: Suggest practical steps to improve humidity conditions in the workspace (e.g., improved ventilation, dehumidifier placement, HVAC tuning, or room reconfiguration).

Return your answer in the following JSON format:
      {{
        "insight": "...",
        "recommendation": "..."
      }}
  """)
      ])

    elif metric == 'CO2':
      floor_table = floor_table[['name', 'Space_Capacity_label', 'Workspace_Capacity_Category',
                                  'Co2(ppm)','Co2(ppm)_90th_percentile','zone']].reset_index(drop=True)
      prompt = ChatPromptTemplate.from_messages([
          ("system", "You are a workplace analytics expert. Respond ONLY in JSON."),
          ("human", """Analyze the attached floor plan chart (image) for Floor {floor}.

Context:
- Workspaces are color-coded based on CO₂ thresholds indicated in the legend.
- **Avg CO₂**: The average carbon dioxide level in the room when occupied.
- **CO2_90th_percentile**: The 90th percentile CO₂ level during occupancy, representing elevated levels under more extreme or sustained conditions.

Floor Plan Details:
{floor_table}

Instructions:
Return your answer as a JSON object with the following fields:

- `insight`: Identify **specific areas or zones** in the floor plan image (e.g., “north wing”, “row of medium rooms in southeast corner”) where CO₂ patterns emerge — either average or 90th percentile values. Quantify insights wherever possible (e.g., “CO₂ levels exceed 1000 ppm in over 60% of rooms in the west section”). Insights must be **entirely grounded in the floor plan image**. Do **not reference 'Other Workspaces'**, and do **not invent or assume rooms not visible in the chart**.

- `recommendation`: Suggest practical steps to improve CO₂ levels in the affected areas. These may include ventilation improvements, sensor reconfiguration, space usage adjustments, or scheduling changes.

Respond using the following format:
      {{
        "insight": "...",
        "recommendation": "..."
      }}
  """)
      ])


    # Step 4: Initialize model
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

    # Step 5: Create chain and run
    chain: Runnable = prompt | llm | parser

    # Step 6: Run chain with image input
    result = chain.invoke({
        "floor": floor_number,
        "utilisation_threshold": utilisation_threshold,
        "occupancy_threshold": occupancy_threshold,
        "floor_table": floor_table.to_markdown(index=False),
        "images": [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64_image}"
                }
            }
        ]
    })

    return result["insight"],result["recommendation"]


import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import contextily as ctx
from shapely.geometry import Point
from adjustText import adjust_text

def plot_metric_map_with_ai_insight(
    gdf,
    selected_floor,
    metric_col,
    color_map,
    legend_title,
    label_formatter,
    image_output_path,
    ai_metric_key,
    util_threshold=0,
    occ_threshold=0
):
    """
    Generalized function to visualize any spatial metric and generate AI insights.

    Parameters:
    - gdf: GeoDataFrame to visualize
    - selected_floor: Floor number (for title/context)
    - metric_col: Column in gdf for color mapping (e.g., 'Occupancy_Utilisation_Legend')
    - color_map: Dict mapping category → color
    - legend_title: Legend title string
    - label_formatter: Function that returns the label string for a row
    - image_output_path: Where to save the image
    - ai_metric_key: Passed to AI insight generator to distinguish metric
    - util_threshold, occ_threshold: Passed to AI model for threshold-based reasoning
    """

    # Filter out unwanted sTypes
    exclude_types = [
        'Workstation Desk', 'Workstation Chair', 'Workstation', 'Stairs', 'Cafeteria', 'Entertainment',
        'Storage', 'Chair', 'Meeting Room Desk', 'Meeting Room Chair', 'Restroom', 'Support Space',
        'Elevator', 'Storage Cabinet', 'Child Care', 'POU Chair'
    ]
    gdf_filtered = gdf.loc[~gdf['sType'].isin(exclude_types)].reset_index(drop=True)

    # Apply color mapping
    gdf_filtered['color'] = gdf_filtered[metric_col].map(color_map)

    # Start plotting
    fig, ax = plt.subplots(figsize=(16, 12))
    default_color = "#eeeeee"  # fallback for unmapped categories
    gdf_filtered['color'] = gdf_filtered[metric_col].map(color_map).fillna(default_color)
    gdf_filtered.plot(ax=ax, color=gdf_filtered['color'], edgecolor='black', linewidth=0.5)

    # Add labels using the formatter
    texts = []
    for _, row in gdf_filtered.iterrows():
        label = label_formatter(row)
        if label:
            centroid = row.geometry.centroid
            text = ax.text(centroid.x, centroid.y, label, fontsize=7, ha='center', va='center')
            texts.append(text)

    adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='-', color='red', lw=0.5))

    # Add basemap
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, crs=gdf_filtered.crs.to_string())
    
    # Get axis bounds
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    x_range = x_max - x_min
    y_range = y_max - y_min

    # Whitewash the borders (add semi-transparent white rectangles)
    border_width = 0.05  # 5% of axis range

    # Top
    ax.add_patch(Rectangle((x_min, y_max - y_range * border_width), x_range, y_range * border_width,
                          color='white', zorder=5))
    # Bottom
    ax.add_patch(Rectangle((x_min, y_min), x_range, y_range * border_width,
                          color='white', zorder=5))
    # Left
    ax.add_patch(Rectangle((x_min, y_min), x_range * border_width, y_range,
                          color='white', zorder=5))
    # Right
    ax.add_patch(Rectangle((x_max - x_range * border_width, y_min), x_range * border_width, y_range,
                          color='white', zorder=5))

    # Add direction text
    ax.text((x_min + x_max) / 2, y_max - y_range * 0.02, 'North', fontsize=14, ha='center', va='top', weight='bold', zorder=6)
    ax.text((x_min + x_max) / 2, y_min + y_range * 0.02, 'South', fontsize=14, ha='center', va='bottom', weight='bold', zorder=6)
    ax.text(x_min + x_range * 0.02, (y_min + y_max) / 2, 'West', fontsize=14, ha='left', va='center', rotation=90, weight='bold', zorder=6)
    ax.text(x_max - x_range * 0.02, (y_min + y_max) / 2, 'East', fontsize=14, ha='right', va='center', rotation=270, weight='bold', zorder=6)

    # Add legend
    legend_patches = [
        mpatches.Patch(color=color, label=label)
        for label, color in color_map.items()
    ]
    ax.legend(handles=legend_patches, title=legend_title, loc='lower right',
              fontsize=9, title_fontsize=10, frameon=True)

    # Final layout
    ax.set_title(f"{legend_title} – Floor {selected_floor}", fontsize=16)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(image_output_path, bbox_inches="tight")

    # Generate AI insight
    def get_cached_insight(gdf_filtered, image_path, floor, util_thresh, occ_thresh, metric):
        return analyze_floor_plan(
            floor_table = gdf_filtered,
            image_path=image_path,
            floor_number=floor,
            utilisation_threshold=util_thresh,
            occupancy_threshold=occ_thresh,
            metric=metric
        )
    with st.spinner("Generating AI Insight..."):
      result = get_cached_insight(gdf_filtered[['name', 'Space_Capacity_label', 'Workspace_Capacity_Category',
                                                'Monthly_Occupancy','Monthly_Utilisation', 'Co2(ppm)','Temp(C)',
                                                'Humidity(%)','ApparentTemp(C)','Co2(ppm)_90th_percentile',
                                                'Temp(C)_90th_percentile','Humidity(%)_90th_percentile',
                                                'ApparentTemp(C)_90th_percentile','zone']], image_output_path, selected_floor, util_threshold, occ_threshold, ai_metric_key)
    # st.write(f"AI Insight: {result[0]}")
    # st.write(f"AI Recommendation: {result[1]}")
    try:
        return result["insight"], result["recommendation"]
    except:
        return result[0], result[1]





def get_zone(centroid):
    x, y = centroid.x, centroid.y
    # Check if in central zone first
    if (x_mid - x_margin <= x <= x_mid + x_margin) and (y_mid - y_margin <= y <= y_mid + y_margin):
        return "central"
    elif x <= x_mid - x_margin:
        if y >= y_mid + y_margin:
            return "northwest"
        elif y <= y_mid - y_margin:
            return "southwest"
        else:
            return "west"
    elif x >= x_mid + x_margin:
        if y >= y_mid + y_margin:
            return "northeast"
        elif y <= y_mid - y_margin:
            return "southeast"
        else:
            return "east"
    else:
        if y >= y_mid + y_margin:
            return "north"
        elif y <= y_mid - y_margin:
            return "south"
        else:
            return "central"



##################################################################################################################################
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
                        'CO2_90percentile': 'Co2(ppm)_90th_percentile',
                        'Temp_90percentile': 'Temp(C)_90th_percentile',
                        'Humidity_90percentile': 'Humidity(%)_90th_percentile',
                        'ApparentTemp_90percentile': 'ApparentTemp(C)_90th_percentile'
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
    df_agg['ApparentTemp(C)_90th_percentile'] = df_agg['ApparentTemp(C)_90th_percentile'].apply(
        lambda x: f"{x:.1f} C" if pd.notnull(x) else x
    )

    # Humidity
    df_agg['Humidity(%)'] = df_agg['Humidity(%)'].apply(
        lambda x: f"{x:.1f}%" if pd.notnull(x) else x
    )
    df_agg['Humidity(%)_90th_percentile'] = df_agg['Humidity(%)_90th_percentile'].apply(
        lambda x: f"{x:.1f}%" if pd.notnull(x) else x
    )

    # Temperature
    df_agg['Temp(C)'] = df_agg['Temp(C)'].apply(
        lambda x: f"{x:.1f} C" if pd.notnull(x) else x
    )
    df_agg['Temp(C)_90th_percentile'] = df_agg['Temp(C)_90th_percentile'].apply(
        lambda x: f"{x:.1f} C" if pd.notnull(x) else x
    )

    # CO2
    df_agg['Co2(ppm)'] = df_agg['Co2(ppm)'].apply(
        lambda x: f"{x:.0f} ppm" if pd.notnull(x) else x
    )
    df_agg['Co2(ppm)_90th_percentile'] = df_agg['Co2(ppm)_90th_percentile'].apply(
        lambda x: f"{x:.0f} ppm" if pd.notnull(x) else x
    )

    df_area['Area_Sqft'] = df_area['Area_Sqft'].apply(
        lambda x: f"{round(x):.0f} sqft" if pd.notnull(x) else x
    )
    # repeated_gdf['lvl'] = repeated_gdf['lvl'].astype(str)
    gdf_mod = pd.merge(gdf, df_agg[['Floor Name','lvl','Workspace Name', 'Workspace Type',
                                    'Space_Capacity_label', 'Workspace_Capacity_Category',
                                    'Monthly_Utilisation','Monthly_Occupancy','ApparentTemp(C)',
                                    'ApparentTemp(C)_90th_percentile','Humidity(%)','Humidity(%)_90th_percentile',
                                    'Temp(C)','Temp(C)_90th_percentile','Co2(ppm)','Co2(ppm)_90th_percentile']], left_on=['lvl','name'],
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


# Floor filter
floor_options = ['1','2','3','4','5']
selected_floor = st.sidebar.selectbox("Select Floor:", floor_options)

# Apply filter button inside the expander
# apply_filter1 = st.sidebar.button("Apply Filter")


# if apply_filter1 ==False:
#   selected_floor = '1'
#   selected_weekday = ['Monday','Tuesday', 'Wednesday', 'Thursday', 'Friday']
#   selected_hour = list(range(7, 19))
#   apply_filter1 = True
################################### Load data ###################################

# Show the DataFrame
df_all_aps = load_json_files(json_folder_path)
gdf_mod = geojson_load_data(selected_weekday,selected_hour)




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


gdf_filtered['humidity_pct'] = gdf_filtered['Humidity(%)_90th_percentile'].apply(parse_percentage)
gdf_filtered['temp_pct'] = gdf_filtered['Temp(C)_90th_percentile'].apply(parse_temp)
gdf_filtered['apparent_temp_pct'] = gdf_filtered['ApparentTemp(C)_90th_percentile'].apply(parse_temp)
gdf_filtered['co2_pct'] = gdf_filtered['Co2(ppm)_90th_percentile'].apply(parse_co2)



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



# Create the 'zone' column
x_min, y_min, x_max, y_max = gdf_filtered.total_bounds
x_mid = (x_min + x_max) / 2
y_mid = (y_min + y_max) / 2

# Optional margin to define "central" zone
x_margin = (x_max - x_min) * 0.1
y_margin = (y_max - y_min) * 0.1
gdf_filtered["zone"] = gdf_filtered.geometry.centroid.apply(get_zone)

#################### Charts ##############################
# if apply_filter1:
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

  #############################################################################################################################################

  # gdf_filtered_genai['color'] = gdf_filtered_genai['Occupancy_Utilisation_Legend'].map(color_map)

  # fig, ax = plt.subplots(figsize=(16, 12))
  # gdf_filtered_genai.plot(ax=ax, color=gdf_filtered_genai['color'], edgecolor='black', linewidth=0.5)

  # # 2. Add labels
  # texts = []
  # for _, row in gdf_filtered_genai.iterrows():
  #     if row['name'] != 'none' and str(row['Monthly_Occupancy']).endswith('%'):
  #         label = f"{row['name']}\nOcc {row['Monthly_Occupancy']}\nUtil {row['Monthly_Utilisation']}"
  #         centroid = row.geometry.centroid
  #         text = ax.text(centroid.x, centroid.y, label, fontsize=7, ha='center', va='center')
  #         texts.append(text)

  # adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='-', color='red', lw=0.5))

  # # 3. Add a basemap (optional)
  # ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, crs=gdf_filtered_genai.crs.to_string())

  # # ADD LEGEND HERE
  # legend_patches = [
  #     mpatches.Patch(color=color, label=label)
  #     for label, color in color_map.items()
  # ]
  # ax.legend(handles=legend_patches, title="Occupancy & Utilisation", loc='lower right',
  #           fontsize=9, title_fontsize=10, frameon=True)

  # # 4. Tidy layout
  # ax.set_title(f"Occupancy & Utilisation Analysis – Floor {selected_floor}", fontsize=16)
  # ax.axis('off')
  # plt.tight_layout()
  # image_path = '/content/utilisation_occupancy.png'
  # plt.savefig(image_path, bbox_inches="tight")

  # # 1. Run the AI insight generation
  # with st.spinner("Generating AI Insight and Recommendation..."):
  #     def get_cached_insight_occ_util(image_path, selected_floor, util_threshold, occ_threshold, metric_name ):
  #         return analyze_floor_plan(image_path, floor_number=selected_floor, utilisation_threshold=util_threshold, occupancy_threshold=occ_threshold, metric= metric_name)

  #     insight_text1, recommendation_text1 = get_cached_insight_occ_util(image_path, selected_floor, util_threshold, occ_threshold,'occ_&_util')
  insight_text1, recommendation_text1 = plot_metric_map_with_ai_insight(
    gdf=gdf_filtered,
    selected_floor=selected_floor,
    metric_col='Occupancy_Utilisation_Legend',
    color_map={
    # Occupancy & Utilisation
    "High Utilisation & Occupancy": "#d62728",
    "High Utilisation But Low Occupancy": "#87CEEB",
    "Low Utilisation & Occupancy": "#2ca02c",
    "Low Utilisation But High Occupancy": "#1f77b4",
    "Other Workspaces": "#fafafa",
    # # Humidity
    # 'Low Humidity(<30%)': '#87CEEB',   # Light Blue
    # 'Ideal Humidity(30%-60%)': '#1f77b4', # Blue
    # 'High Humidity(>60%)': '#d62728' ,  # Red
    # # CO2
    # 'Low Co2(<400ppm)': '#87CEEB',   # Light Blue
    # 'Ideal Co2(400ppm-800ppm)': '#1f77b4', # Blue
    # 'High Co2(>800)': '#d62728' ,  # Red
    # # Temperature
    # 'Low Temperature(<20C)': '#87CEEB',   # Light Blue
    # 'Ideal Temperature(20C-26C)': '#1f77b4', # Blue
    # 'High Temperature(>26C)': '#d62728' ,  # Red
    # # Aparent Temperature
    # 'Low Apparent Temperature(<20C)': '#87CEEB',   # Light Blue
    # 'Ideal Apparent Temperature(20C-26C)': '#1f77b4', # Blue
    # 'High Apparent Temperature(>26C)': '#d62728' ,  # Red

},
    legend_title="Occupancy & Utilisation",
    label_formatter=lambda row: (
        f"{row['name']}\nOcc {row['Monthly_Occupancy']}\nUtil {row['Monthly_Utilisation']}\nCapacity {row['Space_Capacity_label']}"
        if row['name'] != 'none' and str(row['Monthly_Occupancy']).endswith('%') else None
    ),
    image_output_path='images/utilisation_occupancy.png',
    ai_metric_key='occ_&_util',
    util_threshold=util_threshold,
    occ_threshold=occ_threshold
)
  # 2. Display it in a styled tile below the chart
  with st.container():
      st.markdown(f"""
      <div style='padding: 1rem; background-color: #f0f8ff; border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.05);'>
          <h4 style='margin-bottom: 0.5rem;'>🔍 AI Insight</h4>
          <p style='font-size: 1rem;'>{insight_text1}</p>
          <h4 style='margin-top: 1rem; margin-bottom: 0.5rem;'>✅ Recommendation</h4>
          <p style='font-size: 1rem;'>{recommendation_text1}</p>
      </div>
      """, unsafe_allow_html=True)
  #################################################################################################################################################################
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
                  'Humidity(%)','Humidity(%)_90th_percentile'],
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
####################################################
  insight_text2, recommendation_text2 = plot_metric_map_with_ai_insight(
    gdf=gdf_filtered,
    selected_floor=selected_floor,
    metric_col='Humidity(%)_Legend',
    color_map={
    # Occupancy & Utilisation
    # "High Utilisation & Occupancy": "#d62728",
    # "High Utilisation But Low Occupancy": "#87CEEB",
    # "Low Utilisation & Occupancy": "#2ca02c",
    # "Low Utilisation But High Occupancy": "#1f77b4",
    "Other Workspaces": "#fafafa",
    # # Humidity
    'Low Humidity(<30%)': '#87CEEB',   # Light Blue
    'Ideal Humidity(30%-60%)': '#1f77b4', # Blue
    'High Humidity(>60%)': '#d62728' ,  # Red
    # # CO2
    # 'Low Co2(<400ppm)': '#87CEEB',   # Light Blue
    # 'Ideal Co2(400ppm-800ppm)': '#1f77b4', # Blue
    # 'High Co2(>800)': '#d62728' ,  # Red
    # # Temperature
    # 'Low Temperature(<20C)': '#87CEEB',   # Light Blue
    # 'Ideal Temperature(20C-26C)': '#1f77b4', # Blue
    # 'High Temperature(>26C)': '#d62728' ,  # Red
    # # Aparent Temperature
    # 'Low Apparent Temperature(<20C)': '#87CEEB',   # Light Blue
    # 'Ideal Apparent Temperature(20C-26C)': '#1f77b4', # Blue
    # 'High Apparent Temperature(>26C)': '#d62728' ,  # Red

},
    legend_title="Workspace Humidity",
    label_formatter=lambda row: (
        f"{row['name']}\nAvg. Humidity: {row['Humidity(%)']}\nHumidity_90th_percentile: {row['Humidity(%)_90th_percentile']}\nCapacity {row['Space_Capacity_label']}"
        if row['name'] != 'none' and str(row['Humidity(%)']).endswith('ppm') else None
    ),
    image_output_path='images/humidity.png',
    ai_metric_key='Humidity',
    util_threshold=util_threshold,
    occ_threshold=occ_threshold
)
  # 2. Display it in a styled tile below the chart
  with st.container():
      st.markdown(f"""
      <div style='padding: 1rem; background-color: #f0f8ff; border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.05);'>
          <h4 style='margin-bottom: 0.5rem;'>🔍 AI Insight</h4>
          <p style='font-size: 1rem;'>{insight_text2}</p>
          <h4 style='margin-top: 1rem; margin-bottom: 0.5rem;'>✅ Recommendation</h4>
          <p style='font-size: 1rem;'>{recommendation_text2}</p>
      </div>
      """, unsafe_allow_html=True)
######################

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
                  'Co2(ppm)','Co2(ppm)_90th_percentile'],
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

  ####################################################
  insight_text3, recommendation_text3 = plot_metric_map_with_ai_insight(
    gdf=gdf_filtered,
    selected_floor=selected_floor,
    metric_col='Co2(ppm)_Legend',
    color_map={
    # Occupancy & Utilisation
    # "High Utilisation & Occupancy": "#d62728",
    # "High Utilisation But Low Occupancy": "#87CEEB",
    # "Low Utilisation & Occupancy": "#2ca02c",
    # "Low Utilisation But High Occupancy": "#1f77b4",
    "Other Workspaces": "#fafafa",
    # # Humidity
    # 'Low Humidity(<30%)': '#87CEEB',   # Light Blue
    # 'Ideal Humidity(30%-60%)': '#1f77b4', # Blue
    # 'High Humidity(>60%)': '#d62728' ,  # Red
    # # CO2
    'Low Co2(<400ppm)': '#87CEEB',   # Light Blue
    'Ideal Co2(400ppm-800ppm)': '#1f77b4', # Blue
    'High Co2(>800)': '#d62728' ,  # Red
    # # Temperature
    # 'Low Temperature(<20C)': '#87CEEB',   # Light Blue
    # 'Ideal Temperature(20C-26C)': '#1f77b4', # Blue
    # 'High Temperature(>26C)': '#d62728' ,  # Red
    # # Aparent Temperature
    # 'Low Apparent Temperature(<20C)': '#87CEEB',   # Light Blue
    # 'Ideal Apparent Temperature(20C-26C)': '#1f77b4', # Blue
    # 'High Apparent Temperature(>26C)': '#d62728' ,  # Red

},
    legend_title="Workspace CO2",
    label_formatter=lambda row: (
        f"{row['name']}\nAvg. CO2: {row['Co2(ppm)']}\nCO2_90th_percentile: {row['Co2(ppm)_90th_percentile']}\nCapacity {row['Space_Capacity_label']}"
        if row['name'] != 'none' and str(row['Co2(ppm)']).endswith('ppm') else None
    ),
    image_output_path='images/co2.png',
    ai_metric_key='CO2',
    util_threshold=util_threshold,
    occ_threshold=occ_threshold
)
  # 2. Display it in a styled tile below the chart
  with st.container():
      st.markdown(f"""
      <div style='padding: 1rem; background-color: #f0f8ff; border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.05);'>
          <h4 style='margin-bottom: 0.5rem;'>🔍 AI Insight</h4>
          <p style='font-size: 1rem;'>{insight_text3}</p>
          <h4 style='margin-top: 1rem; margin-bottom: 0.5rem;'>✅ Recommendation</h4>
          <p style='font-size: 1rem;'>{recommendation_text3}</p>
      </div>
      """, unsafe_allow_html=True)
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
                  'Temp(C)','Temp(C)_90th_percentile'],
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
####################################################
  insight_text4, recommendation_text4 = plot_metric_map_with_ai_insight(
    gdf=gdf_filtered,
    selected_floor=selected_floor,
    metric_col='Temp(C)_Legend',
    color_map={
    # Occupancy & Utilisation
    # "High Utilisation & Occupancy": "#d62728",
    # "High Utilisation But Low Occupancy": "#87CEEB",
    # "Low Utilisation & Occupancy": "#2ca02c",
    # "Low Utilisation But High Occupancy": "#1f77b4",
    "Other Workspaces": "#fafafa",
    # # Humidity
    # 'Low Humidity(<30%)': '#87CEEB',   # Light Blue
    # 'Ideal Humidity(30%-60%)': '#1f77b4', # Blue
    # 'High Humidity(>60%)': '#d62728' ,  # Red
    # # CO2
    # 'Low Co2(<400ppm)': '#87CEEB',   # Light Blue
    # 'Ideal Co2(400ppm-800ppm)': '#1f77b4', # Blue
    # 'High Co2(>800)': '#d62728' ,  # Red
    # # Temperature
    'Low Temperature(<20C)': '#87CEEB',   # Light Blue
    'Ideal Temperature(20C-26C)': '#1f77b4', # Blue
    'High Temperature(>26C)': '#d62728' ,  # Red
    # # Aparent Temperature
    # 'Low Apparent Temperature(<20C)': '#87CEEB',   # Light Blue
    # 'Ideal Apparent Temperature(20C-26C)': '#1f77b4', # Blue
    # 'High Apparent Temperature(>26C)': '#d62728' ,  # Red

},
    legend_title="Workspace Temperature(C)",
    label_formatter=lambda row: (
        f"{row['name']}\nAvg. Temp: {row['Temp(C)']}\nTemp_90th_percentile: {row['Temp(C)_90th_percentile']}\nCapacity {row['Space_Capacity_label']}"
        if row['name'] != 'none' and str(row['Temp(C)']).endswith(' C') else None
    ),
    image_output_path='images/temp.png',
    ai_metric_key='Temp',
    util_threshold=util_threshold,
    occ_threshold=occ_threshold
)
  # 2. Display it in a styled tile below the chart
  with st.container():
      st.markdown(f"""
      <div style='padding: 1rem; background-color: #f0f8ff; border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.05);'>
          <h4 style='margin-bottom: 0.5rem;'>🔍 AI Insight</h4>
          <p style='font-size: 1rem;'>{insight_text4}</p>
          <h4 style='margin-top: 1rem; margin-bottom: 0.5rem;'>✅ Recommendation</h4>
          <p style='font-size: 1rem;'>{recommendation_text4}</p>
      </div>
      """, unsafe_allow_html=True)
######################
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
  ####################################################
  insight_text5, recommendation_text5 = plot_metric_map_with_ai_insight(
    gdf=gdf_filtered,
    selected_floor=selected_floor,
    metric_col='ApparentTemp(C)_Legend',
    color_map={
    # Occupancy & Utilisation
    # "High Utilisation & Occupancy": "#d62728",
    # "High Utilisation But Low Occupancy": "#87CEEB",
    # "Low Utilisation & Occupancy": "#2ca02c",
    # "Low Utilisation But High Occupancy": "#1f77b4",
    "Other Workspaces": "#fafafa",
    # # Humidity
    # 'Low Humidity(<30%)': '#87CEEB',   # Light Blue
    # 'Ideal Humidity(30%-60%)': '#1f77b4', # Blue
    # 'High Humidity(>60%)': '#d62728' ,  # Red
    # # CO2
    # 'Low Co2(<400ppm)': '#87CEEB',   # Light Blue
    # 'Ideal Co2(400ppm-800ppm)': '#1f77b4', # Blue
    # 'High Co2(>800)': '#d62728' ,  # Red
    # # Temperature
    # 'Low Temperature(<20C)': '#87CEEB',   # Light Blue
    # 'Ideal Temperature(20C-26C)': '#1f77b4', # Blue
    # 'High Temperature(>26C)': '#d62728' ,  # Red
    # # Aparent Temperature
    'Low Apparent Temperature(<20C)': '#87CEEB',   # Light Blue
    'Ideal Apparent Temperature(20C-26C)': '#1f77b4', # Blue
    'High Apparent Temperature(>26C)': '#d62728' ,  # Red

},
    legend_title="Workspace Apparent Temperature(C)",
    label_formatter=lambda row: (
        f"{row['name']}\nAvg. Temp: {row['ApparentTemp(C)']}\nTemp_90th_percentile: {row['ApparentTemp(C)_90th_percentile']}\nCapacity {row['Space_Capacity_label']}"
        if row['name'] != 'none' and str(row['ApparentTemp(C)']).endswith(' C') else None
    ),
    image_output_path='images/apparent_temp.png',
    ai_metric_key='Apparent_Temp',
    util_threshold=util_threshold,
    occ_threshold=occ_threshold
)
  # 2. Display it in a styled tile below the chart
  with st.container():
      st.markdown(f"""
      <div style='padding: 1rem; background-color: #f0f8ff; border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.05);'>
          <h4 style='margin-bottom: 0.5rem;'>🔍 AI Insight</h4>
          <p style='font-size: 1rem;'>{insight_text5}</p>
          <h4 style='margin-top: 1rem; margin-bottom: 0.5rem;'>✅ Recommendation</h4>
          <p style='font-size: 1rem;'>{recommendation_text5}</p>
      </div>
      """, unsafe_allow_html=True)
######################
