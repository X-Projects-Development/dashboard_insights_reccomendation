
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
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import Runnable
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import contextily as ctx
from shapely.geometry import Point
from adjustText import adjust_text
import base64
from io import BytesIO
import plotly.graph_objects as go

dotenv.load_dotenv()

# Load API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY





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
      st.write(floor_table.to_markdown())
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

def productivity_load_data():
    # Replace with paths to your processed files if needed
    df = pd.read_pickle("Data/monthly_productivity_environmental_metirc.pkl")
    gdf = gpd.read_file( "Data/SFO12.geojson")
    df['Non-Violation_Apparent_Temp'] = df['Count_of_intervals_with_occupancy'] - df['Violation_Apparent_Temp'] 
    df['Non-Violation_CO2'] = df['Count_of_intervals_with_occupancy'] - df['Violation_CO2']
    gdf['lvl'] = gdf['lvl'].astype(str)
    # clean names
    gdf['name'] = (gdf['name'].astype(str).str.strip().str.lower().str.replace(r'\s+', ' ', regex=True) ) # collapse multiple spaces
    df['Workspace Name'] = (df['Workspace Name'].astype(str).str.strip().str.lower().str.replace(r'\s+', ' ', regex=True) ) # collapse multiple spaces
    gdf = gdf.loc[~np.logical_or(gdf['sType'] == 'Floor Outline',
                                    gdf['type'] == 'marker'),].reset_index(drop=True)
    # gdf = gdf.loc[~gdf['sType'].isin(['Floor Outline','Workstation Desk','Workstation Chair','Workstation','Stairs','Cafeteria','Entertainment','Storage','Chair','Meeting Room Desk',
    #                           'Meeting Room Chair','Restroom','Support Space','Elevator','Storage Cabinet','Child Care','POU Chair']) ].reset_index(drop=True)


    return df, gdf





#-----------------------------------------------------------------------------------------




def generate_donut_chart(labels, values, title='Donut Chart', show_total=True):
    total = int(sum(values))
    center_text = f'Total<br>15min-intervals<br>{total}' if show_total else ''

    # Define custom colors based on label content
    custom_colors = []
    for label in labels:
        if 'violation' in label.lower():
            custom_colors.append('goldenrod')
        else:
            custom_colors.append('seagreen')

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.5,
        textinfo='label+percent',
        insidetextorientation='radial',
        marker=dict(
            line=dict(color='white', width=2),
            colors=custom_colors
        ),
        sort=False,
        direction='clockwise',
        pull=[0.05 if v == max(values) else 0 for v in values]
    )])

    fig.update_layout(
        title=dict(text=title),
        showlegend=False,
        annotations=[dict(
            text=center_text,
            x=0.5, y=0.5,
            font_size=20,
            showarrow=False
        )] if center_text else []
    )

    # fig.show()
    st.plotly_chart(fig, use_container_width=True)  # Uncomment for Streamlit




def plot_stacked_violation_chart_by_bins(df, violation_var, non_violation_var, title='Apparent Temp Violations by Meeting Size'):
    # Ensure index is reset and people count is numeric
    df = df.reset_index()
    df['Peak_People_Count_v1'] = pd.to_numeric(df['Peak_People_Count_v1'], errors='coerce')

    # Bin people counts into custom ranges
    def bin_people_count(x):
        if pd.isna(x):
            return 'Unknown'
        elif x <= 5:
            return str(int(x))
        elif x <= 10:
            return '6-10'
        elif x <= 15:
            return '11-15'
        elif x <= 20:
            return '16-20'
        else:
            return '21+'

    df['People_Bin'] = df['Peak_People_Count_v1'].apply(bin_people_count)

    # Aggregate values per bin
    grouped = df.groupby('People_Bin')[[violation_var, non_violation_var]].sum().reset_index()
    grouped['Total'] = grouped[violation_var] + grouped[non_violation_var]

    # Compute percentages
    grouped['Violation_pct'] = (grouped[violation_var] / grouped['Total'] * 100).round(1)
    grouped['NonViolation_pct'] = (grouped[non_violation_var] / grouped['Total'] * 100).round(1)

    # Combine raw count and % for labels
    grouped['Violation_text'] = grouped[violation_var].astype(int).astype(str) + ' (' + grouped['Violation_pct'].astype(str) + '%)'
    grouped['NonViolation_text'] = grouped[non_violation_var].astype(int).astype(str) + ' (' + grouped['NonViolation_pct'].astype(str) + '%)'

    # Ensure proper order of bins
    ordered_bins = ['1' , '2', '3', '4' ,'5', '6-10', '11-15', '16-20', '21+']
    grouped['People_Bin'] = pd.Categorical(grouped['People_Bin'], categories=ordered_bins, ordered=True)
    grouped = grouped.sort_values('People_Bin')

    # Plot
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=grouped['People_Bin'],
        y=grouped[violation_var],
        name='Out of Range',
        marker_color='goldenrod',
        text=grouped['Violation_text'],
        textposition='inside',
        texttemplate='%{text}'
    ))

    fig.add_trace(go.Bar(
        x=grouped['People_Bin'],
        y=grouped[non_violation_var],
        name='Ideal Range',
        marker_color='seagreen',
        text=grouped['NonViolation_text'],
        textposition='inside',
        texttemplate='%{text}'
    ))

    fig.update_layout(
        barmode='stack',
        xaxis_title='Meeting Size',
        yaxis_title='Number of 15min-Intervals',
        title=title,
        legend_title='Status',
        xaxis_tickangle=-45,
        template='plotly_white',
        uniformtext_minsize=8,
        uniformtext_mode='hide',
        xaxis=dict(categoryorder='array', categoryarray=ordered_bins)
    )

    # fig.show()
    st.plotly_chart(fig, use_container_width=True)  # Uncomment for Streamlit



import plotly.graph_objects as go
import pandas as pd

def plot_bar_chart(df, x_col, y_cols, title='Bar Chart', barmode='group', colors=None, show_percent=False):
    """
    Plots a bar chart using Plotly.

    Parameters:
    - df: pandas DataFrame
    - x_col: column name for x-axis (categorical)
    - y_cols: list of column names for y-values (1+ series)
    - title: chart title
    - barmode: 'group' or 'stack'
    - colors: optional list of colors
    - show_percent: if True, shows % of total per bar for each y_col
    """
    fig = go.Figure()

    if isinstance(y_cols, str):
        y_cols = [y_cols]

    for idx, col in enumerate(y_cols):
        color = colors[idx] if colors and idx < len(colors) else None
        text = None

        if show_percent:
            total = df[y_cols].sum(axis=1)
            percent = (df[col] / total * 100).round(1)
            text = df[col].astype(str) + '%'

        fig.add_trace(go.Bar(
            x=df[x_col].astype(int).astype(str),
            y=df[col],
            name=col,
            marker_color=color,
            text=text,
            textposition='auto' if show_percent else None
        ))

    fig.update_layout(
        title=title,
        xaxis_title='Number of People in Meeting',
        yaxis_title='Productivity%',
        barmode=barmode,
        template='plotly_white'
    )

    # fig.show()
    st.plotly_chart(fig, use_container_width=True)  # Uncomment for Streamlit


###  Filter and Clean Data
st.title("Productivity Analysis")
# set wide width
st.set_page_config(layout="wide")

productivity_df, gdf = productivity_load_data()
# add month filter to sidebar from productivity_df
available_months = sorted(productivity_df['Month'].unique())
selected_month = st.sidebar.selectbox("Select Month", available_months, index=len(available_months)-1)

with st.expander("ℹ️ Temperature and Productivity Explained", expanded=False):
    st.markdown("""
### 🧠 How Temperature Affects Productivity

Apparent Temperature (°C): The perceived temperature that combines temperature, humidity, and airflow effects. Monthly medain of apparent temperature readings when workspace is occupied.
For details: https://en.wikipedia.org/wiki/Apparent_temperature
Apparent Temperature plays a significant role in workspace productivity. Based on research-informed modeling:

- ✅ **Optimal temperature range:**  
  Temperatures between **21°C and 24°C** are considered ideal. Within this range, productivity is maintained at **100%**.

- 🔻 **Above 24°C:**  
  For every degree **above 24°C**, productivity **drops by 2%**.  
  For example:
  - At 25°C → 98% productivity  
  - At 27°C → 94% productivity

- 🔻 **Below 21°C:**  
  For every degree **below 21°C**, productivity **drops by 4%**.  
  For example:
  - At 20°C → 96% productivity  
  - At 18°C → 88% productivity

- ❗ Temperatures far outside this range may result in **significant performance decline**.

These thresholds are used in the productivity model shown in the workspace analysis above.

Sources

https://www.merlin-technology.com/en/newsfeed/news/working-at-high-temperatures-reduced-productivity

https://cool-r.eu/the-influence-of-temperatures-on-the-employees/

https://www.jll.ca/en/trends-and-insights/workplace/a-surprising-way-to-cut-real-estate-costs
    """)

# filter data
productivity_df = productivity_df.loc[productivity_df['Month'] == selected_month].reset_index(drop=True)



# donut chart
Violation_value = productivity_df.loc[~productivity_df['avg_temp'].isna()]['Violation_Apparent_Temp'].sum()
Non_Violation_value = productivity_df.loc[~productivity_df['avg_temp'].isna()]['Non-Violation_Apparent_Temp'].sum()
labels = ['Violation', 'Ideal Range']
values = [Violation_value, Non_Violation_value]  

# bar chart
temp_df = productivity_df.loc[~productivity_df['avg_temp'].isna()].groupby('Peak_People_Count_v1')[['Violation_Apparent_Temp', 'Non-Violation_Apparent_Temp']].sum().reset_index()

## Generate Charts
col1, col2 = st.columns([1,2])
with col1:
  generate_donut_chart(labels, values, title='Apparent Temperature Violation Distribution')
with col2:
  plot_stacked_violation_chart_by_bins(temp_df,'Violation_Apparent_Temp', 'Non-Violation_Apparent_Temp',  title='Apparent Temperature Violations by Meeting Size')
  
st.title("Measuring Loss of Productivity")

floor_df = (
    productivity_df
    .groupby(['lvl', 'Workspace Name'])
    .agg({
        'Violation_Apparent_Temp': lambda x: x.fillna(0).sum(),
        'Count_of_intervals_with_occupancy': lambda x: x.fillna(0).sum(),
        'Productivity_Apparent_Temp': lambda x: x.dropna().mean()
    })
    .reset_index()
)

floor_df['percentage_of_violations'] = (floor_df['Violation_Apparent_Temp'] / floor_df['Count_of_intervals_with_occupancy'] * 100).round(1).astype(str) + '%'

# bar chart


### -----------------
### Dollar Loss
### -----------------
# Assume these values are already defined in your script
total_employees = 2950
avg_salary = 200000  # Storing as a number for calculation
avg_building_utilization = 0.30  # Storing as a float for calculation
avg_productivity_loss = ( 100 - round(floor_df['Productivity_Apparent_Temp'].mean(),1) ) /100

# Calculate dollar loss
dollar_loss = total_employees * avg_salary * avg_building_utilization * avg_productivity_loss  * (1/12)

# Format currency for display
formatted_avg_salary = f"{avg_salary / 1000:.0f}K USD" # Formats 200000 as $200K
formatted_dollar_loss = f"{dollar_loss:,.0f} USD" # Adds commas for thousands

with st.container():
    st.markdown(
        f"""
        <div style="background-color:#f0f0f0; padding: 20px; border-radius: 8px; font-size: 18px;">
            <h4 style="margin-top: 0;">💰 Dollar Loss</h4>
            <ul style="line-height: 1.8; list-style-type: disc; padding-left: 20px;">
                <li><strong>Total Employees:</strong> {total_employees}</li>
                <li><strong>Average Salary:</strong> {formatted_avg_salary}</li>
                <li><strong>Average Building Utilisation:</strong> {avg_building_utilization:.0%}</li>
                <li><strong>Productivity Loss (Due to Temperature):</strong> {avg_productivity_loss:.1%}</li>
                <li><strong>Estimated Monthly Productivity Loss:</strong><br>
                    {total_employees} employees × {formatted_avg_salary} salary × {avg_building_utilization:.0%} utilization × {avg_productivity_loss:.1%} loss  
                    = <strong>{formatted_dollar_loss}</strong>
                </li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )
# --- End of container ---


plot_bar_chart(productivity_df.loc[~productivity_df['avg_temp'].isna()].groupby('Peak_People_Count_v1')[['Productivity_Apparent_Temp']].mean().reset_index().round(1), 
               'Peak_People_Count_v1', 'Productivity_Apparent_Temp', title='Productivity By Meeting Size', barmode='group', colors=None, show_percent=True)

floor_df = pd.merge(gdf, floor_df, left_on=['lvl','name'],
                right_on=['lvl','Workspace Name'], how='left')


# create multiple tabs
floor1,floor2,floor3,floor4,floor5 = st.tabs(["Floor1", "Floor2", "Floor3", "Floor4", "Floor5"])
custom_scale = [
    [0.0, "lightgray"],  # 0 = Missing or no productivity
    [0.9, "red"],      # 90 = Low high-productivity range
    [1.0, "yellow"]         # 100 = Peak productivity
]


with floor1:
    selected_floor = '1'
    
    # Filter and prepare
    floor_df_filtered = floor_df[floor_df['lvl'] == selected_floor].reset_index(drop=True)
    
    # Fill NA with 0, and round values
    floor_df_filtered["Productivity_Apparent_Temp"] = floor_df_filtered["Productivity_Apparent_Temp"].fillna(0).round(1)

    # Normalize productivity values to 0–1 for color scale
    floor_df_filtered["ProdNorm"] = floor_df_filtered["Productivity_Apparent_Temp"] / 100

 
    # Plot choropleth
    fig = px.choropleth_mapbox(
        floor_df_filtered,
        geojson=floor_df_filtered.geometry,
        locations=floor_df_filtered.index,
        color="ProdNorm",  # use normalized field
        hover_name="name",
        hover_data=[
            "lvl", 
            "Violation_Apparent_Temp", 
            "percentage_of_violations", 
            "Productivity_Apparent_Temp"
        ],
        color_continuous_scale=custom_scale,
        range_color=[0, 1],
        mapbox_style="carto-positron",
        center={
            "lat": floor_df_filtered.geometry.centroid.y.mean(),
            "lon": floor_df_filtered.geometry.centroid.x.mean()
        },
        zoom=19,
        opacity=0.6,
        title=f"Workspace Productivity Analysis - Floor {selected_floor}"
    )

    # Update layout
    fig.update_layout(
        margin={"r": 0, "t": 30, "l": 0, "b": 0},
        height=800,
        coloraxis_colorbar=dict(
            title="Productivity (%)",
            tickvals=[0, 0.9, 1.0],
            ticktext=["0", "90", "100"]
        ),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )

    st.plotly_chart(fig, use_container_width=True)


with floor2:
    selected_floor = '2'
    
    # Filter and prepare
    floor_df_filtered = floor_df[floor_df['lvl'] == selected_floor].reset_index(drop=True)
    
    # Fill NA with 0, and round values
    floor_df_filtered["Productivity_Apparent_Temp"] = floor_df_filtered["Productivity_Apparent_Temp"].fillna(0).round(1)

    # Normalize productivity values to 0–1 for color scale
    floor_df_filtered["ProdNorm"] = floor_df_filtered["Productivity_Apparent_Temp"] / 100

 
    # Plot choropleth
    fig = px.choropleth_mapbox(
        floor_df_filtered,
        geojson=floor_df_filtered.geometry,
        locations=floor_df_filtered.index,
        color="ProdNorm",  # use normalized field
        hover_name="name",
        hover_data=[
            "lvl", 
            "Violation_Apparent_Temp", 
            "percentage_of_violations", 
            "Productivity_Apparent_Temp"
        ],
        color_continuous_scale=custom_scale,
        range_color=[0, 1],
        mapbox_style="carto-positron",
        center={
            "lat": floor_df_filtered.geometry.centroid.y.mean(),
            "lon": floor_df_filtered.geometry.centroid.x.mean()
        },
        zoom=19,
        opacity=0.6,
        title=f"Workspace Productivity Analysis - Floor {selected_floor}"
    )

    # Update layout
    fig.update_layout(
        margin={"r": 0, "t": 30, "l": 0, "b": 0},
        height=800,
        coloraxis_colorbar=dict(
            title="Productivity (%)",
            tickvals=[0, 0.9, 1.0],
            ticktext=["0", "90", "100"]
        ),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )

    st.plotly_chart(fig, use_container_width=True)


with floor3:
    selected_floor = '3'
    
    # Filter and prepare
    floor_df_filtered = floor_df[floor_df['lvl'] == selected_floor].reset_index(drop=True)
    
    # Fill NA with 0, and round values
    floor_df_filtered["Productivity_Apparent_Temp"] = floor_df_filtered["Productivity_Apparent_Temp"].fillna(0).round(1)

    # Normalize productivity values to 0–1 for color scale
    floor_df_filtered["ProdNorm"] = floor_df_filtered["Productivity_Apparent_Temp"] / 100

 
    # Plot choropleth
    fig = px.choropleth_mapbox(
        floor_df_filtered,
        geojson=floor_df_filtered.geometry,
        locations=floor_df_filtered.index,
        color="ProdNorm",  # use normalized field
        hover_name="name",
        hover_data=[
            "lvl", 
            "Violation_Apparent_Temp", 
            "percentage_of_violations", 
            "Productivity_Apparent_Temp"
        ],
        color_continuous_scale=custom_scale,
        range_color=[0, 1],
        mapbox_style="carto-positron",
        center={
            "lat": floor_df_filtered.geometry.centroid.y.mean(),
            "lon": floor_df_filtered.geometry.centroid.x.mean()
        },
        zoom=19,
        opacity=0.6,
        title=f"Workspace Productivity Analysis - Floor {selected_floor}"
    )

    # Update layout
    fig.update_layout(
        margin={"r": 0, "t": 30, "l": 0, "b": 0},
        height=800,
        coloraxis_colorbar=dict(
            title="Productivity (%)",
            tickvals=[0, 0.9, 1.0],
            ticktext=["0", "90", "100"]
        ),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )

    st.plotly_chart(fig, use_container_width=True)


with floor4:
    selected_floor = '4'
    
    # Filter and prepare
    floor_df_filtered = floor_df[floor_df['lvl'] == selected_floor].reset_index(drop=True)
    
    # Fill NA with 0, and round values
    floor_df_filtered["Productivity_Apparent_Temp"] = floor_df_filtered["Productivity_Apparent_Temp"].fillna(0).round(1)

    # Normalize productivity values to 0–1 for color scale
    floor_df_filtered["ProdNorm"] = floor_df_filtered["Productivity_Apparent_Temp"] / 100

 
    # Plot choropleth
    fig = px.choropleth_mapbox(
        floor_df_filtered,
        geojson=floor_df_filtered.geometry,
        locations=floor_df_filtered.index,
        color="ProdNorm",  # use normalized field
        hover_name="name",
        hover_data=[
            "lvl", 
            "Violation_Apparent_Temp", 
            "percentage_of_violations", 
            "Productivity_Apparent_Temp"
        ],
        color_continuous_scale=custom_scale,
        range_color=[0, 1],
        mapbox_style="carto-positron",
        center={
            "lat": floor_df_filtered.geometry.centroid.y.mean(),
            "lon": floor_df_filtered.geometry.centroid.x.mean()
        },
        zoom=19,
        opacity=0.6,
        title=f"Workspace Productivity Analysis - Floor {selected_floor}"
    )

    # Update layout
    fig.update_layout(
        margin={"r": 0, "t": 30, "l": 0, "b": 0},
        height=800,
        coloraxis_colorbar=dict(
            title="Productivity (%)",
            tickvals=[0, 0.9, 1.0],
            ticktext=["0", "90", "100"]
        ),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )

    st.plotly_chart(fig, use_container_width=True)


with floor5:
    selected_floor = '5'
    
    # Filter and prepare
    floor_df_filtered = floor_df[floor_df['lvl'] == selected_floor].reset_index(drop=True)
    
    # Fill NA with 0, and round values
    floor_df_filtered["Productivity_Apparent_Temp"] = floor_df_filtered["Productivity_Apparent_Temp"].fillna(0).round(1)

    # Normalize productivity values to 0–1 for color scale
    floor_df_filtered["ProdNorm"] = floor_df_filtered["Productivity_Apparent_Temp"] / 100

 
    # Plot choropleth
    fig = px.choropleth_mapbox(
        floor_df_filtered,
        geojson=floor_df_filtered.geometry,
        locations=floor_df_filtered.index,
        color="ProdNorm",  # use normalized field
        hover_name="name",
        hover_data=[
            "lvl", 
            "Violation_Apparent_Temp", 
            "percentage_of_violations", 
            "Productivity_Apparent_Temp"
        ],
        color_continuous_scale=custom_scale,
        range_color=[0, 1],
        mapbox_style="carto-positron",
        center={
            "lat": floor_df_filtered.geometry.centroid.y.mean(),
            "lon": floor_df_filtered.geometry.centroid.x.mean()
        },
        zoom=19,
        opacity=0.6,
        title=f"Workspace Productivity Analysis - Floor {selected_floor}"
    )

    # Update layout
    fig.update_layout(
        margin={"r": 0, "t": 30, "l": 0, "b": 0},
        height=800,
        coloraxis_colorbar=dict(
            title="Productivity (%)",
            tickvals=[0, 0.9, 1.0],
            ticktext=["0", "90", "100"]
        ),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )

    st.plotly_chart(fig, use_container_width=True)

