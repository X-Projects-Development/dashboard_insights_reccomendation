
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
dotenv.load_dotenv()

# Load API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# if "OPENAI_API_KEY" not in os.environ:
#     # os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")
#     os.environ["OPENAI_API_KEY"] = ""

# ---------------------
# Load preprocessed data
# ---------------------
all_monthly_data = os.path.join(os.getcwd(),"Data","all_months_occupancy_probability_distribution.pkl")
monthly_util = os.path.join(os.getcwd(),"Data","monthly_utilisation_when_occupied_vs_occupancy.pkl")
@st.cache_data
def load_data():
    # Replace with paths to your processed files if needed
    melted = pd.read_pickle(all_monthly_data)
    df = pd.read_pickle(monthly_util)
    return melted, df

melted, df = load_data()

st.set_page_config(layout="wide")  # <– Set wide layout here

# ---------------------
# Sidebar UI
# ---------------------
st.sidebar.title("Filter Options")
capacity_options = sorted(melted["Workspace_Capacity_Category"].unique())
# select medium by default
selected_category = st.sidebar.selectbox(
    "Select Workspace Capacity Category:",
    capacity_options,
    index=capacity_options.index("Medium") if "Medium" in capacity_options else 0
)



# ---------------------
# Color Palette
# ---------------------
# Cisco-style color palette
cisco_palette = {
    "Small":       "#0097A7",   # Teal Accent
    "Medium":      "#00778B",   # Cisco Blue Mid
    "Large":       "#004E68",   # Darker Blue
    "Extra Large": "#00263A",    # Very Dark Blue
    "Unknown":     "#CCCCCC"
}


# ---------------------
# Filtered Data
# ---------------------
melted_filtered = melted[melted["Workspace_Capacity_Category"] == selected_category]
df_filtered = df[df["Workspace_Capacity_Category"] == selected_category]


st.title("Cisco Space Occupancy Analysis - SFO12")

# ---------------------
# Chart 1: Smoothed Probability Curve
# ---------------------

st.header(f"Occupancy Probability Distribution – {selected_category} - Jan to May 2025")

melted_filtered = melted_filtered.sort_values("Rooms_Occupied")
melted_filtered["Smoothed_Probability"] = (
    melted_filtered["Probability"]
    .rolling(window=3, min_periods=1, center=True)
    .mean()
)

fig1 = go.Figure()
fig1.add_trace(go.Scatter(
    x=melted_filtered["Rooms_Occupied"],
    y=melted_filtered["Smoothed_Probability"],
    mode='lines',
    customdata=np.stack([
        melted_filtered['Occupied_hrs'],
        melted_filtered['Interval_Count'],
        melted_filtered['Smoothed_Probability']
    ], axis=-1),
    hovertemplate=(
        'Rooms Occupied: %{x}<br>'
        'Smoothed Probability: %{y:.2f}%<br>'
        'Occupied Hours: %{customdata[0]:.1f} hrs<br>'
        '<extra></extra>'
    ),
    line=dict(width=3)
))
fig1.update_layout(
    xaxis_title="Simultaneously Occupied Rooms",
    yaxis_title="Occupancy Probability (%)",
    height=500
)

st.plotly_chart(fig1, use_container_width=True)

########### add insights #################

def generate_insight_and_recommendation_consecutive_rooms(df: pd.DataFrame, capacity: str) -> dict:
    """
    Generate a concise insight and actionable recommendation from a DataFrame.

    Parameters:
        df (pd.DataFrame): The input table (e.g., summarised occupancy data).
        capacity (str): The workspace capacity category.

    Returns:
        dict: Dictionary with 'insight' and 'recommendation' keys.
    """
    if df.empty:
        return {
            "insight": "The input table is empty, so no trends can be detected.",
            "recommendation": "Please provide a non-empty table with relevant data."
        }
    # Set up the chain only once
    llm = ChatOpenAI(model="gpt-4o")  # You can swap with "gpt-3.5-turbo"
    parser = JsonOutputParser()
    total_num_rooms = df.loc[df['Workspace_Capacity_Category']==capacity, "Total_Rooms"].mean()
    df = df[["Workspace_Capacity_Category","Rooms_Occupied","Occupied_hrs",	"Probability"]]
    prompt_template = PromptTemplate.from_template(
        """
    You are a workspace planner. The following table shows concurrent usage of rooms, their sizes , number of room simultaneously occupied, the probability of rooms being simultaneously occupied and the duration it was occupied for :

    {table}

    In SFO12 building we have {total_num_rooms} {size} workspaces.
    For rooms of capacity {size} Generate a clear, concise:
    1. Insight — what pattern do we see in simulataneous room occupancy. How many rooms are enough ensuring rooms are available even during extreme scenarios?
    2. Recommendation — any resizing reccomendation for the workspace planner?

    Return your answer as a JSON object like:
    {{
      "insight": "...",
      "recommendation": "..."
    }}
    """
    )

    # Build the LangChain pipeline
    chain = prompt_template | llm | parser

    table_str = df.to_markdown(index=False)
    result = chain.invoke({"table": table_str,
                           "size": capacity,
                           "total_num_rooms":total_num_rooms})
    return result["insight"], result["recommendation"]

new_df1 = melted.sort_values(["Workspace_Capacity_Category","Rooms_Occupied"])

# @st.cache_data(show_spinner="Generating insight...")
# 1. Run the AI insight generation
with st.spinner("Generating AI Insight and Recommendation..."):
    def get_cached_insight(df, category):
        return generate_insight_and_recommendation_consecutive_rooms(df, category)

    insight_text1, recommendation_text1 = get_cached_insight(new_df1, selected_category)

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

# ---------------------
# Chart 2: Animated Scatter Plot
# ---------------------

st.header(f"Monthly Utilisation vs Occupancy – {selected_category}")

# Compute medians (over the whole dataset)
# median_util = df_filtered["Monthly_Utilisation"].median()
median_occ = round(df_filtered["Monthly_Occupancy%"].median(),0)
median_util = round(df_filtered["Monthly_Utilisation_when_Occupied_mean"].median(),0)
max_yaxis = df_filtered["Monthly_Utilisation_when_Occupied_mean"].max()
min_yaxis = df_filtered["Monthly_Utilisation_when_Occupied_mean"].min()
max_xaxis = df_filtered["Monthly_Occupancy%"].max()
min_xaxis = df_filtered["Monthly_Occupancy%"].min()

fig2 = px.scatter(
    df_filtered,
    x="Monthly_Occupancy%",
    y="Monthly_Utilisation_when_Occupied_mean",
    color="Workspace_Capacity_Category",
    size="Space Capacity",
    hover_name="Workspace Name",
    animation_frame="Month Name",
    color_discrete_map=cisco_palette,
    # title="Monthly Utilisation when Occupied vs Occupancy by Workspace Capacity Category",
    size_max=20,
    # size_min = 20,
    height=800,
    hover_data={

        "Month Name": True,
        "Workspace_Capacity_Category": True,
        "Space_Capacity_label": True,
        "Floor Name": True,
        "Monthly_Occupancy_label": True,
        "Utilisation_when_Occupied_label": True,
        "Max_Occupancy_label": True,
        "Avg_Occupancy_label": True,
        "Median_Occupancy_label": True,
        "Occupancy_90_label": True,
        "Occupancy_85_label": True,
        "Occupancy_80_label": True,
        "Avg_Dwell_Time_label": True,
        # Hide default fields
        "Monthly_Occupancy%": False,
        "Monthly_Utilisation_when_Occupied_mean": False,
        "Space Capacity": False,
        "Workspace Name": False


    }
)


# Add median lines (as static shapes that appear in all frames)
fig2.add_shape(
    type="line",
    x0=median_occ,
    x1=median_occ,
    y0=-50, y1=200,
    line=dict(color="red", dash="dash"),
)

fig2.add_shape(
    type="line",
    x0=-50, x1=100,
    y0=median_util, y1=median_util,
    line=dict(color="red", dash="dash"),
)

# Add annotations (static across all frames)
fig2.add_annotation(x=median_occ, y=-8, text=f"Median Occupancy ({median_occ:.2f}%)", showarrow=True, arrowhead=1)
fig2.add_annotation(x=-8, y=median_util,  text=f"Median Utilisation ({median_util:.2f}%)", showarrow=True, arrowhead=1)

# Fix axes so they don’t rescale per frame
fig2.update_layout(
    xaxis=dict(range=[-10, max_xaxis * 1.1],
               title="Average Occupancy (%)"),
    yaxis=dict(
        range=[-10, max_yaxis * 1.1],
        title="Average Utilisation (%) When Occupied"
    )
)


# Set up animation controls and remove legend
fig2.update_layout(
    showlegend=False,
    updatemenus=[{
        "type": "buttons",
        "showactive": False,
        "buttons": [
            {
                "label": "Play",
                "method": "animate",
                "args": [
                    None,
                    {
                        "frame": {"duration": 2000, "redraw": True},
                        "fromcurrent": True,
                        "transition": {"duration": 500, "easing": "linear"}
                    }
                ]
            },
            {
                "label": "Pause",
                "method": "animate",
                "args": [[None], {"frame": {"duration": 0}, "mode": "immediate"}]
            }
        ]
    }]
)

# Step 1: Remove any annotation from the base layout
# fig2.update_layout(annotations=[])

# Step 2: Add watermark annotation to each frame
for frame in fig2.frames:
    frame.layout = go.Layout(
        annotations=[
            dict(
                text=frame.name,
                xref="paper", yref="paper",
                x=0.2, y=0.95,
                showarrow=False,
                font=dict(size=40, color="rgba(200,200,200,0.8)"),
                xanchor='right', yanchor='bottom'
            )
        ]
    )



st.plotly_chart(fig2, use_container_width=True)

# Add insights


def generate_insight_and_recommendation_utilisation_occupancy(df: pd.DataFrame, capacity: str,median_occ: int,median_util: int) -> dict:
    """
    Generate a concise insight and actionable recommendation from a DataFrame.

    Parameters:
        df (pd.DataFrame): The input table (e.g., summarised occupancy data).
        capacity (str): The workspace capacity category.
        median_occ (int): The median occupancy value.
        median_util (int): The median utilisation value.

    Returns:
        dict: Dictionary with 'insight' and 'recommendation' keys.
    """
    if df.empty:
        return {
            "insight": "The input table is empty, so no trends can be detected.",
            "recommendation": "Please provide a non-empty table with relevant data."
        }
    # Set up the chain only once
    llm = ChatOpenAI(model="gpt-4o")  # You can swap with "gpt-3.5-turbo"
    parser = JsonOutputParser()
    total_num_rooms = df['Workspace Name'].nunique()
    df = df[['Month Name', 'Workspace Name',
             'Monthly_Utilisation_when_Occupied_mean','Monthly_Occupancy%']]
    # rename column
    df.columns = ['Month Name', 'Workspace Name', 'Utilisation', 'Occupancy']
    prompt_template = PromptTemplate.from_template(
        """
    You are a workspace planner. The following table compares the Monthly Utilisation against the Monthly Occupancy for each workspace of capacity {size} in SFO12 building.
    Total Number of workspace of capcity {size} in SFO12 is {total_num_rooms}.


    Table Description:
    Workspace Name: The unique name or identifier of the workspace or meeting room.
    Month Name: The month and year corresponding to the aggregated metrics (e.g., 'January 2025').
    Utilisation: The average percentage of space used when the room was occupied, across all recorded intervals in the month. Utilisation% = (Number of people in room)/(Capacity of the room)
    Occupancy%: The average percentage of times the room is occupied during the day by someone during usual office hours (7am to 7pm) in the month.

    {table}


    The median utlisayion across all rooms is {median_utilisation}%.
    The median occupancy across all rooms is {median_occupancy}%.

    Generate a clear, concise:
    1. Insight — overall usage across rooms. How many are well used and which workspaces showing anomalous behaviour?
    2. Recommendation — what should the business do based on this insight?

    Return your answer as a JSON object like:
    {{
      "insight": "...",
      "recommendation": "..."
    }}
    """
    )

    # Build the LangChain pipeline
    chain = prompt_template | llm | parser

    table_str = df.to_markdown(index=False)
    result = chain.invoke({"table": table_str,
                           "size": capacity,
                           "total_num_rooms":total_num_rooms,
                           "median_utilisation":median_util,
                           "median_occupancy":median_occ})
    return result["insight"], result["recommendation"]

new_df2 = df_filtered.sort_values(["Workspace_Capacity_Category","Month Name"])
new_df2 = new_df2.loc[new_df2['Workspace_Capacity_Category']==selected_category,].reset_index(drop=True)

# @st.cache_data(show_spinner="Generating insight...")

# 1. Run the AI insight generation
with st.spinner("Generating AI Insight and Recommendation..."):
    def get_cached_insight(df, category, median_occ, median_util):
        return generate_insight_and_recommendation_utilisation_occupancy(df, category, median_occ, median_util)

    insight_text2, recommendation_text2 = get_cached_insight(new_df2, selected_category, median_occ, median_util)


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
