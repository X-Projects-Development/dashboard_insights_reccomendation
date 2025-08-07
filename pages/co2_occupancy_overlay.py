
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import plotly.graph_objects as go

st.title("Smart Room Occupancy Analysis")
st.set_page_config(layout="wide")

# Load Data
occupancy_filtered_readings = pd.read_csv('Data/occupancy_filtered_readings.csv')
building_df = pd.read_csv('Data/building_utilisation_df.csv')

# Combine date and time for DateTime column
occupancy_filtered_readings['DateTime'] = pd.to_datetime(
    occupancy_filtered_readings['Local Date'] + ' ' + occupancy_filtered_readings['Local Time']
)

# Filter only rows with valid CO2 data
df_with_co2 = occupancy_filtered_readings[occupancy_filtered_readings['co2'].notnull()]

# Separate Smart Room and Meeting Room lists
smart_rooms = df_with_co2[df_with_co2['Workspace Type'].str.lower().str.contains("smart")]['Space Name'].unique()
meeting_rooms = df_with_co2[df_with_co2['Workspace Type'].str.lower().str.contains("meeting")]['Space Name'].unique()

# Streamlit UI for Room Selection
col1, col2 = st.columns(2)
with col1:
    smart_room = st.selectbox("Select Smart Room", smart_rooms, key='smart')
with col2:
    meeting_room = st.selectbox("Select Meeting Room", meeting_rooms, key='meeting')


## Display Stats Above Plot
def room_stats(df, room_name):
    room_df = df[df['Space Name'] == room_name].reset_index(drop=True)
    space_capacity = room_df['Space Capacity'].iloc[0] if not room_df.empty else np.nan
    avg_people = room_df.loc[room_df['Peak People Count']>0,]['Peak People Count'].mean() if not room_df.empty else np.nan
    utilisation_rate = avg_people / space_capacity if (space_capacity and not np.isnan(avg_people)) else np.nan
    occupancy_rate = len(set(room_df.loc[room_df['Peak People Count']>0,]['DateTime']))/len(set(room_df['DateTime']))
    avg_co2 = room_df.loc[room_df['Peak People Count']>0,]['co2'].mean() if not room_df.empty else np.nan
    return space_capacity, avg_people, occupancy_rate, utilisation_rate, avg_co2

smart_stats = room_stats(df_with_co2, smart_room)
meeting_stats = room_stats(df_with_co2, meeting_room)

###  Display Stats Above Plot
col1, col2 = st.columns(2)
with col1:
    st.subheader(f"Smart Room: {smart_room}")
    st.markdown(f"**Space Capacity:** {smart_stats[0]}")
    st.markdown(f"**Avg People:** {smart_stats[1]:.2f}")
    st.markdown(f"**Occupancy Rate:** {smart_stats[2]*100:.1f}%")
    st.markdown(f"**Utilisation Rate:** {smart_stats[3]*100:.1f}%")
    st.markdown(f"**Avg CO₂:** {smart_stats[4]:.1f} ppm")

with col2:
    st.subheader(f"Meeting Room: {meeting_room}")
    st.markdown(f"**Space Capacity:** {meeting_stats[0]}")
    st.markdown(f"**Avg People:** {meeting_stats[1]:.2f}")
    st.markdown(f"**Occupancy Rate:** {meeting_stats[2]*100:.1f}%")
    st.markdown(f"**Utilisation Rate:** {meeting_stats[3]*100:.1f}%")
    st.markdown(f"**Avg CO₂:** {meeting_stats[4]:.1f} ppm")

# Filter Data for the Selected Rooms
smart_df = occupancy_filtered_readings[
    (occupancy_filtered_readings['Space Name'] == smart_room)
]

meeting_df = occupancy_filtered_readings[
    (occupancy_filtered_readings['Space Name'] == meeting_room)
]


# rename buildng df columns
building_df.rename(columns={'Local Datetime': 'DateTime',
                            'Building Count': "Peak People Count"}, inplace=True)


# Plotly Plot: 3 Y-Axes (People, CO2, Building Occupancy)
fig = go.Figure()

# Smart Room: People (line)
fig.add_trace(go.Scatter(
    x=smart_df['DateTime'],
    y=smart_df['Peak People Count'],
    name=f'{smart_room} - People',
    mode='lines',
    yaxis='y1'
))

# Smart Room: CO2 (line)
fig.add_trace(go.Scatter(
    x=smart_df['DateTime'],
    y=smart_df['co2'],
    name=f'{smart_room} - CO₂',
    mode='lines',
    yaxis='y2'
))

# Meeting Room: People (line)
fig.add_trace(go.Scatter(
    x=meeting_df['DateTime'],
    y=meeting_df['Peak People Count'],
    name=f'{meeting_room} - People',
    mode='lines',
    yaxis='y1',
    # line=dict(dash='lines')
))

# Meeting Room: CO2 (line)
fig.add_trace(go.Scatter(
    x=meeting_df['DateTime'],
    y=meeting_df['co2'],
    name=f'{meeting_room} - CO₂',
    mode='lines',
    yaxis='y2',
    # line=dict(dash='lines')
))

# Building Occupancy (line, 3rd y-axis)
fig.add_trace(go.Scatter(
    x=building_df['DateTime'],
    y=building_df['Peak People Count'],
    name='Building Occupancy',
    mode='lines',
    yaxis='y1',
    line=dict(color='black', width=2)
))

# Update layout for multiple y-axes
fig.update_layout(
    title='Smart Room vs Meeting Room Comparison',
    xaxis=dict(title='DateTime'),
    yaxis=dict(
        title='Number of People',
        side='left'
    ),
    yaxis2=dict(
        title='CO₂ (ppm)',
        overlaying='y',
        side='right'
    ),
    yaxis3=dict(
        title='Building Occupancy',
        anchor='free',
        overlaying='y',
        side='right',
        position=1
    ),
    legend=dict(x=0, y=1.1, orientation='h'),
    margin=dict(t=60, l=60, r=60, b=60)
)

st.plotly_chart(fig, use_container_width=True)
