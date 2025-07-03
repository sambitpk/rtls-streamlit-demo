import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import random
from streamlit_autorefresh import st_autorefresh

# ------------------ CONFIG ------------------
room_size = (8, 8)
num_tags = 10
refresh_interval = 1000  # ms

# Initialize readers and tags
if "readers" not in st.session_state:
    st.session_state.readers = [
        {"id": "RDR-001", "pos": (2, 2), "gain": 1.0},
        {"id": "RDR-002", "pos": (6, 2), "gain": 0.9},
        {"id": "RDR-003", "pos": (4, 6), "gain": 1.1},
        {"id": "RDR-004", "pos": (1, 5), "gain": 0.85},
    ]

if "tags" not in st.session_state:
    st.session_state.tags = [
        {
            "uid": f"TAG-{i+1:03d}",
            "position": [random.uniform(1, 7), random.uniform(1, 7)],
            "velocity": [random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1)]
        }
        for i in range(num_tags)
    ]

# ------------------ FUNCTIONS ------------------
def simulate_tdoa(tag_pos):
    c = 3e8
    readers = st.session_state.readers
    arrival_times = [np.linalg.norm(np.array(tag_pos) - np.array(r["pos"])) / c for r in readers]
    ref_time = arrival_times[0]
    tdoas = [t - ref_time for t in arrival_times]

    def residuals(pos):
        x, y = pos
        ref_dist = np.linalg.norm(np.array([x, y]) - np.array(readers[0]["pos"]))
        return [
            (np.linalg.norm(np.array([x, y]) - np.array(r["pos"])) - ref_dist) / c - tdoas[i]
            for i, r in enumerate(readers[1:], start=1)
        ]

    result = least_squares(residuals, x0=[4, 4])
    return tuple(np.round(result.x, 2))

def move_tags():
    data_rows = []
    true_pos_dict = {}
    est_pos_dict = {}

    for tag in st.session_state.tags:
        tag["position"][0] += tag["velocity"][0]
        tag["position"][1] += tag["velocity"][1]
        tag["position"][0] = np.clip(tag["position"][0], 0.5, room_size[0] - 0.5)
        tag["position"][1] = np.clip(tag["position"][1], 0.5, room_size[1] - 0.5)

        true_pos = tuple(np.round(tag["position"], 2))
        est_pos = simulate_tdoa(tag["position"])
        true_pos_dict[tag["uid"]] = true_pos
        est_pos_dict[tag["uid"]] = est_pos

        for r in st.session_state.readers:
            dx = tag["position"][0] - r["pos"][0]
            dy = tag["position"][1] - r["pos"][1]
            dist = np.sqrt(dx**2 + dy**2) + 0.1
            rssi = (-40 - 10 * np.log10(dist)) * r["gain"]
            data_rows.append({
                "Tag UID": tag["uid"],
                "Reader ID": r["id"],
                "RSSI (dBm)": round(rssi, 2),
                "Distance (m)": round(dist, 2),
                "Estimated Position": f"({est_pos[0]}, {est_pos[1]})"
            })

    return pd.DataFrame(data_rows), true_pos_dict, est_pos_dict

# ------------------ STREAMLIT UI ------------------
st.set_page_config(layout="wide")
st.title("📡 RTLS Simulation Dashboard")

# Auto-refresh
st_autorefresh(interval=refresh_interval, key="refresh")

# Simulate
df, true_pos, est_pos = move_tags()

# Layout
col1, col2 = st.columns([1, 1])

# --- Table ---
with col1:
    st.subheader("📊 Tag-Reader Interaction Table")
    st.dataframe(df, use_container_width=True)

# --- Map Plot ---
with col2:
    st.subheader("📍 RTLS Map (True vs Estimated)")
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, room_size[0])
    ax.set_ylim(0, room_size[1])
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("RTLS Reader + Tag Tracking")
    ax.grid(True)

    for r in st.session_state.readers:
        ax.plot(*r["pos"], 'bs')
        ax.text(*r["pos"], r["id"], color='blue', fontsize=8)

    for uid in true_pos:
        ax.plot(*true_pos[uid], 'ro')  # true
        ax.plot(*est_pos[uid], 'go')   # estimated
        ax.text(*true_pos[uid], uid, fontsize=7, color='black')

    st.pyplot(fig)