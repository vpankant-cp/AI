# frontend.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import json
from datetime import datetime, timedelta
import altair as alt

# Configuration
API_URL = "http://localhost:8000/api"

# Page configuration
st.set_page_config(
    page_title="AI Talent Rediscovery",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E40AF;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #3B82F6;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .card {
        background-color: #F3F4F6;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        border-left: 5px solid #3B82F6;
    }
    .skill-tag {
        display: inline-block;
        background-color: #DBEAFE;
        color: #1E40AF;
        padding: 0.2rem 0.5rem;
        border-radius: 5px;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
        font-size: 0.85rem;
    }
    .match-score {
        font-size: 1.2rem;
        font-weight: bold;
        color: #059669;
        margin-bottom: 0.5rem;
    }
    .highlight {
        background-color: #FEFCE8;
        padding: 0.2rem;
        border-radius: 3px;
    }
    .status-pill {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: bold;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #F9FAFB;
        border-radius: 5px 5px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #DBEAFE !important;
        border-bottom: 4px solid #1E40AF !important;
    }
</style>
""", unsafe_allow_html=True)


def make_api_request(endpoint, method="GET", params=None, data=None):
    """Make API request to backend"""
    url = f"{API_URL}/{endpoint}"

    try:
        if method == "GET":
            response = requests.get(url, params=params)
        elif method == "POST":
            response = requests.post(url, json=data)
        else:
            st.error(f"Unsupported HTTP method: {method}")
            return None

        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Error connecting to API: {str(e)}")
        return None


def format_date(date_str):
    """Format date string to a more readable format"""
    try:
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        return date_obj.strftime("%b %d, %Y")
    except:
        return date_str


def render_candidate_list(candidates):
    """Render a list of candidates with their details"""
    if not candidates:
        st.info("No candidates found matching your criteria.")
        return

    for candidate in candidates:
        with st.container():
            col1, col2 = st.columns([3, 1])

            with col1:
                # Name and position
                st.markdown(f"### {candidate['name']}")
                st.markdown(f"**{candidate['lastPosition']}**")

                # Match score if available
                if 'matchScore' in candidate and candidate['matchScore']:
                    st.markdown(f"<div class='match-score'>{candidate['matchScore']}% Match</div>",
                                unsafe_allow_html=True)

                # Location and contact
                st.markdown(f"📍 {candidate['location']} | 📧 {candidate['email']} | 📱 {candidate['phone']}")

                # Skills
                skill_html = ""
                for skill in candidate['topSkills'][:5]:  # Show top 5 skills
                    skill_html += f"<span class='skill-tag'>{skill}</span>"
                st.markdown(f"<div>{skill_html}</div>", unsafe_allow_html=True)

                # Last application date
                st.markdown(f"Last application: {format_date(candidate['lastApplicationDate'])}")

            with col2:
                st.button("View Details", key=f"view_{candidate['id']}",
                          on_click=lambda cid=candidate['id']: st.session_state.update({
                              'view': 'detail',
                              'candidate_id': cid
                          }))

            st.markdown("---")


def render_candidate_detail(candidate_id):
    """Render detailed view of a candidate"""
    candidate = make_api_request(f"candidates/{candidate_id}")

    if not candidate:
        st.error("Failed to load candidate details.")
        return

    # Header with back button
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"<h1 class='main-header'>{candidate['name']}</h1>", unsafe_allow_html=True)
    with col2:
        st.button("Back to List", on_click=lambda: st.session_state.update({'view': 'list'}))

    # Contact information
    st.markdown(f"<div class='card'>", unsafe_allow_html=True)
    contact_col1, contact_col2, contact_col3 = st.columns(3)
    with contact_col1:
        st.markdown(f"**Email:** {candidate['email']}")
    with contact_col2:
        st.markdown(f"**Phone:** {candidate['phone']}")
    with contact_col3:
        st.markdown(f"**Location:** {candidate['location']}")
    st.markdown("</div>", unsafe_allow_html=True)

    # Tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(["Experience & Skills", "Previous Applications", "Education", "Resume"])

    with tab1:
        # Skills visualization
        st.markdown("<div class='section-header'>Skills</div>", unsafe_allow_html=True)

        skill_data = []
        for skill in candidate['skills']:
            skill_data.append({
                'Skill': skill['name'],
                'Level': skill['level']
            })

        if skill_data:
            skill_df = pd.DataFrame(skill_data)

            # Create horizontal bar chart
