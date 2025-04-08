import streamlit as st
import numpy as np
import pandas as pd

from xaspy.xas.polarized import Lz, Sz

st.set_page_config(page_title="XASpy", page_icon=":sparkles:", layout="wide")
st.title("XASpy:Monte Carlo Simulation of Dichroism X-ray Absorption Spectroscopy")

# read and display the input data
st.subheader("Input Data")
st.write("Upload your data file. The data should be formated the following: energy, xas, xmcd, without header.")
uploaded_file = st.file_uploader("Upload your data file")
if uploaded_file is not None:
    try:
        # Read CSV with proper validation
        data = pd.read_csv(uploaded_file, header=None, names=["energy", "xas", "xmcd"])
        # check if data has exactly 3 columns
        if data.shape[1] != 3:
            st.error("CSV must contain exactly 3 columns: 'energy', 'xas', 'xmcd'")
            st.stop()
        # Validate columns before proceeding
        if all(col in data.columns for col in ["energy", "xas", "xmcd"]):
            st.success("Data successfully loaded:")
            col1, col2 = st.columns(2)

            with col1:
                st.write("Data Preview:")
                st.write(data.head())
                st.write("data points:")
                st.write(len(data), "rows")

            with col2:
                st.write("Plot:")
                # Create properly formatted DataFrame for plotting
                chart_data = data.set_index("energy")

                # Plot with axis labels
                st.line_chart(
                    chart_data[["xas", "xmcd"]],
                    use_container_width=True
                )
                st.caption("Energy (eV) →")  # X-axis label

        else:
            st.error("CSV missing required columns. Needs: 'energy', 'xas', 'xmcd'")

    except Exception as e:
        st.error(f"Error reading CSV file: {str(e)}")

# read in all the input parameters in multiple columns
st.subheader("Parameters for Background Subtraction")
col1, col2, col3, col4 = st.columns(4)
with col1:
    initial_step1 = st.number_input("Initial Step 1", value=1085)
with col2:
    initial_step2 = st.number_input("Initial Step 2", value=1110)
with col3:
    initial_slope = st.number_input("Initial Slope", value=1.0)
with col4:
    initial_branching = st.number_input("Initial Branching", value=1.3/5)

st.subheader("Parameters for Monte Carlo Simulation")

col1, col2, col3, col4 = st.columns(4)
with col1:
    initial_step1_dist = st.number_input("Initial Step 1 Distribution", value=1.0)
    initial_step1_dist_function = st.selectbox("Initial Step 1 Distribution Function", ("normal", "uniform", None))

with col2:
    initial_step2_dist = st.number_input("Initial Step 2 Distribution", value=1.0)
    initial_step2_dist_function = st.selectbox("Initial Step 2 Distribution Function", ("normal", "uniform", None))
with col3:
    initial_slope_dist = st.number_input("Initial Slope Distribution", value=1.0)
    initial_slope_dist_function = st.selectbox("Initial Slope Distribution Function", ("normal", "uniform", None))
with col4:
    initial_branching_dist = st.number_input("Initial Branching Distribution", value=1.0)
    initial_branching_dist_function = st.selectbox("Initial Branching Distribution Function", ("normal", "uniform", None))

monte_parameters = dict({
     'step1_dist': (initial_step1, initial_step1_dist, initial_step1_dist_function),
     'step2_dist': (initial_step2, initial_step2_dist, initial_step2_dist_function),
     'slope_dist': (initial_slope, initial_slope_dist, initial_slope_dist_function),
     'branching_dist': (initial_branching, initial_branching_dist, initial_branching_dist_function)
                        })

