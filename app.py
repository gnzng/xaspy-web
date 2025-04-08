import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# from xaspy.xas.polarized import Lz, Sz
from xaspy.xas.backgrounds import step

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
                # for further handling we will use shorter variables
                x = data["energy"].values
                y = data["xas"].values
                z = data["xmcd"].values

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


# define a two step function:
def _step(x, xas, step1, step2, slope=None, br=2/3):
    final_step_hight = np.array(xas)[-1]
    stepx = step(final_step_hight*br,
                 step1, x, slope=slope) + step(final_step_hight*(1-br),
                                               step2, x, slope=slope)
    xas00 = xas - stepx
    return xas00, stepx


def create_plot(x, y, z):
    fig = go.Figure()

    # Original XAS/XMCD
    fig.add_trace(go.Scatter(x=x, y=y, name='XAS',
                             line=dict(color='firebrick', width=2, dash='solid')))
    fig.add_trace(go.Scatter(x=x, y=z, name='XMCD',
                             line=dict(color='firebrick', width=2, dash='dash')))

    # Calculate modified curves
    corr_xas, step_func = _step(x, y, initial_step1, initial_step2,
                                slope=initial_slope,
                                br=initial_branching)

    fig.add_trace(go.Scatter(x=x, y=corr_xas, name='Corrected XAS', line=dict(color="slateblue", width=1, dash='dot')))
    fig.add_trace(go.Scatter(x=x, y=step_func, name='Step Function', line=dict(color='slategrey', width=1, dash='dot')))

    # Formatting
    fig.update_layout(
        title='XAS/XMCD Analysis',
        xaxis_title='Energy',
        yaxis_title='Intensity',
        hovermode='x unified',
        showlegend=True,
        template='plotly_white',
        legend=dict(orientation='h', yanchor='bottom', y=1.02)
    )
    return fig


try:
    # Display in Streamlit
    st.plotly_chart(create_plot(x, y, z),
                    use_container_width=True)
except Exception:
    st.write("nothing to plot yet")


st.subheader("Parameters for Monte Carlo Simulation")

col1, col2, col3, col4 = st.columns(4)
with col1:
    initial_step1_dist = st.number_input("Initial Step 1 Distribution", value=1.0)
    initial_step1_dist_function = st.selectbox("Initial Step 1 Distribution Function", ("normal", "uniform", "randint", None))

with col2:
    initial_step2_dist = st.number_input("Initial Step 2 Distribution", value=1.0)
    initial_step2_dist_function = st.selectbox("Initial Step 2 Distribution Function", ("normal", "uniform", "randint", None))
with col3:
    initial_slope_dist = st.number_input("Initial Slope Distribution", value=1.0)
    initial_slope_dist_function = st.selectbox("Initial Slope Distribution Function", ("normal", "uniform", "randint", None))
with col4:
    initial_branching_dist = st.number_input("Initial Branching Distribution", value=1.0)
    initial_branching_dist_function = st.selectbox("Initial Branching Distribution Function", ("normal", "uniform", "randint", None))

monte_parameters = dict({
     'step1_dist': (initial_step1, initial_step1_dist, initial_step1_dist_function),
     'step2_dist': (initial_step2, initial_step2_dist, initial_step2_dist_function),
     'slope_dist': (initial_slope, initial_slope_dist, initial_slope_dist_function),
     'branching_dist': (initial_branching, initial_branching_dist, initial_branching_dist_function)
                        })


# define different distributions:
def define_dist(a, b, dist='normal'):
    if b is None:
        return a
    if dist == 'normal':
        return np.random.normal(a, b)
    elif dist == 'randint':
        return np.random.randint(a, b)
    elif dist == 'uniform':
        return np.random.uniform(a, b)
    else:
        raise SyntaxError('distribution not in list, choose normal, uniform or randint distribution')


# TODO I need another input field here

# add further parameters for sum rule analysis:
monte_parameters['nh_dist'] = (9, None, 'normal')
monte_parameters['tz_dist'] = (-.4, 0.1, 'uniform')
monte_parameters['last_number_xas_dist'] = (1, 300, 'randint')
monte_parameters['last_number_xmcd_dist'] = (1, 300, 'randint')
monte_parameters['edge_divider_dist'] = (-30, 300, 'randint')

col1, col2 = st.columns([1, 2])
with col1:
    sampling_size = st.number_input("Sampling Size", min_value=1, value=10000, step=1000, format="%d")
with col2:
    button_clicked = st.button("Setup Monte Carlo Simulation Parameters")
    if button_clicked:
        # run the simulation
        # TODO add a progress bar
        st.write("Settingn up Monte Carlo Parameters...")
        # TODO add a button to stop the simulation

        whole_set = list()
        for i in range(sampling_size):
            param_set = [
                define_dist(*monte_parameters['step1_dist']),
                define_dist(*monte_parameters['step2_dist']),
                define_dist(*monte_parameters['slope_dist']),
                define_dist(*monte_parameters['branching_dist']),
                define_dist(*monte_parameters['nh_dist']),
                define_dist(*monte_parameters['tz_dist']),
                define_dist(*monte_parameters['last_number_xas_dist']),
                define_dist(*monte_parameters['last_number_xmcd_dist']),
                define_dist(*monte_parameters['edge_divider_dist']),
            ]
            whole_set.append(param_set)

        st.write(f"Monte Carlo Simulation parameters setup complet with {len(whole_set)} samples.")


# TODO continue here
