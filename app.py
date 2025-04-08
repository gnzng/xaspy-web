import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# from xaspy.xas.polarized import Lz, Sz
from xaspy.xas.backgrounds import step
import time

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


st.subheader("Parameters for step function and background subtraction")
st.write("The following parameters are used for the step function and background subtraction. The values are used to generate a random distribution of the parameters.")

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
    if b is None or dist is None:
        return a
    if dist == 'normal':
        return np.random.normal(a, b)
    elif dist == 'randint':
        return np.random.randint(a, b)
    elif dist == 'uniform':
        return np.random.uniform(a, b)
    else:
        raise SyntaxError('distribution not in list, choose normal, uniform or randint distribution')


st.subheader("Parameters for Sum Rule Analysis")
st.write("The following parameters are used for the sum rule analysis. The values are used to generate a random distribution of the parameters.")


# Add further parameters for sum rule analysis as input fields:
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    nh_dist_value = st.number_input("Nh Dist Value", value=9.0, step=1.0)
    nh_dist_function = st.selectbox("Nh Dist Function", ("normal", "uniform", None))

with col2:
    tz_dist_value = st.number_input("Tz Dist Value", value=-0.4)
    tz_dist_function = st.selectbox("Tz Dist Function", ("normal", "uniform", None))
    tz_dist_variance = st.number_input("Tz Dist Variance", value=0.1)

with col3:
    last_number_xas_value = st.number_input("Last Number XAS Value", value=1)
    last_number_xas_range = st.number_input("Last Number XAS Range", value=10)

with col4:
    last_number_xmcd_value = st.number_input("Last Number XMCD Value", value=1)
    last_number_xmcd_range = st.number_input("Last Number XMCD Range", value=3)

with col5:
    edge_divider_value = st.number_input("Edge Divider Value", value=-20)
    edge_divider_range = st.number_input("Edge Divider Range", value=5)

# Update monte_parameters dictionary with user inputs
monte_parameters['nh_dist'] = (nh_dist_value, None, nh_dist_function)
monte_parameters['tz_dist'] = (tz_dist_value, tz_dist_variance, tz_dist_function)
monte_parameters['last_number_xas_dist'] = (last_number_xas_value, last_number_xas_range, "randint")
monte_parameters['last_number_xmcd_dist'] = (last_number_xmcd_value, last_number_xmcd_range, "randint")
monte_parameters['edge_divider_dist'] = (edge_divider_value, edge_divider_range, "randint")


sampling_size = st.number_input("Sampling Size", min_value=1, value=10000, step=1000, format="%d")

# run the simulation
# TODO add a progress bar
st.write("Setting up Monte Carlo Parameters...")
# TODO add a button to stop the simulation
whole_set = list()
progress_bar = st.progress(0)
status_text = st.empty()

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
    # Update progress bar and status text
    progress_bar.progress((i + 1) / sampling_size)
    status_text.text(f"Processing sample {i + 1} of {sampling_size}")
    time.sleep(0.001)  # Simulate processing time (optional)

# Clear progress bar and status text after completion
progress_bar.empty()

status_text.text(f"Monte Carlo Simulation parameters setup complet with {len(whole_set)} samples.")


# TODO continue here


def plot_cum_sums(
    x, y, z, whole_set, monte_parameters,
    initial_step1, initial_step2,
    initial_slope, initial_branching,
    number_of_points=50
):
    # Create two columns
    col1, col2 = st.columns(2)

    # --- Left Column (XAS) ---
    with col1:
        fig_xas = go.Figure()

        # Calculate cumulative sums
        initial_corrected_xas_cs = np.cumsum(_step(x, y, initial_step1, initial_step2, slope=initial_slope)[0])
        cs1 = np.cumsum(
            _step(
                x, y,
                monte_parameters['step1_dist'][0] - monte_parameters['step1_dist'][1],
                monte_parameters['step2_dist'][0] - monte_parameters['step2_dist'][1],
                slope=initial_slope, br=initial_branching
            )[0]
        )

        cs2 = np.cumsum(
            _step(
                x, y,
                monte_parameters['step1_dist'][0] + monte_parameters['step1_dist'][1],
                monte_parameters['step2_dist'][0] + monte_parameters['step2_dist'][1],
                slope=initial_slope, br=initial_branching
            )[0]
        )

        # Add traces
        fig_xas.add_trace(go.Scatter(x=x, y=initial_corrected_xas_cs, name='XAS Baseline'))
        fig_xas.add_trace(go.Scatter(x=x, y=cs1, name='XAS Min Variant', line=dict(dash='dot')))
        fig_xas.add_trace(go.Scatter(x=x, y=cs2, name='XAS Max Variant', line=dict(dash='dot')))

        # Add vertical lines
        for n in np.array(whole_set)[:number_of_points, 6]:
            n = int(n)
            fig_xas.add_vline(x=x[len(z)-n], line=dict(color='gray', width=0.5))

        # Format XAS plot
        fig_xas.update_layout(
            title='XAS Cumulative Analysis',
            xaxis_title='Energy [eV]',
            yaxis_title='Cumulative Signal [arb. units]',
            showlegend=True,
            height=500,
            margin=dict(t=40, b=40)
        )

        st.plotly_chart(fig_xas, use_container_width=True)

    # --- Right Column (XMCD) ---
    with col2:
        xmcd_cumulative = np.cumsum(z)
        fig_xmcd = go.Figure()

        # Main XMCD trace
        fig_xmcd.add_trace(go.Scatter(x=x, y=xmcd_cumulative, name='XMCD'))

        # Add scatter markers
        for n in np.array(whole_set)[:number_of_points, 7]:
            n = int(n)
            fig_xmcd.add_trace(go.Scatter(
                x=[x[len(z)-n]],
                y=[xmcd_cumulative[-n]],
                mode='markers',
                marker=dict(symbol='x', color='red'),
                showlegend=False
            ))

        for n in np.array(whole_set)[:number_of_points, 8]:
            n = int(n)
            idx = int(len(z)/2) + n
            fig_xmcd.add_trace(go.Scatter(
                x=[x[idx]],
                y=[xmcd_cumulative[idx]],
                mode='markers',
                marker=dict(symbol='x', color='blue'),
                showlegend=False
            ))

        # Format XMCD plot
        fig_xmcd.update_layout(
            title='XMCD Cumulative Analysis',
            xaxis_title='Energy [eV]',
            yaxis_title='Cumulative Signal [arb. units]',
            showlegend=True,
            height=500,
            margin=dict(t=40, b=40)
        )

        st.plotly_chart(fig_xmcd, use_container_width=True)


plot_cum_sums(
    x, y, z, whole_set, monte_parameters,
    initial_step1, initial_step2,
    initial_slope, initial_branching
)
