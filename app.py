import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from xaspy.xas.polarized import Lz, Sz
from xaspy.xas.backgrounds import step


st.set_page_config(page_title="XASpy", page_icon=":sparkles:", layout="wide")
st.title("XASpy: Monte Carlo Simulation of Dichroism X-ray Absorption Spectroscopy Sum Rule Analysis")

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

if uploaded_file is None:
    st.warning("Please upload a data file to proceed.")
    st.stop()

# read in all the input parameters in multiple columns
st.subheader("Parameters for Background Subtraction")
col1, col2, col3, col4 = st.columns(4)
with col1:
    initial_step1 = st.number_input("Initial Step 1", value=708.0, step=0.1)
with col2:
    initial_step2 = st.number_input("Initial Step 2", value=721.0, step=0.1)
with col3:
    initial_slope = st.number_input("Initial Slope", value=1.0)
with col4:
    initial_branching = st.number_input("Initial Branching", value=0.33)


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
    initial_branching_dist_function = st.selectbox("Initial Branching Distribution Function", ("normal", "uniform", "randint", None), index=3)

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
    tz_dist_value = st.number_input("Tz Dist Value", value=0.0, step=0.1)
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


sampling_size = st.number_input("Sampling Size", min_value=1, value=1000, step=1000, format="%d")


def setup_monte_carlo_parameters(monte_parameters, sampling_size):
    """
    Setup the parameters for the Monte Carlo Simulation.
    """
    # setup the oarameters
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

    # Clear progress bar and status text after completion
    progress_bar.empty()

    status_text.text(f"Monte Carlo Simulation parameters setup complet with {len(whole_set)} samples.")

    return whole_set


whole_set = setup_monte_carlo_parameters(monte_parameters, sampling_size)


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

        # Add traces
        fig_xas.add_trace(go.Scatter(x=x, y=initial_corrected_xas_cs, name='XAS'))

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


def plot_parameter_distributions(monte_parameters):
    # Calculate grid dimensions
    n_params = len(monte_parameters)
    for_grid = int(np.ceil(np.sqrt(n_params)))

    # Create Plotly subplot grid
    fig = make_subplots(
        rows=for_grid,
        cols=for_grid,
        horizontal_spacing=0.1,
        vertical_spacing=0.1
    )

    # Track valid parameters and their positions
    valid_params = []
    index = 0

    for param_name in monte_parameters:
        try:
            # Generate distribution data
            dist_values = [define_dist(*monte_parameters[param_name]) for _ in range(500)]

            # Calculate grid position
            row = (index % for_grid) + 1  # Plotly uses 1-based indexing
            col = (index // for_grid) + 1

            # Add histogram to subplot
            fig.add_trace(
                go.Histogram(
                    x=dist_values,
                    marker_color='slategrey',
                    showlegend=False
                ),
                row=row,
                col=col
            )

            # Add title annotation
            fig.add_annotation(
                xref=f"x{(row-1)*for_grid + col}",
                yref=f"y{(row-1)*for_grid + col}",
                text=param_name,
                showarrow=False,
                font=dict(size=10),
                xanchor="center",
                yanchor="bottom",
                y=1.1  # Position above subplot
            )

            valid_params.append(param_name)
            index += 1

        except Exception as e:
            st.error(f"Could not plot {param_name}: {str(e)}")
            continue

    # Hide empty subplot axes
    total_plots = for_grid ** 2
    for i in range(len(valid_params), total_plots):
        row = (i % for_grid) + 1
        col = (i // for_grid) + 1
        fig.update_xaxes(showticklabels=False, showgrid=False, row=row, col=col)
        fig.update_yaxes(showticklabels=False, showgrid=False, row=row, col=col)

    # Set layout properties
    fig.update_layout(
        height=800,
        title_text="Parameter Distributions",
        title_x=0.5,
        margin=dict(t=100),
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)


st.subheader("Parameter Distributions going into the Monte Carlo Simulation")
# Usage in your Streamlit app:
plot_parameter_distributions(monte_parameters)

st.write("The following parameters are used for the Monte Carlo Simulation. The values are used to generate a random distribution of the parameters.")
# First row
col1, col2 = st.columns(2)
with col1:
    # g-factor
    g = st.number_input("g-factor", value=2, step=1)
with col2:
    # c-factor
    c = st.number_input("c=1 for p -> d, c=2 for d -> f", value=1, step=1)

l_value = c + 1


###################
lz_list = list()
sz_list = list()
mu_tot_list = list()
# ratio of orbital to spin moment,
# independent of XAS and nh if Tz = 0
mu_rat_list = list()

for n in whole_set:
    try:
        # calculate:
        xas_corr = np.array(_step(
            x, y, n[0], n[1],
            slope=float(n[2]),
            br=n[3]
        )[0])
        print(n)
        lz = Lz(z, xas_corr,
                c=c, l=l_value,
                nh=n[4],
                last_number_xas=n[6],
                last_number_xmcd=n[7])

        sz = Sz(z, xas_corr,
                c=c, l=l_value,
                nh=n[4], tz=n[5],
                last_number_xas=n[6],
                last_number_xmcd=n[7],
                edge_div=n[8])

        mu_tot = -(g * sz + lz)
        mu_rat = lz / (g * sz)

        # save in lists:
        lz_list.append(lz)
        sz_list.append(sz)
        mu_tot_list.append(mu_tot)
        mu_rat_list.append(mu_rat)

    except Exception as e:
        st.error(f"Error processing: {str(e)}")
        continue

# Create a DataFrame to display the results
results_df = pd.DataFrame({
    "Parameter": ["Lz", "Sz", "µtot", "µratio (µl/µs)"],
    "Mean": [
        np.around(np.nanmean(lz_list), 5),
        np.around(np.nanmean(sz_list), 5),
        np.around(np.nanmean(mu_tot_list), 5),
        np.around(np.nanmean(mu_rat_list) * 100, 5)
    ],
    "Standard Deviation": [
        np.around(np.nanstd(lz_list), 5),
        np.around(np.nanstd(sz_list), 5),
        np.around(np.nanstd(mu_tot_list), 5),
        np.around(np.nanstd(mu_rat_list) * 100, 5)
    ],
    "Units": ["µB", "µB", "µB", "%"]
})

# Display the results as a Streamlit table
st.subheader("Monte Carlo Simulation Results")
st.table(results_df)
