import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from loguru import logger

from xaspy.xas.polarized import Lz_cumsum, Sz_cumsum
from xaspy.xas.backgrounds import step
from xaspy.utils.utils import cumtrapz


# colors:
blue = "#0074D9"
orange = "#FF851B"
grey = "#6D6D6D"
purple = "#B10DC9"
cyan = "#01FF70"
red = "#FF4136"


st.set_page_config(
    page_title="XASpy - Monte Carlo", page_icon=":sparkles:", layout="wide"
)
st.title(
    "XASpy: Monte Carlo Simulation of Dichroism X-ray Absorption Spectroscopy Sum Rule Analysis"
)

# read and display the input data
st.subheader("Input Data")
st.write(
    "Upload your data file. The data should be formated the following: energy, xas, xmcd, without header."
)

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
                # Create a Plotly figure for XAS and XMCD
                fig = go.Figure()

                # Add XAS trace
                fig.add_trace(
                    go.Scatter(
                        x=data["energy"],
                        y=data["xas"],
                        mode="lines",
                        name="XAS",
                        line=dict(color=grey, width=2),
                    )
                )

                # Add XMCD trace
                fig.add_trace(
                    go.Scatter(
                        x=data["energy"],
                        y=data["xmcd"],
                        mode="lines",
                        name="XMCD",
                        line=dict(color=cyan, width=2),
                    )
                )

                # Update layout
                fig.update_layout(
                    title="XAS and XMCD Plot",
                    xaxis_title="Energy (eV)",
                    yaxis_title="Intensity",
                    hovermode="x unified",
                    template="plotly_white",
                    legend=dict(
                        orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
                    ),
                )

                # Display the Plotly chart
                st.plotly_chart(fig, use_container_width=True)
                # for further handling we will use shorter variables
                x = data["energy"].values
                y = data["xas"].values
                z = data["xmcd"].values

        else:
            st.error("CSV missing required columns. Needs: 'energy', 'xas', 'xmcd'")

    except Exception as e:
        st.error(f"Error reading text file: {str(e)}")
        st.stop()

if uploaded_file is None:
    st.warning("Please upload a data file to proceed.")
    st.stop()

# read in all the input parameters in multiple columns
st.subheader("Parameters for Step Function as Background Subtraction")
col1, col2, col3, col4 = st.columns(4)
with col1:
    initial_step1 = st.number_input("Initial Step 1", value=708.0, step=0.1)
with col2:
    initial_step2 = st.number_input("Initial Step 2", value=721.0, step=0.1)
with col3:
    initial_slope = st.number_input("Initial Slope", value=1.0)
with col4:
    initial_branching = st.number_input("Initial Branching", value=0.33)
logger.info(
    f"Initial Step 1: {initial_step1}, Initial Step 2: {initial_step2}, Initial Slope: {initial_slope}, Initial Branching: {initial_branching}"
)


# define a two step function:
def _step(x, xas, step1, step2, slope=None, br=2 / 3):
    final_step_hight = np.array(xas)[-1]
    stepx = step(final_step_hight * br, step1, x, slope=slope) + step(
        final_step_hight * (1 - br), step2, x, slope=slope
    )
    xas00 = xas - stepx
    return xas00, stepx


# define different distributions:
def define_dist(a, b, dist="normal"):
    if b is None or dist is None:
        return a
    if dist == "normal":
        return np.random.normal(a, b)
    elif dist == "randint":
        return np.random.randint(a, b)
    elif dist == "uniform":
        return np.random.uniform(a, b)
    else:
        raise SyntaxError(
            "distribution not in list, choose normal, uniform or randint distribution"
        )


def create_plot(x, y, z):
    fig = go.Figure()

    # Original XAS/XMCD
    fig.add_trace(
        go.Scatter(x=x, y=y, name="XAS", line=dict(color=grey, width=2, dash="solid"))
    )
    fig.add_trace(
        go.Scatter(x=x, y=z, name="XMCD", line=dict(color=cyan, width=2, dash="solid"))
    )

    # Calculate modified curves
    corr_xas, step_func = _step(
        x, y, initial_step1, initial_step2, slope=initial_slope, br=initial_branching
    )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=corr_xas,
            name="Corrected XAS",
            line=dict(color=grey, width=1, dash="dot"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=step_func,
            name="Step Function",
            line=dict(color=red, width=1, dash="dot"),
        )
    )

    # Formatting
    fig.update_layout(
        title="XAS/XMCD Analysis",
        xaxis_title="Energy",
        yaxis_title="Intensity",
        hovermode="x unified",
        showlegend=True,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


try:
    # Display in Streamlit
    st.plotly_chart(create_plot(x, y, z), use_container_width=True)
except Exception:
    st.write("nothing to plot yet")
    logger.warning("No data to plot. Please upload a data file and set the parameters.")


st.subheader("Parameters for step function and background subtraction")
st.write(
    "The following parameters are used for the step function and background subtraction. The values are used to generate a random distribution of the parameters."
)

col1, col2, col3, col4 = st.columns(4)
with col1:
    initial_step1_dist = st.number_input("Initial Step 1 Distribution", value=1.0)
    initial_step1_dist_function = st.selectbox(
        "Initial Step 1 Distribution Function", ("normal", "uniform", "randint", None)
    )

with col2:
    initial_step2_dist = st.number_input("Initial Step 2 Distribution", value=1.0)
    initial_step2_dist_function = st.selectbox(
        "Initial Step 2 Distribution Function", ("normal", "uniform", "randint", None)
    )
with col3:
    initial_slope_dist = st.number_input("Initial Slope Distribution", value=1.0)
    initial_slope_dist_function = st.selectbox(
        "Initial Slope Distribution Function", ("normal", "uniform", "randint", None)
    )
with col4:
    initial_branching_dist = st.number_input(
        "Initial Branching Distribution", value=1.0
    )
    initial_branching_dist_function = st.selectbox(
        "Initial Branching Distribution Function",
        ("normal", "uniform", "randint", None),
        index=3,
    )

monte_parameters = dict(
    {
        "step1_dist": (initial_step1, initial_step1_dist, initial_step1_dist_function),
        "step2_dist": (initial_step2, initial_step2_dist, initial_step2_dist_function),
        "slope_dist": (initial_slope, initial_slope_dist, initial_slope_dist_function),
        "branching_dist": (
            initial_branching,
            initial_branching_dist,
            initial_branching_dist_function,
        ),
    }
)


st.subheader("Parameters for Sum Rule Analysis")
st.write(
    "The following parameters are used for the sum rule analysis. The values are used to generate a random distribution of the parameters."
)


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
monte_parameters["nh_dist"] = (nh_dist_value, None, nh_dist_function)
monte_parameters["tz_dist"] = (tz_dist_value, tz_dist_variance, tz_dist_function)
monte_parameters["last_number_xas_dist"] = (
    last_number_xas_value,
    last_number_xas_range,
    "randint",
)
monte_parameters["last_number_xmcd_dist"] = (
    last_number_xmcd_value,
    last_number_xmcd_range,
    "randint",
)
monte_parameters["edge_divider_dist"] = (
    edge_divider_value,
    edge_divider_range,
    "randint",
)
logger.info("All parameters:")
for key, value in monte_parameters.items():
    logger.info(f"  {key}: {value}")


sampling_size = st.number_input(
    "Sampling Size", min_value=1, value=1000, step=1000, format="%d"
)


def setup_monte_carlo_parameters(monte_parameters, sampling_size):
    """
    Setup the parameters for the Monte Carlo Simulation.
    """
    # setup the oarameters
    st.write("Setting up Monte Carlo Parameters...")
    whole_set = list()
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i in range(sampling_size):
        param_set = [
            define_dist(*monte_parameters["step1_dist"]),
            define_dist(*monte_parameters["step2_dist"]),
            define_dist(*monte_parameters["slope_dist"]),
            define_dist(*monte_parameters["branching_dist"]),
            define_dist(*monte_parameters["nh_dist"]),
            define_dist(*monte_parameters["tz_dist"]),
            define_dist(*monte_parameters["last_number_xas_dist"]),
            define_dist(*monte_parameters["last_number_xmcd_dist"]),
            define_dist(*monte_parameters["edge_divider_dist"]),
        ]
        whole_set.append(param_set)
        # Update progress bar and status text
        progress_bar.progress((i + 1) / sampling_size)
        status_text.text(f"Processing sample {i + 1} of {sampling_size}")

    # Clear progress bar and status text after completion
    progress_bar.empty()

    status_text.text(
        f"Monte Carlo Simulation parameters setup complet with {len(whole_set)} samples."
    )

    return whole_set


whole_set = setup_monte_carlo_parameters(monte_parameters, sampling_size)


def plot_cum_sums(
    x,
    y,
    z,
    whole_set,
    monte_parameters,
    initial_step1,
    initial_step2,
    initial_slope,
    initial_branching,
    number_of_points=50,
):
    # Create two columns
    col1, col2 = st.columns(2)

    # --- Left Column (XAS) ---
    with col1:
        fig_xas = go.Figure()

        # Calculate cumulative sums
        initial_corrected_xas_cs = cumtrapz(
            _step(x, y, initial_step1, initial_step2, slope=initial_slope)[0], x
        )

        # Add traces
        fig_xas.add_trace(
            go.Scatter(
                x=x, y=initial_corrected_xas_cs, name="XAS", line=dict(color="grey")
            )
        )

        n = int(last_number_xas_value)
        fig_xas.add_trace(
            go.Scatter(
                x=[x[len(z) - n]],
                y=[initial_corrected_xas_cs[-n]],
                mode="markers",
                marker=dict(symbol="triangle-left", color=purple),
                showlegend=True,
                name="last number value",
            )
        )
        n = int(last_number_xas_range)
        fig_xas.add_trace(
            go.Scatter(
                x=[x[len(z) - n]],
                y=[initial_corrected_xas_cs[-n]],
                mode="markers",
                marker=dict(symbol="triangle-right", color=purple),
                showlegend=True,
                name="last number range",
            )
        )

        # Format XAS plot
        fig_xas.update_layout(
            title="XAS Cumulative Analysis",
            xaxis_title="Energy [eV]",
            yaxis_title="Cumulative Signal [arb. units]",
            showlegend=True,
            height=500,
            margin=dict(t=40, b=40),
        )

        st.plotly_chart(fig_xas, use_container_width=True)

    # --- Right Column (XMCD) ---
    with col2:
        xmcd_cumulative = cumtrapz(z, x)
        fig_xmcd = go.Figure()

        # Main XMCD trace
        fig_xmcd.add_trace(
            go.Scatter(
                x=x,
                y=xmcd_cumulative,
                name="XMCD",
                line=dict(color=grey, width=2, dash="solid"),
            )
        )

        xmcd_cum_marker_color_ln = orange
        n = last_number_xmcd_value
        n = int(n)
        fig_xmcd.add_trace(
            go.Scatter(
                x=[x[len(z) - n]],
                y=[xmcd_cumulative[-n]],
                mode="markers",
                marker=dict(symbol="triangle-left", color=xmcd_cum_marker_color_ln),
                showlegend=True,
                name="last number value",
            )
        )
        n = int(last_number_xmcd_range)
        fig_xmcd.add_trace(
            go.Scatter(
                x=[x[len(z) - n]],
                y=[xmcd_cumulative[-n]],
                mode="markers",
                marker=dict(symbol="triangle-right", color=xmcd_cum_marker_color_ln),
                showlegend=True,
                name="last number range",
            )
        )

        xmcd_cum_marker_color = blue
        n = int(edge_divider_value)
        idx = int(len(z) / 2) + n
        fig_xmcd.add_trace(
            go.Scatter(
                x=[x[idx]],
                y=[xmcd_cumulative[idx]],
                mode="markers",
                marker=dict(symbol="triangle-right", color=xmcd_cum_marker_color),
                showlegend=True,
                name="edge div. value",
            )
        )
        n = int(edge_divider_range)
        idx = int(len(z) / 2) + n
        fig_xmcd.add_trace(
            go.Scatter(
                x=[x[idx]],
                y=[xmcd_cumulative[idx]],
                mode="markers",
                marker=dict(symbol="triangle-left", color=xmcd_cum_marker_color),
                showlegend=True,
                name="edge div. range",
            )
        )
        fig_xmcd.update_layout(
            title="XMCD Cumulative Analysis",
            xaxis_title="Energy [eV]",
            yaxis_title="Cumulative Signal [arb. units]",
            showlegend=True,
            height=500,
            margin=dict(t=40, b=40),
        )

        st.plotly_chart(fig_xmcd, use_container_width=True)


plot_cum_sums(
    x,
    y,
    z,
    whole_set,
    monte_parameters,
    initial_step1,
    initial_step2,
    initial_slope,
    initial_branching,
)


def plot_parameter_distributions(monte_parameters):
    # Calculate grid dimensions
    n_params = len(monte_parameters)
    for_grid = int(np.ceil(np.sqrt(n_params)))

    # Create Plotly subplot grid
    fig = make_subplots(
        rows=for_grid, cols=for_grid, horizontal_spacing=0.1, vertical_spacing=0.1
    )

    # Track valid parameters and their positions
    valid_params = []
    index = 0

    for param_name in monte_parameters:
        try:
            # Generate distribution data
            dist_values = [
                define_dist(*monte_parameters[param_name])
                for _ in range(len(whole_set))
            ]

            # Calculate grid position
            row = (index % for_grid) + 1  # Plotly uses 1-based indexing
            col = (index // for_grid) + 1

            # Add histogram to subplot
            # TODO use auto limits
            fig.add_trace(
                go.Histogram(x=dist_values, marker_color=grey, showlegend=False),
                row=row,
                col=col,
            )

            # Add title annotation
            fig.add_annotation(
                xref=f"x{(row-1)*for_grid + col}",
                yref=f"y{(row-1)*for_grid + col}",
                text=param_name,
                showarrow=False,
                font=dict(size=30, color="orange"),
                xanchor="center",
                yanchor="bottom",
                y=1.1,  # Position above subplot
            )

            valid_params.append(param_name)
            index += 1

        except Exception as e:
            st.error(f"Could not plot {param_name}: {str(e)}")
            continue

    # Hide empty subplot axes
    total_plots = for_grid**2
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
        template="plotly_white",
    )

    st.plotly_chart(fig, use_container_width=True)


st.subheader("Parameter Distributions going into the Monte Carlo Simulation")
# Usage in your Streamlit app:
plot_parameter_distributions(monte_parameters)

st.write(
    "The following parameters are used for the Monte Carlo Simulation. The values are used to generate a random distribution of the parameters."
)
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
        xas_corr_cs = cumtrapz(
            _step(x, y, n[0], n[1], slope=float(n[2]), br=n[3])[0], x
        )
        lz = Lz_cumsum(
            z,
            xas_corr_cs,
            c=c,
            l=l_value,
            nh=n[4],
            last_number_xas=n[6],
            last_number_xmcd=n[7],
        )

        sz = Sz_cumsum(
            z,
            xas_corr_cs,
            c=c,
            l=l_value,
            nh=n[4],
            tz=n[5],
            last_number_xas=n[6],
            last_number_xmcd=n[7],
            edge_div=n[8],
        )

        mu_tot = -(g * sz + lz)
        mu_rat = lz / (g * sz)

        # save in lists:
        lz_list.append(lz)
        sz_list.append(sz)
        mu_tot_list.append(mu_tot)
        mu_rat_list.append(mu_rat)

    except Exception as e:
        st.error(f"Error processing: {str(e)}")
        logger.error(f"Error processing: {str(e)}")
        continue

# Create a DataFrame to display the results
results_df = pd.DataFrame(
    {
        "Parameter": ["Lz", "Sz", "µtot", "µratio (µl/µs)"],
        "Mean": [
            np.around(np.nanmean(lz_list), 5),
            np.around(np.nanmean(sz_list), 5),
            np.around(np.nanmean(mu_tot_list), 5),
            np.around(np.nanmean(mu_rat_list) * 100, 5),
        ],
        "Standard Deviation": [
            np.around(np.nanstd(lz_list), 5),
            np.around(np.nanstd(sz_list), 5),
            np.around(np.nanstd(mu_tot_list), 5),
            np.around(np.nanstd(mu_rat_list) * 100, 5),
        ],
        "Units": ["µB", "µB", "µB", "%"],
    }
)

st.header("Monte Carlo Simulation Results")
# Create two columns to display results
col1, col2, col3 = st.columns(3)

# Display the results in the first column
with col1:

    def color_row(row):
        colors = [
            f"background-color: {blue}",
            f"background-color: {orange}",
            f"background-color: {cyan}",
            f"background-color: {purple}",
        ]
        return [colors[row.name % len(colors)]] * len(row)

    styled_results_df = results_df.style.apply(color_row, axis=1)
    st.dataframe(styled_results_df)

# Display histograms of the results in the second column
with col2:
    fig = make_subplots(
        rows=2, cols=2, subplot_titles=["Lz", "Sz", "µtot", "µratio (µl/µs)"]
    )

    # Add histograms for each parameter
    fig.add_trace(go.Histogram(x=lz_list, name="Lz", marker_color=blue), row=1, col=1)
    fig.add_trace(go.Histogram(x=sz_list, name="Sz", marker_color=orange), row=1, col=2)
    fig.add_trace(
        go.Histogram(x=mu_tot_list, name="µtot", marker_color=cyan), row=2, col=1
    )
    fig.add_trace(
        go.Histogram(x=mu_rat_list, name="µratio", marker_color=purple), row=2, col=2
    )

    # Update layout
    fig.update_layout(
        title="Monte Carlo Simulation Parameter Distributions",
        showlegend=False,
        height=600,
        template="plotly_white",
    )

    # Display the plot
    st.plotly_chart(fig, use_container_width=True)

# Create a correlation matrix
df = pd.DataFrame(
    whole_set,
    columns=[
        "step1",
        "step2",
        "slope",
        "br",
        "nh",
        "tz",
        "ln_xas",
        "ln_xmcd",
        "edge_divider",
    ],
)
df2 = pd.DataFrame(
    {"lz": lz_list, "sz": sz_list, "mu_tot": mu_tot_list, "mu_rat": mu_rat_list}
)

df3 = pd.concat([df2, df], axis=1)
corr = df3.corr().round(2)
corr = corr.dropna(axis=0, how="all")
corr = corr.dropna(axis=1, how="all")


with col3:
    st.subheader("Correlation Matrix")
    st.dataframe(
        corr.style.format("{:.2f}").background_gradient(cmap="coolwarm"),
        use_container_width=True,
    )


# Add a download button for the results
st.subheader("Download Results")
csv = results_df.to_csv(index=False)
st.download_button(
    label="Download Results as CSV",
    data=csv,
    file_name="monte_carlo_results.csv",
    mime="text/csv",
)
