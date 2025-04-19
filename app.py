import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from loguru import logger
import json
import io
import h5py
import base64
from datetime import datetime
import plotly.io as pio

from xaspy.xas.polarized import Lz_cumsum, Sz_cumsum
from xaspy.xas.backgrounds import step
from xaspy.utils.utils import cumtrapz

# ---------- CONSTANTS ----------
COLORS = {
    "blue": "#0074D9",
    "orange": "#FF851B",
    "grey": "#6D6D6D",
    "purple": "#B10DC9",
    "cyan": "#01FF70",
    "red": "#FF4136",
}


# ---------- UTILITY FUNCTIONS ----------
def _step(x, xas, step1, step2, slope=None, br=2 / 3):
    """Define a two step function for background subtraction."""
    final_step_hight = np.array(xas)[-1]
    stepx = step(final_step_hight * br, step1, x, slope=slope) + step(
        final_step_hight * (1 - br), step2, x, slope=slope
    )
    xas00 = xas - stepx
    return xas00, stepx


def define_dist(a, b, dist="normal"):
    """Define different distributions for Monte Carlo simulation."""
    if b is None or dist is None:
        return a
    if dist == "normal":
        return np.random.normal(a, b)
    elif dist == "randint":
        return np.random.randint(a, b + 1)
    elif dist == "uniform":
        return np.random.uniform(a, b)
    else:
        raise SyntaxError(
            "distribution not in list, choose normal, uniform or randint distribution"
        )


# ---------- FILE HANDLING FUNCTIONS ----------
def save_to_h5(file_path_or_buffer, data_dict, curves_dict, results_dict):
    """Save all data to an H5 file."""
    with h5py.File(file_path_or_buffer, "w") as f:
        # Create parameters group
        params_group = f.create_group("parameters")
        for key, value in data_dict.items():
            # Convert value to appropriate type for H5 storage
            if isinstance(value, (list, tuple)):
                # Convert lists and tuples to numpy arrays
                try:
                    # Try to convert to a numeric array
                    arr = np.array(value, dtype=np.float64)
                except (ValueError, TypeError):
                    # If it contains strings or mixed types, store as strings
                    arr = np.array([str(item) for item in value], dtype="S")
                params_group.create_dataset(key, data=arr)
            elif isinstance(value, (int, float, bool)):
                # Store numeric values as attributes
                params_group.attrs[key] = value
            elif value is None:
                # Store None as an empty array with a special attribute
                dset = params_group.create_dataset(key, data=np.array([]))
                dset.attrs["is_none"] = True
            else:
                # Convert anything else to string
                params_group.attrs[key] = str(value)

        # Create curves group
        curves_group = f.create_group("curves")
        for key, value in curves_dict.items():
            if value is not None:
                curves_group.create_dataset(key, data=value)

        # Create results group
        results_group = f.create_group("results")
        for key, value in results_dict.items():
            if isinstance(value, np.ndarray):
                results_group.create_dataset(key, data=value)
            elif isinstance(value, list):
                try:
                    results_group.create_dataset(key, data=np.array(value))
                except (ValueError, TypeError):
                    # If conversion fails, store as strings
                    results_group.create_dataset(
                        key, data=np.array([str(item) for item in value], dtype="S")
                    )
            elif isinstance(value, (int, float, bool)):
                results_group.attrs[key] = value
            elif value is None:
                # Store None as an empty array with a special attribute
                dset = results_group.create_dataset(key, data=np.array([]))
                dset.attrs["is_none"] = True
            else:
                # Convert anything else to string
                results_group.attrs[key] = str(value)


def get_h5_download_button(data_dict, curves_dict, results_dict):
    """Generate a temporary H5 file and provide a Streamlit download button."""
    # Create file in memory
    with io.BytesIO() as buffer:
        save_to_h5(buffer, data_dict, curves_dict, results_dict)
        buffer.seek(0)
        b64 = base64.b64encode(buffer.read()).decode()

    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"xaspy_monte_carlo_{timestamp}.h5"

    # Create download button
    st.download_button(
        label="Download all results as H5 File",
        data=base64.b64decode(b64),
        file_name=filename,
        mime="application/octet-stream",
    )
    return ""


def get_html_download_button(fig_list, title="XASpy Monte Carlo Results"):
    """Generate a standalone HTML file with all plots and provide a Streamlit download button."""
    # Create HTML content
    html_content = f"<html><head><title>{title}</title></head><body><h1>{title}</h1>"

    # Add each figure
    for i, fig in enumerate(fig_list):
        html_content += (
            f"<div id='fig{i}'>{pio.to_html(fig, include_plotlyjs='cdn')}</div>"
        )

    html_content += "</body></html>"

    # Encode as base64
    b64 = base64.b64encode(html_content.encode()).decode()

    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"xaspy_monte_carlo_{timestamp}.html"

    # Create download button
    st.download_button(
        label="Download figures as HTML",
        data=base64.b64decode(b64),
        file_name=filename,
        mime="text/html",
    )
    return "only the plots will be saved in the html, not the parameters"


# ---------- DATA LOADING FUNCTIONS ----------
def load_data(uploaded_file):
    """Load and validate data from uploaded file."""
    try:
        # Read CSV with proper validation
        data = pd.read_csv(uploaded_file, header=None, names=["energy", "xas", "xmcd"])

        # Check if data has exactly 3 columns
        if data.shape[1] != 3:
            st.error("CSV must contain exactly 3 columns: 'energy', 'xas', 'xmcd'")
            st.stop()
        # Check if all columns are numeric
        if not all(pd.api.types.is_numeric_dtype(data[col]) for col in data.columns):
            st.error("All columns in the CSV must be numeric.")
            st.stop()

        # Validate columns before proceeding
        if all(col in data.columns for col in ["energy", "xas", "xmcd"]):
            st.success("Data successfully loaded:")
            return data
        else:
            st.error("CSV missing required columns. Needs: 'energy', 'xas', 'xmcd'")
            st.stop()

    except Exception as e:
        st.error(f"Error reading text file: {str(e)}")
        st.stop()


def load_parameters_from_json(uploaded_file):
    """Load parameters from uploaded JSON file."""
    try:
        parameters = json.load(uploaded_file)
        st.success("Parameters successfully loaded from JSON file")
        return parameters
    except Exception as e:
        st.error(f"Error loading parameters from JSON: {str(e)}")
        return None


# ---------- UI COMPONENTS ----------
def display_data_preview(data, flip_xmcd=False):
    """Display data preview and basic information."""
    col1, col2 = st.columns(2)

    with col1:
        st.write("Data Preview:")
        st.write(data.head())
        st.write("data points:")
        st.write(len(data), "rows")

        # For further handling we will use shorter variables
        x = data["energy"].values
        y = data["xas"].values
        z = data["xmcd"].values

        # Add a toggle switch to flip XMCD
        flip_xmcd = st.checkbox("Flip XMCD (Multiply by -1)", value=flip_xmcd)
        if flip_xmcd:
            z = -1 * z

        show_energy_trace = st.checkbox(
            "Show Energy Trace (good for checking steps in energy)",
            value=False,
        )

        return x, y, z, show_energy_trace, col2, flip_xmcd


def get_step_function_parameters(saved_params=None):
    """Get parameters for step function background subtraction."""
    st.subheader("Parameters for Step Function as Background Subtraction")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        initial_step1 = st.number_input(
            "Initial Step 1",
            value=(
                float(saved_params.get("initial_step1", 708.0))
                if saved_params
                else 708.0
            ),
            step=0.1,
        )
    with col2:
        initial_step2 = st.number_input(
            "Initial Step 2",
            value=(
                float(saved_params.get("initial_step2", 721.0))
                if saved_params
                else 721.0
            ),
            step=0.1,
        )
    with col3:
        initial_slope = st.number_input(
            "Initial Slope",
            value=(
                float(saved_params.get("initial_slope", 1.0)) if saved_params else 1.0
            ),
        )
    with col4:
        initial_branching = st.number_input(
            "Initial Branching",
            value=(
                float(saved_params.get("initial_branching", 0.33))
                if saved_params
                else 0.33
            ),
        )

    logger.info(
        f"Initial Step 1: {initial_step1}, Initial Step 2: {initial_step2}, "
        f"Initial Slope: {initial_slope}, Initial Branching: {initial_branching}"
    )

    return initial_step1, initial_step2, initial_slope, initial_branching


def get_distribution_parameters(
    initial_step1, initial_step2, initial_slope, initial_branching, saved_params=None
):
    """Get parameters for distribution in Monte Carlo simulation."""
    st.subheader("Parameters for step function and background subtraction")
    st.write(
        "The following parameters are used for the step function and background subtraction. "
        "The values are used to generate a random distribution of the parameters."
    )

    use_seed = st.checkbox(
        "Use `np.random.seed(21)` for reproducibility",
        value=saved_params.get("use_seed", False) if saved_params else False,
    )
    if use_seed:
        seed = 21
        np.random.seed(seed)
        logger.info(f"Using seed: {seed}")
    else:
        seed = None
        logger.info("No seed used for random number generation")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        initial_step1_dist = st.number_input(
            "Initial Step 1 Distribution",
            value=(
                float(saved_params.get("initial_step1_dist", 1.0))
                if saved_params
                else 1.0
            ),
        )
        initial_step1_dist_function = st.selectbox(
            "Initial Step 1 Distribution Function",
            options=["normal", "uniform", "randint", None],
            index=["normal", "uniform", "randint", None].index(
                saved_params.get("initial_step1_dist_function", "normal")
                if saved_params
                else "normal"
            ),
        )

    with col2:
        initial_step2_dist = st.number_input(
            "Initial Step 2 Distribution",
            value=(
                float(saved_params.get("initial_step2_dist", 1.0))
                if saved_params
                else 1.0
            ),
        )
        initial_step2_dist_function = st.selectbox(
            "Initial Step 2 Distribution Function",
            options=["normal", "uniform", "randint", None],
            index=["normal", "uniform", "randint", None].index(
                saved_params.get("initial_step2_dist_function", "normal")
                if saved_params
                else "normal"
            ),
        )

    with col3:
        initial_slope_dist = st.number_input(
            "Initial Slope Distribution",
            value=(
                float(saved_params.get("initial_slope_dist", 1.0))
                if saved_params
                else 1.0
            ),
        )
        initial_slope_dist_function = st.selectbox(
            "Initial Slope Distribution Function",
            options=["normal", "uniform", "randint", None],
            index=["normal", "uniform", "randint", None].index(
                saved_params.get("initial_slope_dist_function", "normal")
                if saved_params
                else "normal"
            ),
        )

    with col4:
        initial_branching_dist = st.number_input(
            "Initial Branching Distribution",
            value=(
                float(saved_params.get("initial_branching_dist", 0.1))
                if saved_params
                else 0.1
            ),
        )
        initial_branching_dist_function = st.selectbox(
            "Initial Branching Distribution Function",
            options=["normal", "uniform", "randint", None],
            index=["normal", "uniform", "randint", None].index(
                saved_params.get("initial_branching_dist_function", None)
                if saved_params
                else None
            ),
        )

    dist_params = {
        "step1_dist": (initial_step1, initial_step1_dist, initial_step1_dist_function),
        "step2_dist": (initial_step2, initial_step2_dist, initial_step2_dist_function),
        "slope_dist": (initial_slope, initial_slope_dist, initial_slope_dist_function),
        "branching_dist": (
            initial_branching,
            initial_branching_dist,
            initial_branching_dist_function,
        ),
        # Store individual values for export
        "initial_step1_dist": initial_step1_dist,
        "initial_step1_dist_function": initial_step1_dist_function,
        "initial_step2_dist": initial_step2_dist,
        "initial_step2_dist_function": initial_step2_dist_function,
        "initial_slope_dist": initial_slope_dist,
        "initial_slope_dist_function": initial_slope_dist_function,
        "initial_branching_dist": initial_branching_dist,
        "initial_branching_dist_function": initial_branching_dist_function,
    }

    # Store the seed setting
    dist_params["use_seed"] = use_seed

    return dist_params


def get_sum_rule_parameters(saved_params=None):
    """Get parameters for sum rule analysis."""
    st.subheader("Parameters for Sum Rule Analysis")
    st.write(
        "The following parameters are used for the sum rule analysis. "
        "The values are used to generate a random distribution of the parameters."
    )

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        nh_dist_value = st.number_input(
            "Nh Dist Value",
            value=(
                float(saved_params.get("nh_dist_value", 4.0)) if saved_params else 4.0
            ),
            step=1.0,
        )
        nh_dist_function = st.selectbox(
            "Nh Dist Function",
            options=["normal", "uniform", None],
            index=["normal", "uniform", None].index(
                saved_params.get("nh_dist_function", "normal")
                if saved_params
                else "normal"
            ),
        )
        nh_dist_variance = st.number_input(
            "Nh Dist Variance",
            value=(
                float(saved_params.get("nh_dist_variance", 0.1))
                if saved_params
                else 0.1
            ),
            step=0.01,
        )

    with col2:
        tz_dist_value = st.number_input(
            "Tz Dist Value",
            value=(
                float(saved_params.get("tz_dist_value", 0.0)) if saved_params else 0.0
            ),
            step=0.1,
        )
        tz_dist_function = st.selectbox(
            "Tz Dist Function",
            options=["normal", "uniform", None],
            index=["normal", "uniform", None].index(
                saved_params.get("tz_dist_function", "normal")
                if saved_params
                else "normal"
            ),
        )
        tz_dist_variance = st.number_input(
            "Tz Dist Variance",
            value=(
                float(saved_params.get("tz_dist_variance", 0.1))
                if saved_params
                else 0.1
            ),
            step=0.1,
        )

    with col3:
        last_number_xas_value = st.number_input(
            "Last Number XAS Value",
            value=(
                int(saved_params.get("last_number_xas_value", 1)) if saved_params else 1
            ),
        )
        last_number_xas_range = st.number_input(
            "Last Number XAS Range",
            value=(
                int(saved_params.get("last_number_xas_range", 10))
                if saved_params
                else 10
            ),
        )

    with col4:
        last_number_xmcd_value = st.number_input(
            "Last Number XMCD Value",
            value=(
                int(saved_params.get("last_number_xmcd_value", 1))
                if saved_params
                else 1
            ),
        )
        last_number_xmcd_range = st.number_input(
            "Last Number XMCD Range",
            value=(
                int(saved_params.get("last_number_xmcd_range", 3))
                if saved_params
                else 3
            ),
        )

    with col5:
        edge_divider_value = st.number_input(
            "Edge Divider Value",
            value=(
                int(saved_params.get("edge_divider_value", -20))
                if saved_params
                else -20
            ),
        )
        edge_divider_range = st.number_input(
            "Edge Divider Range",
            value=int(saved_params.get("edge_divider_range", 5)) if saved_params else 5,
        )

    sum_rule_params = {
        "nh_dist": (nh_dist_value, nh_dist_variance, nh_dist_function),
        "tz_dist": (tz_dist_value, tz_dist_variance, tz_dist_function),
        "last_number_xas_dist": (
            last_number_xas_value,
            last_number_xas_range,
            "randint",
        ),
        "last_number_xmcd_dist": (
            last_number_xmcd_value,
            last_number_xmcd_range,
            "randint",
        ),
        "edge_divider_dist": (edge_divider_value, edge_divider_range, "randint"),
        # Store individual values for JSON export
        "nh_dist_value": nh_dist_value,
        "nh_dist_function": nh_dist_function,
        "nh_dist_variance": nh_dist_variance,
        "tz_dist_value": tz_dist_value,
        "tz_dist_function": tz_dist_function,
        "tz_dist_variance": tz_dist_variance,
        "last_number_xas_value": last_number_xas_value,
        "last_number_xas_range": last_number_xas_range,
        "last_number_xmcd_value": last_number_xmcd_value,
        "last_number_xmcd_range": last_number_xmcd_range,
        "edge_divider_value": edge_divider_value,
        "edge_divider_range": edge_divider_range,
    }

    return sum_rule_params


def get_sampling_size(saved_params=None):
    """Get sampling size for Monte Carlo simulation."""
    colsampling_size, _, _, _, _ = st.columns(5)

    with colsampling_size:
        st.write(
            "Sampling size input for Monte Carlo. For adjusting ~100, for full simulation ~10000."
        )
        sampling_size = st.number_input(
            "Sampling Size",
            min_value=1,
            max_value=20000,
            value=int(saved_params.get("sampling_size", 100)) if saved_params else 100,
            step=1000,
            format="%d",
            key="sampling_size",
            help="Number of samples to generate for the Monte Carlo simulation. "
            "Higher values will take longer to compute. Recommended highest value is 10000.",
        )

    return sampling_size


def get_additional_parameters(saved_params=None):
    """Get additional parameters for calculations."""
    st.write(
        "The following parameters are used for the Monte Carlo Simulation. "
        "The values are used to generate a random distribution of the parameters."
    )
    col1, col2 = st.columns(2)

    with col1:
        # g-factor
        g = st.number_input(
            "g-factor",
            value=int(saved_params.get("g_factor", 2)) if saved_params else 2,
            step=1,
        )

    with col2:
        # c-factor
        c = st.number_input(
            "c=1 for p -> d, c=2 for d -> f",
            value=int(saved_params.get("c_factor", 1)) if saved_params else 1,
            step=1,
        )

    l_value = c + 1

    return g, c, l_value


def display_parameters_summary(all_params):
    """Display a summary of all parameters and provide download options."""
    st.subheader("All Input Parameters")

    # Convert tuple parameters to strings for display
    display_params = {}
    for key, value in all_params.items():
        if isinstance(value, tuple):
            display_params[key] = str(value)
        else:
            display_params[key] = value

    # Display parameters in a dataframe
    df_params = pd.DataFrame(
        {"Parameter": display_params.keys(), "Value": display_params.values()}
    )

    st.dataframe(df_params)

    st.write(
        "Download the parameters as JSON file, which can be uploaded for future analysis."
    )
    # JSON download button
    json_str = json.dumps(all_params, indent=2, default=str)
    st.download_button(
        label="Download Parameters as JSON",
        data=json_str,
        file_name="xaspy_parameters.json",
        mime="application/json",
    )


# ---------- PLOTTING FUNCTIONS ----------
def plot_data(x, y, z, show_energy_trace, column):
    """Plot the uploaded data."""
    with column:
        st.write("Plot:")
        # Create a Plotly figure for XAS and XMCD
        fig = go.Figure()

        # Add XAS trace
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name="XAS",
                line=dict(color=COLORS["grey"], width=2),
            )
        )

        # Add XMCD trace
        fig.add_trace(
            go.Scatter(
                x=x,
                y=z,
                mode="lines",
                name="XMCD",
                line=dict(color=COLORS["cyan"], width=2),
            )
        )

        # Add Energy trace (deactivated by default)
        if show_energy_trace:
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=np.arange(len(x)) / len(x),
                    mode="markers",
                    name="energy",
                    marker=dict(symbol="x", color=COLORS["purple"], size=8),
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

        return fig


def create_step_function_plot(
    x, y, z, initial_step1, initial_step2, initial_slope, initial_branching
):
    """Create plot showing the step function and corrected XAS."""
    fig = go.Figure()

    # Original XAS/XMCD
    fig.add_trace(
        go.Scatter(
            x=x, y=y, name="XAS", line=dict(color=COLORS["grey"], width=2, dash="solid")
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=z,
            name="XMCD",
            line=dict(color=COLORS["cyan"], width=2, dash="solid"),
        )
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
            line=dict(color=COLORS["grey"], width=1, dash="dot"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=step_func,
            name="Step Function",
            line=dict(color=COLORS["red"], width=1, dash="dot"),
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

    return fig, corr_xas, step_func


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
    last_number_xas_value,
    last_number_xas_range,
    last_number_xmcd_value,
    last_number_xmcd_range,
    edge_divider_value,
    edge_divider_range,
):
    """Plot cumulative sums for XAS and XMCD."""
    # Create two columns
    col1, col2 = st.columns(2)

    # --- Left Column (XAS) ---
    with col1:
        fig_xas = go.Figure()

        # Calculate cumulative sums
        initial_corrected_xas, _ = _step(
            x,
            y,
            initial_step1,
            initial_step2,
            slope=initial_slope,
            br=initial_branching,
        )
        initial_corrected_xas_cs = cumtrapz(initial_corrected_xas, x)

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
                marker=dict(symbol="triangle-left", color=COLORS["purple"]),
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
                marker=dict(symbol="triangle-right", color=COLORS["purple"]),
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
                line=dict(color=COLORS["grey"], width=2, dash="solid"),
            )
        )

        xmcd_cum_marker_color_ln = COLORS["orange"]
        n = int(last_number_xmcd_value)
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

        xmcd_cum_marker_color = COLORS["blue"]
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

    return fig_xas, fig_xmcd, initial_corrected_xas_cs, xmcd_cumulative


def plot_parameter_distributions(monte_parameters, whole_set):
    """Plot distributions of Monte Carlo parameters."""
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
            # Skip parameters that are not distributions
            if param_name in ["use_seed"] or not isinstance(
                monte_parameters[param_name], tuple
            ):
                continue

            # Generate distribution data
            dist_values = [
                define_dist(*monte_parameters[param_name])
                for _ in range(len(whole_set))
            ]

            # Calculate grid position
            row = (index % for_grid) + 1
            col = (index // for_grid) + 1

            # Check if this is a single value parameter
            is_single_value = len(set(dist_values)) == 1

            if is_single_value:
                # For single value, create a more visible representation
                single_value = dist_values[0]

                # Use a bar chart instead of histogram for single values
                fig.add_trace(
                    go.Bar(
                        x=[single_value],
                        y=[len(dist_values)],  # Count all occurrences
                        marker_color="rgba(255, 165, 0, 0.7)",  # Orange with transparency
                        width=0.1,  # Narrow bar
                        showlegend=False,
                    ),
                    row=row,
                    col=col,
                )

                # Set appropriate axis range for single value
                margin = abs(single_value) * 0.1 if single_value != 0 else 0.5
                fig.update_xaxes(
                    range=[single_value - margin, single_value + margin],
                    row=row,
                    col=col,
                )

                # Add a marker to indicate this is a fixed value
                fig.add_annotation(
                    x=single_value,
                    y=len(dist_values) / 2,
                    text="Fixed Value",
                    showarrow=False,
                    font=dict(size=12),
                    xref=f"x{index+1}",
                    yref=f"y{index+1}",
                )

            else:
                # Regular histogram for distributions
                min_val = min(dist_values)
                max_val = max(dist_values)
                data_range = max_val - min_val

                # If the data spans small values, ensure we have at least 10 bins
                bin_size = max(data_range / 20, data_range * 0.01)

                fig.add_trace(
                    go.Histogram(
                        x=dist_values,
                        marker_color=COLORS["grey"],
                        showlegend=False,
                        xbins=dict(
                            start=min_val - bin_size,
                            end=max_val + bin_size,
                            size=bin_size,
                        ),
                    ),
                    row=row,
                    col=col,
                )

                # Set x-axis range with small margin
                margin = data_range * 0.05
                fig.update_xaxes(
                    range=[min_val - margin, max_val + margin], row=row, col=col
                )

            # Add title annotation
            fig.add_annotation(
                xref=f"x{index+1}",
                yref=f"y{index+1}",
                text=param_name,
                showarrow=False,
                font=dict(size=20, color="orange"),
                xanchor="center",
                yanchor="bottom",
                y=1.1,
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
    return fig


def display_results(lz_list, sz_list, mu_tot_list, mu_rat_list, monte_carlo_params):
    """Display Monte Carlo simulation results."""
    st.header("Monte Carlo Simulation Results")

    # Create results DataFrame
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

    # Create columns to display results
    col1, col2, col3 = st.columns(3)

    # Display the results table
    with col1:

        def color_row(row):
            colors = [
                f"background-color: {COLORS['blue']}",
                f"background-color: {COLORS['orange']}",
                f"background-color: {COLORS['cyan']}",
                f"background-color: {COLORS['purple']}",
            ]
            return [colors[row.name % len(colors)]] * len(row)

        styled_results_df = results_df.style.apply(color_row, axis=1)
        st.dataframe(styled_results_df)

    # Display histograms of the results
    with col2:
        fig = make_subplots(
            rows=2, cols=2, subplot_titles=["Lz", "Sz", "µtot", "µratio (µl/µs)"]
        )

        # Add histograms for each parameter
        fig.add_trace(
            go.Histogram(x=lz_list, name="Lz", marker_color=COLORS["blue"]),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Histogram(x=sz_list, name="Sz", marker_color=COLORS["orange"]),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Histogram(x=mu_tot_list, name="µtot", marker_color=COLORS["cyan"]),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Histogram(x=mu_rat_list, name="µratio", marker_color=COLORS["purple"]),
            row=2,
            col=2,
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
    with col3:
        # Create dataframe with parameters
        df = pd.DataFrame(
            monte_carlo_params,
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

        # Create dataframe with results
        df2 = pd.DataFrame(
            {"lz": lz_list, "sz": sz_list, "mu_tot": mu_tot_list, "mu_rat": mu_rat_list}
        )

        df3 = pd.concat([df2, df], axis=1)
        corr = df3.corr().round(2)
        corr = corr.dropna(axis=0, how="all")
        corr = corr.dropna(axis=1, how="all")

        st.subheader("Correlation Matrix")
        st.dataframe(
            corr.style.format("{:.2f}").background_gradient(cmap="coolwarm"),
            use_container_width=True,
        )

    # Prepare detailed results for download
    results_dict = {
        "lz_list": np.array(lz_list),
        "sz_list": np.array(sz_list),
        "mu_tot_list": np.array(mu_tot_list),
        "mu_rat_list": np.array(mu_rat_list),
        "lz_mean": float(np.nanmean(lz_list)),
        "sz_mean": float(np.nanmean(sz_list)),
        "mu_tot_mean": float(np.nanmean(mu_tot_list)),
        "mu_rat_mean": float(np.nanmean(mu_rat_list)),
        "lz_std": float(np.nanstd(lz_list)),
        "sz_std": float(np.nanstd(sz_list)),
        "mu_tot_std": float(np.nanstd(mu_tot_list)),
        "mu_rat_std": float(np.nanstd(mu_rat_list)),
    }

    return results_df, results_dict, fig


def provide_download_options(
    results_df, all_params, results_dict, curves_dict, all_figures
):
    """Provide download options for results."""
    st.subheader("Download Options")

    # Results CSV download
    col1, col2, col3 = st.columns(3)

    with col1:
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="Download magnetic results as CSV",
            data=csv,
            file_name="monte_carlo_results.csv",
            mime="text/csv",
        )

    # H5 file download
    with col2:
        get_h5_download_button(all_params, curves_dict, results_dict),

    # HTML download (optional)
    with col3:
        get_html_download_button(all_figures)


# ---------- MONTE CARLO SIMULATION FUNCTIONS ----------
def setup_monte_carlo_parameters(monte_parameters, sampling_size):
    """Setup parameters for Monte Carlo simulation."""
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
        f"Monte Carlo Simulation parameters setup complete with {len(whole_set)} samples."
    )

    return whole_set


def run_monte_carlo_simulation(x, y, z, whole_set, g, c, l_value):
    """Run Monte Carlo simulation and return results."""
    lz_list = list()
    sz_list = list()
    mu_tot_list = list()
    mu_rat_list = list()

    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, n in enumerate(whole_set):
        try:
            # Calculate
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

            # Save in lists
            lz_list.append(lz)
            sz_list.append(sz)
            mu_tot_list.append(mu_tot)
            mu_rat_list.append(mu_rat)

            # Update progress
            progress_bar.progress((i + 1) / len(whole_set))
            status_text.text(f"Running simulation {i + 1} of {len(whole_set)}")

        except Exception as e:
            st.error(f"Error in simulation run {i+1}: {str(e)}")
            logger.error(f"Error in simulation run {i+1}: {str(e)}")
            continue

    # Clear progress indicators
    progress_bar.empty()
    status_text.text(
        f"Monte Carlo simulation completed with {len(lz_list)} valid results."
    )

    return lz_list, sz_list, mu_tot_list, mu_rat_list


def collect_all_parameters(
    initial_step1,
    initial_step2,
    initial_slope,
    initial_branching,
    monte_parameters,
    g,
    c,
    sampling_size,
    flip_xmcd,
):
    """Collect all input parameters into a single dictionary."""
    all_params = {
        "initial_step1": initial_step1,
        "initial_step2": initial_step2,
        "initial_slope": initial_slope,
        "initial_branching": initial_branching,
        "g_factor": g,
        "c_factor": c,
        "l_value": c + 1,
        "sampling_size": sampling_size,
        "flip_xmcd": flip_xmcd,
    }

    # Add all monte carlo parameters
    for key, value in monte_parameters.items():
        if key not in all_params:
            all_params[key] = value

    return all_params


# ---------- MAIN APPLICATION ----------
def main():
    """Main application function."""
    st.set_page_config(
        page_title="XASpy - Monte Carlo", page_icon=":sparkles:", layout="wide"
    )

    st.title(
        "XASpy: Monte Carlo Simulation of Dichroism X-ray Absorption Spectroscopy Sum Rule Analysis"
    )
    st.write(
        "Upload your data file. The data should be formatted as follows: energy, xas, xmcd, without header. "
        "The first column should be energy, the second column should be xas and the third column should be xmcd. "
        "The data should be in CSV format. "
        "Use at your own risk. Changing any field will rerun the simulation."
    )

    upload_col1, upload_col2 = st.columns(2)
    with upload_col1:
        # Load input data
        st.subheader("Spectrum Input Data")
        uploaded_file = st.file_uploader(
            "Upload your data file", type=["csv", "txt", "dat"]
        )
    with upload_col2:
        # Parameter upload option
        st.subheader("Upload Parameters (Optional)")
        param_file = st.file_uploader(
            "Upload a parameter JSON file to restore settings", type=["json"]
        )
        saved_params = None
        if param_file is not None:
            saved_params = load_parameters_from_json(param_file)

    if uploaded_file is None:
        st.warning("Please upload a data file to proceed.")
        st.stop()

    # Load and process data
    try:
        data = load_data(uploaded_file)
        x, y, z, show_energy_trace, plot_column, flip_xmcd = display_data_preview(
            data,
            flip_xmcd=saved_params.get("flip_xmcd", False) if saved_params else False,
        )
        data_plot = plot_data(x, y, z, show_energy_trace, plot_column)
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        logger.error(f"Error loading data: {str(e)}")
        st.stop()
    # Get step function parameters
    initial_step1, initial_step2, initial_slope, initial_branching = (
        get_step_function_parameters(saved_params)
    )

    # Display step function plot
    try:
        step_function_plot, corr_xas, step_func = create_step_function_plot(
            x, y, z, initial_step1, initial_step2, initial_slope, initial_branching
        )
        st.plotly_chart(step_function_plot, use_container_width=True)
    except Exception as e:
        st.write("Nothing to plot yet")
        logger.warning(f"No data to plot: {str(e)}")

    # Get distribution parameters
    monte_parameters = get_distribution_parameters(
        initial_step1, initial_step2, initial_slope, initial_branching, saved_params
    )

    # Get sum rule parameters and update monte_parameters
    sum_rule_params = get_sum_rule_parameters(saved_params)
    monte_parameters.update(sum_rule_params)

    # Log all parameters
    logger.info("All parameters:")
    for key, value in monte_parameters.items():
        logger.info(f"  {key}: {value}")

    # Get sampling size
    sampling_size = get_sampling_size(saved_params)

    # Setup Monte Carlo parameters
    whole_set = setup_monte_carlo_parameters(monte_parameters, sampling_size)

    # Plot cumulative sums
    cum_xas_plot, cum_xmcd_plot, initial_corrected_xas_cs, xmcd_cumulative = (
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
            monte_parameters["last_number_xas_dist"][0],
            monte_parameters["last_number_xas_dist"][1],
            monte_parameters["last_number_xmcd_dist"][0],
            monte_parameters["last_number_xmcd_dist"][1],
            monte_parameters["edge_divider_dist"][0],
            monte_parameters["edge_divider_dist"][1],
        )
    )

    # Plot parameter distributions
    st.subheader("Parameter Distributions going into the Monte Carlo Simulation")
    param_dist_plot = plot_parameter_distributions(monte_parameters, whole_set)

    # Get additional parameters
    g, c, l_value = get_additional_parameters(saved_params)

    # Collect all parameters
    all_params = collect_all_parameters(
        initial_step1,
        initial_step2,
        initial_slope,
        initial_branching,
        monte_parameters,
        g,
        c,
        sampling_size,
        flip_xmcd,
    )

    # Run Monte Carlo simulation
    lz_list, sz_list, mu_tot_list, mu_rat_list = run_monte_carlo_simulation(
        x, y, z, whole_set, g, c, l_value
    )

    # Display results
    results_df, results_dict, results_plot = display_results(
        lz_list, sz_list, mu_tot_list, mu_rat_list, whole_set
    )

    # Collect curves for H5 storage
    curves_dict = {
        "energy": x,
        "xas": y,
        "xmcd": z,
        "corrected_xas": corr_xas,
        "step_function": step_func,
        "xas_cumulative": initial_corrected_xas_cs,
        "xmcd_cumulative": xmcd_cumulative,
    }

    # Collect all figures for HTML export
    all_figures = [
        data_plot,
        step_function_plot,
        cum_xas_plot,
        cum_xmcd_plot,
        param_dist_plot,
        results_plot,
    ]

    # Display parameter summary and provide download options
    display_parameters_summary(all_params)

    # Provide download options
    provide_download_options(
        results_df, all_params, results_dict, curves_dict, all_figures
    )


if __name__ == "__main__":
    main()
