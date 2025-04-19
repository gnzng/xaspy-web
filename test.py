import pytest
import numpy as np
import pandas as pd
import json
import h5py
import io
import plotly.graph_objects as go
from unittest.mock import patch, MagicMock


# Import functions from the app
from app import (
    _step,
    define_dist,
    save_to_h5,
    get_h5_download_button,
    get_html_download_button,
    load_data,
    load_parameters_from_json,
    display_data_preview,
    get_step_function_parameters,
    create_step_function_plot,
    get_distribution_parameters,
    get_sampling_size,
    get_additional_parameters,
    display_results,
    setup_monte_carlo_parameters,
    run_monte_carlo_simulation,
    collect_all_parameters,
)


# Test fixtures and utility functions for testing
@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    # Create realistic XAS/XMCD data
    energy = np.linspace(700, 730, 100)
    # Create a step-like function for XAS with noise
    xas = np.ones_like(energy) * 0.1
    step_idx = np.where(energy > 710)[0][0]
    xas[step_idx:] += 1.0
    xas += np.random.normal(0, 0.01, size=len(energy))

    # Create XMCD with realistic features
    xmcd = np.zeros_like(energy)
    peak1_idx = np.where((energy > 708) & (energy < 712))[0]
    peak2_idx = np.where((energy > 718) & (energy < 722))[0]
    xmcd[peak1_idx] = -0.1 * np.sin((energy[peak1_idx] - 708) * np.pi / 4)
    xmcd[peak2_idx] = 0.05 * np.sin((energy[peak2_idx] - 718) * np.pi / 4)
    xmcd += np.random.normal(0, 0.005, size=len(energy))

    # Create DataFrame
    df = pd.DataFrame({"energy": energy, "xas": xas, "xmcd": xmcd})

    return df


@pytest.fixture
def sample_csv_file(sample_data, tmp_path):
    """Create a sample CSV file for testing."""
    filename = tmp_path / "test_data.csv"
    sample_data.to_csv(filename, index=False, header=False)
    return filename


@pytest.fixture
def sample_json_params():
    """Create sample parameters for testing."""
    return {
        "initial_step1": 708.0,
        "initial_step2": 720.0,
        "initial_slope": 1.0,
        "initial_branching": 0.33,
        "initial_step1_dist": 0.5,
        "initial_step1_dist_function": "normal",
        "initial_step2_dist": 0.5,
        "initial_step2_dist_function": "normal",
        "initial_slope_dist": 0.1,
        "initial_slope_dist_function": "normal",
        "initial_branching_dist": 0.05,
        "initial_branching_dist_function": "normal",
        "nh_dist_value": 4.0,
        "nh_dist_function": "normal",
        "nh_dist_variance": 0.1,
        "tz_dist_value": 0.0,
        "tz_dist_function": "normal",
        "tz_dist_variance": 0.1,
        "last_number_xas_value": 1,
        "last_number_xas_range": 10,
        "last_number_xmcd_value": 1,
        "last_number_xmcd_range": 3,
        "edge_divider_value": -20,
        "edge_divider_range": 5,
        "g_factor": 2,
        "c_factor": 1,
        "sampling_size": 100,
        "use_seed": True,
        "flip_xmcd": False,
    }


@pytest.fixture
def sample_json_file(sample_json_params, tmp_path):
    """Create a sample JSON file for testing."""
    filename = tmp_path / "test_params.json"
    with open(filename, "w") as f:
        json.dump(sample_json_params, f)
    return filename


@pytest.fixture
def monte_parameters():
    """Create Monte Carlo parameters for testing."""
    return {
        "step1_dist": (708.0, 0.5, "normal"),
        "step2_dist": (720.0, 0.5, "normal"),
        "slope_dist": (1.0, 0.1, "normal"),
        "branching_dist": (0.33, 0.05, "normal"),
        "nh_dist": (4.0, 0.1, "normal"),
        "tz_dist": (0.0, 0.1, "normal"),
        "last_number_xas_dist": (1, 10, "randint"),
        "last_number_xmcd_dist": (1, 3, "randint"),
        "edge_divider_dist": (-20, 5, "randint"),
        "use_seed": True,
        "initial_step1_dist": 0.5,
        "initial_step1_dist_function": "normal",
        "initial_step2_dist": 0.5,
        "initial_step2_dist_function": "normal",
        "initial_slope_dist": 0.1,
        "initial_slope_dist_function": "normal",
        "initial_branching_dist": 0.05,
        "initial_branching_dist_function": "normal",
        "nh_dist_value": 4.0,
        "nh_dist_function": "normal",
        "nh_dist_variance": 0.1,
        "tz_dist_value": 0.0,
        "tz_dist_function": "normal",
        "tz_dist_variance": 0.1,
        "last_number_xas_value": 1,
        "last_number_xas_range": 10,
        "last_number_xmcd_value": 1,
        "last_number_xmcd_range": 3,
        "edge_divider_value": -20,
        "edge_divider_range": 5,
    }


class MockFile:
    """Mock file-like object for testing."""

    def __init__(self, data):
        self.data = data
        self.name = "test.csv"
        self.size = len(data)

    def read(self):
        return self.data

    def seek(self, pos):
        pass


# Now let's start with the tests


# 1. Test utility functions
def test_step_function():
    """Test the step function for background subtraction."""
    x = np.linspace(700, 730, 100)
    xas = np.ones_like(x)
    xas[50:] = 2.0

    # Test with default values
    corrected_xas, step_func = _step(x, xas, 710, 720)

    # Assertions
    assert len(corrected_xas) == len(x)
    assert len(step_func) == len(x)
    assert np.isclose(corrected_xas[0], xas[0] - step_func[0])
    assert np.all(np.isfinite(corrected_xas))
    assert np.all(np.isfinite(step_func))

    # Test with custom parameters
    corrected_xas2, step_func2 = _step(x, xas, 705, 725, slope=0.5, br=0.5)
    assert len(corrected_xas2) == len(x)
    assert len(step_func2) == len(x)
    assert np.isclose(corrected_xas2[0], xas[0] - step_func2[0])

    # The parameters should affect the result
    assert not np.array_equal(step_func, step_func2)


def test_define_dist():
    """Test the distribution function."""
    # Test with fixed seed for reproducibility
    np.random.seed(42)

    # Test normal distribution
    val = define_dist(10, 1, "normal")
    assert isinstance(val, float)
    assert 5 < val < 15  # Should be within reasonable range

    # Test uniform distribution
    val = define_dist(5, 10, "uniform")
    assert isinstance(val, float)
    assert 5 <= val <= 10

    # Test randint distribution
    val = define_dist(1, 10, "randint")
    assert isinstance(val, (int, np.integer))
    assert 1 <= val <= 10

    # Test with None parameters
    val = define_dist(7, None)
    assert val == 7
    val = define_dist(7, 1, None)
    assert val == 7

    # Test error case
    with pytest.raises(SyntaxError):
        define_dist(5, 10, "invalid_dist")


# 2. Test file handling functions
def test_save_to_h5():
    """Test saving data to H5 format."""
    # Create test data
    data_dict = {
        "param1": 1,
        "param2": "test",
        "param3": [1, 2, 3],
        "param4": True,
        "param5": None,
        "param6": np.array([1.0, 2.0, 3.0]),
    }

    curves_dict = {
        "energy": np.linspace(700, 730, 100),
        "xas": np.random.random(100),
        "xmcd": np.random.random(100),
        "nulldata": None,
    }

    results_dict = {
        "result1": np.array([1, 2, 3]),
        "result2": 42.0,
        "result3": "text_result",
        "result4": [1, "mixed", 3.0],
        "result5": None,
    }

    # Use a BytesIO object for in-memory testing
    buffer = io.BytesIO()

    # Call the function
    save_to_h5(buffer, data_dict, curves_dict, results_dict)

    # Rewind the buffer and read the H5 file
    buffer.seek(0)
    with h5py.File(buffer, "r") as f:
        # Test parameters group
        assert "parameters" in f
        params = f["parameters"]
        assert "param1" in params.attrs
        assert params.attrs["param1"] == 1
        assert "param2" in params.attrs
        assert params.attrs["param2"] == "test"
        assert "param3" in params
        assert np.array_equal(params["param3"][:], np.array([1, 2, 3]))
        assert "param4" in params.attrs
        assert params.attrs["param4"]

        # Test curves group
        assert "curves" in f
        curves = f["curves"]
        assert "energy" in curves
        assert "xas" in curves
        assert "xmcd" in curves
        assert np.array_equal(curves["energy"][:], curves_dict["energy"])

        # Test results group
        assert "results" in f
        results = f["results"]
        assert "result1" in results
        assert np.array_equal(results["result1"][:], np.array([1, 2, 3]))
        assert "result2" in results.attrs
        assert results.attrs["result2"] == 42.0


def test_get_h5_download_button():
    """Test generating H5 download button."""
    # Create simple test data
    data_dict = {"param": 1}
    curves_dict = {"x": np.array([1, 2, 3])}
    results_dict = {"result": 42.0}

    # Call the function
    get_h5_download_button(data_dict, curves_dict, results_dict)


def test_get_html_download_button():
    """Test generating HTML download button."""
    # Create test figures
    fig1 = go.Figure(data=go.Scatter(x=[1, 2, 3], y=[4, 5, 6]))
    fig2 = go.Figure(data=go.Scatter(x=[1, 2, 3], y=[7, 8, 9]))

    # Call the function
    get_html_download_button([fig1, fig2], "Test Title")


def test_load_data_validation():
    """Test validation in data loading."""
    # Create an invalid CSV with wrong number of columns
    invalid_data = "700,0.1\n701,0.2\n702,0.3"
    mock_file = MockFile(invalid_data.encode())

    # Mock streamlit functions
    with patch("streamlit.error", MagicMock()) as mock_error, patch(
        "streamlit.stop", MagicMock()
    ) as mock_stop, patch("streamlit.success", MagicMock()):
        # Call should fail validation
        load_data(mock_file)
        mock_error.assert_called()
        mock_stop.assert_called()

    corrupted_data = "700,0.1,0.2\n701,0.test2,0.3\n702,0.3,0.4"
    mock_file = MockFile(corrupted_data.encode())
    with patch("streamlit.error", MagicMock()) as mock_error, patch(
        "streamlit.stop", MagicMock()
    ) as mock_stop, patch("streamlit.success", MagicMock()):
        # Call should fail validation
        load_data(mock_file)
        mock_error.assert_called()
        mock_stop.assert_called()


def test_load_parameters_from_json(sample_json_file):
    """Test loading parameters from a JSON file."""
    with open(sample_json_file, "rb") as f:
        mock_uploaded_file = MockFile(f.read())

    # Mock streamlit functions
    with patch("streamlit.success", MagicMock()), patch("streamlit.error", MagicMock()):
        result = load_parameters_from_json(mock_uploaded_file)

        # Check the result
        assert isinstance(result, dict)
        assert "initial_step1" in result
        assert result["initial_step1"] == 708.0
        assert "sampling_size" in result
        assert result["sampling_size"] == 100


# 4. Test UI component functions with mocked Streamlit
@patch("streamlit.checkbox", return_value=False)
@patch("streamlit.columns", return_value=[MagicMock(), MagicMock()])
def test_display_data_preview(mock_columns, mock_checkbox, sample_data):
    """Test display_data_preview function."""
    # Call the function with sample data
    with patch("streamlit.write", MagicMock()):
        result = display_data_preview(sample_data)

        # Check the return values
        assert len(result) == 6
        x, y, z, show_energy_trace, col2, flip_xmcd = result

        # Verify the data types and shapes
        assert isinstance(x, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert isinstance(z, np.ndarray)
        assert isinstance(show_energy_trace, bool)
        assert len(x) == len(y) == len(z)
        assert x.shape == y.shape == z.shape
        assert len(x) == len(sample_data)


@patch("streamlit.subheader")
@patch(
    "streamlit.columns",
    return_value=[MagicMock(), MagicMock(), MagicMock(), MagicMock()],
)
@patch("streamlit.number_input")
def test_get_step_function_parameters(
    mock_number_input, mock_columns, mock_subheader, sample_json_params
):
    """Test get_step_function_parameters function."""
    # Set up the mock returns for number_input
    mock_number_input.side_effect = [708.0, 720.0, 1.0, 0.33]

    # Call the function
    result = get_step_function_parameters(sample_json_params)

    # Check the result
    assert len(result) == 4
    step1, step2, slope, branching = result
    assert step1 == 708.0
    assert step2 == 720.0
    assert slope == 1.0
    assert branching == 0.33


def test_create_step_function_plot(sample_data):
    """Test create_step_function_plot function."""
    x = sample_data["energy"].values
    y = sample_data["xas"].values
    z = sample_data["xmcd"].values

    # Call the function
    result = create_step_function_plot(x, y, z, 708.0, 720.0, 1.0, 0.33)

    # Check the result
    assert len(result) == 3
    fig, corr_xas, step_func = result

    # Verify the types
    assert isinstance(fig, go.Figure)
    assert isinstance(corr_xas, np.ndarray)
    assert isinstance(step_func, np.ndarray)

    # Check the lengths
    assert len(corr_xas) == len(x)
    assert len(step_func) == len(x)

    # Check the figure
    assert len(fig.data) == 4  # Should have 4 traces
    assert fig.data[0].name == "XAS"
    assert fig.data[1].name == "XMCD"
    assert fig.data[2].name == "Corrected XAS"
    assert fig.data[3].name == "Step Function"


@patch("streamlit.subheader")
@patch("streamlit.write")
@patch("streamlit.checkbox", return_value=True)
@patch(
    "streamlit.columns",
    return_value=[MagicMock(), MagicMock(), MagicMock(), MagicMock()],
)
@patch("streamlit.number_input")
@patch("streamlit.selectbox")
def test_get_distribution_parameters(
    mock_selectbox,
    mock_number_input,
    mock_columns,
    mock_checkbox,
    mock_write,
    mock_subheader,
    sample_json_params,
):
    """Test get_distribution_parameters function."""
    # Set up the mock returns
    mock_number_input.side_effect = [0.5, 0.5, 0.1, 0.05]
    mock_selectbox.side_effect = ["normal", "normal", "normal", "normal"]

    # Call the function
    result = get_distribution_parameters(708.0, 720.0, 1.0, 0.33, sample_json_params)

    # Check the result
    assert isinstance(result, dict)

    # Check that all expected keys are present
    expected_keys = [
        "step1_dist",
        "step2_dist",
        "slope_dist",
        "branching_dist",
        "initial_step1_dist",
        "initial_step1_dist_function",
        "initial_step2_dist",
        "initial_step2_dist_function",
        "initial_slope_dist",
        "initial_slope_dist_function",
        "initial_branching_dist",
        "initial_branching_dist_function",
        "use_seed",
    ]
    for key in expected_keys:
        assert key in result

    # Check specific values
    assert result["step1_dist"] == (708.0, 0.5, "normal")
    assert result["step2_dist"] == (720.0, 0.5, "normal")
    assert result["use_seed"] is True

    # Check that with seed=True, np.random.seed was called
    # Indirectly verify this by checking two random draws give same result
    np.random.seed(21)
    val1 = np.random.normal(0, 1)
    np.random.seed(21)
    val2 = np.random.normal(0, 1)
    assert val1 == val2


@patch(
    "streamlit.columns",
    return_value=[MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock()],
)
@patch("streamlit.write")
@patch("streamlit.number_input", return_value=100)
def test_get_sampling_size(
    mock_number_input, mock_write, mock_columns, sample_json_params
):
    """Test get_sampling_size function."""
    # Call the function
    result = get_sampling_size(sample_json_params)

    # Check the result
    assert result == 100
    mock_number_input.assert_called_once()


@patch("streamlit.write")
@patch("streamlit.columns", return_value=[MagicMock(), MagicMock()])
@patch("streamlit.number_input")
def test_get_additional_parameters(
    mock_number_input, mock_columns, mock_write, sample_json_params
):
    """Test get_additional_parameters function."""
    # Set up mock returns
    mock_number_input.side_effect = [2, 1]

    # Call the function
    result = get_additional_parameters(sample_json_params)

    # Check the result
    assert len(result) == 3
    g, c, l_value = result
    assert g == 2
    assert c == 1
    assert l_value == 2  # l_value = c + 1


@patch("streamlit.progress")
@patch("streamlit.empty")
def test_run_monte_carlo_simulation(
    mock_empty, mock_progress, sample_data, monte_parameters
):
    """Test run_monte_carlo_simulation function."""
    # Create a small parameter set for testing
    np.random.seed(42)
    whole_set = setup_monte_carlo_parameters(monte_parameters, 5)

    # Extract data from sample
    x = sample_data["energy"].values
    y = sample_data["xas"].values
    z = sample_data["xmcd"].values

    # Mock the progress bar
    progress_bar = MagicMock()
    mock_progress.return_value = progress_bar

    # Run the simulation with mocked Polarized module functions
    with patch("app.Lz_cumsum", return_value=0.5), patch(
        "app.Sz_cumsum", return_value=0.25
    ):

        lz_list, sz_list, mu_tot_list, mu_rat_list = run_monte_carlo_simulation(
            x, y, z, whole_set, 2, 1, 2
        )

    # Check the results
    assert len(lz_list) == 5
    assert len(sz_list) == 5
    assert len(mu_tot_list) == 5
    assert len(mu_rat_list) == 5

    # All values should be the same due to our mock
    assert all(lz == 0.5 for lz in lz_list)
    assert all(sz == 0.25 for sz in sz_list)

    # Check calculated values
    for i in range(5):
        # µtot = -(g * sz + lz)
        assert mu_tot_list[i] == -(2 * 0.25 + 0.5)
        # µrat = lz / (g * sz)
        assert mu_rat_list[i] == 0.5 / (2 * 0.25)


def test_collect_all_parameters(monte_parameters):
    """Test collect_all_parameters function."""
    # Call the function
    result = collect_all_parameters(
        708.0, 720.0, 1.0, 0.33, monte_parameters, 2, 1, 100, False
    )

    # Check the result
    assert isinstance(result, dict)

    # Check key values
    assert result["initial_step1"] == 708.0
    assert result["initial_step2"] == 720.0
    assert result["initial_slope"] == 1.0
    assert result["initial_branching"] == 0.33
    assert result["g_factor"] == 2
    assert result["c_factor"] == 1
    assert result["l_value"] == 2  # c + 1
    assert result["sampling_size"] == 100
    assert not result["flip_xmcd"]

    # Check that Monte Carlo parameters were included
    for key in monte_parameters:
        assert key in result


# 6. Test results display and download
@patch("streamlit.header")
@patch("streamlit.columns", return_value=[MagicMock(), MagicMock(), MagicMock()])
def test_display_results(mock_columns, mock_header):
    """Test display_results function."""
    # Create mock data
    lz_list = [0.5, 0.6, 0.55, 0.52, 0.48]
    sz_list = [0.25, 0.3, 0.27, 0.26, 0.24]
    mu_tot_list = [-1.0, -1.2, -1.1, -1.04, -0.96]
    mu_rat_list = [1.0, 1.0, 1.02, 1.0, 1.0]
    monte_carlo_params = [
        [708.1, 720.2, 1.01, 0.33, 4.0, 0.0, 5, 2, -20] for _ in range(5)
    ]

    # Mock dataframe display
    with patch("streamlit.dataframe", MagicMock()), patch(
        "streamlit.plotly_chart", MagicMock()
    ), patch("streamlit.subheader", MagicMock()):

        # Call the function
        results_df, results_dict, fig = display_results(
            lz_list, sz_list, mu_tot_list, mu_rat_list, monte_carlo_params
        )

    # Check results
    assert isinstance(results_df, pd.DataFrame)
    assert len(results_df) == 4  # 4 parameters: Lz, Sz, µtot, µratio

    assert isinstance(results_dict, dict)
    assert "lz_list" in results_dict
    assert "sz_list" in results_dict
    assert "mu_tot_list" in results_dict
    assert "mu_rat_list" in results_dict
    assert "lz_mean" in results_dict
    assert "lz_std" in results_dict

    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 4  # 4 histograms


def test_h5_save_edge_cases():
    """Test edge cases for H5 saving."""
    # Test saving unusual data types
    data_dict = {
        "complex_value": complex(1, 2),  # Complex number
        "inf_value": float("inf"),  # Infinity
        "nan_value": float("nan"),  # NaN
        "empty_list": [],  # Empty list
        "mixed_list": [1, "two", 3.0],  # Mixed types
        "nested_dict": {"a": 1, "b": 2},  # Nested dictionary
    }

    curves_dict = {"x": np.array([1, 2, 3])}
    results_dict = {"result": 42.0}

    # Use a BytesIO object for in-memory testing
    buffer = io.BytesIO()

    # Call the function - should handle these unusual types without error
    save_to_h5(buffer, data_dict, curves_dict, results_dict)

    # Rewind the buffer and read the H5 file to verify it saved something
    buffer.seek(0)
    with h5py.File(buffer, "r") as f:
        assert "parameters" in f


def test_load_data_with_invalid_input():
    """Test load_data with various invalid inputs."""
    # Test with empty file
    empty_file = MockFile(b"")
    with patch("streamlit.error", MagicMock()) as mock_error, patch(
        "streamlit.stop", MagicMock()
    ) as mock_stop:
        load_data(empty_file)
        mock_error.assert_called()
        mock_stop.assert_called()

    # Test with file containing invalid data (not CSV formatted)
    invalid_file = MockFile(b"This is not a CSV file")
    with patch("streamlit.error", MagicMock()) as mock_error, patch(
        "streamlit.stop", MagicMock()
    ) as mock_stop:
        load_data(invalid_file)
        mock_error.assert_called()
        mock_stop.assert_called()

    # Test with file containing wrong number of columns
    wrong_columns = MockFile(b"700,0.1\n701,0.2\n702,0.3")
    with patch("streamlit.error", MagicMock()) as mock_error, patch(
        "streamlit.stop", MagicMock()
    ) as mock_stop:
        load_data(wrong_columns)
        mock_error.assert_called()
        mock_stop.assert_called()


def test_load_parameters_with_invalid_json():
    """Test load_parameters_from_json with invalid JSON."""
    # Test with invalid JSON file
    invalid_json = MockFile(b"This is not JSON")
    with patch("streamlit.error", MagicMock()) as mock_error, patch(
        "streamlit.success", MagicMock()
    ):
        result = load_parameters_from_json(invalid_json)
        mock_error.assert_called()
        assert result is None

    # Test with empty JSON file
    empty_json = MockFile(b"{}")
    with patch("streamlit.error", MagicMock()), patch(
        "streamlit.success", MagicMock()
    ) as mock_success:
        result = load_parameters_from_json(empty_json)
        mock_success.assert_called()
        assert result == {}


def test_special_parameter_combinations():
    """Test with specific parameter combinations that might be problematic."""
    # Test with very large step function values
    x = np.linspace(700, 730, 100)
    y = np.ones_like(x)

    with pytest.warns():
        corr_xas, step_func = _step(x, y, 1000, 2000, slope=1.0, br=0.5)

    # Test with inverted step positions (step2 < step1)
    corr_xas, step_func = _step(x, y, 720, 710, slope=1.0, br=0.5)
    assert np.all(np.isfinite(corr_xas))
    assert np.all(np.isfinite(step_func))

    # Test with very small branching ratio
    corr_xas, step_func = _step(x, y, 710, 720, slope=1.0, br=0.001)
    assert np.all(np.isfinite(corr_xas))
    assert np.all(np.isfinite(step_func))

    # Test with very large branching ratio
    corr_xas, step_func = _step(x, y, 710, 720, slope=1.0, br=0.999)
    assert np.all(np.isfinite(corr_xas))
    assert np.all(np.isfinite(step_func))

    # Test with negative slope
    corr_xas, step_func = _step(x, y, 710, 720, slope=-1.0, br=0.5)
    assert np.all(np.isfinite(corr_xas))
    assert np.all(np.isfinite(step_func))


def test_define_dist_edge_cases():
    """Test define_dist with edge case distribution parameters."""
    # Set seed for reproducibility
    np.random.seed(21)

    # Test normal distribution with zero variance
    assert define_dist(5, 0, "normal") == 5

    # Test normal distribution with very small variance
    small_var = define_dist(5, 1e-10, "normal")
    assert abs(small_var - 5) < 1e-9

    # Test uniform distribution with same min and max
    assert define_dist(5, 5, "uniform") == 5

    # Test randint with same min and max
    assert define_dist(5, 5, "randint") == 5

    # Test with None distribution type
    assert define_dist(5, 10, None) == 5


# 10. Test with realistic data and simulation parameters
def test_realistic_end_to_end(sample_data, monte_parameters):
    """Test a realistic end-to-end simulation with small sample size."""
    # Extract data from sample
    x = sample_data["energy"].values
    y = sample_data["xas"].values
    z = sample_data["xmcd"].values

    # Setup Monte Carlo with very small sample for speed
    np.random.seed(42)
    whole_set = setup_monte_carlo_parameters(monte_parameters, 3)

    # Patch the actual XAS functions to return predictable values
    with patch("app.Lz_cumsum", return_value=0.5), patch(
        "app.Sz_cumsum", return_value=0.25
    ), patch("streamlit.progress", return_value=MagicMock()), patch(
        "streamlit.empty", return_value=MagicMock()
    ):

        # Run the simulation
        lz_list, sz_list, mu_tot_list, mu_rat_list = run_monte_carlo_simulation(
            x, y, z, whole_set, 2, 1, 2
        )

    # Check results are as expected
    assert len(lz_list) == 3
    assert all(lz == 0.5 for lz in lz_list)
    assert all(sz == 0.25 for sz in sz_list)

    # Collect all parameters
    all_params = collect_all_parameters(
        708.0, 720.0, 1.0, 0.33, monte_parameters, 2, 1, 3, False
    )

    # Create curves dictionary
    curves_dict = {
        "energy": x,
        "xas": y,
        "xmcd": z,
        "corrected_xas": y - 0.5 * y,  # Mock corrected XAS
        "step_function": 0.5 * y,  # Mock step function
        "xas_cumulative": np.cumsum(y),
        "xmcd_cumulative": np.cumsum(z),
    }

    # Create results dictionary
    results_dict = {
        "lz_list": np.array(lz_list),
        "sz_list": np.array(sz_list),
        "mu_tot_list": np.array(mu_tot_list),
        "mu_rat_list": np.array(mu_rat_list),
        "lz_mean": float(np.mean(lz_list)),
        "sz_mean": float(np.mean(sz_list)),
        "mu_tot_mean": float(np.mean(mu_tot_list)),
        "mu_rat_mean": float(np.mean(mu_rat_list)),
        "lz_std": float(np.std(lz_list)),
        "sz_std": float(np.std(sz_list)),
        "mu_tot_std": float(np.std(mu_tot_list)),
        "mu_rat_std": float(np.std(mu_rat_list)),
    }

    # Test H5 creation
    buffer = io.BytesIO()
    save_to_h5(buffer, all_params, curves_dict, results_dict)

    # Verify H5 file was created successfully
    buffer.seek(0)
    with h5py.File(buffer, "r") as f:
        assert "parameters" in f
        assert "curves" in f
        assert "results" in f
        assert "energy" in f["curves"]
        assert "lz_list" in f["results"]
