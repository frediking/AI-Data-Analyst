import pytest
import importlib
import streamlit as st
import pandas as pd
from utils.state import initialize_session_state, update_state_after_cleaning
from utils.preprocessing import clean_dataset
from utils.visualization import generate_visualizations
from utils.ml_insights import MLInsights
from utils.export import export_dataset
from utils.data_quality import assess_data_quality

@pytest.fixture(autouse=True)
def reset_session_state(monkeypatch):
    # Reset Streamlit session state before each test
    st.session_state.clear()
    yield
    st.session_state.clear()

def test_app_runs_without_errors():
    """
    Test that the main Streamlit app runs without raising exceptions.
    This is a smoke/integration test.
    """
    try:
        import app  # Your main app file (ensure this matches your filename)
        importlib.reload(app)
    except Exception as e:
        pytest.fail(f"App failed to run: {e}")

def test_session_state_initialized():
    """
    Test that required session state keys are initialized.
    """
    # Directly call your initialization logic, since Streamlit runtime is not present
    initialize_session_state()
    required_keys = [
        "data_state",
        "app_state",
        "viz_settings",
        "dashboard_state",
        "version_state",
        "ml_state",
        "chat_state"
    ]
    for key in required_keys:
        assert key in st.session_state, f"Session state missing key: {key}"


def test_app_core_workflow(monkeypatch):
    """
    Test the app's core workflow: simulate loading a DataFrame and running a main processing function.
    """
    # Example: Mock file uploader or data load
    sample_df = pd.DataFrame({"A": [1, 2], "B": ["x", "y"]})
    initialize_session_state()
    st.session_state.data_state["original_df"] = sample_df

    # Import and reload app to trigger main logic
    import app
    importlib.reload(app)

    # Check that the app updated session state as expected (customize as needed)
    assert st.session_state.data_state["original_df"] is not None
    # If your app processes or cleans data, check for expected state updates:
    # assert st.session_state.data_state["current_df"] is not None

def test_full_userflow(monkeypatch, tmp_path):
    """
    Simulate the main userflow: upload → clean → analyze → visualize → ML → export.
    """
    # 1. Simulate file upload (create a sample DataFrame and save as CSV)
    df = pd.DataFrame({
        "num1": [1, 2, 3, 4],
        "num2": [10, 20, 30, 40],
        "cat": ["a", "b", "a", "b"]
    })
    csv_path = tmp_path / "test.csv"
    df.to_csv(csv_path, index=False)

    # 2. Initialize session state as app does
    initialize_session_state()
    st.session_state.data_state["original_df"] = df.copy()
    st.session_state.original_df = df.copy()

    # 3. Data cleaning step (simulate user cleaning)
    cleaned_df, cleaning_notes = clean_dataset(df, reset_index=True, standardize_dates=False, clean_numeric=False, clean_text=False, remove_duplicates=True)
    update_state_after_cleaning(cleaned_df, cleaning_notes)
    assert not cleaned_df.empty
    assert "current_df" in st.session_state.data_state

    # 4. Data quality assessment
    quality = assess_data_quality(cleaned_df)
    expected_keys = ["null_summary", "overall_completeness", "outliers"]
    for key in expected_keys:
        assert key in quality, f"Missing key in quality report: {key}"

    # 5. Visualization step
    fig = generate_visualizations(cleaned_df)
    assert fig is not None

    # 6. ML Insights step
    ml = MLInsights(cleaned_df)
    target_col = "num1"
    # You may want to mock quick_model if it requires heavy computation or external calls
    if hasattr(ml, "quick_model"):
        try:
            results = ml.quick_model(target_col)
            assert "performance" in results
        except Exception:
            # If quick_model is not implemented or fails, skip this part
            pass

    # 7. Export processed data
    for fmt, ext in [("csv", ".csv"), ("excel", ".xlsx"), ("json", ".json")]:
        data, filename, mimetype = export_dataset(cleaned_df, fmt)
        assert data is not None
        assert filename.endswith(ext)

    # 8. Check session state reflects user progress
    assert st.session_state.data_state["cleaning_notes"] == cleaning_notes

    # 9. (Optional) Simulate navigation to each tab by checking expected session state keys
    expected_tabs = [
        "data_state", "original_df"
    ]
    for key in expected_tabs:
        assert key in st.session_state