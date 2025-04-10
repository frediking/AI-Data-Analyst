import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for non-GUI environments
import matplotlib.pyplot as plt
import seaborn as sns
import os
import tempfile
from utils.DeepSeek import DeepSeekChat
from utils.loader import get_csv_preview, validate_dataframe
from utils.preprocessing import clean_dataset
from utils.visualization import create_advanced_visualization, generate_visualizations
from utils.export import export_dataset, export_quality_report
from utils.logger import setup_logging
from utils.cache import cache_dataframe, cache_analysis_results, clear_cache
from utils.data_quality import assess_data_quality
from utils.dashboard import InteractiveDashboard
from utils.version_control import DataVersionControl
from utils.ml_insights import MLInsights
from utils.state import initialize_session_state, update_state_after_cleaning




# Initialize logger 
logger = setup_logging()

def initialize_chat():
    """Initialize chat functionality"""
    try:
        return DeepSeekChat()
    except Exception as e:
        st.error(f"Failed to initialize chat: {str(e)}")
        return None


def check_system_requirements():
    """Verify system requirements are met"""
    try:
        import requests
        return True
    except ImportError:
        st.error("Request not installed. Please install with: pip install replicate")
        return False

def validate_dataframe(df: pd.DataFrame) -> tuple[bool, str]:
    """Enhanced data validation"""
    if df.empty:
        return False, "DataFrame is empty"
    if df.columns.duplicated().any():
        return False, "Duplicate column names found"
    if df.isnull().all().any():
        return False, "Empty columns found"
    if df.shape[1] == 0:
        return False, "No columns found"
    return True, "Validation passed"


def validate_session_state():
    """Validate session state integrity"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'data_state' not in st.session_state:
        st.session_state.data_state = {}
    if 'original_df' not in st.session_state:
        st.session_state.original_df = None
    return True


@st.cache_data(ttl=3600)  # Cache for 1 hour
def generate_summary_stats(df: pd.DataFrame):
    """Generate cached summary statistics"""
    return {
        'shape': df.shape,
        'memory_usage': df.memory_usage().sum() / 1024,
        'summary': df.describe(include="all").fillna("").to_string()
    }


@st.cache_data(max_entries=2)
def load_and_process_data(file):
    """
    Load and process CSV data with caching and validation
    
    Args:
        file: File object or path to CSV file
        
    Returns:
        pd.DataFrame: Validated and processed dataframe
        
    Raises:
        RuntimeError: If file loading fails
        ValueError: If dataframe validation fails
    """
    try:
        df = pd.read_csv(file, low_memory=True)
        
        # Validate before returning
        if not validate_dataframe(df):
            raise ValueError("Invalid DataFrame: empty or contains duplicate columns")
            
        return df
        
    except pd.errors.EmptyDataError:
        raise RuntimeError("The CSV file is empty")
    except pd.errors.ParserError:
        raise RuntimeError("Failed to parse CSV file - invalid format")
    except Exception as e:
        raise RuntimeError(f"Failed to load CSV file: {str(e)}")
    


def setup_page():
    st.set_page_config(page_title="AI Data Analyst", layout="wide")
    st.title("📊 f-AI Data Analyst")
    st.markdown("""
    Upload a **CSV file** to:
    - 🧹 Clean and preprocess your data
    - 📊 Get statistical insights
    - 🤖 Quick Predictive Insights
    - 📈 Create visualizations
    - 💬 Chat with your data using DeepSeeek R1
    """)

def create_visualization(df):
    try:
        plt.close('all')  # Close any existing figures
        numeric_cols = df.select_dtypes(include="number")
        if not numeric_cols.empty:
            # Create a more compact visualization
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(numeric_cols.corr(), annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)
            plt.close(fig)
            # Add pairplot as an expandable section
            with st.expander("Show detailed pairplot"):
                st.pyplot(sns.pairplot(numeric_cols).fig)
        else:
            st.warning("No numeric columns found for visualization.")
    except Exception as viz_error:
        logger.error(f"Visualization error: {str(viz_error)}", exc_info=True)
        st.error(f"Chart error: {str(viz_error)}")
    finally:
        plt.close('all')


def validate_file_security(file) -> tuple[bool, str]:
    """Basic security validation"""
    ALLOWED_EXTENSIONS = {'.csv'}
    MAX_SIZE = 200 * 1024 * 1024  # 200MB
    ALLOWED_MIME_TYPES = {'text/csv'}
    
    if not file.name.lower().endswith('.csv'):
        return False, "Only CSV files are allowed"
    if file.size > MAX_SIZE:
        return False, f"File too large (max {MAX_SIZE/1024/1024}MB)"
        # Add MIME type check
    try:
        content_type = file.type
        if content_type not in ALLOWED_MIME_TYPES:
            return False, "Invalid file type"
    except AttributeError:
        logger.warning("Could not verify file MIME type")

    return True, "File passes security checks"



__version__ = "1.0.0"
logger.info(f"Starting FrozAI Data Analyst v{__version__}")

def main():
    try:
        # Initialize logging and app state
        logger.info("Starting FrozAI Data Analyst...")
        setup_page()

        if not validate_session_state():
         initialize_session_state()  # Initialize session state

        # Check system requirements
        logger.info("Checking system requirements...")
        if not check_system_requirements():
            logger.error("System requirements not met")
            st.error("Please install required dependencies first.")
            return

    
        uploaded_file = st.file_uploader("Upload your CSV", type=["csv"])
    
        if not uploaded_file:
            st.info("Please upload a CSV file to begin.")
            return
        
        # Clear previous cache before processing new file
        clear_cache('dataframe')
        clear_cache('analysis')
        
        # Validate file security first
        is_secure, security_msg = validate_file_security(uploaded_file)
        if not is_secure:
            logger.warning(f"Security check failed: {security_msg}")
            st.error(security_msg)
            return
        
        # Show progress bar
        progress = st.progress(0)
        progress.progress(25)

        # Add size check before processing
        MAX_FILE_SIZE = 200 * 1024 * 1024  # 200MB
        if uploaded_file.size > MAX_FILE_SIZE:
            logger.warning(f"File too large: {uploaded_file.name} ({uploaded_file.size / 1024 / 1024:.1f} MB)")
            st.error("File too large. Please upload a file smaller than 200MB.")
            return
            
        try:
            logger.info(f"Processing file: {uploaded_file.name}")
            # Get file path from uploaded file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                temp_path = tmp_file.name
            
             # Use cached loading
            df = load_and_process_data(temp_path)
            df = cache_dataframe(df)  # Cache after successful load
            logger.info(f"Successfully loaded file: {uploaded_file.name}")
            progress.progress(50)  # After loading

            # Generate and cache analysis results
            stats = generate_summary_stats(df)
            stats = cache_analysis_results('summary_stats', stats)  # Cache analysis results
             
            # Validate DataFrame
            valid_df, validation_msg = validate_dataframe(df)
            if not valid_df:
               logger.error(f"Invalid data format in file: {uploaded_file.name}")
               st.error(f"Invalid data format. {validation_msg}")
               return
            progress.progress(75)  # After validation

            # Cache the initial state
            if 'original_df' not in st.session_state:
                st.session_state.original_df = df.copy()

            # Create tabs for different functionalities
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "🧹 Clean Data",
                "📊 Statistical Summary",
                "📈 Visualizations",
                "📊 Interactive Dashboard",
                "🤖 ML Predictive Insights", 
                "💬 Chat with Data"
            ])

            with tab1:
                # Add preview at the top of tab1
                st.subheader("Data Preview")
                preview_df = get_csv_preview(temp_path)
                st.dataframe(preview_df)
                
                # Cleaning Options section
                st.subheader("Data Cleaning Options")
                reset_index = st.checkbox("Reset DataFrame index")
                standardize_dates = st.checkbox("Standardize date formats (YYYY-MM-DD)") 
                clean_numeric = st.checkbox("Clean numeric columns (remove outliers, fill missing values)")
                clean_text = st.checkbox("Clean text columns (remove special characters, standardize case)")
                remove_duplicates = st.checkbox("Remove duplicate rows")
                
                if st.button("Apply Cleaning"):
                    with st.spinner("Applying cleaning operations..."):    
                        df_clean, cleaning_notes = clean_dataset(
                            df, 
                            reset_index=reset_index,
                            standardize_dates=standardize_dates,
                            clean_numeric=clean_numeric,
                            clean_text=clean_text,
                            remove_duplicates=remove_duplicates,
                        )

                
                    # Update the main dataframe
                    df = df_clean.copy()  

                    # Initialize version control
                    version_control = DataVersionControl()            
        
                    # Save new version after cleaning
                    try:
                        version_id, version_hash = version_control.save_version(
                            df_clean,
                            "Cleaned dataset",
                            cleaning_notes
                        )
                        st.success(f"Created version {version_id} (hash: {version_hash})")

                        # Update session state
                        update_state_after_cleaning(df_clean, cleaning_notes)

                        with st.expander("📋 Cleaning Report"):
                            for note in cleaning_notes:
                                st.write(f"- {note}")

                    except Exception as e:
                        logger.error(f"Version control error: {str(e)}")
                        st.error("Failed to save version")


                # Add version history viewer
            with st.expander("📚 Version History"):
                try:
                    history = version_control.get_version_history()
                    for version in history:
                        st.write(f"**Version:** {version['id']}")
                        st.write(f"- Time: {version['timestamp']}")
                        st.write(f"- Description: {version['description']}")
                        st.write(f"- Changes: {len(version['transformations'])} transformations")
                        if st.button(f"Restore to {version['id']}"):
                            try:
                                df = version_control.load_version(version['id'])
                                st.success(f"Restored to version {version['id']}")
                            except Exception as e:
                                st.error(f"Failed to restore version: {str(e)}")            
                        st.divider()
                except Exception as e:
                    st.error("No version history available yet")
            
               
            
            with tab2:
                st.subheader("Data Quality & Statistical Summary")
                
                # Data Quality Section - Move this first
                with st.expander("📊 Data Quality Assessment", expanded=True):
                    try:
                        with st.spinner("Analyzing data quality..."):
                            quality_metrics = assess_data_quality(df)
                            
                            # Overall metrics
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric(
                                    "Data Completeness",
                                    f"{quality_metrics['completeness']['overall_completeness']}%"
                                )
                            with col2:
                                st.metric(
                                    "Duplicate Rows",
                                    quality_metrics['uniqueness']['duplicate_rows']
                                )
                            with col3:
                                st.metric(
                                    "Mixed Type Columns",
                                    len(quality_metrics['type_consistency']['mixed_types'])
                                )
                            
                            # Quality Metric Tabs
                            q_tab1, q_tab2, q_tab3 = st.tabs(["Completeness", "Uniqueness", "Data Types"])
                            
                            with q_tab1:
                                st.write("Missing Values by Column:")
                                missing_df = pd.DataFrame({
                                    'Missing Count': quality_metrics['completeness']['missing_counts'],
                                    'Missing %': quality_metrics['completeness']['missing_percentages']
                                })
                                st.dataframe(missing_df.style.background_gradient(cmap='RdYlGn_r'))
                            
                            with q_tab2:
                                st.write("Uniqueness Analysis:")
                                unique_df = pd.DataFrame({
                                    'Unique Values': quality_metrics['uniqueness']['unique_values_per_column'],
                                    'Unique %': quality_metrics['uniqueness']['unique_percentages']
                                })
                                st.dataframe(unique_df.style.background_gradient(cmap='RdYlGn'))
                            
                            with q_tab3:
                                st.write("Data Types and Consistency:")
                                for col, types in quality_metrics['type_consistency']['mixed_types'].items():
                                    st.warning(f"Mixed types in '{col}': {', '.join(types)}")
                            
                            # Export options
                            col1, col2 = st.columns(2)
                            with col1:
                                report_format = st.selectbox(
                                    "Export Format",
                                    ["markdown", "json", "yaml"]
                                )
                            with col2:
                                if st.button("Export Quality Report"):
                                    try:
                                        report_bytes, filename, mime_type = export_quality_report(
                                            quality_metrics,
                                            format=report_format
                                        )
                                        st.download_button(
                                            label="📥 Download Quality Report",
                                            data=report_bytes,
                                            file_name=filename,
                                            mime=mime_type
                                        )
                                    except Exception as e:
                                        st.error(f"Failed to export report: {str(e)}")
                                        
                    except Exception as e:
                        logger.error(f"Failed to generate quality report: {str(e)}")
                        st.error(f"Failed to generate quality report: {str(e)}")
                
                # Statistical Summary Section - After quality assessment
                st.divider()
                st.subheader("Statistical Summary")
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.dataframe(df.head(10))
                with col2:
                    st.write("Dataset Info:")
                    st.write(f"- Rows: {stats['shape'][0]:,}")
                    st.write(f"- Columns: {stats['shape'][1]}")
                    st.write(f"- Memory Usage: {stats['memory_usage']:.2f} KB")

            with tab3:
                try:
                    st.subheader("Data Visualization")
                    
                    # Check if we have numeric columns
                    numeric_cols = df.select_dtypes(include=['number']).columns
                    if len(numeric_cols) < 2:
                        st.warning("Need at least 2 numeric columns for visualization")
                        return
                        
                    viz_type = st.selectbox(
                        "Select visualization type:",
                        ["Correlation Heatmap", "Advanced Charts"]
                    )
                    
                    if viz_type == "Correlation Heatmap":
                        fig = generate_visualizations(df)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        create_advanced_visualization(df)
                        
                except Exception as e:
                    logger.error(f"Visualization error: {str(e)}", exc_info=True)
                    st.error(f"Failed to create visualization: {str(e)}")
            
            with tab4:
                st.subheader("📊 Interactive Dashboard")
        
                try:
                    dashboard = InteractiveDashboard(df)
                    
                    # Dashboard Layout
                    st.write("### Data Explorer Dashboard")
                    
                    # Sidebar for controls
                    chart_type = st.selectbox(
                        "Select Chart Type",
                        ["Time Series", "Distribution Analysis", "Correlation Analysis", "Scatter Analysis"]
                    )
                    
                    if chart_type == "Time Series":
                        col1, col2, col3 = st.columns([1,1,1])
                        with col1:
                            x_col = st.selectbox("Select X-axis (Time)", dashboard.date_cols)
                        with col2:
                            y_col = st.selectbox("Select Y-axis (Metric)", dashboard.numeric_cols)
                        with col3:
                            group_col = st.selectbox("Group By (optional)", 
                                                ["None"] + list(dashboard.categorical_cols))
                        
                        fig = dashboard.create_time_series_plot(
                            x_col, 
                            y_col, 
                            None if group_col == "None" else group_col
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                    elif chart_type == "Distribution Analysis":
                        col = st.selectbox("Select Column", dashboard.numeric_cols)
                        fig = dashboard.create_distribution_plot(col)
                        st.plotly_chart(fig, use_container_width=True)
                        
                    elif chart_type == "Correlation Analysis":
                        fig = dashboard.create_correlation_matrix()
                        st.plotly_chart(fig, use_container_width=True)
                        
                    elif chart_type == "Scatter Analysis":
                        cols = st.multiselect(
                            "Select Columns (2-4 recommended)", 
                            dashboard.numeric_cols,
                            default=list(dashboard.numeric_cols)[:3]
                        )
                        if len(cols) >= 2:
                            fig = dashboard.create_scatter_matrix(cols)
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("Please select at least 2 columns")
                    
                except Exception as e:
                    logger.error(f"Dashboard error: {str(e)}")
                    st.error(f"Failed to create dashboard: {str(e)}")    


            with tab5:
                st.subheader("🤖 Quick Predictive Insights")
    
                try:
                    ml_insights = MLInsights(df)
                    
                    # Select target variable
                    target_col = st.selectbox(
                        "Select Target Variable",
                        df.columns.tolist(),
                        help="Select the variable you want to predict"
                    )
                    
                    if st.button("Generate ML Insights"):
                        with st.spinner("Analyzing data and generating insights..."):
                            results = ml_insights.quick_model(target_col)
                            
                            # Display results
                            st.write("### Model Performance")
                            if results['task_type'] == 'classification':
                                st.metric("Accuracy", f"{results['performance']['accuracy']:.2%}")
                                st.code(results['performance']['classification_report'])
                            else:
                                st.metric("R² Score", f"{results['performance']['r2_score']:.2%}")
                            
                            # Plot feature importance
                            st.write("### Feature Importance")
                            fig = ml_insights.plot_feature_importance(results['feature_importance'])
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Download detailed report
                            report = f"""
                            # Machine Learning Insights Report

                            ## Model Information
                            - Task Type: {results['task_type']}
                            - Target Variable: {target_col}

                            ## Performance Metrics
                            {results['performance']}

                            ## Top Features
                            {results['feature_importance'].to_string()}
                            """
                            
                            st.download_button(
                                label="📥 Download ML Report",
                                data=report.encode('utf-8'),
                                file_name='ml_insights_report.txt',
                                mime='text/plain'
                            )
                            
                except Exception as e:
                    logger.error(f"ML Insights error: {str(e)}")
                    st.error(f"Failed to generate ML insights: {str(e)}")




            with tab6:
                st.subheader("💬 Chat with Your Data")
                
                chat = initialize_chat()
                if chat:
                    question = st.text_input("Ask a question about your data:")
                    if question:
                        df_info = {
                            'rows': len(df),
                            'columns': list(df.columns),
                            'dtypes': df.dtypes.to_dict(),
                            'descriptions': {col: str(df[col].describe()) for col in df.columns}
                        }
                        
                        with st.spinner("Generating response..."):
                            try:
                                response = chat.chat_with_data(df_info, question)
                                # Add message to chat history
                                st.session_state.messages.append({"role": "user", "content": question})
                                st.session_state.messages.append({"role": "assistant", "content": response})
                                st.write(response)
                            except Exception as e:
                                logger.error(f"Chat error: {str(e)}")
                                st.error(f"Failed to generate response: {str(e)}")
                
                # Display chat history
                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.write(message["content"])
                
                # Chat input
                if prompt := st.chat_input("Ask about your data..."):
                    # Add user message
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    
                    # Display user message
                    with st.chat_message("user"):
                        st.write(prompt)
                    
                    # Generate response
                    with st.chat_message("assistant"):
                        with st.spinner("Analyzing..."):
                            try:
                                context = f"""
                                Dataset Summary:
                                - Rows: {df.shape[0]}
                                - Columns: {', '.join(df.columns)}
                                - Numeric columns: {df.select_dtypes(include=['number']).columns.tolist()}
                                - Categories: {df.select_dtypes(include=['object']).columns.tolist()}
                                """
                                
                                response = st.session_state.chat_chain.run({
                                    "context": context,
                                    "question": prompt
                                })
                                
                                st.write(response)
                                st.session_state.messages.append({
                                    "role": "assistant",
                                    "content": response
                                })
                            except Exception as e:
                                st.error(f"Failed to generate response: {str(e)}")

            progress.progress(100)  # Processing complete
            progress.empty()  # Clear progress bar after completion
            
            
            # Add export functionality
            st.divider()
            st.subheader("📥 Export Options")
            col1, col2, col3 = st.columns(3)

            with col1:
                export_format = st.selectbox(
                    "Select processed data format:",
                    ["csv", "excel", "json", "yaml"]
                )
                data, filename, mimetype = export_dataset(df, export_format)
                st.download_button(
                   label=f"📥 Download Processed Data",
                   data=data,
                   file_name=filename,
                   mime=mimetype
                )

            with col2:
                st.download_button(
                   label="📥 Download Raw Data",
                   data=uploaded_file.getvalue(),
                   file_name='raw_data.csv',
                   mime='text/csv'
                )
            with col3:
                report_text = f"""Dataset Analysis Report
                - Total Rows: {df.shape[0]:,}
                - Total Columns: {df.shape[1]}
                - Cleaning Steps: {len(st.session_state.data_state.get('cleaning_notes', []))} changes made
                Data Summary:
                {stats['summary']}
                
                Cleaning Report:
                    {chr(10).join([f'- {note}' for note in st.session_state.data_state.get('cleaning_notes', [])])}
                """
                st.download_button(
                    label="📥 Download Analysis Report",
                    data=report_text.encode('utf-8'),
                    file_name='analysis_report.txt',
                    mime='text/plain'
                )


        except pd.errors.EmptyDataError:
            logger.error("Empty file uploaded")
            st.error("The uploaded file is empty.")
        except pd.errors.ParserError:
            logger.error("Invalid CSV format")
            st.error("Invalid CSV format. Please check your file.")
        except Exception as e:
            logger.error(f"Error processing file {uploaded_file.name}: {str(e)}", exc_info=True)
            st.error(f"Failed to process file: {str(e)}")
        finally:
            if 'progress' in locals():
                progress.empty()
            # Clean up temporary file
            if 'temp_path' in locals():
                try:
                    os.unlink(temp_path)
                except Exception as e:
                    st.warning(f"Warning: Could not delete temporary file: {str(e)}")
                    logger.warning(f"Could not delete temporary file: {str(e)}")
   
    except Exception as e:
        logger.error(f"Application error: {str(e)}", exc_info=True)
        st.error("An unexpected error occurred. Please try again.")

  
# Clean up resources on shutdown
def cleanup_resources():
    """Clean up resources on shutdown"""
    try:
        plt.close('all')
        if 'temp_path' in st.session_state:
            clear_cache()
            os.unlink(st.session_state.temp_path)
    except Exception as e:
        logger.warning(f"Cleanup error: {str(e)}")

if __name__ == "__main__":
    try:
        main()
    finally:
        cleanup_resources()
