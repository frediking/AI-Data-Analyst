# AI Data Analyst (Streamlit)

![Screenshot 2025-04-18 at 00 28 23](https://github.com/user-attachments/assets/f1d2ea7a-7eb1-4a9f-94cf-1b25bc83e738)

## Overview
The **AI Data Analyst** project is an intelligent data analysis tool that leverages artificial intelligence to automate and enhance data analytics tasks. This project provides a framework for performing exploratory data analysis (EDA), generating insights, creating visualizations, and building predictive models using AI-driven techniques. It is designed for data analysts, data scientists, and businesses looking to streamline data processing and decision-making processes.

The project is built primarily in **Python** and utilizes popular libraries such as **Pandas**, **NumPy**, **Matplotlib**, **Seaborn**, and **Scikit-learn**, along with AI-powered tools like **PandasAI** or **LLM-based agents** for natural language data queries and automated analysis.

## Features
- **Automated Data Analysis**: Perform EDA with minimal coding using AI-driven insights.
- **Interactive Visualizations**: Generate dynamic charts (e.g., bar, line, scatter, histograms) using Plotly and Matplotlib.
- **AI-Powered Queries**: Query datasets in natural language using AI agents (e.g., powered by LLMs).
- **Predictive Modeling**: Build and evaluate machine learning models for classification, regression, or clustering.
- **Data Cleaning & Preprocessing**: Handle missing values, outliers, and data transformations automatically.
- **Report Generation**: Export analysis results and visualizations as downloadable reports.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [Testing](#testing)
- [License](#license)
- [Contact](#contact)

## Installation

### Prerequisites
Ensure you have the following installed:
- **Python** (version 3.8 or higher)
- **pip** (Python package manager)
- **Git** (for cloning the repository)
- Optional: **Docker** (if running in a containerized environment)

### Steps
1. **Clone the repository**:
   ```bash
   git clone https://github.com/frediking/AI-Data-Analyst.git
   cd AI-Data-Analyst
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   - Create a `.env` file in the root directory.
   - Add necessary API keys (e.g., for OpenAI, if using LLMs):
     ```plaintext
     OPENAI_API_KEY=your_openai_api_key
     ```

5. **Verify installation**:
   Run the main script to ensure everything is set up correctly:
   ```bash
   python main.py
   ```

## Usage

### Running the Application
To start the AI Data Analyst tool, run:
```bash
streamlit run app.py
```
This launches a web-based interface (powered by **Streamlit**) at `http://localhost:8501`.

### Example Workflow
1. **Upload Data**:
   - Use the Streamlit interface to upload a CSV or Excel file.
2. **Perform Analysis**:
   - Run automated EDA to generate summary statistics and visualizations.
   - Use natural language queries (e.g., "What is the average sales by region?") to get instant results.
3. **Build Models**:
   - Train machine learning models via the interface or scripts in the `models/` directory.
4. **Export Results**:
   - Download reports or visualizations as PDF/PNG files.

### Example Commands
- Run EDA on a dataset:
  ```bash
  python scripts/eda.py --file data/sample_data.csv
  ```
- Train a predictive model:
  ```bash
  python scripts/train_model.py --model logistic --data data/sample_data.csv
  ```

## Configuration
- **Data Sources**: Place your datasets in the `data/` directory. Supported formats: CSV, Excel, JSON.
- **Model Parameters**: Edit `config.yaml` to customize model hyperparameters or visualization settings:
  ```yaml
  model:
    type: logistic
    max_iterations: 100
  visualization:
    theme: plotly
    colors: ["#1f77b4", "#ff7f0e"]
  ```
- **API Keys**: Ensure API keys for AI services (e.g., OpenAI, Google Cloud) are set in the `.env` file.

## Project Structure
```plaintext
AI-Data-Analyst/
├── data/                 # Directory for datasets
├── scripts/              # Python scripts for analysis and modeling
│   ├── eda.py           # Script for exploratory data analysis
│   ├── train_model.py   # Script for training ML models
├── models/               # Trained model files and pipelines
├── notebooks/            # Jupyter notebooks for experimentation
├── src/                  # Source code for the application
│   ├── __init__.py
│   ├── analysis.py      # Core analysis functions
│   ├── visualization.py  # Visualization utilities
├── app.py                # Main Streamlit application
├── requirements.txt      # Python dependencies
├── config.yaml           # Configuration file
├── .env                  # Environment variables (not tracked)
├── README.md             # This file
└── LICENSE               # License file
```

## Contributing
We welcome contributions to improve the AI Data Analyst project! To contribute:
1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. Make your changes and commit:
   ```bash
   git commit -m "Add your feature description"
   ```
4. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```
5. Open a pull request on GitHub.

Please follow the [Code of Conduct](CODE_OF_CONDUCT.md) and review the [Contributing Guidelines](CONTRIBUTING.md).

## Testing
To run the test suite:
```bash
pytest tests/
```
Tests are located in the `tests/` directory and cover core functionality, including data processing, model training, and visualization generation.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact
For questions, suggestions, or issues, please reach out:
- **GitHub Issues**: https://github.com/frediking/AI-Data-Analyst/issues
- **Email**: frediking@example.com
- **Discord**: Join our community at [Your Discord link, if applicable]
