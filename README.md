# PREDICTING EXCHANGE RATE SHOCK WINDOWS IN GHANA: A MACHINE LEARNING EARLY WARNING SYSTEM FOR IMPORT-DEPENDENT SMES

## Overview
This project aims to develop a machine learning-based early warning system to predict exchange rate shock windows in Ghana, specifically targeting import-dependent small and medium-sized enterprises (SMEs). By leveraging historical exchange rate data and relevant economic indicators, the system will provide timely alerts to SMEs, enabling them to make informed decisions regarding their import activities and financial planning.

## Project Setup
To set up the project, follow these steps:
1. Clone the repository to your local machine:
   ```bash
   git clone https://github.com/achifred/cedishockprediction.git
   ```
2. Navigate to the project directory:
   ```bash
   cd ghshockwindow
3. Create a virtual environment to manage dependencies for mac/linux:
   ```bash
   python -m venv venv
   ```
4. Activate the virtual environment:
   ```bash
   source venv/bin/activate  
   ```
5. Create a virtual environment to manage dependencies for Windows:
   ```bash
   python -m venv venv
   ```
6. Activate virtual environment for Windows:
    ```bash
    source myenv/Scripts/activate
    or
    #powershell
    myenv\Scripts\Activate.ps1
    or
    #cmd
    myenv\Scripts\activate.bat
    ```
7. Install the required dependencies using pip:
   ```bash
   pip install -r requirements.txt
8. Ensure you have Python 3.13 or higher installed on your system.


9. run the following command to make the scripts executable:
   ```bash
   chmod +x scripts/*.sh
   ```

## To Run the Project
1. Ensure that the virtual environment is activated.

2. To run the entire pipeline, execute the following command:
   ```bash
   bash scripts/run_pipeline.sh
   ```
3. To run the data preprocessing step only, execute:
   ```bash
   bash scripts/preprocessing_data.sh
   ```
4. To run the feature engineering step only, execute:
   ```bash
   bash scripts/feature_engineer.sh
   ```
5. To run the model training and evaluation step only, execute:
   ```bash
   bash scripts/run_models.sh
   ```
6. To run Xgboost model only, execute:
   ```bash
   bash scripts/run_xgb.sh
   ```
7. To run LSTM model only, execute:
   ```bash
   bash scripts/run_lstm.sh
   ```
8. To run the data summary step only, execute:
   ```bash
   bash scripts/run_data_summary.sh
   ```
9. To run the metric graph generation step only, execute:
   ```bash
   bash scripts/run_metric_graph.sh
   ```