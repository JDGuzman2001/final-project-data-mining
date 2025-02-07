# Premier League Performance Predictor

## Overview
This project uses machine learning to predict Premier League team performance for the 2025 season based on historical data from 2020-2024. It combines Random Forest and Linear Regression models to generate accurate predictions while considering various performance metrics such as expected goals (xG), possession statistics, and shooting accuracy.

## Features
- Historical data analysis from 2020-2024 seasons
- Advanced feature engineering and data preprocessing
- Ensemble prediction model (Random Forest + Linear Regression)
- Interactive visualizations using Plotly
- Confidence interval analysis for predictions
- Comprehensive model evaluation metrics

## Requirements
```python
pandas
numpy
scikit-learn
matplotlib
seaborn
plotly
scipy
```

## Installation
1. Clone this repository
```bash
git clone https://github.com/JDGuzman2001/final-project-data-mining.git
cd final-project-data-mining
```

2. Install required packages
```bash
pip install -r requirements.txt
```

## Usage
1. Place your Premier League data file (`matches.csv`) in the project directory
2. Run the main script:
```bash
python main.py
```

## Output Files
The script generates several output files:
- `predicciones_2025.csv`: Detailed predictions for each team
- `evaluacion_modelos.png`: Model evaluation visualizations
- `metricas_evaluacion.csv`: Detailed performance metrics
- `predicciones_2025.html`: Interactive visualization of predictions
- `intervalos_confianza.html`: Confidence intervals visualization

## Model Details
The prediction system uses two models:
- **Random Forest Regressor**: Accounts for 70% of the final prediction
- **Linear Regression**: Accounts for 30% of the final prediction

Features used for prediction include:
- Goals scored and conceded
- Expected goals (xG) and expected goals against (xGA)
- Possession statistics
- Shot accuracy
- Total shots and shots on target
- Non-penalty expected goals (npxG)

## Evaluation Metrics
The model's performance is evaluated using:
- R² Score
- Root Mean Square Error (RMSE)
- Mean Absolute Error (MAE)

## Visualizations
The project includes several visualizations:
1. Predictions vs Actual Values
2. Residual Analysis
3. Residual Distribution
4. Q-Q Plot
5. Interactive Bar Charts of Predictions
6. Confidence Interval Plots

## Project Structure
```
final-project-data-mining/
│
├── main.py                    # Main script
├── matches.csv               # Input data
├── requirements.txt          # Package dependencies
├── README.md                # Project documentation
│
├── output/
│   ├── predicciones_2025.csv
│   ├── evaluacion_modelos.png
│   ├── metricas_evaluacion.csv
│   ├── predicciones_2025.html
│   └── intervalos_confianza.html
```


