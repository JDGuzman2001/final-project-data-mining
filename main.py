# import kagglehub
#
# # Download latest version
# path = kagglehub.dataset_download("chrisdsilva007/premier-league-from-2020-2024")
#
# print("Path to dataset files:", path)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import scipy.stats as stats

warnings.filterwarnings('ignore')


def load_and_prepare_data(file_path):
    """
    Carga y prepara los datos iniciales
    """
    df = pd.read_csv(file_path)
    print(f"Registros totales: {len(df)}")
    return df


def clean_and_transform_data(df):
    """
    Limpia y transforma los datos para el análisis
    """
    # Convertir resultados a puntos
    result_points = {'W': 3, 'D': 1, 'L': 0}
    df['points'] = df['result_x'].map(result_points)

    # Convertir columnas a numéricas
    numeric_columns = ['gf_x', 'ga_x', 'xg_x', 'xga', 'poss', 'sh', 'sot', 'npxg']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Crear características agregadas por temporada y equipo
    seasonal_stats = df.groupby(['season', 'team']).agg({
        'points': 'sum',  # Puntos totales
        'gf_x': 'sum',  # Goles a favor
        'ga_x': 'sum',  # Goles en contra
        'xg_x': 'mean',  # Expected goals promedio
        'xga': 'mean',  # Expected goals against promedio
        'poss': 'mean',  # Posesión promedio
        'sh': 'sum',  # Tiros totales
        'sot': 'sum',  # Tiros al arco
        'npxg': 'sum',  # Non-penalty expected goals
        'result_x': lambda x: (x == 'W').sum()  # Victorias totales
    }).reset_index()

    # Renombrar columnas
    seasonal_stats.columns = [
        'season', 'team', 'total_points', 'goals_for', 'goals_against',
        'avg_xg', 'avg_xga', 'avg_possession', 'total_shots',
        'shots_on_target', 'total_npxg', 'total_wins'
    ]

    # Calcular métricas adicionales
    seasonal_stats['goal_difference'] = seasonal_stats['goals_for'] - seasonal_stats['goals_against']
    seasonal_stats['xg_difference'] = seasonal_stats['avg_xg'] - seasonal_stats['avg_xga']
    seasonal_stats['shot_accuracy'] = seasonal_stats['shots_on_target'] / seasonal_stats['total_shots']
    seasonal_stats['goals_per_shot'] = seasonal_stats['goals_for'] / seasonal_stats['total_shots']
    seasonal_stats['points_per_win'] = seasonal_stats['total_points'] / seasonal_stats['total_wins']
    seasonal_stats['expected_points'] = seasonal_stats['total_npxg'] * 3  # Aproximación basada en xG

    # Seleccionar solo columnas numéricas
    numeric_cols = seasonal_stats.select_dtypes(include=[np.number]).columns

    # Llenar valores NaN solo en columnas numéricas
    seasonal_stats[numeric_cols] = seasonal_stats[numeric_cols].fillna(seasonal_stats[numeric_cols].mean())

    return seasonal_stats


def prepare_for_ml(seasonal_stats):
    """
    Prepara los datos para el entrenamiento de modelos
    """
    features = [
        'goals_for', 'goals_against', 'avg_xg', 'avg_xga',
        'avg_possession', 'total_shots', 'shots_on_target',
        'total_npxg', 'goal_difference', 'xg_difference',
        'shot_accuracy', 'goals_per_shot', 'expected_points'
    ]

    X = seasonal_stats[features]
    y = seasonal_stats['total_points']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, features


def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """
    Entrena y evalúa múltiples modelos
    """
    # Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_test = rf_model.predict(X_test)
    rf_train = rf_model.predict(X_train)
    rf_score = r2_score(y_test, rf_pred)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
    rf_mae = mean_absolute_error(y_test, rf_pred)

    # Regresión Lineal
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_pred = lr_test = lr_model.predict(X_test)
    lr_train = lr_model.predict(X_train)
    lr_score = r2_score(y_test, lr_pred)
    lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
    lr_mae = mean_absolute_error(y_test, lr_pred)

    # Visualizar resultados de evaluación
    plt.figure(figsize=(15, 10))
    
    # Subplot 1: Predicciones vs Reales
    plt.subplot(2, 2, 1)
    plt.scatter(y_test, rf_test, alpha=0.5, label='Random Forest')
    plt.scatter(y_test, lr_test, alpha=0.5, label='Linear Regression')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Puntos Reales')
    plt.ylabel('Puntos Predichos')
    plt.title('Predicciones vs Valores Reales')
    plt.legend()

    # Subplot 2: Residuos vs Predicciones
    plt.subplot(2, 2, 2)
    plt.scatter(rf_test, y_test - rf_test, alpha=0.5, label='Random Forest')
    plt.scatter(lr_test, y_test - lr_test, alpha=0.5, label='Linear Regression')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicciones')
    plt.ylabel('Residuos')
    plt.title('Análisis de Residuos')
    plt.legend()

    # Subplot 3: Distribución de Residuos
    plt.subplot(2, 2, 3)
    sns.kdeplot(data=y_test - rf_test, label='Random Forest', fill=True)
    sns.kdeplot(data=y_test - lr_test, label='Linear Regression', fill=True)
    plt.xlabel('Residuos')
    plt.ylabel('Densidad')
    plt.title('Distribución de Residuos')
    plt.legend()

    # Subplot 4: Q-Q Plot de Residuos
    plt.subplot(2, 2, 4)
    stats.probplot(y_test - rf_test, dist="norm", plot=plt)
    plt.title('Q-Q Plot de Residuos (Random Forest)')

    plt.tight_layout()
    plt.savefig('evaluacion_modelos.png')
    plt.close()

    # Crear DataFrame con métricas de evaluación
    metrics_df = pd.DataFrame({
        'Modelo': ['Random Forest', 'Linear Regression'],
        'R² Score': [rf_score, lr_score],
        'RMSE': [rf_rmse, lr_rmse],
        'MAE': [rf_mae, lr_mae]
    })

    # Guardar métricas en un archivo
    metrics_df.to_csv('metricas_evaluacion.csv', index=False)
    
    print("\nMétricas de Evaluación:")
    print(metrics_df.to_string(index=False))

    return rf_model, lr_model, rf_score, lr_score, rf_rmse, lr_rmse


def predict_2025_performance(rf_model, lr_model, current_stats, features):
    """
    Realiza predicciones para la temporada 2025
    """
    latest_season = current_stats['season'].max()
    latest_stats = current_stats[current_stats['season'] == latest_season].copy()

    pred_features = latest_stats[features]

    # Realizar predicciones
    rf_predictions = rf_model.predict(pred_features)
    lr_predictions = lr_model.predict(pred_features)

    # Combinar predicciones (70% RF, 30% LR)
    latest_stats['predicted_points_2025'] = (0.7 * rf_predictions + 0.3 * lr_predictions)

    # Preparar resultados
    results = latest_stats[['team', 'total_points', 'total_wins',
                            'predicted_points_2025', 'goal_difference']].copy()
    results['predicted_points_2025'] = results['predicted_points_2025'].round(1)
    results['predicted_position_change'] = results['predicted_points_2025'] - results['total_points']

    # Visualizar predicciones
    fig = px.bar(results.head(20), 
                 x='team', 
                 y='predicted_points_2025',
                 title='Predicción de Puntos para 2025',
                 color='predicted_position_change',
                 labels={'predicted_points_2025': 'Puntos Predichos',
                        'team': 'Equipo',
                        'predicted_position_change': 'Cambio en Posición'},
                 color_continuous_scale='RdYlBu')
    fig.write_html('predicciones_2025.html')

    # Calcular y visualizar intervalos de confianza
    predictions_std = np.std([
        rf_model.predict(pred_features),
        lr_model.predict(pred_features)
    ], axis=0)
    
    results['confidence_lower'] = results['predicted_points_2025'] - 1.96 * predictions_std
    results['confidence_upper'] = results['predicted_points_2025'] + 1.96 * predictions_std

    fig_confidence = px.scatter(results.head(10), 
                              x='team', 
                              y='predicted_points_2025',
                              error_y='confidence_upper',
                              error_y_minus='confidence_lower',
                              title='Predicciones con Intervalos de Confianza')
    fig_confidence.write_html('intervalos_confianza.html')

    return results.sort_values('predicted_points_2025', ascending=False)


def main(file_path):
    """
    Función principal que ejecuta todo el proceso
    """
    print("1. Cargando datos...")
    df = load_and_prepare_data(file_path)

    print("\n2. Limpiando y transformando datos...")
    seasonal_stats = clean_and_transform_data(df)

    print("\n3. Preparando datos para Machine Learning...")
    X_train, X_test, y_train, y_test, features = prepare_for_ml(seasonal_stats)

    print("\n4. Entrenando y evaluando modelos...")
    rf_model, lr_model, rf_score, lr_score, rf_rmse, lr_rmse = train_and_evaluate_models(
        X_train, X_test, y_train, y_test)
    print(f"Random Forest - R² Score: {rf_score:.4f}, RMSE: {rf_rmse:.2f}")
    print(f"Linear Regression - R² Score: {lr_score:.4f}, RMSE: {lr_rmse:.2f}")

    print("\n5. Prediciendo resultados para 2025...")
    predictions_2025 = predict_2025_performance(rf_model, lr_model, seasonal_stats, features)
    print("\nTop 5 equipos predichos para 2025:")
    print(predictions_2025.head().to_string(index=False))

    # Guardar predicciones
    predictions_2025.to_csv('predicciones_2025.csv', index=False)
    print("\nPredicciones guardadas en 'predicciones_2025.csv'")

    print("\n6. Generando visualizaciones...")
    print("- Evaluación completa: evaluacion_modelos.png")
    print("- Métricas de evaluación: metricas_evaluacion.csv")
    print("- Gráfico interactivo: predicciones_2025.html")
    print("- Intervalos de confianza: intervalos_confianza.html")

    return predictions_2025


# Ejecutar el análisis
if __name__ == "__main__":
    predictions = main('matches.csv')
