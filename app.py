import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from flask import Flask, render_template, request, jsonify
import pickle
import json

# Flask app initialization
app = Flask(__name__)

# Path for saving models
MODEL_DIR = 'models'
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# Function to generate synthetic movie data
def generate_movie_data(n_samples=1000):
    np.random.seed(42)
    
    # Features
    years = np.random.randint(1950, 2025, n_samples)
    budgets = np.random.normal(50, 30, n_samples) * 1000000  # Budget in millions
    durations = np.random.normal(120, 30, n_samples)  # Duration in minutes
    
    # Genre encoding (one-hot)
    genres = ['Action', 'Comedy', 'Drama', 'Sci-Fi', 'Horror', 'Romance', 'Thriller']
    genre_matrix = np.zeros((n_samples, len(genres)))
    
    for i in range(n_samples):
        # Each movie has 1-3 genres
        num_genres = np.random.randint(1, 4)
        genre_indices = np.random.choice(len(genres), num_genres, replace=False)
        genre_matrix[i, genre_indices] = 1
    
    # Actor popularity (scale 1-10)
    actor_popularity = np.random.normal(7, 2, n_samples)
    actor_popularity = np.clip(actor_popularity, 1, 10)
    
    # Director experience (years)
    director_experience = np.random.normal(15, 8, n_samples)
    director_experience = np.clip(director_experience, 1, 40)
    
    # Create the target variable (rating from 1-10)
    # Complex rating formula based on features
    base_rating = 5.0
    year_effect = (years - 1950) * 0.01
    budget_effect = budgets / 10000000 * 0.1
    duration_effect = (durations - 90) * 0.01
    genre_effect = np.sum(genre_matrix * np.array([0.2, 0.1, 0.3, 0.2, -0.1, 0.1, 0.0]), axis=1)
    actor_effect = (actor_popularity - 5) * 0.3
    director_effect = (director_experience - 10) * 0.02
    
    # Add some noise
    noise = np.random.normal(0, 0.7, n_samples)
    
    ratings = base_rating + year_effect + budget_effect + duration_effect + genre_effect + actor_effect + director_effect + noise
    ratings = np.clip(ratings, 1, 10)  # Clamp ratings to 1-10 range
    
    # Combine features
    features = np.column_stack([years, budgets, durations, genre_matrix, actor_popularity, director_experience])
    
    # Create feature names
    feature_names = ['year', 'budget', 'duration'] + genres + ['actor_popularity', 'director_experience']
    
    # Create DataFrame
    df = pd.DataFrame(features, columns=feature_names)
    df['rating'] = ratings
    
    # Generate movie titles
    adjectives = ['Amazing', 'Dark', 'Eternal', 'Lost', 'Shining', 'Silent', 'Frozen', 'Hidden', 'Raging', 'Mystic']
    nouns = ['Dreams', 'Hero', 'Journey', 'Legend', 'Storm', 'Forest', 'Future', 'Paradise', 'Secrets', 'Knight']
    
    titles = []
    for i in range(n_samples):
        adj = adjectives[np.random.randint(0, len(adjectives))]
        noun = nouns[np.random.randint(0, len(nouns))]
        titles.append(f"The {adj} {noun}")
    
    df['title'] = titles
    
    return df

# Function to train and save models
def train_models(df):
    # Prepare data
    X = df.drop(['rating', 'title'], axis=1)
    y = df['rating']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler
    with open(os.path.join(MODEL_DIR, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    
    # Define models
    models = {
        'linear_regression': LinearRegression(),
        'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'svr': SVR(kernel='rbf', C=100, gamma=0.1)
    }
    
    # Train and evaluate models
    results = {}
    feature_names = X.columns.tolist()
    
    for name, model in models.items():
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        
        # Evaluate model
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Store results
        results[name] = {
            'mse': mse,
            'mae': mae,
            'r2': r2
        }
        
        # Save model
        with open(os.path.join(MODEL_DIR, f'{name}.pkl'), 'wb') as f:
            pickle.dump(model, f)
    
    # Save feature names
    with open(os.path.join(MODEL_DIR, 'feature_names.json'), 'w') as f:
        json.dump(feature_names, f)
    
    # Save model performance metrics
    with open(os.path.join(MODEL_DIR, 'model_results.json'), 'w') as f:
        json.dump(results, f)
    
    return results, feature_names

# Function to make predictions
def predict_rating(input_data, model_name):
    # Load scaler
    with open(os.path.join(MODEL_DIR, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    
    # Load model
    with open(os.path.join(MODEL_DIR, f'{model_name}.pkl'), 'rb') as f:
        model = pickle.load(f)
    
    # Load feature names
    with open(os.path.join(MODEL_DIR, 'feature_names.json'), 'r') as f:
        feature_names = json.load(f)
    
    # Create input DataFrame
    input_df = pd.DataFrame([input_data], columns=feature_names)
    
    # Scale input
    input_scaled = scaler.transform(input_df)
    
    # Make prediction
    prediction = model.predict(input_scaled)[0]
    
    # Ensure prediction is within range
    prediction = min(max(prediction, 1), 10)
    
    return prediction

# Flask routes
@app.route('/')
def index():
    # Load model results
    try:
        with open(os.path.join(MODEL_DIR, 'model_results.json'), 'r') as f:
            results = json.load(f)
    except FileNotFoundError:
        results = None
    
    return render_template('index.html', model_results=results)

@app.route('/generate_data', methods=['POST'])
def generate_data_route():
    n_samples = int(request.form.get('n_samples', 1000))
    df = generate_movie_data(n_samples)
    
    # Save data to CSV
    df.to_csv('movie_data.csv', index=False)
    
    # Train and save models
    results, _ = train_models(df)
    
    return jsonify({
        'success': True,
        'message': f'Generated {n_samples} samples and trained models',
        'results': results
    })

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    year = int(request.form.get('year'))
    budget = float(request.form.get('budget')) * 1000000  # Convert to actual budget
    duration = float(request.form.get('duration'))
    
    # Genre selection (one-hot encoding)
    genres = ['Action', 'Comedy', 'Drama', 'Sci-Fi', 'Horror', 'Romance', 'Thriller']
    genre_values = [1 if request.form.get(genre.lower(), 'off') == 'on' else 0 for genre in genres]
    
    actor_popularity = float(request.form.get('actor_popularity'))
    director_experience = float(request.form.get('director_experience'))
    
    # Combine all inputs
    input_data = [year, budget, duration] + genre_values + [actor_popularity, director_experience]
    
    # Get selected model
    model_name = request.form.get('model', 'random_forest')
    
    # Make prediction
    prediction = predict_rating(input_data, model_name)
    
    return jsonify({
        'rating': round(prediction, 2),
        'model': model_name
    })

# Function to initialize the application
def initialize_app():
    if not os.path.exists('movie_data.csv'):
        print("Generating initial movie data...")
        df = generate_movie_data(1000)
        df.to_csv('movie_data.csv', index=False)
        print("Training initial models...")
        train_models(df)
        print("Initialization complete.")

if __name__ == '__main__':
    # Run initialization before starting the app
    initialize_app()
    app.run(debug=True)
