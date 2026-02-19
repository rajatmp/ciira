import os
import datetime
import numpy as np
import pandas as pd
import joblib
import sqlite3
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    session,
    send_from_directory,
    jsonify
)
from flask_login import (
    LoginManager,
    UserMixin,
    login_user,
    logout_user,
    login_required,
    current_user
)
from werkzeug.security import generate_password_hash, check_password_hash
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier


# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_super_secret_key_here'  # IMPORTANT: Change this to a strong, random key in production
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MODEL_FOLDER'] = 'models'
app.config['ALLOWED_EXTENSIONS'] = {'csv'}

# Database configuration
DATABASE = 'users.db'
ACCURACIES_FILE = os.path.join(app.config['MODEL_FOLDER'], 'accuracies.joblib') # File to store accuracies

# Ensure upload and model folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['MODEL_FOLDER'], exist_ok=True)

# Flask-Login setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Database connection helper
def get_db_connection():
    """Establishes a connection to the SQLite database."""
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row  # This allows accessing columns by name
    return conn

# Initialize database schema
def init_db():
    """Initializes the database schema if tables do not exist."""
    conn = sqlite3.connect(DATABASE)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

# Call init_db when the application starts
with app.app_context():
    init_db()

# User class for Flask-Login, now interacting with SQLite
class User(UserMixin):
    def __init__(self, id, name, email, password_hash):
        self.id = id
        self.name = name
        self.email = email
        self.password_hash = password_hash

    def get_id(self):
        return str(self.id)

    @staticmethod
    def get(user_id):
        conn = get_db_connection()
        user_data = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
        conn.close()
        if user_data:
            # Explicitly cast to str to prevent TypeError with Undefined
            return User(user_data['id'], str(user_data['name']), str(user_data['email']), user_data['password'])
        return None

    @staticmethod
    def get_by_email(email):
        conn = get_db_connection()
        user_data = conn.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()
        conn.close()
        if user_data:
            # Explicitly cast to str to prevent TypeError with Undefined
            return User(user_data['id'], str(user_data['name']), str(user_data['email']), user_data['password'])
        return None

@login_manager.user_loader
def load_user(user_id):
    return User.get(user_id)


# Helper function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Model loading and preprocessing functions
def load_model_components(algorithm):
    try:
        model_path = os.path.join(app.config['MODEL_FOLDER'], f'{algorithm}_model.joblib')
        scaler_path = os.path.join(app.config['MODEL_FOLDER'], f'{algorithm}_scaler.joblib')
        features_path = os.path.join(app.config['MODEL_FOLDER'], f'{algorithm}_features.joblib')

        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        features = joblib.load(features_path)
        return model, scaler, features
    except FileNotFoundError:
        flash(f'Error: Model components for {algorithm.replace("_", " ").title()} not found. Please train the model first.', 'danger')
        return None, None, None
    except Exception as e:
        flash(f'Error loading model components: {e}', 'danger')
        return None, None, None

def preprocess_data(df, features, scaler):
    # Ensure all required features are present
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        # Fill missing features with 0 or a suitable default
        for f in missing_features:
            df[f] = 0
        flash(f"Warning: Missing features {missing_features} in uploaded file. Filling with 0.", 'warning')

    # Select and reorder columns according to training features
    X = df[features]
    
    # Scale the data
    X_scaled = scaler.transform(X)
    return X_scaled

# Function to load accuracies from file
def load_accuracies():
    if os.path.exists(ACCURACIES_FILE):
        try:
            return joblib.load(ACCURACIES_FILE)
        except Exception as e:
            print(f"Error loading accuracies: {e}")
            return {} # Return empty dict on error
    return {} # Return empty dict if file doesn't exist

# Function to save accuracies to file
def save_accuracies(accuracies_dict):
    try:
        joblib.dump(accuracies_dict, ACCURACIES_FILE)
    except Exception as e:
        print(f"Error saving accuracies: {e}")

# Route for login
@app.route('/', methods=['GET', 'POST'])
@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    if request.method == 'POST':
        email = request.form['email'] # Changed from 'username' to 'email'
        password = request.form['password']
        user = User.get_by_email(email) # Fetch user by email
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            flash('Logged in successfully.', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid email or password.', 'danger')
    # Pass a default 'Guest' username if not logged in
    return render_template('homepage.html', username='Guest')

# Route for user registration
@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        flash('You are already logged in.', 'info')
        return redirect(url_for('home'))

    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirmPassword'] # Corrected field name from homepage.html

        if not name or not email or not password or not confirm_password:
            flash('All fields are required.', 'danger')
            return render_template('homepage.html', username='Guest') # Pass username
            
        if password != confirm_password:
            flash('Passwords do not match.', 'danger')
            return render_template('homepage.html', username='Guest') # Pass username

        conn = None
        try:
            conn = get_db_connection()
            # Check if user already exists
            existing_user = conn.execute('SELECT id FROM users WHERE email = ?', (email,)).fetchone()
            if existing_user:
                flash('Email already registered. Please use a different email.', 'danger')
                return render_template('homepage.html', username='Guest') # Pass username

            hashed_password = generate_password_hash(password)
            conn.execute('INSERT INTO users (name, email, password) VALUES (?, ?, ?)',
                         (name, email, hashed_password))
            conn.commit()
            flash('Registration successful! Please log in.', 'success')
            # After successful registration, redirect to login page (which is homepage.html)
            # This will trigger the modal to open on login form.
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('An account with this email already exists.', 'danger')
            return render_template('homepage.html', username='Guest') # Pass username
        except Exception as e:
            flash(f'An error occurred during registration: {str(e)}', 'danger')
            print(f"Registration error: {e}") # For debugging
            return render_template('homepage.html', username='Guest') # Pass username
        finally:
            if conn:
                conn.close()
    
    # For GET request to /register, just render the homepage (which contains the modals)
    return render_template('homepage.html', username='Guest')


# Route for logout
@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

# Route for home/dashboard
@app.route('/homee')
@app.route('/home')
@login_required
def home():
    # Pass current_user's name or email to the template
    user_display_name = 'Guest'
    if current_user.is_authenticated:
        if hasattr(current_user, 'name') and current_user.name is not None: # Added is not None check
            user_display_name = str(current_user.name)
        elif hasattr(current_user, 'email') and current_user.email is not None: # Added is not None check
            user_display_name = str(current_user.email)
        else:
            user_display_name = 'Authenticated User' # Fallback for authenticated user without name/email
    return render_template('homee.html', username=user_display_name)

# Route for data upload
@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload_file():
    user_display_name = 'Guest'
    if current_user.is_authenticated:
        if hasattr(current_user, 'name') and current_user.name is not None: # Added is not None check
            user_display_name = str(current_user.name)
        elif hasattr(current_user, 'email') and current_user.email is not None: # Added is not None check
            user_display_name = str(current_user.email)
        else:
            user_display_name = 'Authenticated User'
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part', 'danger')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file', 'danger')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            session['uploaded_file'] = filepath
            flash('File uploaded successfully!', 'success')
            return redirect(url_for('model_page'))
        else:
            flash('Invalid file type. Please upload a CSV file.', 'danger')
    return render_template('upload.html', username=user_display_name)

# Route for model selection/training
@app.route('/model', methods=['GET', 'POST'])
@login_required
def model_page():
    user_display_name = 'Guest'
    if current_user.is_authenticated:
        if hasattr(current_user, 'name') and current_user.name is not None: # Added is not None check
            user_display_name = str(current_user.name)
        elif hasattr(current_user, 'email') and current_user.email is not None: # Added is not None check
            user_display_name = str(current_user.email)
        else:
            user_display_name = 'Authenticated User'

    # Load existing accuracies at the beginning of the function
    accuracies = load_accuracies()
    # Ensure all expected algorithms are present, initialize to 0 if not found in loaded data
    for algo in [
    'random_forest', 'decision_tree', 'k_neighbors', 'logistic_regression',
    'gradient_boosting', 'svm', 'extra_trees_classifier', 'deep_neural_network(mlp)'
]:
        if algo not in accuracies:
            accuracies[algo] = 0

    if request.method == 'POST':
        algorithm = request.form.get('algorithm')
        if not algorithm:
            flash('Please select an algorithm.', 'danger')
            # Pass existing accuracies even on error
            return render_template('model.html', username=user_display_name, accuracies=accuracies)

        filepath = session.get('uploaded_file')
        if not filepath or not os.path.exists(filepath):
            flash('No CSV file uploaded or file not found. Please upload a file first.', 'danger')
            return redirect(url_for('upload_file'))

        try:
            df = pd.read_csv(filepath, low_memory=False) # Ensure low_memory=False for larger datasets
            
            # --- Feature Engineering (as per previous context if needed) ---
            # This part would typically be based on your dataset's specifics.
            # Example:
            if 'ip.len' in df.columns:
                df['http_content_length'] = df['ip.len'] - 40 # Assuming typical IP/TCP header size for HTTP content length approx
            if 'tcp.seq' in df.columns and 'tcp.ack' in df.columns:
                 # Simple duration / packet count proxies if actual duration/packet columns are missing
                df['duration_approx'] = df['tcp.seq'].diff().fillna(0).abs()
                df['packet_count_approx'] = df['tcp.ack'].diff().fillna(0).abs() # or just a count
            
            # Define features and target based on common network traffic datasets
            # ADJUST THESE COLUMN NAMES TO MATCH YOUR ACTUAL CSV FILE'S COLUMNS
            # AND YOUR MODEL'S EXPECTED FEATURES AND TARGET
            target_column = 'label' # Or 'attack_type', 'Class', etc.
            
            # Common features, adjust based on your dataset's actual columns
            common_features = [
                'src_port', 'dst_port', 'tcp_seq', 'tcp_len',
                'http_content_length', 'tcp_ack', 'tcp_raw_ack',
                # Add other features from your dataset if available and relevant
                # 'duration', 'protocol', 'flow_bytes_s', 'flow_packets_s',
                # 'fwd_pkt_len_max', 'bwd_pkt_len_max', 'fin_flag_count', 'syn_flag_count',
                # 'rst_flag_count', 'psh_flag_count', 'ack_flag_count', 'urg_flag_count',
                # 'cwe_flag_count', 'ece_flag_count'
            ]
            
            # Filter features that are actually in the DataFrame
            features = [col for col in common_features if col in df.columns]

            if target_column not in df.columns:
                flash(f'Error: Target column "{target_column}" not found in the uploaded CSV.', 'danger')
                return redirect(url_for('upload_file'))
            if not features:
                flash('Error: No common features found in the uploaded CSV. Please check column names or upload a different dataset.', 'danger')
                return redirect(url_for('upload_file'))

            X = df[features]
            y = df[target_column]

            # Handle non-numeric data in feature columns by converting to numeric where possible, or dropping
            for col in X.columns:
                if X[col].dtype == 'object':
                    try:
                        X[col] = pd.to_numeric(X[col], errors='coerce')
                    except ValueError:
                        # If truly non-numeric and not convertible, consider one-hot encoding or dropping
                        flash(f"Warning: Non-numeric data in column '{col}'. This column might be dropped or treated as 0 for model training.", 'warning')
                        X[col] = 0 # Simple handling for demo; advanced would be encoding or dropping
            X = X.fillna(0) # Fill any NaN values introduced by coerce

            # Initialize and train StandardScaler
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Initialize and train the selected model
            model = None
            if algorithm == 'random_forest':    
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            elif algorithm == 'decision_tree':
                model = DecisionTreeClassifier(random_state=42)
            elif algorithm == 'extra_trees_classifier':
                model = ExtraTreesClassifier(n_estimators=100, random_state=42)
            elif algorithm == 'deep_neural_network(mlp)':
                model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
            elif algorithm == 'k_neighbors':
                model = KNeighborsClassifier(n_neighbors=5)
            elif algorithm == 'logistic_regression':
                model = LogisticRegression(max_iter=1000, random_state=42)
            elif algorithm == 'gradient_boosting': # Assuming this is the value from your select
                model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
            elif algorithm == 'svm': # Assuming this is the value from your select
                model = SVC(probability=True, random_state=42) # probability=True is needed for predict_proba
            else:
                flash('Invalid algorithm selected.', 'danger')
                # Pass existing accuracies even on error
                return render_template('model.html', username=user_display_name, accuracies=accuracies)

            if model:
                model.fit(X_scaled, y)

                # Calculate accuracy and store it
                accuracy = model.score(X_scaled, y) * 100 # Example accuracy
                accuracies[algorithm] = round(accuracy, 2) # Update the dictionary
                save_accuracies(accuracies) # Save updated accuracies

                # Save the trained model, scaler, and feature list
                joblib.dump(model, os.path.join(app.config['MODEL_FOLDER'], f'{algorithm}_model.joblib'))
                joblib.dump(scaler, os.path.join(app.config['MODEL_FOLDER'], f'{algorithm}_scaler.joblib'))
                joblib.dump(features, os.path.join(app.config['MODEL_FOLDER'], f'{algorithm}_features.joblib')) # Save features list

                flash(f'{algorithm.replace("_", " ").title()} model trained and saved successfully!', 'success')
                return redirect(url_for('predict_page', algorithm=algorithm)) # Pass algorithm to predict page
        except pd.errors.EmptyDataError:
            flash('Error: The uploaded CSV file is empty.', 'danger')
            return redirect(url_for('upload_file'))
        except KeyError as e:
            flash(f'Error: Missing expected column for training. Please check your CSV file. {e}', 'danger')
            return redirect(url_for('upload_file'))
        except Exception as e:
            flash(f'Error training model: {str(e)}', 'danger')
            print(f"Error training model: {e}") # Log error for debugging
            # Pass existing accuracies even on error
            return render_template('model.html', username=user_display_name, accuracies=accuracies)

    # Always pass accuracies to the template for GET requests
    return render_template('model.html', username=user_display_name, accuracies=accuracies)

# Route for prediction
@app.route('/predict', methods=['GET'])
@login_required
def predict_page():
    """
    Renders the prediction page.
    If an algorithm is selected (from model.html), it proceeds with batch prediction.
    Otherwise, it shows the real-time prediction form.
    """
    algorithm = request.args.get('algorithm')
    records = []
    filename = None
    # NEW: Get threat_only flag from URL. Defaults to 'false' if not present.
    threat_only = request.args.get('threat_only', 'false').lower() == 'true'
    # NEW: Get top_n parameter from URL. Defaults to 10 if not provided or invalid.
    top_n_str = request.args.get('top_n')
    top_n = 10  # Default to 10 rows
    if top_n_str and top_n_str.isdigit():
        top_n = int(top_n_str)
    
    user_display_name = 'Guest'
    if current_user.is_authenticated:
        if hasattr(current_user, 'name') and current_user.name is not None: # Added is not None check
            user_display_name = str(current_user.name)
        elif hasattr(current_user, 'email') and current_user.email is not None: # Added is not None check
            user_display_name = str(current_user.email)
        else:
            user_display_name = 'Authenticated User'

    if algorithm and 'uploaded_file' in session:
        filepath = session['uploaded_file']
        try:
            df = pd.read_csv(filepath, low_memory=False) # Corrected: Use False instead of false for boolean

            # Load model components
            model, scaler, features = load_model_components(algorithm)

            if model is None or scaler is None or features is None:
                # Flash message is already handled in load_model_components
                return redirect(url_for('model_page'))

            # Ensure all required features are present in the dataframe
            # And fill missing features with 0 (or a suitable value) before preprocessing
            for f in features:
                if f not in df.columns:
                    df[f] = 0
                    # You might want to flash a warning here too, or decide on a more robust default.
                    # flash(f"Warning: Missing feature '{f}' in prediction data. Filled with 0.", 'warning')

            X_scaled = preprocess_data(df.copy(), features, scaler)

            print(f"Number of records after preprocessing: {X_scaled.shape[0]}")

            # Perform prediction
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(X_scaled)
                max_probs = np.max(probs, axis=1)
                predicted_labels = model.predict(X_scaled)
                df['prediction'] = [
                    f"{label} ({prob:.1%})" for label, prob in zip(predicted_labels, max_probs)
                ]
                for i, label in enumerate(model.classes_):
                    df[f'prob_{label}'] = probs[:, i]
            else:
                df['prediction'] = model.predict(X_scaled)
            
            # DEBUG: Print all unique predictions to help identify benign labels
            print("--- DEBUG: All unique predictions before filtering ---")
            print(df['prediction'].unique())
            print("--- End DEBUG ---")

            # NEW: Filter for threats if threat_only is true
            if threat_only:
                # ADJUST THIS LINE: Ensure 'Benign' and 'benign' match your model's exact non-threat output labels.
                # E.g., if your model outputs 'No_Attack', change to 'No_Attack' not in x.lower()
                df_filtered = df[df['prediction'].apply(lambda x: 'Benign' not in x.lower() and 'benign' not in x.lower())] 
                
                print("--- DEBUG: Filtered predictions (threats only) ---")
                print(df_filtered['prediction'].tolist())
                print("--- End DEBUG ---")
                
                df_to_display = df_filtered # Use filtered DataFrame
            else:
                df_to_display = df # Use original DataFrame

            # NEW: Apply top_n limit if specified
            if top_n is not None:
                df_to_display = df_to_display.head(top_n)

            records = df_to_display.to_dict(orient='records') # Convert the (potentially filtered and limited) DataFrame to records

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"prediction_output_{timestamp}.csv"
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_file)
            df.to_csv(output_path, index=False) # Always save the full original prediction output

            flash(f'Batch prediction complete using {algorithm.replace("_", " ").title()} algorithm.', 'success')
            filename = output_file
            session.pop('uploaded_file', None) # Clear uploaded file from session after processing

        except FileNotFoundError as e:
            flash(f'Error: {e}. Please ensure the model files exist and the CSV file is correctly uploaded.', 'danger')
            return redirect(url_for('model_page'))
        except pd.errors.EmptyDataError:
            flash('Error: The uploaded CSV file is empty.', 'danger')
            return redirect(url_for('upload_file'))
        except Exception as e:
            flash(f'Batch prediction error: {str(e)}', 'danger')
            print(f"Batch prediction error: {e}") # Log error for debugging
            return redirect(url_for('model_page'))

    # Pass the threat_only flag to the template to maintain state for the checkbox
    # Also pass top_n to the template if you want to reflect it in the UI (e.g., in a text field)
    return render_template('predict.html', algorithm=algorithm, records=records, filename=filename, username=user_display_name, threat_only=threat_only, top_n=top_n)


# Route for real-time prediction (example, needs to match your form fields)
@app.route('/predict_realtime', methods=['POST'])
@login_required
def predict_realtime():
    # Placeholder for real-time prediction logic
    # You would get input from request.form, preprocess it,
    # load a pre-selected model, and make a prediction.
    # For a simple demo, we'll just return a dummy response.

    data = {
        'src_port': float(request.form['src_port']),
        'dst_port': float(request.form['dst_port']),
        'tcp_seq': float(request.form['tcp_seq']),
        'tcp_len': float(request.form['tcp_len']),
        'http_content_length': float(request.form['http_content_length']),
        'tcp_ack': float(request.form['tcp_ack']),
        'tcp_raw_ack': float(request.form['tcp_raw_ack'])
    }
    
    # Assuming you want to use the last used algorithm or a default one
    # For a real-time prediction, you'd likely pick a default or let the user choose.
    # For simplicity, we'll just use a dummy prediction.
    
    # You would ideally load a model here, preprocess 'data', and make a prediction
    # model, scaler, features = load_model_components('random_forest') # Example
    # if model and scaler and features:
    #     input_df = pd.DataFrame([data])
    #     X_scaled = preprocess_data(input_df, features, scaler)
    #     prediction = model.predict(X_scaled)[0]
    #     if hasattr(model, 'predict_proba'):
    #         prob = np.max(model.predict_proba(X_scaled)[0])
    #         prediction_text = f"{prediction} ({prob:.1%})"
    #     else:
    #         prediction_text = str(prediction)
    # else:
    #     prediction_text = "Model not loaded, cannot predict."
    
    prediction_text = "Sample Prediction: Benign (99.5%)" # Dummy prediction
    if data['src_port'] > 50000 or data['dst_port'] > 50000: # Simple dummy logic for "threat"
        prediction_text = "Sample Prediction: DDoS Attack (85.2%)"

    user_display_name = 'Guest'
    if current_user.is_authenticated:
        if hasattr(current_user, 'name') and current_user.name is not None: # Added is not None check
            user_display_name = str(current_user.name)
        elif hasattr(current_user, 'email') and current_user.email is not None: # Added is not None check
            user_display_name = str(current_user.email)
        else:
            user_display_name = 'Authenticated User'

    flash(f'Real-time prediction: {prediction_text}', 'info')
    return render_template('predict.html', realtime_prediction=True, username=user_display_name, records=[{'Prediction': prediction_text}])


# Route for downloading prediction output
@app.route('/download/<filename>')
@login_required
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)