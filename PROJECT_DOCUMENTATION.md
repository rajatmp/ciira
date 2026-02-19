# CIRA on IIoT - Cybersecurity Intrusion Recognition and Analysis for Industrial IoT

## 📋 Project Overview

**CIRA on IIoT** is a comprehensive web-based cybersecurity threat detection system specifically designed for Industrial Internet of Things (IIoT) environments. The platform leverages multiple machine learning algorithms to analyze network traffic patterns and identify potential cyber threats in real-time.

### Key Features

- **Multi-Algorithm Support**: Implements 8 different machine learning algorithms for threat detection
- **Real-time Threat Detection**: Analyzes network traffic data instantly
- **Batch Processing**: Upload CSV files for bulk threat analysis
- **Interactive Dashboard**: Modern, responsive UI with dark/light mode support
- **User Authentication**: Secure login and registration system
- **Wireshark Integration**: Process and analyze Wireshark packet captures
- **Severity Classification**: Categorizes threats by severity levels (low, medium, high, very high)

---

## 🏗️ Project Architecture

### Technology Stack

**Backend:**
- Python 3.x
- Flask (Web Framework)
- Flask-Login (Authentication)
- SQLite (User Database)

**Machine Learning:**
- scikit-learn (ML Models)
- pandas & numpy (Data Processing)
- joblib (Model Serialization)
- TensorFlow/Keras (Deep Learning Model)

**Frontend:**
- HTML5, CSS3, JavaScript
- Tailwind CSS
- AOS (Animate On Scroll)
- Font Awesome Icons
- Custom CSS Animations

### Directory Structure

```
CODE/
├── APP/
│   ├── app.py                          # Main Flask application
│   ├── prepare_wireshark_data.py       # Wireshark data preprocessing
│   ├── show_columns.py                 # Utility script
│   ├── users.db                        # SQLite user database
│   ├── models/                         # Trained ML models (68 files)
│   │   ├── *_model.joblib             # Trained model files
│   │   ├── *_scaler.joblib            # Feature scalers
│   │   ├── *_features.joblib          # Feature lists
│   │   ├── *_label_binarizer.joblib   # Label encoders
│   │   └── deep_model.h5              # Deep learning model
│   ├── templates/                      # HTML templates
│   │   ├── homepage.html              # Landing/login page
│   │   ├── homee.html                 # Dashboard
│   │   ├── upload.html                # Data upload interface
│   │   ├── model.html                 # Model selection/training
│   │   └── predict.html               # Threat detection interface
│   ├── uploads/                        # User uploaded files
│   └── *.csv                          # Training/test datasets
├── JUPYTER/
│   ├── IIOT_Cyber_model.ipynb         # Model training notebook
│   └── *.csv                          # Dataset files
└── models/                             # Additional model artifacts
    └── graphs/                         # Performance visualizations
```

---

## 🤖 Machine Learning Models

The system implements **8 different machine learning algorithms**, each with two variants:
1. **Attack Type Classification** - Identifies specific attack types
2. **Severity Classification** - Categorizes threat severity levels

### Supported Algorithms

| Algorithm | Type | Use Case |
|-----------|------|----------|
| **Random Forest** | Ensemble | High accuracy, handles non-linear patterns |
| **Decision Tree** | Tree-based | Interpretable, fast predictions |
| **Extra Trees** | Ensemble | Reduces overfitting, robust |
| **Gradient Boosting** | Ensemble | Sequential learning, high performance |
| **K-Neighbors (KNN)** | Instance-based | Pattern matching, simple |
| **Logistic Regression** | Linear | Fast, baseline model |
| **SVM** | Kernel-based | Effective in high-dimensional spaces |
| **MLP (Neural Network)** | Deep Learning | Complex pattern recognition |

### Attack Types Detected

1. **Normal** - Benign traffic (Severity: Low)
2. **DDoS_TCP** - Distributed Denial of Service attacks (Severity: Medium)
3. **Password** - Password-based attacks (Severity: High)
4. **Port_Scanning** - Network reconnaissance (Severity: Very High)

---

## 📊 Data Features

The system analyzes network traffic using **7 key features**:

| Feature | Description | Source |
|---------|-------------|--------|
| `tcp.srcport` | Source port number | TCP header |
| `tcp.dstport` | Destination port number | TCP header |
| `tcp.len` | TCP segment length | TCP header |
| `http.content_length` | HTTP content size | HTTP header |
| `tcp.seq` | TCP sequence number | TCP header |
| `tcp.ack` | TCP acknowledgment number | TCP header |
| `tcp.raw_ack` | Raw TCP ACK value | TCP header |

### Data Preprocessing

1. **Feature Extraction**: Extracts relevant fields from network packets
2. **Normalization**: Standardizes features using StandardScaler
3. **Missing Value Handling**: Fills missing values with 0
4. **Type Conversion**: Converts non-numeric data to numeric format

---

## 🔐 Security Features

### User Authentication
- **Registration System**: Secure user account creation
- **Password Hashing**: Uses Werkzeug's `generate_password_hash`
- **Session Management**: Flask-Login for secure sessions
- **SQLite Database**: Stores user credentials securely

### Database Schema

```sql
CREATE TABLE users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    email TEXT UNIQUE NOT NULL,
    password TEXT NOT NULL
);
```

---

## 🚀 Application Workflow

### 1. User Authentication
```
Login/Register → Session Creation → Dashboard Access
```

### 2. Data Upload
```
Upload CSV → Validate Format → Store in uploads/ → Proceed to Model Selection
```

### 3. Model Training
```
Select Algorithm → Load Dataset → Train Model → Save Artifacts → Display Accuracy
```

### 4. Threat Detection

**Batch Prediction:**
```
Upload CSV → Preprocess Data → Apply Model → Generate Results → Download Report
```

**Real-time Prediction:**
```
Enter Parameters → Validate Input → Predict → Display Result
```

---

## 🎨 User Interface

### Pages

1. **Homepage (homepage.html)**
   - Landing page with login/registration modals
   - Animated HUD-style graphics
   - Feature showcase
   - Dark/Light mode toggle

2. **Dashboard (homee.html)**
   - System overview
   - Quick access to features
   - User profile display
   - Navigation menu

3. **Upload (upload.html)**
   - Drag-and-drop file upload
   - CSV format validation
   - File preview
   - Progress indication

4. **Model Selection (model.html)**
   - Algorithm comparison
   - Training interface
   - Accuracy display
   - Model management

5. **Threat Detection (predict.html)**
   - Batch prediction results table
   - Real-time detection form
   - Filtering options (threats only, top N results)
   - Download results as CSV

### Design Features

- **Responsive Design**: Works on desktop, tablet, and mobile
- **Dark/Light Mode**: User preference saved in localStorage
- **Animations**: AOS library for smooth scroll animations
- **Sci-Fi Theme**: Cybersecurity-inspired visual design
- **Flash Messages**: User feedback for actions
- **Custom Message Boxes**: Enhanced user notifications

---

## 📈 Model Performance

### Training Process

1. **Data Loading**: Reads CSV with network traffic data
2. **Feature Engineering**: Creates derived features
3. **Train-Test Split**: Splits data for validation
4. **Scaling**: Applies StandardScaler to features
5. **Model Training**: Fits selected algorithm
6. **Evaluation**: Calculates accuracy metrics
7. **Serialization**: Saves model artifacts using joblib

### Model Artifacts

For each algorithm, the system saves:
- `{algorithm}_model.joblib` - Trained model
- `{algorithm}_scaler.joblib` - Feature scaler
- `{algorithm}_features.joblib` - Feature list
- `{algorithm}_label_binarizer.joblib` - Label encoder

---

## 🔧 Wireshark Integration

### Data Preparation Script

The `prepare_wireshark_data.py` script converts Wireshark packet captures to the required format:

**Input:** Wireshark CSV export with columns:
- Source Port
- Destination Port
- TCP Length
- HTTP Content Length
- TCP Sequence
- TCP ACK Number
- TCP Raw ACK Number

**Output:** Prepared CSV with normalized column names matching model requirements

**Process:**
1. Normalize column names (lowercase, replace spaces/dots)
2. Map Wireshark columns to model features
3. Convert to numeric format
4. Fill missing values with 0
5. Save prepared dataset

---

## 🛠️ Installation & Setup

### Prerequisites

```bash
Python 3.8+
pip (Python package manager)
```

### Installation Steps

1. **Clone/Download the project**
   ```bash
   cd CODE/APP
   ```

2. **Install dependencies**
   ```bash
   pip install flask flask-login pandas numpy scikit-learn joblib werkzeug
   ```

3. **Initialize database**
   ```bash
   # Database is auto-created on first run
   python app.py
   ```

4. **Access the application**
   ```
   Open browser: http://localhost:5000
   ```

### Configuration

Edit `app.py` to configure:
- `SECRET_KEY`: Change to a strong random key for production
- `UPLOAD_FOLDER`: Directory for uploaded files
- `MODEL_FOLDER`: Directory for model artifacts
- `DATABASE`: SQLite database file path

---

## 📝 Usage Guide

### For End Users

1. **Register/Login**
   - Create account or login with credentials
   - Credentials stored securely in SQLite

2. **Upload Network Traffic Data**
   - Navigate to "Upload Data"
   - Select CSV file with network traffic
   - Supported format: Wireshark export or prepared CSV

3. **Train/Select Model**
   - Choose from 8 ML algorithms
   - View training accuracy
   - Models saved automatically

4. **Detect Threats**
   - **Batch Mode**: Upload CSV for bulk analysis
   - **Real-time Mode**: Enter parameters manually
   - Filter results (threats only, top N)
   - Download results as CSV

### For Developers

1. **Add New Algorithm**
   ```python
   # In app.py, model_page() function
   elif algorithm == 'new_algorithm':
       model = NewAlgorithm(params)
   ```

2. **Modify Features**
   ```python
   # Update common_features list in model_page()
   common_features = ['feature1', 'feature2', ...]
   ```

3. **Customize UI**
   - Edit HTML templates in `templates/`
   - Modify CSS in `<style>` sections
   - Add JavaScript in `<script>` sections

---

## 🔍 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` or `/login` | GET, POST | Login page |
| `/register` | POST | User registration |
| `/logout` | GET | User logout |
| `/home` or `/homee` | GET | Dashboard |
| `/upload` | GET, POST | File upload |
| `/model` | GET, POST | Model selection/training |
| `/predict` | GET | Batch prediction results |
| `/predict_realtime` | POST | Real-time prediction |
| `/download/<filename>` | GET | Download results |

---

## 📊 Sample Data Format

### Input CSV Format

```csv
tcp.srcport,tcp.dstport,tcp.len,http.content_length,tcp.seq,tcp.ack,tcp.raw_ack,attack_type
52345,80,45,0,1020,150,1020,DDoS_TCP
80,443,68,120,750,1800,900,Normal
9001,22,55,80,600,1100,700,Password
20000,25000,30,0,300,50,300,Port_Scanning
```

### Output Prediction Format

```csv
tcp.srcport,tcp.dstport,tcp.len,http.content_length,tcp.seq,tcp.ack,tcp.raw_ack,prediction
52345,80,45,0,1020,150,1020,DDoS_TCP (85.2%)
80,443,68,120,750,1800,900,Normal (99.5%)
```

---

## 🐛 Troubleshooting

### Common Issues

**Issue:** Model not found error
- **Solution**: Train the model first using the Model Selection page

**Issue:** CSV upload fails
- **Solution**: Ensure CSV has required columns and proper format

**Issue:** Prediction shows all zeros
- **Solution**: Check if features are correctly mapped in preprocessing

**Issue:** Login fails
- **Solution**: Verify database exists and credentials are correct

---

## 🔮 Future Enhancements

- [ ] Real-time packet capture integration
- [ ] Advanced visualization dashboards
- [ ] Multi-user role management (admin, analyst, viewer)
- [ ] API for external integrations
- [ ] Automated model retraining
- [ ] Alert notification system
- [ ] Historical threat analysis
- [ ] Export reports in multiple formats (PDF, Excel)
- [ ] Integration with SIEM systems
- [ ] Support for additional attack types

---

## 📚 Technical Details

### Model Training Parameters

**Random Forest:**
- n_estimators: 100
- random_state: 42

**Gradient Boosting:**
- n_estimators: 100
- learning_rate: 0.1
- max_depth: 3

**MLP (Neural Network):**
- hidden_layer_sizes: (100,)
- max_iter: 300

**SVM:**
- probability: True (for predict_proba)

### Performance Considerations

- **Batch Processing**: Handles large CSV files efficiently
- **Model Caching**: Models loaded once and reused
- **Session Management**: Secure session handling with Flask-Login
- **Database**: SQLite for lightweight deployment

---

## 🤝 Contributing

This project is designed for cybersecurity research and IIoT threat detection. Contributions can include:

- New ML algorithms
- Additional attack type detection
- UI/UX improvements
- Performance optimizations
- Documentation enhancements

---

## ⚠️ Security Considerations

### Production Deployment

1. **Change SECRET_KEY**: Use a strong, random secret key
2. **HTTPS**: Deploy with SSL/TLS certificates
3. **Database**: Consider PostgreSQL for production
4. **Input Validation**: Implement comprehensive validation
5. **Rate Limiting**: Add rate limiting for API endpoints
6. **Logging**: Implement comprehensive logging
7. **Backup**: Regular database and model backups

---

## 📄 License

This project is designed for educational and research purposes in cybersecurity and IIoT threat detection.

---

## 👥 Credits

**Project Name:** CIRA on IIoT (Cybersecurity Intrusion Recognition and Analysis for Industrial IoT)

**Technologies Used:**
- Flask Framework
- scikit-learn
- TensorFlow/Keras
- Tailwind CSS
- Font Awesome
- AOS Animation Library

---

## 📞 Support

For issues, questions, or contributions, please refer to the project repository or contact the development team.

---

**Last Updated:** 2025
**Version:** 1.0
**Status:** Active Development
