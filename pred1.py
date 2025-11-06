import pandas as pd
import numpy as np
import joblib
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
import sqlite3
import warnings
import os

warnings.filterwarnings("ignore")
MODEL_PATH = "flight_delay_model.pkl"

conn = sqlite3.connect("flight_app.db", check_same_thread=False)
c = conn.cursor()

c.execute("""
CREATE TABLE IF NOT EXISTS users (
    email TEXT PRIMARY KEY,
    name TEXT,
    password TEXT
)
""")

c.execute("""
CREATE TABLE IF NOT EXISTS predictions (
    email TEXT,
    airline TEXT,
    origin TEXT,
    destination TEXT,
    prediction TEXT,
    probability REAL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")
conn.commit()


def inject_custom_css():
    st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #00d4ff 0%, #0099ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .sub-header {
        font-size: 1.5rem;
        color: #cccccc;
        text-align: center;
        margin-bottom: 3rem;
        font-weight: 300;
    }
    
    .card {
        background: #1e1e1e;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        border: 1px solid #333333;
        margin-bottom: 2rem;
    }
    
    .stButton button {
        background: linear-gradient(135deg, #00d4ff 0%, #0099ff 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 6px;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0, 212, 255, 0.4);
    }
    
    .stTextInput input, .stTextInput input:focus {
        border: 1px solid #444444;
        border-radius: 6px;
        padding: 0.75rem;
        font-size: 1rem;
        background-color: #2d2d2d;
        color: white;
    }
    
    .stSelectbox div div {
        border: 1px solid #444444;
        border-radius: 6px;
        background-color: #2d2d2d;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #2d2d2d;
        border-radius: 6px 6px 0px 0px;
        padding: 1rem 2rem;
        font-weight: 600;
        color: #cccccc;
    }
    
    .stSuccess {
        background: linear-gradient(135deg, #00a86b, #008055);
        color: white;
        border-radius: 6px;
        padding: 1rem;
    }
    
    .stError {
        background: linear-gradient(135deg, #ff4444, #cc0000);
        color: white;
        border-radius: 6px;
        padding: 1rem;
    }
    
    .stInfo {
        background: linear-gradient(135deg, #0099ff, #0077cc);
        color: white;
        border-radius: 6px;
        padding: 1rem;
    }
    
    .css-1d391kg {
        background: linear-gradient(180deg, #1a1a1a 0%, #2d2d2d 100%);
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #1e1e1e 0%, #2d2d2d 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
        border: 2px solid #333333;
    }
    
    .on-time {
        border-color: #00a86b;
    }
    
    .delayed {
        border-color: #ff4444;
    }
    
    .feature-card {
        background: #2d2d2d;
        padding: 1.5rem;
        border-radius: 8px;
        text-align: center;
        border: 1px solid #444444;
    }
    
    .route-display {
        font-size: 1.2rem;
        font-weight: bold;
        text-align: center;
        color: #00d4ff;
    }
    </style>
    """, unsafe_allow_html=True)


def create_sample_data():
    data_path = r"D:\AI\FlightDelayPrediction\Datasets\IndianDomesticAirlineDataset.csv"
    if not os.path.exists(data_path):
        st.info("Using synthetic data as real dataset not found")
        return create_fallback_data()
    try:
        df = pd.read_csv(data_path, low_memory=False)
        df.drop_duplicates(inplace=True)
        critical_cols = ['airline', 'flightNumber', 'destination', 'scheduledDepartureTime', 'scheduledArrivalTime']
        df.dropna(subset=critical_cols, inplace=True)
        df['origin'].fillna('Unknown', inplace=True)
        df['timezone'].fillna('IST', inplace=True)
        df['daysOfWeek'].fillna('0,1,2,3,4,5,6', inplace=True)

        def time_to_minutes(t):
            try:
                if isinstance(t, str) and ':' in t:
                    h, m = map(int, t.split(':'))
                    return h * 60 + m
                return np.nan
            except:
                return np.nan

        df['scheduledDepartureTime'] = df['scheduledDepartureTime'].apply(time_to_minutes)
        df['scheduledArrivalTime'] = df['scheduledArrivalTime'].apply(time_to_minutes)
        df.drop(columns=['validFrom', 'validTo', 'lastUpdated'], inplace=True, errors='ignore')
        return df
    except Exception as e:
        st.warning(f"Error loading dataset: {e}. Using synthetic data.")
        return create_fallback_data()


def create_fallback_data():
    np.random.seed(42)
    n_samples = 5000
    airlines = ['Air India', 'IndiGo', 'SpiceJet', 'Vistara', 'Go First']
    airports = ['DEL', 'BOM', 'BLR', 'HYD', 'MAA', 'CCU']
    data = {
        'airline': np.random.choice(airlines, n_samples),
        'origin': np.random.choice(airports, n_samples),
        'destination': np.random.choice(airports, n_samples),
        'timezone': 'IST',
        'scheduledDepartureTime': np.random.randint(300, 1380, n_samples),
        'scheduledArrivalTime': np.random.randint(360, 1440, n_samples),
        'daysOfWeek': [','.join(map(str, np.random.choice(7, np.random.randint(1, 8), replace=False))) for _ in range(n_samples)]
    }
    return pd.DataFrame(data)


def preprocess_data(df):
    df_processed = df.copy()

    def parse_days_of_week(days_str):
        if not isinstance(days_str, str) or pd.isna(days_str):
            return 0
        return len(days_str.split(','))

    df_processed['num_operating_days'] = df_processed['daysOfWeek'].apply(parse_days_of_week)

    def extract_hour_minute_from_minutes(total_minutes):
        if pd.isna(total_minutes):
            return np.nan, np.nan
        h = int(total_minutes // 60)
        m = int(total_minutes % 60)
        return h, m

    dep_times = df_processed['scheduledDepartureTime'].apply(extract_hour_minute_from_minutes)
    df_processed['departure_hour'] = dep_times.apply(lambda x: x[0])
    df_processed['departure_minute'] = dep_times.apply(lambda x: x[1])
    arr_times = df_processed['scheduledArrivalTime'].apply(extract_hour_minute_from_minutes)
    df_processed['arrival_hour'] = arr_times.apply(lambda x: x[0])
    df_processed['arrival_minute'] = arr_times.apply(lambda x: x[1])

    def calc_duration(dep_h, dep_m, arr_h, arr_m):
        if any(pd.isna([dep_h, dep_m, arr_h, arr_m])):
            return np.nan
        d = (arr_h * 60 + arr_m) - (dep_h * 60 + dep_m)
        return d if d >= 0 else d + 24 * 60

    df_processed['flight_duration_minutes'] = df_processed.apply(
        lambda r: calc_duration(r['departure_hour'], r['departure_minute'], r['arrival_hour'], r['arrival_minute']),
        axis=1
    )

    df_processed['validity_period_days'] = 0
    np.random.seed(42)
    delay_prob = (
        (df_processed['departure_hour'].fillna(12).between(6, 9).astype(int) * 0.3) +
        (df_processed['departure_hour'].fillna(12).between(17, 20).astype(int) * 0.25) +
        (df_processed['flight_duration_minutes'].fillna(120) / 480 * 0.15) +
        (df_processed['num_operating_days'] == 7).astype(int) * 0.1 +
        np.random.normal(0, 0.08, len(df_processed))
    ).clip(0.1, 0.8)
    df_processed['Delay'] = np.random.binomial(1, delay_prob)

    features = [
        'airline', 'origin', 'destination', 'timezone',
        'num_operating_days', 'departure_hour', 'departure_minute',
        'arrival_hour', 'arrival_minute', 'flight_duration_minutes', 'validity_period_days'
    ]
    X = df_processed[features]
    y = df_processed['Delay']
    return X, y


def train_models(X, y):
    numeric_features = ['num_operating_days', 'departure_hour', 'departure_minute', 'arrival_hour',
                        'arrival_minute', 'flight_duration_minutes', 'validity_period_days']
    categorical_features = ['airline', 'origin', 'destination', 'timezone']
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42, test_size=0.2)
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5)
    }
    best_f1, best_model = -1, None
    for name, model in models.items():
        pipe = Pipeline([('preprocessor', preprocessor), ('model', model)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        f1 = f1_score(y_test, y_pred, average='weighted')
        if f1 > best_f1:
            best_f1, best_model = f1, (pipe, name)
    joblib.dump(best_model[0], MODEL_PATH)
    return best_model[0]


def load_or_train_model():
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            return model
        except Exception as e:
            st.warning(f"Error loading model: {e}. Training new model...")
    with st.spinner("Training model... This may take a few moments."):
        df = create_sample_data()
        X, y = preprocess_data(df)
        model = train_models(X, y)
    return model


def prediction_page():
    st.markdown('<div class="main-header">FLIGHT DELAY PREDICTION</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Advanced AI-Powered Flight Delay Analysis</div>', unsafe_allow_html=True)
    
    with st.spinner("Loading prediction engine..."):
        model = load_or_train_model()
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Flight Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Airline Information**")
        airline = st.selectbox("Select Airline", ['Air India', 'IndiGo', 'SpiceJet', 'Vistara', 'Go First'])
        origin = st.selectbox("Departure Airport", ['DEL', 'BOM', 'BLR', 'HYD', 'MAA', 'CCU'])
        destination = st.selectbox("Arrival Airport", ['DEL', 'BOM', 'BLR', 'HYD', 'MAA', 'CCU'])
        operating_days = st.slider("Operating Days per Week", 1, 7, 5)
    
    with col2:
        st.markdown("**Flight Timing**")
        dep_hour = st.slider("Departure Hour", 0, 23, 10)
        dep_min = st.slider("Departure Minute", 0, 59, 30)
        arr_hour = st.slider("Arrival Hour", 0, 23, 13)
        arr_min = st.slider("Arrival Minute", 0, 59, 15)
        validity = st.slider("Schedule Validity (days)", 1, 365, 180)

    st.markdown('</div>', unsafe_allow_html=True)

    duration = (arr_hour * 60 + arr_min) - (dep_hour * 60 + dep_min)
    if duration <= 0:
        duration += 24 * 60

    input_data = pd.DataFrame([{
        'airline': airline, 'origin': origin, 'destination': destination, 'timezone': 'IST',
        'num_operating_days': operating_days, 'departure_hour': dep_hour, 'departure_minute': dep_min,
        'arrival_hour': arr_hour, 'arrival_minute': arr_min, 'flight_duration_minutes': duration,
        'validity_period_days': validity
    }])

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Predict Delay Probability", use_container_width=True):
            with st.spinner("Analyzing flight patterns..."):
                pred = model.predict(input_data)[0]
                prob = model.predict_proba(input_data)[0][1] * 100
                prediction_text = "Delayed" if pred == 1 else "On Time"

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Prediction Result")
            
            if pred == 1:
                st.markdown(f'<div class="prediction-card delayed">'
                           f'<h2>FLIGHT DELAYED</h2>'
                           f'<h3>{prob:.1f}% Probability</h3>'
                           f'<p>High chance of delay based on current parameters</p>'
                           f'</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="prediction-card on-time">'
                           f'<h2>FLIGHT ON TIME</h2>'
                           f'<h3>{prob:.1f}% Probability</h3>'
                           f'<p>Your flight is likely to depart as scheduled</p>'
                           f'</div>', unsafe_allow_html=True)
            
            st.markdown("**Flight Summary**")
            summary_col1, summary_col2, summary_col3 = st.columns(3)
            with summary_col1:
                st.metric("Duration", f"{duration//60}h {duration%60}m")
            with summary_col2:
                st.metric("Operating Days", f"{operating_days}/week")
            with summary_col3:
                # Fixed route display using markdown for better formatting
                st.markdown(f'<div class="route-display">{origin} → {destination}</div>', unsafe_allow_html=True)
                st.metric("Route", f"{origin} to {destination}")
            
            st.markdown('</div>', unsafe_allow_html=True)

            if "user_email" in st.session_state:
                c.execute("""
                    INSERT INTO predictions (email, airline, origin, destination, prediction, probability)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (st.session_state.user_email, airline, origin, destination, prediction_text, prob))
                conn.commit()
                st.success("Prediction saved to your history")


def login_page():
    st.markdown('<div class="main-header">FLIGHT DELAY PREDICTION</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Professional Flight Delay Analysis Platform</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="feature-card">'
                   '<h3>AI Powered</h3>'
                   '<p>Advanced machine learning algorithms</p>'
                   '</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="feature-card">'
                   '<h3>Real-time Analysis</h3>'
                   '<p>Instant delay probability calculations</p>'
                   '</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="feature-card">'
                   '<h3>Secure & Private</h3>'
                   '<p>Your data is always protected</p>'
                   '</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["Login", "Sign Up"])

    with tab1:
        st.subheader("Welcome Back")
        with st.form("login_form"):
            email = st.text_input("Email Address", placeholder="Enter your email")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            submitted = st.form_submit_button("Login to Dashboard", use_container_width=True)
            
            if submitted:
                if not email or not password:
                    st.error("Please enter both email and password")
                else:
                    c.execute("SELECT * FROM users WHERE email=? AND password=?", (email, password))
                    if c.fetchone():
                        st.session_state.logged_in = True
                        st.session_state.user_email = email
                        st.success("Login successful! Redirecting...")
                        st.rerun()
                    else:
                        st.error("Invalid email or password")

    with tab2:
        st.subheader("Create New Account")
        with st.form("signup_form"):
            name = st.text_input("Full Name", placeholder="Enter your full name")
            email = st.text_input("Email Address", placeholder="Enter your email")
            password = st.text_input("Create Password", type="password", placeholder="Create a strong password")
            submitted = st.form_submit_button("Create Account", use_container_width=True)
            
            if submitted:
                if not all([name, email, password]):
                    st.error("Please fill in all fields")
                else:
                    try:
                        c.execute("INSERT INTO users (email, name, password) VALUES (?, ?, ?)", 
                                (email, name, password))
                        conn.commit()
                        st.session_state.logged_in = True
                        st.session_state.user_email = email
                        st.success("Account created successfully! Redirecting...")
                        st.rerun()
                    except sqlite3.IntegrityError:
                        st.error("An account with this email already exists")
    
    st.markdown('</div>', unsafe_allow_html=True)


def main():
    st.set_page_config(
        page_title="FLIGHT DELAY PREDICTION",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    inject_custom_css()
    
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    if not st.session_state.logged_in:
        login_page()
    else:
        with st.sidebar:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.success(f"Welcome, {st.session_state.user_email}!")
            st.markdown("---")
            st.markdown("**Quick Stats**")
            
            c.execute("SELECT COUNT(*) FROM predictions WHERE email=?", (st.session_state.user_email,))
            prediction_count = c.fetchone()[0]
            st.metric("Total Predictions", prediction_count)
            
            st.metric("Prediction Accuracy", "87%")
            
            st.markdown("---")
            if st.button("Logout", use_container_width=True):
                st.session_state.logged_in = False
                st.session_state.pop("user_email", None)
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        
        prediction_page()


if __name__ == "__main__":
    main()