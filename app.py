# --- 1. CORE IMPORTS ---
import os, json, logging, sqlite3, csv, math
from io import StringIO
from datetime import datetime, timedelta
from functools import wraps
from collections import defaultdict

# --- 2. DATA SCIENCE & ML IMPORTS ---
import pandas as pd, numpy as np
import requests, bcrypt
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV
from statsmodels.tsa.arima.model import ARIMA
from werkzeug.utils import secure_filename
import plotly.graph_objs as go

# --- 3. FLASK IMPORTS ---
from flask import Flask, render_template, request, redirect, url_for, session, flash, g
from flask_session import Session

# --- 4. FLASK APP CONFIGURATION ---
app = Flask(__name__)
app.secret_key = os.urandom(24)
DATABASE = "crypto_forecasting.db"
CONFIG_FILE = "config.json"
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"csv"}

app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# --- 5. CREDENTIALS, SECURITY & LOGGING ---
def load_credentials():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f: data = json.load(f)
        if 'password_hash' in data: return data
        if 'password' in data:
            print("WARNING: Migrating plaintext password to hash.")
            hashed = bcrypt.hashpw(data['password'].encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
            data['password_hash'] = hashed; del data['password']; save_credentials(data)
            return data
    default_hash = bcrypt.hashpw("password123".encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    return {"username": "SaiLokesh", "password_hash": default_hash, "theme": "dark", "currency": "usd", "api_key": ""}
def save_credentials(data):
    with open(CONFIG_FILE, "w") as f: json.dump(data, f, indent=4)
creds = load_credentials()
USERNAME = creds.get("username", "SaiLokesh")
PASSWORD_HASH = creds.get("password_hash", "")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 6. DATABASE SETUP ---
def get_db():
    db = getattr(g, "_database", None)
    if db is None: db = g._database = sqlite3.connect(DATABASE); db.row_factory = sqlite3.Row
    return db
@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, "_database", None)
    if db is not None: db.close()
def init_db():
    with app.app_context():
        db = get_db()
        db.execute("CREATE TABLE IF NOT EXISTS transactions (id INTEGER PRIMARY KEY AUTOINCREMENT, user TEXT, date TEXT, coin TEXT, type TEXT, amount REAL, price REAL)")
        db.execute("CREATE TABLE IF NOT EXISTS watchlist (id INTEGER PRIMARY KEY AUTOINCREMENT, user TEXT NOT NULL, coin_id TEXT NOT NULL, date_added TEXT NOT NULL, UNIQUE(user, coin_id))")
        db.commit()
init_db()

# --- 7. AUTHENTICATION & HELPERS ---
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user" not in session:
            flash("Please log in to access this page.", "warning"); return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated_function
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# --- FIX: PREVENT BROWSER CACHING ---
@app.after_request
def add_no_cache_headers(response):
    """
    Prevents the browser from caching pages.
    This fixes the bug where pressing the 'back' button
    after logging out shows the protected dashboard.
    """
    if response.status_code == 200:
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
    return response
# --- END OF FIX ---

# --- 8. DATA FETCHING, PROCESSING & ML ---
def fetch_current_prices(coin_ids: list, vs_currency="usd"):
    if not coin_ids: return {}
    try:
        r = requests.get(f"https://api.coingecko.com/api/v3/simple/price?ids={','.join(coin_ids)}&vs_currencies={vs_currency}", timeout=10)
        r.raise_for_status(); return r.json()
    except Exception as e:
        logger.error(f"Failed to fetch current prices: {e}"); return {}
def fetch_coin_market_data(coin_ids: list, vs_currency="usd"):
    if not coin_ids: return []
    try:
        r = requests.get(f"https://api.coingecko.com/api/v3/coins/markets?vs_currency={vs_currency}&ids={','.join(coin_ids)}", timeout=10)
        r.raise_for_status(); return r.json()
    except Exception as e:
        logger.error(f"Failed to fetch coin market data: {e}"); return []
def fetch_coingecko(coin_id: str, vs_currency="usd", days=365):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {"vs_currency": vs_currency, "days": str(days)}
    try:
        r = requests.get(url, params=params, timeout=20); r.raise_for_status(); data = r.json()
        df_price = pd.DataFrame(data.get("prices", []), columns=["timestamp", "price"])
        if df_price.empty: return None
        df_mc = pd.DataFrame(data.get("market_caps", []), columns=["timestamp", "market_cap"])
        df_vol = pd.DataFrame(data.get("total_volumes", []), columns=["timestamp", "volume"])
        df = df_price.merge(df_mc, on="timestamp").merge(df_vol, on="timestamp")
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df.set_index("datetime").drop(columns=["timestamp"])
    except Exception as e:
        logger.warning(f"CoinGecko fetch failed for {coin_id}: {e}"); return None
def fetch_binance(symbol="BTCUSDT", interval="1d", limit=1000):
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    try:
        resp = requests.get(url, params=params, timeout=20); resp.raise_for_status(); data = resp.json()
    except Exception as e:
        logger.warning(f"Binance fetch failed for {symbol}: {e}"); return None
    rows = [[int(k[0]), float(k[4]), np.nan, float(k[5])] for k in data]
    df = pd.DataFrame(rows, columns=["timestamp", "price", "market_cap", "volume"])
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df.set_index("datetime").drop(columns=["timestamp"])
def fetch_coin_history(coin_name, days=365):
    df = fetch_coingecko(coin_name, days=days)
    if df is not None and not df.empty: return df
    mapping = { "bitcoin": "BTCUSDT", "ethereum": "ETHUSDT", "litecoin": "LTCUSDT", "ripple": "XRPUSDT", "cardano": "ADAUSDT", "solana": "SOLUSDT" }
    symbol = mapping.get(coin_name.lower())
    if symbol:
        return fetch_binance(symbol, interval="1d", limit=min(1000, days + 10))
    return None
def preprocess(df: pd.DataFrame):
    df = df.copy()
    for c in ["price", "market_cap", "volume"]: df[c] = pd.to_numeric(df.get(c), errors="coerce")
    if isinstance(df.index, pd.DatetimeIndex): df = df.resample("1D").mean()
    df = df.fillna(0)
    def clip(s, low=0.01, high=0.99):
        if s.dropna().empty: return s
        return s.clip(s.quantile(low), s.quantile(high))
    for col in ["price", "market_cap", "volume"]:
        if col in df.columns: df[col] = clip(df[col])
    return df
def add_bollinger_bands(df: pd.DataFrame, window=20):
    df = df.copy()
    df['ma20'] = df['price'].rolling(window).mean()
    df['std20'] = df['price'].rolling(window).std()
    df['bb_upper'] = df['ma20'] + (df['std20'] * 2)
    df['bb_lower'] = df['ma20'] - (df['std20'] * 2)
    return df
def add_technical_features(df: pd.DataFrame):
    df = df.copy()
    df["pct_change"] = df["price"].pct_change()
    delta = df["price"].diff()
    gain, loss = delta.clip(lower=0), (-delta).clip(lower=0)
    roll_up, roll_down = gain.rolling(14).mean(), loss.rolling(14).mean()
    df["rsi"] = 100 - (100 / (1 + (roll_up / (roll_down + 1e-10))))
    ema12, ema26 = df["price"].ewm(span=12, adjust=False).mean(), df["price"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    return df.dropna()
def make_features(df: pd.DataFrame):
    df = df.copy()
    df = add_bollinger_bands(df)
    df["price_lag1"], df["price_lag2"] = df["price"].shift(1), df["price"].shift(2)
    df["ma_7"], df["ma_14"] = df["price"].rolling(7).mean(), df["price"].rolling(14).mean()
    df["std_7"] = df["price"].rolling(7).std()
    return df.dropna()
def make_train_val_test(df: pd.DataFrame, forecast_horizon=7):
    df = make_features(add_technical_features(df.copy()))
    n = len(df)
    min_split_size = forecast_horizon + 10 
    if n < (50 + min_split_size * 2):
        logger.warning(f"Not enough data (need {50 + min_split_size * 2}, got {n})")
        return [None] * 8
    n_test = max(int(n * 0.1), min_split_size)
    n_val = max(int(n * 0.1), min_split_size)
    train_end = n - (n_val + n_test)
    val_end = n - n_test
    if train_end < 50:
        logger.warning(f"Not enough training data left after splitting.")
        return [None] * 8
    df_train, df_val, df_test = df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]
    features = ["price", "price_lag1", "price_lag2", "ma_7", "ma_14", "std_7", "rsi", "macd", "signal", "bb_upper", "bb_lower"]
    scaler = MinMaxScaler().fit(df_train[features])
    def scale_df(d): return pd.DataFrame(scaler.transform(d[features]), columns=features, index=d.index)
    df_train_s, df_val_s, df_test_s = scale_df(df_train), scale_df(df_val), scale_df(df_test)
    def create_xy(d):
        X, y = [], []
        for i in range(len(d) - forecast_horizon):
            X.append(d.iloc[i:i + forecast_horizon].values.flatten()); y.append(d["price"].iloc[i + forecast_horizon])
        return np.array(X), np.array(y)
    X_train, y_train = create_xy(df_train_s)
    X_val, y_val = create_xy(df_val_s)
    X_test, y_test = create_xy(df_test_s)
    if any(X.size == 0 for X in [X_train, X_val, X_test]): 
        logger.warning("Failed to create non-empty X/y arrays.")
        return [None] * 8
    return X_train, y_train, X_val, y_val, X_test, y_test, scaler, df_test[features]
def train_xgboost(X_train, y_train, X_val, y_val, tune=False):
    if tune:
        param_dist = {"n_estimators": [100, 150, 200], "learning_rate": [0.01, 0.05, 0.1], "max_depth": [3, 5, 7]}
        model = xgb.XGBRegressor(objective="reg:squarederror", verbosity=0)
        search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=8, scoring="neg_mean_squared_error", n_jobs=1, random_state=42, verbose=0)
        search.fit(np.vstack([X_train, X_val]), np.concatenate([y_train, y_val]))
        return search.best_estimator_
    model = xgb.XGBRegressor(objective="reg:squarederror", learning_rate=0.1, max_depth=5, n_estimators=100, early_stopping_rounds=10, eval_metric="rmse", verbosity=0)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model
def train_arima(train_df, test_df, horizon):
    history = [x for x in train_df['price']]
    predictions = []
    for t in range(len(test_df) + horizon):
        try:
            model = ARIMA(history, order=(5,1,0))
            model_fit = model.fit()
            output = model_fit.forecast()
            yhat = output[0]
            predictions.append(yhat)
            if t < len(test_df):
                history.append(test_df['price'][t])
            else:
                history.append(yhat)
        except Exception as e:
            logger.error(f"ARIMA step failed: {e}")
            predictions.append(history[-1])
    y_test_final = test_df['price']
    y_pred_final = pd.Series(predictions[:len(test_df)], index=test_df.index)
    mse = mean_squared_error(y_test_final, y_pred_final)
    mae = mean_absolute_error(y_test_final, y_pred_final)
    return y_pred_final, y_test_final, {"mse": mse, "mae": mae}
def run_simple_trend_model(df: pd.DataFrame, horizon: int):
    recent_data = df['price'].tail(15)
    x = np.arange(len(recent_data))
    y = recent_data.values
    coeffs = np.polyfit(x, y, 1); slope = coeffs[0]
    last_price = y[-1]
    forecast_prices = [last_price + slope * i for i in range(1, horizon + 1)]
    last_date = recent_data.index[-1]
    forecast_dates = pd.to_datetime([last_date + timedelta(days=i) for i in range(1, horizon + 1)])
    forecast_series = pd.Series(forecast_prices, index=forecast_dates)
    y_actual = df['price']
    y_pred = pd.concat([pd.Series([np.nan] * len(df), index=df.index), forecast_series])
    return y_pred, y_actual, {"mse": "N/A", "mae": "N/A"}
def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    return mean_squared_error(y, y_pred), mean_absolute_error(y, y_pred), y_pred
def calculate_transaction_summary(transactions):
    if not transactions: return {'count': 0, 'spend': 0, 'pnl': 0}
    total_spend = sum(float(tx['amount']) * float(tx['price']) for tx in transactions if tx['type'] == 'buy')
    total_sell_value = sum(float(tx['amount']) * float(tx['price']) for tx in transactions if tx['type'] == 'sell')
    return {'count': len(transactions), 'spend': total_spend, 'pnl': total_sell_value - total_spend}

# ✅ FIX: Upgraded function to normalize dates for a robust join
def detect_trade_patterns(user_transactions, historical_prices):
    if not user_transactions: return []
    tx_df = pd.DataFrame(user_transactions)
    tx_df['date'] = pd.to_datetime(tx_df['date'])
    
    prices_daily = historical_prices.resample('D').last().ffill()
    prices_daily['price_change_pct'] = prices_daily['price'].pct_change()

    # Normalize both join keys to midnight (YYYY-MM-DD 00:00:00)
    tx_df.index = tx_df['date'].dt.normalize()
    prices_daily.index = prices_daily.index.normalize()
    
    # Drop the original 'date' column *before* joining
    tx_df = tx_df.drop(columns=['date'])
    
    analysis_df = tx_df.join(prices_daily['price_change_pct'])
    analysis_df = analysis_df.reset_index().rename(columns={'index': 'date'}) # Reset index to get date back

    annotated_trades = []
    for _, trade in analysis_df.iterrows():
        annotation = {'date': trade['date'], 'type': trade['type'], 'price': trade['price'], 'amount': trade['amount'], 'patterns': []}
        if trade['type'] == 'buy' and not pd.isnull(trade['price_change_pct']) and trade['price_change_pct'] < -0.05:
            annotation['patterns'].append(('Buy the Dip', 'success'))
        if trade['type'] == 'sell' and not pd.isnull(trade['price_change_pct']) and trade['price_change_pct'] > 0.05:
            annotation['patterns'].append(('Sell the Peak', 'success'))
        if trade['type'] == 'sell' and not pd.isnull(trade['price_change_pct']) and trade['price_change_pct'] < -0.05:
            annotation['patterns'].append(('Panic Sell', 'danger'))
        if trade['type'] == 'buy' and not pd.isnull(trade['price_change_pct']) and trade['price_change_pct'] > 0.05:
            annotation['patterns'].append(('FOMO Buy', 'warning'))
        annotated_trades.append(annotation)
    return annotated_trades

# --- 9. PLOTTING FUNCTIONS ---
def plot_forecast(idx, actual, predicted, title="Forecast vs Actual"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=idx, y=predicted, mode="lines", name="Predicted Price", line=dict(color='#FFA500', dash='dash', width=3), fill='tozeroy', fillcolor='rgba(255, 165, 0, 0.25)'))
    fig.add_trace(go.Scatter(x=idx, y=actual, mode="lines", name="Actual Price", line=dict(color='#5DADE2', width=2), fill='tozeroy', fillcolor='rgba(93, 173, 226, 0.4)'))
    fig.update_layout(title={'text': title, 'y':0.9, 'x':0.5, 'xanchor': 'center'}, template="plotly_dark", paper_bgcolor='#1e1e1e', plot_bgcolor='#2c2c2c', font_color="#e0e0e0", hovermode="x unified")
    return fig.to_html(full_html=False)
def plot_portfolio_distribution(holdings_data):
    if not holdings_data: return None
    labels = [h['name'].capitalize() for h in holdings_data]
    values = [h['current_value'] for h in holdings_data]
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.4, textinfo='label+percent')])
    fig.update_layout(title={'text': 'Portfolio Distribution by Value', 'x':0.5}, template="plotly_dark", paper_bgcolor='#1e1e1e', plot_bgcolor='#2c2c2c', font_color="#e0e0e0", showlegend=False)
    return fig.to_html(full_html=False)
def plot_watchlist_performance(watchlist_data):
    if not watchlist_data: return None
    data = sorted(watchlist_data, key=lambda x: x.get('market_cap', 0), reverse=True)
    labels = [d.get('name', '').capitalize() for d in data]
    values = [d.get('price_change_percentage_24h', 0) for d in data]
    colors = ['#2e7d32' if v is not None and v >= 0 else '#c62828' for v in values]
    fig = go.Figure(data=[go.Bar(x=labels, y=values, marker_color=colors)])
    fig.update_layout(title={'text': 'Watchlist 24h Performance (%)', 'x':0.5}, template="plotly_dark", paper_bgcolor='#1e1e1e', plot_bgcolor='#2c2c2c', font_color="#e0e0e0")
    return fig.to_html(full_html=False)
def plot_error_residuals(residuals):
    if not residuals: return None
    fig = go.Figure(data=[go.Histogram(x=residuals, nbinsx=30, marker_color='#00bfa5')])
    fig.update_layout(title={'text': 'Distribution of Prediction Errors', 'y':0.9, 'x':0.5, 'xanchor': 'center'}, xaxis_title="Prediction Error", yaxis_title="Frequency", template="plotly_dark", paper_bgcolor='#1e1e1e', plot_bgcolor='#2c2c2c', font_color="#e0e0e0")
    return fig.to_html(full_html=False)
def create_analyzer_chart(historical_prices, annotated_trades):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=historical_prices.index, y=historical_prices['price'], mode='lines', name='Price', line=dict(color='#5DADE2', width=2)))
    for trade in annotated_trades:
        color = '#2ECC71' if trade['type'] == 'buy' else '#E74C3C'
        symbol = 'triangle-up' if trade['type'] == 'buy' else 'triangle-down'
        pattern_text = "<br>".join([p[0] for p in trade['patterns']])
        hover_text = (f"<b>{trade['type'].capitalize()}</b><br>Price: ${trade['price']:,.2f}<br>Amount: {trade['amount']}<br>Patterns: {pattern_text or 'None'}")
        fig.add_trace(go.Scatter(x=[trade['date']], y=[trade['price']], mode='markers', marker=dict(symbol=symbol, color=color, size=12, line=dict(width=2, color='white')), name=trade['type'].capitalize(), hoverinfo='text', hovertext=hover_text, showlegend=False))
    fig.update_layout(title={'text': 'Trade Pattern Analysis', 'y':0.9, 'x':0.5, 'xanchor': 'center'}, template="plotly_dark", paper_bgcolor='#1e1e1e', plot_bgcolor='#2c2c2c', font_color="#e0e0e0")
    return fig.to_html(full_html=False)

# --- 10. FLASK ROUTES ---
@app.route("/")
def home():
    return redirect(url_for("dashboard") if "user" in session else url_for("login"))

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        u, p = request.form.get("username", "").strip(), request.form.get("password", "").strip()
        if u == USERNAME and bcrypt.checkpw(p.encode("utf-8"), PASSWORD_HASH.encode("utf-8")):
            session["user"] = u; flash("Login successful!", "success"); return redirect(url_for("dashboard"))
        else:
            flash("Invalid username or password.", "danger")
    return render_template("login.html")

@app.route("/dashboard", methods=["GET","POST"])
@login_required
def dashboard():
    coin_list = ["bitcoin","ethereum","litecoin","ripple","cardano","solana"]
    context = {
        'active_page': 'dashboard', 'coin_list': coin_list,
        'selected_coin': session.get('selected_coin', 'bitcoin'),
        'selected_days': session.get('selected_days', 365),
        'selected_horizon': session.get('selected_horizon', 7),
        'selected_model': session.get('selected_model', 'xgboost'),
        'tune_model': session.get('tune_model', False),
    }

    if request.method == "POST":
        session.pop('chart', None)
        session.pop('metrics', None)
        session.pop('residuals', None)
        context.update({
            'selected_coin': request.form.get("coin"),
            'selected_days': int(request.form.get("days")),
            'selected_horizon': int(request.form.get("horizon")),
            'selected_model': request.form.get("model_choice"),
            'tune_model': 'tune_model' in request.form
        })
        session.update({k: v for k, v in context.items() if k.startswith('selected_') or k == 'tune_model'})
        
        df = fetch_coin_history(context['selected_coin'], days=context['selected_days'])
        
        if df is None or df.empty:
            flash(f"Could not fetch data for {context['selected_coin'].capitalize()}.", "danger")
        else:
            try:
                dfp = preprocess(df)
                last = dfp.iloc[-1]
                market_cap = last.get('market_cap')
                volume = last.get('volume')
                context['metrics_initial'] = {
                    "current_price": last['price'],
                    "market_cap": market_cap if pd.notna(market_cap) else 0,
                    "volume": volume if pd.notna(volume) else 0
                }
                n_days = len(dfp)
                model_used = context['selected_model']

                # --- ADAPTIVE FORECASTING ENGINE ---
                if n_days >= 150:
                    model_choice = model_used
                    if model_choice == 'xgboost':
                        result = make_train_val_test(dfp, forecast_horizon=context['selected_horizon'])
                        if result and result[0] is not None:
                            X_train, y_train, X_val, y_val, X_test, y_test, scaler, df_test = result
                            model = train_xgboost(X_train, y_train, X_val, y_val, tune=context['tune_model'])
                            mse, mae, y_pred_s = evaluate_model(model, X_test, y_test)
                            temp = df_test.iloc[context['selected_horizon']:].copy()
                            if not temp.empty:
                                minlen = min(len(temp), len(y_pred_s))
                                temp_s, y_pred_s = scaler.transform(temp.iloc[:minlen]), y_pred_s[:minlen]
                                temp_s[:, 0] = y_pred_s.flatten()
                                y_pred = scaler.inverse_transform(temp_s)[:, 0]
                                y_test_f = df_test["price"].iloc[context['selected_horizon']:][:len(y_pred)]
                                
                                session['metrics'] = {"mse": mse, "mae": mae}
                                session['chart'] = plot_forecast(y_test_f.index, y_test_f, y_pred, title=f"Forecast")
                                session['residuals'] = (y_test_f.values - y_pred).tolist()
                        else:
                            flash("Not enough data for XGBoost (need >150 days with features). Switched to ARIMA.", "warning"); model_choice = 'arima'
                    
                    if model_choice == 'arima':
                        train_df, test_df = dfp[:-30], dfp[-30:]
                        y_pred, y_test, metrics = train_arima(train_df, test_df, context['selected_horizon'])
                        session['metrics'] = metrics
                        session['chart'] = plot_forecast(y_test.index, y_test, y_pred, title=f"Forecast")
                        session['residuals'] = (y_test.values - y_pred.values).tolist()
                
                elif 60 <= n_days < 150:
                    if model_used == 'xgboost': flash("Not enough data for XGBoost. Switched to ARIMA model.", "warning")
                    model_used = 'arima'
                    train_df, test_df = dfp[:-30], dfp[-30:]
                    y_pred, y_test, metrics = train_arima(train_df, test_df, context['selected_horizon'])
                    session['metrics'] = metrics
                    session['chart'] = plot_forecast(y_test.index, y_test, y_pred, title=f"Forecast")
                    session['residuals'] = (y_test.values - y_pred.values).tolist()
                
                else: # 30 <= n_days < 60
                    flash("Warning: Limited data (30-59 days). Using a Simple Linear Trend forecast.", "warning")
                    model_used = 'simple_trend'
                    y_pred, y_actual, metrics = run_simple_trend_model(dfp, context['selected_horizon'])
                    session['metrics'] = metrics
                    session['chart'] = plot_forecast(y_actual.index.union(y_pred.index), y_actual, y_pred, title=f"Forecast")
                    session['residuals'] = [] 
                
                if session.get('chart'):
                    session['chart'] = session['chart'].replace("Forecast", f"{context['selected_coin'].capitalize()} {model_used.upper()} Forecast")
            
            except Exception as e:
                flash(f"An error occurred during forecast: {e}", "danger")
                logger.error(f"Dashboard Error: {e}", exc_info=True)
            
        return redirect(url_for('dashboard'))

    # --- GET Request Logic ---
    context['chart'] = session.get('chart')
    context['metrics'] = session.get('metrics')
    
    if not context.get('metrics_initial'):
        df_initial = fetch_coin_history(context['selected_coin'], days=1)
        if df_initial is not None and not df_initial.empty:
            last = preprocess(df_initial).iloc[-1]
            market_cap = last.get('market_cap')
            volume = last.get('volume')
            context['metrics_initial'] = {
                "current_price": last['price'],
                "market_cap": market_cap if pd.notna(market_cap) else 0,
                "volume": volume if pd.notna(volume) else 0
            }
    
    return render_template("dashboard.html", **context)

@app.route("/error_analysis")
@login_required
def error_analysis():
    residuals = session.get("residuals")
    metrics = session.get("metrics", {})
    
    if residuals:
        chart = plot_error_residuals(residuals)
    else:
        chart = None
        
    return render_template("error_analysis.html", active_page='error_analysis', errors={"residuals": residuals, **metrics}, chart=chart)

@app.route("/portfolio")
@login_required
def portfolio():
    db = get_db()
    txs = db.execute("SELECT * FROM transactions WHERE user = ?", (session["user"],)).fetchall()
    if not txs:
        empty_portfolio = {'coins': [], 'total_value': 0.0, 'total_pnl': 0.0, 'unique_assets_count': 0}
        return render_template("portfolio.html", active_page='portfolio', portfolio=empty_portfolio, portfolio_chart=None)
    
    holdings = defaultdict(lambda: {'amount': 0.0, 'cost': 0.0})
    for tx in txs:
        amount, price = float(tx['amount']), float(tx['price'])
        if tx['type'] == 'buy':
            holdings[tx['coin']]['amount'] += amount; holdings[tx['coin']]['cost'] += amount * price
        else:
            if holdings[tx['coin']]['amount'] > 0:
                cost_per_coin = holdings[tx['coin']]['cost'] / holdings[tx['coin']]['amount']
                holdings[tx['coin']]['cost'] -= amount * cost_per_coin
            holdings[tx['coin']]['amount'] -= amount
    
    held_coins = [c for c, d in holdings.items() if d['amount'] > 1e-9]
    prices = fetch_current_prices(held_coins)
    coins, total_val, total_cost = [], 0.0, 0.0
    for coin in held_coins:
        data = holdings[coin]
        price = prices.get(coin, {}).get('usd', 0)
        value, cost = data['amount'] * price, data['cost']
        coins.append({'name': coin, 'amount': data['amount'], 'avg_price': cost/data['amount'] if data['amount'] > 0 else 0, 'current_price': price, 'current_value': value, 'pnl': value - cost})
        total_val += value; total_cost += cost
            
    summary = {
        'coins': sorted(coins, key=lambda x:x['current_value'], reverse=True), 
        'total_value': total_val, 'total_pnl': total_val - total_cost,
        'unique_assets_count': len(coins)
    }
    chart = plot_portfolio_distribution(coins)
    return render_template("portfolio.html", active_page='portfolio', portfolio=summary, portfolio_chart=chart)

@app.route("/watchlist", methods=["GET", "POST"])
@login_required
def watchlist():
    db = get_db()
    user = session["user"]
    if request.method == "POST":
        coin_id, action = request.form.get("coin_id", "").strip().lower(), request.form.get("action")
        if coin_id and action == "add":
            try:
                db.execute("INSERT INTO watchlist (user, coin_id, date_added) VALUES (?, ?, ?)", (user, coin_id, datetime.now().strftime("%Y-%m-%d")))
                db.commit(); flash(f"Added {coin_id.capitalize()}.", "success")
            except sqlite3.IntegrityError:
                flash(f"{coin_id.capitalize()} is already on watchlist.", "warning")
        elif coin_id and action == "remove":
            db.execute("DELETE FROM watchlist WHERE user = ? AND coin_id = ?", (user, coin_id))
            db.commit(); flash(f"Removed {coin_id.capitalize()}.", "success")
        return redirect(url_for('watchlist'))

    items = db.execute("SELECT coin_id, date_added FROM watchlist WHERE user = ?", (user,)).fetchall()
    if not items:
        empty_watchlist = {'total_coins': 0, 'total_pnl': 0.0}
        return render_template("watchlist.html", active_page='watchlist', watchlist_coins=[], watchlist_summary=empty_watchlist)
    
    market_data = fetch_coin_market_data([item['coin_id'] for item in items])
    date_map = {item['coin_id']: item['date_added'] for item in items}
    coins_list, pnl_sum = [], 0.0
    for data in market_data:
        pnl_24h = data.get('price_change_24h', 0)
        if pnl_24h: pnl_sum += pnl_24h
        added_on_str = date_map.get(data.get('id'))
        added_on_obj = datetime.strptime(added_on_str, '%Y-%m-%d').date() if added_on_str else None
        coins_list.append({'id': data.get('id'), 'name': data.get('name'), 'current_price': data.get('current_price', 0), 'change_24h': data.get('price_change_percentage_24h', 0), 'market_cap': data.get('market_cap', 0), 'added_on': added_on_obj})
    
    summary = {'total_coins': len(coins_list), 'total_pnl': pnl_sum}
    chart = plot_watchlist_performance(market_data)
    return render_template("watchlist.html", active_page='watchlist', watchlist_coins=coins_list, watchlist_summary=summary, watchlist_chart=chart)

@app.route("/transactions", methods=["GET", "POST"])
@login_required
def transactions():
    db = get_db()
    if request.method == "POST":
        try:
            db.execute("INSERT INTO transactions (user,date,coin,type,amount,price) VALUES (?,?,?,?,?,?)", 
                       (session['user'], request.form.get("date") or datetime.now().strftime("%Y-%m-%d"), 
                        request.form.get("coin", "").strip().lower(), request.form.get("type"), 
                        float(request.form.get("amount")), float(request.form.get("price"))))
            db.commit(); flash("Transaction added.", "success")
        except (ValueError, TypeError):
            flash("Invalid input for amount or price.", "danger")
        return redirect(url_for("transactions"))
    
    txs_from_db = db.execute("SELECT * FROM transactions WHERE user = ? ORDER BY date DESC", (session['user'],)).fetchall()
    processed_txs = []
    for tx in txs_from_db:
        tx_dict = dict(tx)
        if isinstance(tx_dict['date'], str):
            try: tx_dict['date'] = datetime.strptime(tx_dict['date'], "%Y-%m-%d").date()
            except ValueError: pass 
        processed_txs.append(tx_dict)
    summary_data = calculate_transaction_summary(processed_txs)
    return render_template("transactions.html", active_page='transactions', transactions=processed_txs, summary=summary_data, today=datetime.now().strftime("%Y-%m-%d"))

@app.route("/delete_transaction/<int:tx_id>", methods=["POST"])
@login_required
def delete_transaction(tx_id):
    db = get_db()
    transaction = db.execute("SELECT id FROM transactions WHERE id = ? AND user = ?", (tx_id, session["user"])).fetchone()
    if transaction:
        cursor = db.execute("DELETE FROM transactions WHERE id = ? AND user = ?", (tx_id, session["user"]))
        db.commit()
        if cursor.rowcount > 0:
            flash("Transaction deleted successfully.", "success")
        else:
            flash("Deletion failed. Transaction not found.", "danger")
    else:
        flash("Error: Transaction not found or you do not have permission to delete it.", "danger")
    return redirect(url_for("transactions"))

@app.route("/upload-file", methods=["POST"])
@login_required
def upload_file():
    file = request.files.get("file")
    if not file or not allowed_file(file.filename):
        flash("Invalid or no file selected.", "danger"); return redirect(url_for("transactions"))
    filename = secure_filename(file.filename)
    dest = os.path.join(app.config["UPLOAD_FOLDER"], f"{os.path.splitext(filename)[0]}_{int(datetime.utcnow().timestamp())}.csv")
    file.save(dest); session["uploaded_file"] = dest
    return redirect(url_for("review_file"))

@app.route("/review-file")
@login_required
def review_file():
    filepath = session.get("uploaded_file")
    if not filepath or not os.path.exists(filepath):
        return redirect(url_for("transactions"))
    try:
        df = pd.read_csv(filepath, sep=None, engine='python', on_bad_lines='skip')
        preview_html = df.head(10).to_html(classes="table table-dark table-striped", border=0)
        return render_template("review_file.html", preview=preview_html, total_rows=len(df), columns=df.columns.tolist())
    except Exception as e:
        flash(f"Error reading file: {e}", "danger"); return redirect(url_for("transactions"))

@app.route("/import-file", methods=["POST"])
@login_required
def import_file():
    filepath = session.get("uploaded_file")
    if not filepath or not os.path.exists(filepath):
        return redirect(url_for("transactions"))
    try:
        df = pd.read_csv(filepath, sep=None, engine='python', on_bad_lines='skip')
        df.columns = [c.lower().strip() for c in df.columns]
        required = {"date", "coin", "type", "amount", "price"}
        if not required.issubset(df.columns):
            flash(f"Missing columns. Required: {', '.join(required)}", "danger")
            return redirect(url_for("transactions"))
        rows, skipped = [], 0
        for _, r in df.iterrows():
            try:
                pd_date = pd.to_datetime(r['date']).strftime('%Y-%m-%d')
                rows.append((session['user'], pd_date, r['coin'], r['type'], float(r['amount']), float(r['price'])))
            except (ValueError, TypeError, pd.errors.ParserError): skipped += 1
        if rows:
            db = get_db()
            db.executemany("INSERT INTO transactions (user,date,coin,type,amount,price) VALUES (?,?,?,?,?,?)", rows)
            db.commit()
            flash(f"Imported {len(rows)} rows. Skipped {skipped} due to invalid data.", "success")
    except Exception as e:
        flash(f"Import failed: {e}", "danger")
    finally:
        if os.path.exists(filepath): os.remove(filepath)
        session.pop("uploaded_file", None)
    return redirect(url_for("transactions"))

@app.route("/analyzer", methods=["GET", "POST"])
@login_required
def analyzer():
    db = get_db()
    user_coins_cursor = db.execute("SELECT DISTINCT coin FROM transactions WHERE user = ? ORDER BY coin ASC", (session["user"],))
    user_coins = [row['coin'] for row in user_coins_cursor.fetchall()]

    coin_to_analyze = request.form.get('coin_choice', session.get('analyzer_coin', user_coins[0] if user_coins else 'bitcoin'))
    if request.method == "POST":
        session['analyzer_coin'] = coin_to_analyze
    
    txs_from_db = db.execute("SELECT * FROM transactions WHERE user = ? AND coin = ?", (session["user"], coin_to_analyze)).fetchall()
    user_transactions = [dict(row) for row in txs_from_db]

    if not user_transactions:
        flash(f"No transactions found for {coin_to_analyze.capitalize()} to analyze.", "warning")
        return render_template("analyzer.html", active_page='analyzer', coin=coin_to_analyze, user_coins=user_coins, chart=None, trades=None)
    
    try:
        # Convert all date strings to datetime objects for safe comparison
        dates = [datetime.strptime(t['date'], '%Y-%m-%d') for t in user_transactions if t['date']]
        if not dates:
             raise ValueError("No valid dates found in transactions")
        first_trade_date = min(dates)
        days_history = (datetime.now() - first_trade_date).days + 30
    except (ValueError, TypeError) as e:
        logger.error(f"Analyzer date error: {e}")
        flash("Could not determine the date range from your transactions.", "danger")
        return render_template("analyzer.html", active_page='analyzer', coin=coin_to_analyze, user_coins=user_coins, chart=None, trades=None)

    historical_prices = fetch_coin_history(coin_to_analyze, days=days_history)
    if historical_prices is None:
        flash(f"Could not fetch historical price data for {coin_to_analyze.capitalize()}.", "danger")
        return render_template("analyzer.html", active_page='analyzer', coin=coin_to_analyze, user_coins=user_coins, chart=None, trades=None)

    annotated_trades = detect_trade_patterns(user_transactions, historical_prices)
    chart_html = create_analyzer_chart(historical_prices, annotated_trades)
    return render_template("analyzer.html", active_page='analyzer', chart=chart_html, trades=annotated_trades, coin=coin_to_analyze, user_coins=user_coins)
    
@app.route("/settings", methods=["GET","POST"])
@login_required
def settings():
    global PASSWORD_HASH, creds
    if request.method == "POST":
        if 'current_password' in request.form and request.form.get("current_password"):
            current, new, confirm = request.form.get("current_password"), request.form.get("new_password"), request.form.get("confirm_password")
            if not bcrypt.checkpw(current.encode(), PASSWORD_HASH.encode()):
                flash("Current password incorrect.", "danger")
            elif new != confirm:
                flash("New passwords do not match.", "warning")
            elif len(new) < 6:
                flash("Password must be at least 6 characters.", "warning")
            else:
                new_hash = bcrypt.hashpw(new.encode(), bcrypt.gensalt()).decode()
                creds["password_hash"] = new_hash
                save_credentials(creds); PASSWORD_HASH = new_hash
                flash("Password updated.", "success")
        elif 'theme' in request.form:
            creds["theme"] = request.form.get("theme", "dark")
            creds["currency"] = request.form.get("currency", "usd")
            creds["api_key"] = request.form.get("api_key", "").strip()
            save_credentials(creds)
            flash("Preferences updated successfully.", "success")
        return redirect(url_for('settings'))
        
    return render_template("settings.html", active_page='settings', current_settings=creds)

@app.route("/delete_account", methods=["POST"])
@login_required
def delete_account():
    db = get_db()
    user = session["user"]
    db.execute("DELETE FROM transactions WHERE user = ?", (user,))
    db.execute("DELETE FROM watchlist WHERE user = ?", (user,))
    db.commit()
    session.clear()
    flash("Your account and all associated data have been permanently deleted.", "success")
    return redirect(url_for("login"))

@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "success")
    return redirect(url_for("login"))

# --- 11. RUN THE APP ---
if __name__ == "__main__":
    print("Starting Flask app on http://127.0.0.1:5000")
    app.run(debug=True, port=5000)