# ☀️ Solar Panel Output Predictor (SPOP)

A modular machine learning forecasting system that predicts daily solar energy output for a given location.

SPOP predicts **Global Horizontal Irradiance (GHI)** using weather data, converts irradiance into expected solar panel energy output (kWh), and provides uncertainty bands to communicate forecast confidence.

This is not a notebook experiment — it is a structured, reproducible ML system with clean architecture and formal documentation.

---

## 🚀 Core Capabilities

- Predict daily Global Horizontal Irradiance (GHI)
- Convert GHI → solar energy output (kWh)
- Generate uncertainty intervals (±MAE, ±1–3 STD)
- Train and cache location-specific models
- Interactive 5-day forecast via Streamlit
- Fully documented using Sphinx

---

## 🧠 System Architecture

SPOP is structured as a modular pipeline:

```
Data Sources
    ↓
Data Pipeline
    ↓
Feature Engineering
    ↓
Model Training
    ↓
Forecasting
    ↓
Uncertainty Bands
    ↓
Streamlit UI
```

### Source Structure

```
src/main/
    app/
    data_pipeline/
    data_sources/
    features/
    math/
    models/
    config.py
    paths.py
    schemas.py
```

Each layer is isolated to ensure maintainability, reproducibility, and scalability.

---

## 📊 Forecast Methodology

1. Retrieve weather forecasts for a specified latitude and longitude.
2. Generate engineered features (seasonality encoding, derived weather predictors).
3. Predict daily GHI using a Random Forest model.
4. Convert GHI to energy output:

```
Energy (kWh) = GHI × panel_area × efficiency
```

5. Compute uncertainty bands using residual statistics:
   - ±MAE
   - ±1 STD
   - ±2 STD
   - ±3 STD

---

## 🛠 Technology Stack

- Python
- scikit-learn (Random Forest)
- Pandas / NumPy
- Streamlit (UI)
- Sphinx (Documentation)

---

## ⚙️ Running Locally

### 1. Clone

```
git clone https://github.com/Aabdi22k/Solar-Panel-Output-Predictor-Remake.git
cd Solar-Panel-Output-Predictor-Remake
```

### 2. Create Environment

```
python -m venv .venv
source .venv/Scripts/activate   # Windows
# or
source .venv/bin/activate       # macOS/Linux
```

### 3. Install Dependencies

```
pip install -r requirements.txt
```

### 4. Set Required Environment Variables

Create a `.env` file:

```
NREL_API_KEY=your_key_here
NREL_EMAIL=your_email_here
```

### 5. Run the App

```
streamlit run src/main/app/streamlit_app.py
```

The application will:

- Load a cached model if available
- Otherwise build the dataset and train automatically
- Display a 5-day forecast with uncertainty bands

---

## 📁 Documentation

Full project documentation (architecture, API reference, methodology) is available in the `/docs` directory.

To build locally:

```
cd docs
make html
```

---

## 🔮 Future Improvements

- Time-series cross-validation
- Hyperparameter tuning
- Model comparison (RF vs XGBoost)
- Quantile regression for true prediction intervals
- Backtesting framework
- CI/CD pipeline
- Public deployment

---

## 🎯 Project Scope

SPOP is designed as a production-aware ML forecasting foundation.

It demonstrates:

- Modular ML architecture
- Reproducible training pipeline
- Feature engineering rationale
- Model evaluation discipline
- Uncertainty modeling
- Structured documentation


## 📌 License

MIT 