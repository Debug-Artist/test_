"""
================================================================================
CROP YIELD PREDICTION & MOMENTUM OPTIMIZATION PLATFORM
Backend Service: FastAPI + Pure NumPy Vector Calculus Engine
File: app.py

PROBLEM STATEMENT:
"To develop and evaluate an interpretable crop yield prediction model and
determine whether momentum optimization trains it faster and more accurately
than standard gradient descent."
================================================================================
"""

import os
import io
import base64
import logging
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend safe for server threads
import matplotlib.pyplot as plt

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

# Setup structured logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("AgronomicOptimizer")

# Absolute Path Resolution & Defensive CSV Discovery
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

PROBLEM_STATEMENT = (
    "To develop and evaluate an interpretable crop yield prediction model and "
    "determine whether momentum optimization trains it faster and more accurately "
    "than standard gradient descent."
)

FEATURE_COLUMNS = [
    "Rainfall_mm",
    "Temperature_C",
    "Soil_pH",
    "Humidity_pct",
    "Fertilizer_Used_kg"
]
TARGET_COLUMN = "Yield_ton_per_ha"

# Default Model Hyperparameters
DEFAULT_TEST_SIZE = 0.20
DEFAULT_RANDOM_STATE = 42
DEFAULT_EPOCHS = 1500
DEFAULT_LEARNING_RATE = 0.01
DEFAULT_MOMENTUM_FACTOR = 0.90

def resolve_dataset_path() -> Tuple[str, bool]:
    """
    Dynamically searches for the intended dataset across multiple standard locations
    supporting both 'data/crop_data' and 'data/crop_data.csv'.
    """
    search_candidates = [
        os.path.join(BASE_DIR, "data", "crop_data.csv"),
        os.path.join(BASE_DIR, "data", "crop_data"),
        os.path.join(os.getcwd(), "data", "crop_data.csv"),
        os.path.join(os.getcwd(), "data", "crop_data"),
        os.path.join(BASE_DIR, "crop_data.csv"),
        os.path.join(BASE_DIR, "crop_data")
    ]
    for path in search_candidates:
        if os.path.exists(path) and os.path.isfile(path):
            return path, True
    return search_candidates[0], False

class ModelTelemetry:
    def __init__(self):
        self.problem_statement: str = PROBLEM_STATEMENT
        self.is_demo_mode: bool = False
        self.dataset_path: str = ""
        self.dataset_rows: int = 0
        self.columns_detected: List[str] = []
        
        # Scaling Statistics (computed strictly on training partition)
        self.mu_train: np.ndarray = np.zeros(len(FEATURE_COLUMNS))
        self.sigma_train: np.ndarray = np.ones(len(FEATURE_COLUMNS))
        
        # Training Parameters
        self.epochs: int = DEFAULT_EPOCHS
        self.learning_rate: float = DEFAULT_LEARNING_RATE
        self.beta: float = DEFAULT_MOMENTUM_FACTOR
        
        # Learned Parameters - Vanilla Gradient Descent
        self.w_gd: np.ndarray = np.zeros(len(FEATURE_COLUMNS))
        self.b_gd: float = 0.0
        self.train_loss_gd: List[float] = []
        self.test_mse_gd: float = 0.0
        
        # Learned Parameters - Momentum Optimization
        self.w_mom: np.ndarray = np.zeros(len(FEATURE_COLUMNS))
        self.b_mom: float = 0.0
        self.velocity_w: np.ndarray = np.zeros(len(FEATURE_COLUMNS))
        self.velocity_b: float = 0.0
        self.train_loss_mom: List[float] = []
        self.test_mse_mom: float = 0.0
        
        # Comparative Research Evaluation Outcomes
        self.epochs_to_threshold_gd: Optional[int] = None
        self.epochs_to_threshold_mom: Optional[int] = None
        self.is_faster: bool = True
        self.is_more_accurate: bool = True
        self.speedup_factor: float = 1.0

state = ModelTelemetry()

def load_and_prepare_dataset() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Ingests tabular CSV data, validates schema, applies strict train/test splitting,
    and scales features strictly on the training set to prevent data leakage.
    """
    global state
    resolved_path, exists = resolve_dataset_path()
    state.dataset_path = resolved_path
    
    if not exists:
        logger.warning(f"Target CSV file not found at '{resolved_path}'. Enabling Demo Mode.")
        state.is_demo_mode = True
        return fallback_synthetic_data()

    try:
        df = pd.read_csv(resolved_path)
        df.columns = df.columns.str.strip()
        state.columns_detected = list(df.columns)
        
        # Schema Validation
        missing_features = [col for col in FEATURE_COLUMNS if col not in df.columns]
        if missing_features or TARGET_COLUMN not in df.columns:
            logger.error(f"Schema mismatch: missing {missing_features}. Activating Demo Mode.")
            state.is_demo_mode = True
            return fallback_synthetic_data()
        
        # Clean non-numeric entries and drop missing rows
        df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])
        for col in FEATURE_COLUMNS + [TARGET_COLUMN]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])

        state.dataset_rows = len(df)
        state.is_demo_mode = False
        logger.info(f"Loaded verified Kaggle dataset: {state.dataset_rows} rows from '{resolved_path}'.")

        X = df[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
        y = df[TARGET_COLUMN].to_numpy(dtype=np.float64)

        # Train-Test Partitioning (80/20)
        np.random.seed(DEFAULT_RANDOM_STATE)
        permutation = np.random.permutation(len(X))
        split_point = int(len(X) * (1.0 - DEFAULT_TEST_SIZE))
        
        train_indices = permutation[:split_point]
        test_indices = permutation[split_point:]

        X_train_raw = X[train_indices]
        y_train = y[train_indices]
        X_test_raw = X[test_indices]
        y_test = y[test_indices]

        # Prevent Data Leakage: Standardize strictly using training set parameters
        state.mu_train = np.mean(X_train_raw, axis=0)
        state.sigma_train = np.std(X_train_raw, axis=0)
        state.sigma_train[state.sigma_train == 0.0] = 1.0

        X_train_scaled = (X_train_raw - state.mu_train) / state.sigma_train
        X_test_scaled = (X_test_raw - state.mu_train) / state.sigma_train

        return X_train_scaled, y_train, X_test_scaled, y_test

    except Exception as exc:
        logger.error(f"Failed to parse CSV dataset: {exc}. Falling back to Demo Mode.")
        state.is_demo_mode = True
        return fallback_synthetic_data()

def fallback_synthetic_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generates synthetic agronomic dataset when CSV is missing or invalid."""
    np.random.seed(DEFAULT_RANDOM_STATE)
    m = 250
    X_syn = np.random.randn(m, len(FEATURE_COLUMNS))
    true_weights = np.array([0.75, -0.40, 0.60, 0.25, 0.85])
    true_bias = 4.20
    y_syn = np.dot(X_syn, true_weights) + true_bias + np.random.randn(m) * 0.25

    state.dataset_rows = m
    state.columns_detected = FEATURE_COLUMNS + [TARGET_COLUMN]
    state.mu_train = np.zeros(len(FEATURE_COLUMNS))
    state.sigma_train = np.ones(len(FEATURE_COLUMNS))

    split = int(m * (1.0 - DEFAULT_TEST_SIZE))
    return X_syn[:split], y_syn[:split], X_syn[split:], y_syn[split:]

# ==============================================================================
# FIRST-PRINCIPLES CALCULUS & MOMENTUM (PURE NUMPY)
# ==============================================================================
def compute_loss_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Squared Error: L(w, b) = (1/m) * sum((y_hat - y)^2)."""
    return float(np.mean((y_pred - y_true) ** 2))

def compute_partial_derivatives(X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Multivariable chain rule evaluation:
      dL/dw_j = (2/m) * sum((y_hat_i - y_i) * x_ij) => (2/m) * X^T (y_hat - y)
      dL/db   = (2/m) * sum(y_hat_i - y_i)
    """
    m = X.shape[0]
    residuals = y_pred - y_true
    dL_dw = (2.0 / m) * np.dot(X.T, residuals)
    dL_db = float((2.0 / m) * np.sum(residuals))
    return dL_dw, dL_db

def run_vanilla_gradient_descent(
    X_train: np.ndarray, y_train: np.ndarray, epochs: int, lr: float
) -> Tuple[np.ndarray, float, List[float]]:
    """Standard Gradient Descent parameter update loop: theta_t = theta_{t-1} - lr * grad."""
    m, n = X_train.shape
    w = np.zeros(n, dtype=np.float64)
    b = 0.0
    history = []

    for _ in range(epochs):
        y_pred = np.dot(X_train, w) + b
        loss = compute_loss_mse(y_train, y_pred)
        history.append(loss)

        dL_dw, dL_db = compute_partial_derivatives(X_train, y_train, y_pred)
        w -= lr * dL_dw
        b -= lr * dL_db

    return w, b, history

def run_momentum_optimization(
    X_train: np.ndarray, y_train: np.ndarray, epochs: int, lr: float, beta: float
) -> Tuple[np.ndarray, float, List[float], np.ndarray, float]:
    """
    Momentum Optimization update loop using the exact user-specified convention:
      V_t = V_{t-1} + (1 - beta) * grad(L)
      theta_t = theta_{t-1} - lr * V_t
    """
    m, n = X_train.shape
    w = np.zeros(n, dtype=np.float64)
    b = 0.0
    V_w = np.zeros(n, dtype=np.float64)
    V_b = 0.0
    history = []

    for _ in range(epochs):
        y_pred = np.dot(X_train, w) + b
        loss = compute_loss_mse(y_train, y_pred)
        history.append(loss)

        dL_dw, dL_db = compute_partial_derivatives(X_train, y_train, y_pred)
        
        # Exact velocity accumulation
        V_w = V_w + (1.0 - beta) * dL_dw
        V_b = V_b + (1.0 - beta) * dL_db
        
        w -= lr * V_w
        b -= lr * V_b

    return w, b, history, V_w, V_b

def render_loss_plot_base64() -> str:
    """Renders dark-themed convergence plot to Base64 string."""
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=120)
    fig.patch.set_facecolor("#070D0A")
    ax.set_facecolor("#0E1712")

    epochs_range = list(range(len(state.train_loss_gd)))
    ax.plot(epochs_range, state.train_loss_gd, label="Vanilla Gradient Descent", color="#F87171", lw=2.2)
    ax.plot(epochs_range, state.train_loss_mom, label=f"Momentum Optimization (β = {state.beta})", color="#34D399", lw=2.4)

    ax.set_title("Empirical Loss Trajectory: Vanilla GD vs. Momentum Optimization", fontsize=12, fontweight=600, color="#F1F5F9", pad=12)
    ax.set_xlabel("Optimization Epochs", fontsize=10, color="#94A3B8")
    ax.set_ylabel("Mean Squared Error (tons/ha)²", fontsize=10, color="#94A3B8")
    ax.grid(True, linestyle=":", alpha=0.25, color="#34D399")
    
    legend = ax.legend(frameon=True, facecolor="#070D0A", edgecolor="rgba(52,211,153,0.2)", fontsize=9)
    for text in legend.get_texts():
        text.set_color("#F1F5F9")

    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def execute_model_training(epochs: int, lr: float, beta: float):
    global state
    state.epochs = epochs
    state.learning_rate = lr
    state.beta = beta

    X_train, y_train, X_test, y_test = load_and_prepare_dataset()

    # 1. Train Vanilla GD
    w_gd, b_gd, loss_gd = run_vanilla_gradient_descent(X_train, y_train, epochs, lr)
    state.w_gd, state.b_gd, state.train_loss_gd = w_gd, b_gd, loss_gd

    # 2. Train Momentum
    w_mom, b_mom, loss_mom, v_w, v_b = run_momentum_optimization(X_train, y_train, epochs, lr, beta)
    state.w_mom, state.b_mom, state.velocity_w, state.velocity_b, state.train_loss_mom = w_mom, b_mom, v_w, v_b, loss_mom

    # 3. Test Evaluation (Generalization Test)
    pred_test_gd = np.dot(X_test, w_gd) + b_gd
    pred_test_mom = np.dot(X_test, w_mom) + b_mom
    state.test_mse_gd = compute_loss_mse(y_test, pred_test_gd)
    state.test_mse_mom = compute_loss_mse(y_test, pred_test_mom)

    # 4. Benchmarking Answers to the Problem Statement: Faster & More Accurately?
    target_thresh = min(loss_gd[-1], loss_mom[-1]) * 1.25
    state.epochs_to_threshold_gd = next((i for i, v in enumerate(loss_gd) if v <= target_thresh), epochs)
    state.epochs_to_threshold_mom = next((i for i, v in enumerate(loss_mom) if v <= target_thresh), epochs)
    
    # Speedup calculation
    if state.epochs_to_threshold_mom and state.epochs_to_threshold_mom > 0:
        state.speedup_factor = round(state.epochs_to_threshold_gd / state.epochs_to_threshold_mom, 2)
    else:
        state.speedup_factor = 1.0

    state.is_faster = (state.epochs_to_threshold_mom <= state.epochs_to_threshold_gd)
    state.is_more_accurate = (state.test_mse_mom <= state.test_mse_gd)

# FastAPI App
app = FastAPI(title="Agronomic Momentum Optimization Engine")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

class PredictionPayload(BaseModel):
    Rainfall_mm: float = Field(..., ge=0.0, le=3000.0)
    Temperature_C: float = Field(..., ge=-10.0, le=60.0)
    Soil_pH: float = Field(..., ge=0.0, le=14.0)
    Humidity_pct: float = Field(..., ge=0.0, le=100.0)
    Fertilizer_Used_kg: float = Field(..., ge=0.0, le=1000.0)

class RetrainPayload(BaseModel):
    epochs: int = Field(DEFAULT_EPOCHS, ge=100, le=10000)
    learning_rate: float = Field(DEFAULT_LEARNING_RATE, ge=0.0001, le=0.5)
    momentum_factor: float = Field(DEFAULT_MOMENTUM_FACTOR, ge=0.0, le=0.99)

class MathCalculatorPayload(BaseModel):
    rainfall: float
    temperature: float
    soil_ph: float
    humidity: float
    fertilizer: float
    actual_yield: float

@app.on_event("startup")
def app_startup():
    execute_model_training(DEFAULT_EPOCHS, DEFAULT_LEARNING_RATE, DEFAULT_MOMENTUM_FACTOR)

@app.get("/", response_class=HTMLResponse)
async def serve_index(request: Request):
    plot_b64 = render_loss_plot_base64()
    weights_summary = [
        {
            "feature": col,
            "mean": round(float(state.mu_train[i]), 2),
            "std": round(float(state.sigma_train[i]), 2),
            "weight_gd": round(float(state.w_gd[i]), 4),
            "weight_mom": round(float(state.w_mom[i]), 4),
            "velocity": round(float(state.velocity_w[i]), 4)
        }
        for i, col in enumerate(FEATURE_COLUMNS)
    ]
        return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "problem_statement": state.problem_statement,
            "is_demo_mode": state.is_demo_mode,
            "dataset_rows": state.dataset_rows,
            "dataset_path": state.dataset_path,
            "epochs": state.epochs,
            "learning_rate": state.learning_rate,
            "beta": state.beta,
            "bias_gd": round(state.b_gd, 4),
            "bias_mom": round(state.b_mom, 4),
            "train_mse_gd": round(state.train_loss_gd[-1], 4) if state.train_loss_gd else 0.0,
            "train_mse_mom": round(state.train_loss_mom[-1], 4) if state.train_loss_mom else 0.0,
            "test_mse_gd": round(state.test_mse_gd, 4),
            "test_mse_mom": round(state.test_mse_mom, 4),
            "epochs_to_threshold_gd": state.epochs_to_threshold_gd,
            "epochs_to_threshold_mom": state.epochs_to_threshold_mom,
            "speedup_factor": state.speedup_factor,
            "is_faster": state.is_faster,
            "is_more_accurate": state.is_more_accurate,
            "weights_summary": weights_summary,
            "plot_b64": plot_b64
        }
    )

@app.post("/api/predict")
async def api_predict(payload: PredictionPayload):
    raw_input = np.array([
        payload.Rainfall_mm,
        payload.Temperature_C,
        payload.Soil_pH,
        payload.Humidity_pct,
        payload.Fertilizer_Used_kg
    ], dtype=np.float64)

    scaled_input = (raw_input - state.mu_train) / state.sigma_train
    pred_gd = float(np.dot(scaled_input, state.w_gd) + state.b_gd)
    pred_mom = float(np.dot(scaled_input, state.w_mom) + state.b_mom)

    return {
        "prediction_momentum": round(max(0.0, pred_mom), 2),
        "prediction_vanilla_gd": round(max(0.0, pred_gd), 2),
        "unit": "ton/ha",
        "raw_scaled_features": [round(float(z), 3) for z in scaled_input],
        "is_demo_mode": state.is_demo_mode
    }

@app.post("/api/retrain")
async def api_retrain(payload: RetrainPayload):
    execute_model_training(payload.epochs, payload.learning_rate, payload.momentum_factor)
    return {
        "status": "success",
        "train_mse_gd": round(state.train_loss_gd[-1], 4),
        "train_mse_mom": round(state.train_loss_mom[-1], 4),
        "test_mse_gd": round(state.test_mse_gd, 4),
        "test_mse_mom": round(state.test_mse_mom, 4),
        "epochs_to_threshold_gd": state.epochs_to_threshold_gd,
        "epochs_to_threshold_mom": state.epochs_to_threshold_mom,
        "speedup_factor": state.speedup_factor,
        "is_faster": state.is_faster,
        "is_more_accurate": state.is_more_accurate,
        "plot_b64": render_loss_plot_base64()
    }

@app.post("/api/calculate_math")
async def api_calculate_math(payload: MathCalculatorPayload):
    raw_vec = np.array([payload.rainfall, payload.temperature, payload.soil_ph, payload.humidity, payload.fertilizer], dtype=np.float64)
    scaled_vec = (raw_vec - state.mu_train) / state.sigma_train
    y_hat = float(np.dot(scaled_vec, state.w_mom) + state.b_mom)
    residual = float(y_hat - payload.actual_yield)
    sample_grad_w = 2.0 * residual * scaled_vec
    sample_grad_b = 2.0 * residual
    simulated_V_w = state.velocity_w + (1.0 - state.beta) * sample_grad_w
    simulated_w_update = state.w_mom - state.learning_rate * simulated_V_w

    return {
        "scaled_features": [round(float(v), 4) for v in scaled_vec],
        "predicted_yield": round(y_hat, 4),
        "residual_error": round(residual, 4),
        "squared_error": round(residual ** 2, 4),
        "partial_derivatives_w": [round(float(g), 4) for g in sample_grad_w],
        "partial_derivative_b": round(sample_grad_b, 4),
        "simulated_velocity_w": [round(float(v), 4) for v in simulated_V_w],
        "simulated_weights_next": [round(float(w), 4) for w in simulated_w_update]
    }

@app.get("/api/status")
async def api_status():
    return {
        "status": "healthy",
        "problem_statement": state.problem_statement,
        "demo_mode": state.is_demo_mode,
        "dataset_path": state.dataset_path,
        "dataset_rows": state.dataset_rows,
        "columns_detected": state.columns_detected,
        "test_mse_gd": round(state.test_mse_gd, 4),
        "test_mse_mom": round(state.test_mse_mom, 4),
        "speedup_factor": state.speedup_factor,
        "is_faster": state.is_faster,
        "is_more_accurate": state.is_more_accurate,
        "epochs": state.epochs,
        "learning_rate": state.learning_rate,
        "beta": state.beta
    }
