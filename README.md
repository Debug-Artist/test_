# Utilizing Partial Derivatives and Python Libraries for Momentum Optimization

An applied mathematical workbench and crop yield prediction application implementing multivariable calculus, analytical Mean Squared Error derivatives, and momentum optimization from first principles using pure NumPy and FastAPI.

---

## Project Structure

```
crop_yield_momentum/
│
├── app.py                      # FastAPI application, vector calculus engine, ML pipeline
├── run.py                      # One-click desktop launcher (runs Uvicorn & opens browser)
├── requirements.txt            # Dependency specification
├── README.md                   # System documentation & mathematical formulas
│
├── data/
│   └── crop_data.csv           # Kaggle agronomic dataset (13 columns, verified schema)
│
├── templates/
│   └── index.html              # Custom research console with Integrated Mathematics Suite
│
└── static/
    ├── css/
    │   └── style.css           # Bespoke obsidian & forest slate design (no AI clichés)
    └── js/
        └── main.js             # Dual slider-number sync, live API client, math sandbox
```

---

## Mathematical Architecture

### 1. Multivariable Linear Hypothesis
$$\hat{y} = \mathbf{w}^T \mathbf{z} + b = \sum_{j=1}^5 w_j z_j + b$$
Where $z_j = \frac{x_j - \mu_j}{\sigma_j}$ are features standardized strictly using training set distributions.

### 2. Analytical Loss Gradients (Mean Squared Error)
$$L(\mathbf{w}, b) = \frac{1}{m} \sum_{i=1}^m (\hat{y}_i - y_i)^2$$
$$\frac{\partial L}{\partial w_j} = \frac{2}{m} \sum_{i=1}^m (\hat{y}_i - y_i) x_{ij} \implies \nabla_{\mathbf{w}} L = \frac{2}{m} \mathbf{X}^T (\mathbf{\hat{y}} - \mathbf{y})$$
$$\frac{\partial L}{\partial b} = \frac{2}{m} \sum_{i=1}^m (\hat{y}_i - y_i)$$

### 3. Exact Momentum Optimization Update
$$\mathbf{V}_t = \mathbf{V}_{t-1} + (1 - \beta)\nabla L(\theta_{t-1})$$
$$\theta_t = \theta_{t-1} - \eta \mathbf{V}_t$$

---

## How to Run

1. Open your terminal in this directory:
   ```bash
   cd "C:\Users\Akshat Kohli\.gemini\antigravity\scratch\crop_yield_momentum"
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Launch the application:
   ```bash
   python run.py
   ```
   Or via Uvicorn directly:
   ```bash
   uvicorn app:app --host 127.0.0.1 --port 8000 --reload
   ```

4. Navigate to `http://127.0.0.1:8000` in your web browser.
