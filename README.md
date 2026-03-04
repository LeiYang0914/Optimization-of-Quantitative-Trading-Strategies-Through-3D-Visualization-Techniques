# 📊 Optimization of Quantitative Trading Strategies Through 3D Visualization Techniques

![Python](https://img.shields.io/badge/Python-3.9-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![License](https://img.shields.io/badge/License-MIT-green)

This project investigates how **quantitative trading strategies can be optimized using data-driven techniques and visual analytics**.  
The research integrates **Bayesian Optimization, Grid Search Cross Validation, and 3D visualization** to analyze relationships between strategy parameters and trading performance.

The objective is to **improve the risk-adjusted performance of algorithmic trading strategies** by identifying optimal parameter configurations through systematic experimentation.

---

# 📌 Research Motivation

Financial markets are highly dynamic and volatile, making it difficult to design robust trading strategies.

Traditional parameter tuning techniques often fail to capture the **complex relationships between trading parameters and performance metrics**.

This research explores how **advanced optimization techniques combined with visual analytics** can help researchers and traders better understand the parameter space and improve trading strategy performance.

### Research Question

> How can trading strategies be optimized in volatile cryptocurrency markets using **data-driven optimization techniques and visual analytics**?

---

# 🔬 Methodology

The system follows a structured research pipeline combining **data engineering, financial modeling, optimization algorithms, and visualization techniques**.

---

## 1️⃣ Data Collection

Historical cryptocurrency data is collected using the **CCXT API**, covering several years of Bitcoin market activity.

The dataset includes:

- Cryptocurrency exchange historical data  
- OHLCV price data  
- Hourly market time series  

---

## 2️⃣ Data Preprocessing

The raw market data undergoes several preprocessing steps:

- Missing value cleaning  
- Timestamp alignment  
- Log return transformation  
- Feature engineering  

These steps ensure the dataset is suitable for **financial modeling and algorithmic trading research**.

---

## 3️⃣ Strategy Development

Two categories of trading strategies are implemented.

### Momentum Strategies

Momentum strategies attempt to capture **persistent market trends**.

Indicators used include:

- Moving Average  
- MACD  

These indicators help identify **trend continuation opportunities**.

---

### Mean Reversion Strategies

Mean reversion strategies assume that asset prices tend to **return to their historical average levels**.

Indicators used include:

- Bollinger Bands  
- RSI  

These strategies are particularly effective in **sideways or ranging market environments**.

---

## 4️⃣ Parameter Optimization

Two optimization approaches are used to improve strategy performance.

### Grid Search Cross Validation

- Exhaustive parameter search  
- Evaluates multiple combinations of strategy parameters  
- Provides baseline optimization results  

### Bayesian Optimization

- Efficient exploration of parameter space  
- Faster convergence toward optimal parameters  
- Reduces computational cost compared to grid search  

Both methods aim to maximize the **Sharpe Ratio**, a key metric for evaluating **risk-adjusted trading performance**.

---

# 📈 3D Visualization of Optimization Results

A key contribution of this project is the use of **3D visualization techniques** to analyze the optimization landscape.

Instead of relying solely on numerical tables, the system generates **3D surface plots** to illustrate relationships between:

- Strategy parameters  
- Trading performance metrics  
- Sharpe Ratio  

These visualizations allow researchers to:

- Identify optimal parameter regions  
- Understand parameter sensitivity  
- Detect potential overfitting  

This approach improves **interpretability and decision-making** in quantitative strategy development.

---

# 🖥 Streamlit Interactive Interface

The project includes a **Streamlit-based dashboard** for interactive experimentation.

Users can:

- Upload market datasets  
- Select trading strategies  
- Run optimization algorithms  
- Visualize results in interactive 3D plots  

The dashboard provides a **user-friendly research environment** for exploring strategy performance.

---

# 🌍 Real-World Testing

Optimized strategies are tested using the **Cybotrade Cloud platform**, which connects directly to cryptocurrency exchanges.

This enables:

- Live strategy deployment  
- Automated trading execution  
- Validation beyond historical backtesting  

Real-world testing ensures that the strategies remain **robust under live market conditions**.

---

# 🧠 System Architecture
Market Data (CCXT)
↓
Data Preprocessing
↓
Strategy Development
↓
Parameter Optimization
(Grid Search / Bayesian Optimization)
↓
Backtesting
↓
3D Visualization
↓
Streamlit Dashboard
↓
Real Market Testing (Cybotrade)

---

# ⚙️ Technologies Used

| Technology | Purpose |
|------------|--------|
| Python | Core programming language |
| Pandas | Data processing |
| NumPy | Numerical computation |
| Scikit-learn | Optimization tools |
| Bayesian Optimization | Strategy parameter tuning |
| CCXT | Cryptocurrency market data API |
| Streamlit | Interactive dashboard |
| Plotly | 3D visualization |
| Matplotlib | Data visualization |

---

# 📂 Project Structure
Optimization-of-Quantitative-Trading-Strategies
│
├── data
│ └── btc_1H_may2020_to_current.csv
│
├── notebooks
│ └── Extract_data.ipynb
│
├── src
│ └── main.py
│
├── requirements.txt
│
├── docs
│ ├── project_poster.pdf
│ └── final_report.pdf
│
└── README.md


---

# 🚀 Installation

Clone the repository

git clone https://github.com/LeiYang0914/Optimization-of-Quantitative-Trading-Strategies-Through-3D-Visualization-Techniques.git


Install dependencies
pip install -r requirements.txt

Run the project
python main.py

Run Streamlit dashboard
streamlit run app.py


---

# 🔭 Future Research Directions

Possible future improvements include:

- Reinforcement learning based trading strategies  
- Multi-asset portfolio optimization  
- Real-time strategy optimization  
- Advanced visualization for financial analytics  
- Machine learning based alpha generation  

---

# 👨‍💻 Author

**Chan Li Yang**

Bachelor of Computer Science (Data Science)

Research Interests:

- Quantitative Finance  
- Financial Machine Learning  
- Algorithmic Trading  
- Data Visualization  

---

# 📄 Documentation

Full research materials:

- Project Poster  
- Final Year Project Report  

These documents describe the **complete methodology, experimental design, and evaluation results**.