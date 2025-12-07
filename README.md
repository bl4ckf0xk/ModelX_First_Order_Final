# First Order: Real-Time Situational Awareness Platform
**First Order** is an AI-powered dashboard designed to provide Sri Lankan business leaders with real-time intelligence. It aggregates signals from news, social media, and financial markets to detect risks (e.g., currency volatility, social unrest) and operational opportunities.

## Features
- **Live Data Ingestion:** Scrapes local news (RSS) and Reddit (r/srilanka) in real-time.
- **Market Intelligence:** Tracks LKR/USD exchange rates and Brent Crude Oil trends.
- **AI Analytics:**
  - **Sentiment Analysis:** Detects social unrest using VADER.
  - **Anomaly Detection:** Identifies abnormal Forex volatility using Isolation Forests.
  - **Trend Analysis:** Decomposes oil price trends using STL (Seasonal-Trend decomposition).
- **Actionable Advice:** Provides color-coded, role-based "AI Commander" briefs for Finance, HR, and Supply Chain.

## Installation

### Prerequisites
- Python 3.8 or higher

### 1. Clone the Repository
```bash
git clone [https://github.com/yourusername/modelx.git](https://github.com/yourusername/modelx.git)
cd modelx
```

### 2. Set up Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install streamlit pandas numpy feedparser requests yfinance tenacity statsmodels scikit-learn vaderSentiment streamlit-autorefresh
```

### 4. Run the file using this command (enter blank if prompt for email)
```bash
streamlit run First_Order_Situation_Awareness.py
```

The dashboard will automatically open in your browser at http://localhost:8501.

### Project Structure
First_Order_Situation_Awareness.py: Main application code.
Tier 1 (Ingestion): Scrapers for RSS, Reddit, and Yahoo Finance.
Tier 2 (Processing): ML models for anomaly detection and trend analysis.
Tier 3 (UI): Streamlit frontend with auto-refresh logic.
