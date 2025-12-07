import streamlit as st
import pandas as pd
import numpy as np
import feedparser
import requests
import time
import random
import yfinance as yf
from datetime import datetime, timedelta
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from statsmodels.tsa.seasonal import STL
from sklearn.ensemble import IsolationForest
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Configure Page
st.set_page_config(page_title="ModelX: Live Situational Awareness", layout="wide", page_icon="🇱🇰")

# --- TIER 1: RESILIENT SENSOR LAYER ---

class BaseScraper:
    USER_AGENTS = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15'
    ]

    def _get_headers(self):
        return {'User-Agent': random.choice(self.USER_AGENTS)}

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((requests.exceptions.RequestException, TimeoutError))
    )
    def fetch_url(self, url):
        response = requests.get(url, headers=self._get_headers(), timeout=10)
        response.raise_for_status()
        return response

class NewsIngestor(BaseScraper):
    RSS_SOURCES = {
        "AdaDerana": "http://www.adaderana.lk/rss.php",
        "DailyMirror": "https://www.dailymirror.lk/RSS_Feeds/breaking-news"
    }

    def fetch_live_news(self):
        news_items = []
        for source, url in self.RSS_SOURCES.items():
            try:
                feed = feedparser.parse(url)
                if feed.bozo: pass 
                for entry in feed.entries[:10]:
                    published = datetime.now()
                    if 'published_parsed' in entry:
                        published = datetime.fromtimestamp(time.mktime(entry.published_parsed))
                    news_items.append({
                        "source": source,
                        "title": entry.title,
                        "link": entry.link,
                        "published": published
                    })
            except Exception as e:
                print(f"Failed to fetch {source}: {e}") 

        if not news_items:
            return pd.DataFrame(columns=["source", "title", "link", "published"])

        return pd.DataFrame(news_items).sort_values(by="published", ascending=False)

class MarketDataIngestor:
    def fetch_usd_lkr(self):
        try:
            ticker = yf.Ticker("LKR=X")
            hist = ticker.history(period="1mo")
            if hist.empty:
                # Fallback only if API fails, but ideally, this keeps retrying
                return pd.Series(dtype=float)
            return hist['Close']
        except:
            return pd.Series(dtype=float)

    def fetch_oil_price(self):
        try:
            # Brent Crude Oil Futures (Global Benchmark)
            # Impact: Rising oil -> Higher transport/energy costs in SL
            ticker = yf.Ticker("BZ=F") 
            hist = ticker.history(period="1mo")
            if hist.empty:
                return pd.Series(dtype=float)
            return hist['Close']
        except:
            return pd.Series(dtype=float)

# --- TIER 2: INTELLIGENT PROCESSING LAYER ---

def train_and_detect_anomalies(history_series):
    if len(history_series) < 10: return False
    X = history_series.values.reshape(-1, 1)
    clf = IsolationForest(random_state=42, contamination='auto')
    clf.fit(X)
    latest_value = X[-1].reshape(1, -1)
    prediction = clf.predict(latest_value)[0]
    return prediction == -1 

def analyze_news_sentiment(news_df):
    analyzer = SentimentIntensityAnalyzer()
    
    if news_df.empty: return 0, False, []

    scores = []
    crisis_keywords = ["protest", "strike", "crisis", "curfew", "violence", "shortage"]
    triggered_keywords = []

    for title in news_df['title'].head(15):
        vs = analyzer.polarity_scores(title)
        scores.append(vs['compound'])
        for k in crisis_keywords:
            if k in title.lower():
                triggered_keywords.append(k)

    avg_sentiment = np.mean(scores) if scores else 0
    is_critical = (avg_sentiment < -0.2) or (len(triggered_keywords) > 0)
    
    return avg_sentiment, is_critical, list(set(triggered_keywords))

def compute_analytics(news_df, usd_series, oil_series):
    # 1. ML Anomaly Detection
    fx_anomaly = train_and_detect_anomalies(usd_series)
    
    # 2. NLP Sentiment Analysis
    sentiment_score, social_risk, keywords = analyze_news_sentiment(news_df)

    # 3. Statistical Trend on Oil
    slope = 0
    trend = oil_series
    if not oil_series.empty:
        try:
            # Decompose to find the underlying trend (ignoring daily noise)
            stl = STL(oil_series, period=5, robust=True)
            res = stl.fit()
            trend = res.trend
            # Calculate slope (Price change over last 4 days)
            slope = trend.iloc[-1] - trend.iloc[-4]
        except:
            pass # Keep default

    # 4. SL-BSI (Business Stability Index) Calculation
    base_score = 100

    # Penalty: FX Volatility (Unstable currency)
    if fx_anomaly: base_score -= 25

    # Penalty: Social Unrest (Protests/Strikes)
    if social_risk: base_score -= 25

    # Penalty: Rising Oil Prices (Operational Cost Spike)
    # If oil price increased by more than $2 in the trend recently
    if slope > 2.0: base_score -= 15
    
    # Bonus: Good Sentiment
    if not news_df.empty and sentiment_score > 0.2: base_score += 10 

    bsi = max(0, min(100, base_score))

    return {
        "bsi": bsi,
        "oil_slope": slope,
        "sentiment": sentiment_score,
        "social_risk": social_risk,
        "keywords": keywords,
        "fx_anomaly": fx_anomaly,
        "trend_data": trend,
        "current_usd": usd_series.iloc[-1] if not usd_series.empty else 0,
        "current_oil": oil_series.iloc[-1] if not oil_series.empty else 0
    }

# --- TIER 3: INSIGHT GENERATION (NLG) ---

def generate_nlg(analytics):
    msgs = []

    # Forex Logic
    if analytics['fx_anomaly']:
        msgs.append("💸 **Forex Warning:** AI model detected abnormal volatility in LKR/USD. Hedge exposure.")

    # Social Logic
    if analytics['social_risk']:
        kws = ", ".join(analytics['keywords'])
        msgs.append(f"🔥 **Social Risk:** Negative sentiment detected. Keywords: {kws}.")
    # Oil/Energy Logic
    # Slope > 1 means prices are trending UP
    if analytics['oil_slope'] > 1.5:
        msgs.append(f"🛢️ **Supply Chain Risk:** Global Oil prices are rallying (Trend: +${analytics['oil_slope']:.2f}). Expect transport cost increases.")
    elif analytics['oil_slope'] < -1.5:
        msgs.append(f"✅ **Cost Benefit:** Oil prices are trending down. Potential savings on logistics.")

    # General Stability Logic
    if not analytics['fx_anomaly'] and analytics['bsi'] > 80:
        msgs.append("✅ **Opportunity:** Market conditions are highly stable. Good window for capital investments.")
    if analytics['sentiment'] > 0.5:
        msgs.append("📈 **Sentiment:** Public sentiment is positive. Potential for consumer confidence rebound.")

    if not msgs:
        msgs.append("🛡️ **Status Quo:** No immediate threats or major opportunities detected. Standard operations advised.")

    return "  \n".join(msgs)

# --- UI EXECUTION ---

def main():
    st.title("🇱🇰 ModelX: Real-Time Intelligence")
    
    if st.button("🔄 Refresh Data Now"):
        st.rerun()

    header = st.empty()
    header.markdown(f"*System Live | Last Update: {datetime.now().strftime('%H:%M:%S')}*")

    with st.spinner("Aggregating multi-source intelligence..."):
        news_ingestor = NewsIngestor()
        market_ingestor = MarketDataIngestor()

        # Fetch Data
        news_df = news_ingestor.fetch_live_news()
        usd_series = market_ingestor.fetch_usd_lkr()
        oil_series = market_ingestor.fetch_oil_price()

        # Compute
        metrics = compute_analytics(news_df, usd_series, oil_series)
        narrative = generate_nlg(metrics)

    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("SL-BSI (Stability Index)", f"{metrics['bsi']:.0f}/100", 
                delta="Stable" if metrics['bsi'] > 75 else "-Risk",
                delta_color="normal" if metrics['bsi'] > 75 else "inverse")
    
    kpi2.metric("LKR/USD", f"{metrics['current_usd']:.2f}", 
                delta="Volatility Alert" if metrics['fx_anomaly'] else "Normal",
                delta_color="inverse" if metrics['fx_anomaly'] else "off")
    
    kpi3.metric("Global Oil (Brent)", f"${metrics['current_oil']:.2f}", 
                delta=f"{metrics['oil_slope']:.2f} Trend",
                delta_color="inverse" if metrics['oil_slope'] > 0 else "normal")

    st.success(f"### 🤖 AI Commander's Brief\n{narrative}")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("📡 Live News Signals")
        if not news_df.empty:
            for i, row in news_df.head(5).iterrows():
                title = row['title']
                source = row['source']
                st.markdown(f"**{row['source']}**: {title} *({row['published'].strftime('%H:%M')})*")
        else:
            st.warning("No live news available right now.")
    
    with c2:
        st.subheader("🛢️ Oil Price Trend (STL)")
        st.line_chart(metrics['trend_data'])
        st.caption("Underlying price trend (removing daily market noise). Rising trend = Inflationary Pressure.")

if __name__ == "__main__":
    main()
