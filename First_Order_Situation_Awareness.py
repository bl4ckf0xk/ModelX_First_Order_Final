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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Attempt to import autorefresh, but handle failure gracefully if not installed
try:
    from streamlit_autorefresh import st_autorefresh
    HAS_AUTOREFRESH = True
except ImportError:
    HAS_AUTOREFRESH = False

# Configure Page
st.set_page_config(page_title="First Order: Sri Lanka Situational Awareness", layout="wide", page_icon="🇱🇰")

# --- TIER 1: DATA INGESTION LAYERS ---

class BaseScraper:
    USER_AGENTS = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15'
    ]
    def _get_headers(self):
        return {'User-Agent': random.choice(self.USER_AGENTS)}

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
                for entry in feed.entries[:6]:
                    published = datetime.now()
                    if 'published_parsed' in entry:
                        published = datetime.fromtimestamp(time.mktime(entry.published_parsed))
                    news_items.append({
                        "source": source,
                        "title": entry.title,
                        "link": entry.link,
                        "published": published,
                        "type": "news",
                        "score": 0
                    })
            except Exception: pass
        return pd.DataFrame(news_items).sort_values(by="published", ascending=False)

class SocialPulseIngestor(BaseScraper):
    def fetch_reddit_trends(self):
        # Fetch 100 posts to build a time-series trend
        url = "https://www.reddit.com/r/srilanka/new.json?limit=100"
        try:
            headers = {'User-Agent': 'ModelX-Dashboard/1.0'}
            resp = requests.get(url, headers=headers, timeout=5)
            if resp.status_code != 200: return pd.DataFrame()
            
            data = resp.json()
            posts = []
            for item in data['data']['children']:
                post = item['data']
                posts.append({
                    "source": "Reddit (r/srilanka)",
                    "title": post['title'],
                    "link": f"https://reddit.com{post['permalink']}",
                    "published": datetime.fromtimestamp(post['created_utc']),
                    "score": post['score'] + post['num_comments'],
                    "type": "social"
                })
            return pd.DataFrame(posts)
        except Exception: return pd.DataFrame()

class MarketDataIngestor:
    def fetch_usd_lkr(self):
        try:
            ticker = yf.Ticker("LKR=X")
            hist = ticker.history(period="1mo")
            if hist.empty: return pd.Series(dtype=float)
            return hist['Close']
        except: return pd.Series(dtype=float)

    def fetch_oil_price(self):
        try:
            ticker = yf.Ticker("BZ=F") 
            hist = ticker.history(period="1mo")
            if hist.empty: return pd.Series(dtype=float)
            return hist['Close']
        except: return pd.Series(dtype=float)

# --- TIER 2: INTELLIGENCE & ANALYTICS ---

def process_social_trend(social_df):
    """Generates a time-series of social engagement over the last 24h"""
    if social_df.empty: return pd.Series(dtype=float)
    
    # 1. Convert to datetime
    df = social_df.copy()
    df['published'] = pd.to_datetime(df['published'])
    df.set_index('published', inplace=True)
    
    # 2. Resample by Hour to get 'Velocity' of discussions
    # We sum the 'score' (Upvotes + Comments) to see intensity
    trend = df['score'].resample('2H').sum().fillna(0)
    
    # Return the last 12 periods (24 hours)
    return trend.tail(12)

def train_and_detect_anomalies(history_series):
    if len(history_series) < 10: return False
    X = history_series.values.reshape(-1, 1)
    clf = IsolationForest(random_state=42, contamination='auto')
    clf.fit(X)
    return clf.predict(X[-1].reshape(1, -1))[0] == -1

def analyze_sentiment(text_df):
    analyzer = SentimentIntensityAnalyzer()
    if text_df.empty: return 0, False, []
    
    scores = []
    crisis_keywords = ["protest", "strike", "shortage", "power", "fuel", "crisis", "inflation", "curfew"]
    triggered = []
    
    # Analyze sentiment of top 20 items
    for title in text_df['title'].head(20):
        vs = analyzer.polarity_scores(title)
        scores.append(vs['compound'])
        for k in crisis_keywords:
            if k in title.lower(): triggered.append(k)
            
    avg = np.mean(scores) if scores else 0
    return avg, (avg < -0.2 or len(triggered) > 1), list(set(triggered))

def compute_analytics(combined_df, social_trend, usd, oil):
    # 1. Forex Anomaly
    fx_anomaly = train_and_detect_anomalies(usd)
    
    # 2. Text Sentiment
    sent_score, social_risk, kws = analyze_sentiment(combined_df)
    
    # 3. Oil Trend Slope
    oil_slope = 0
    trend_data_oil = oil # Default to raw data
    if not oil.empty:
        try:
            stl = STL(oil, period=5, robust=True)
            res = stl.fit()
            trend_data_oil = res.trend # Use smooth trend
            oil_slope = res.trend.iloc[-1] - res.trend.iloc[-4]
        except: pass

    # 4. Social Trend Slope (Is discussion heating up?)
    social_spike = False
    social_slope = 0
    if len(social_trend) > 2:
        current_vol = social_trend.iloc[-1]
        avg_vol = social_trend.mean()
        social_slope = current_vol - avg_vol
        if current_vol > (avg_vol * 1.5): social_spike = True 

    # 5. BSI Score
    score = 100
    if fx_anomaly: score -= 20
    if social_risk: score -= 20
    if social_spike: score -= 15 
    if oil_slope > 1.5: score -= 15
    if sent_score > 0.2: score += 10
    
    return {
        "bsi": max(0, min(100, score)),
        "oil_slope": oil_slope,
        "social_slope": social_slope,
        "social_spike": social_spike,
        "fx_anomaly": fx_anomaly,
        "social_risk": social_risk,
        "sentiment": sent_score,
        "keywords": kws,
        "trend_data_oil": trend_data_oil,
        "trend_data_social": social_trend,
        "curr_usd": usd.iloc[-1] if not usd.empty else 0,
        "curr_oil": oil.iloc[-1] if not oil.empty else 0,
        "total_posts": len(combined_df)
    }

def render_scenario_brief(a):
    st.markdown("### 🤖 AI Commander's Advice")
    
    alerts_triggered = False

    # RED ALERTS (CRITICAL RISKS)
    if a['fx_anomaly']:
        st.error("**💰 FINANCE ALERT:** Abnormal currency volatility detected.\n\n**Action:** Pause non-essential USD payments. Hedge currency exposure.", icon="💸")
        alerts_triggered = True

    if a['social_risk'] or a['social_spike']:
        risk_type = "Viral Negative Discourse" if a['social_spike'] else "Public Unrest Signals"
        st.error(f"**🛡️ OPERATIONS ALERT:** High social risk detected ({risk_type}).\n\n**Action:** Review employee commute safety. Prepare for absenteeism.", icon="🔥")
        alerts_triggered = True

    if a['oil_slope'] > 1.5:
        st.error("**🚚 LOGISTICS ALERT:** Oil prices are trending UP rapidly.\n\n**Action:** Lock in forward transport contracts now.", icon="🛢️")
        alerts_triggered = True

    # GREEN ALERTS (OPPORTUNITIES)
    if a['oil_slope'] < -1.5:
        st.success("**🚚 LOGISTICS OPPORTUNITY:** Fuel costs are trending DOWN.\n\n**Action:** Renegotiate logistics rates for next month.", icon="📉")
        alerts_triggered = True

    if not a['fx_anomaly'] and a['bsi'] > 80:
        st.success("**📈 STRATEGIC OPPORTUNITY:** Market conditions are highly stable.\n\n**Action:** Favorable window for Capital Expenditure (CapEx).", icon="✅")
        alerts_triggered = True

    # GREY (STATUS QUO)
    if not alerts_triggered:
        st.info("**🛡️ STATUS QUO:** No immediate high-impact threats detected.\n\n**Action:** Continue standard operations. Monitor feeds.", icon="ℹ️")
        
# --- UI EXECUTION ---

def main():
    st.title("🇱🇰 First Order: Real-Time Intelligence Platform")
    
    # Optional: Use streamlit-autorefresh if installed
    if HAS_AUTOREFRESH:
        st_autorefresh(interval=100000, key="datarefresh") # 10 seconds

    # Layout Header
    c1, c2 = st.columns([3, 1])
    c1.markdown(f"**🔴 LIVE MONITORING** | Last Update: {datetime.now().strftime('%H:%M:%S')}")
    if c2.button("🔄 Force Refresh", type="primary"): st.rerun()

    with st.spinner("Analyzing real-time signals..."):
        # Init
        news = NewsIngestor()
        social = SocialPulseIngestor()
        market = MarketDataIngestor()

        # Fetch
        news_df = news.fetch_live_news()
        social_df = social.fetch_reddit_trends()
        
        # Process
        social_trend = process_social_trend(social_df)
        combined_text = pd.concat([news_df, social_df], ignore_index=True)
        usd_series = market.fetch_usd_lkr()
        oil_series = market.fetch_oil_price()
        
        # Compute
        metrics = compute_analytics(combined_text, social_trend, usd_series, oil_series)

    # --- KPI ROW ---
    st.divider()
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Stability Index (BSI)", f"{metrics['bsi']:.0f}/100", delta="Stable" if metrics['bsi']>75 else "Risk", delta_color="normal" if metrics['bsi']>75 else "inverse")
    k2.metric("LKR/USD", f"{metrics['curr_usd']:.2f}", delta="Volatile" if metrics['fx_anomaly'] else "Normal", delta_color="inverse" if metrics['fx_anomaly'] else "off")
    k3.metric("Oil (Brent)", f"${metrics['curr_oil']:.2f}", delta=f"{metrics['oil_slope']:.2f} Trend", delta_color="inverse" if metrics['oil_slope']>0 else "normal")
    k4.metric("Social Velocity", f"{metrics['social_slope']:.1f}", delta="Surge" if metrics['social_spike'] else "Normal", delta_color="inverse" if metrics['social_spike'] else "off")

    # --- COLOR-CODED ADVICE SECTION ---
    # We call the renderer function here instead of st.success
    render_scenario_brief(metrics)
    
    # --- CHARTS ROW ---
    left, right = st.columns(2)
    
    with left:
        st.subheader("📡 Live Feed (News & Social)")
        if not combined_text.empty:
            df_display = combined_text.sort_values(by="published", ascending=False).head(5)
            for i, row in df_display.iterrows():
                icon = "📰" if row['type'] == "news" else "💬"
                # Handle missing links
                link = row['link'] if 'link' in row else '#'
                st.markdown(f"{icon} **{row['source']}**: [{row['title']}]({link})")
        else:
            st.warning("No live data.")

    with right:
        # Chart 1: Oil
        st.subheader("🛢️ Oil Price Trend")
        st.line_chart(metrics['trend_data_oil'], height=150)
        
        # Chart 2: Social Media Volume
        st.subheader("📊 Social Media Volume (r/srilanka)")
        if not metrics['trend_data_social'].empty:
            st.bar_chart(metrics['trend_data_social'], height=150)
            st.caption("Hourly volume of discussions. Spikes indicate emerging events.")
        else:
            st.caption("Insufficient social data to plot trend.")

    # --- FALLBACK REFRESH LOGIC ---
    # If st_autorefresh is not present, use this manual loop
    if not HAS_AUTOREFRESH:
        st.divider()
        st.caption("Auto-refreshing in 10 seconds...")
        time.sleep(10)
        st.rerun()

if __name__ == "__main__":
    main()
