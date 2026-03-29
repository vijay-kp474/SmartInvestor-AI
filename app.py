from flask import Flask, render_template, request, jsonify
import yfinance as yf
import numpy as np
from openai import OpenAI

# 🔑 GROQ API key 
client = OpenAI(
    api_key="key",
    base_url="https://api.groq.com/openai/v1"
)

app = Flask(__name__)

# ─────────────────────────────────────────
# 📊 Core Stock Analysis
# ─────────────────────────────────────────
def analyze_stock(symbol):
    stock = yf.Ticker(symbol)
    df = stock.history(period="6mo")

    if df.empty:
        return {"error": f"Invalid stock symbol: {symbol}"}

    # Moving Averages
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()

    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df['Close'].ewm(span=12).mean()
    ema26 = df['Close'].ewm(span=26).mean()
    df['MACD'] = ema12 - ema26
    df['Signal_Line'] = df['MACD'].ewm(span=9).mean()

    # Bollinger Bands
    df['BB_Mid'] = df['Close'].rolling(window=20).mean()
    df['BB_Std'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Mid'] + 2 * df['BB_Std']
    df['BB_Lower'] = df['BB_Mid'] - 2 * df['BB_Std']

    latest = df.iloc[-1]
    prev = df.iloc[-2]

    latest_price = latest['Close']
    latest_ma20  = latest['MA20'] if not np.isnan(latest['MA20']) else None
    latest_ma50  = latest['MA50'] if not np.isnan(latest['MA50']) else None
    latest_rsi   = latest['RSI'] if not np.isnan(latest['RSI']) else None
    latest_macd  = latest['MACD'] if not np.isnan(latest['MACD']) else None
    latest_sig   = latest['Signal_Line'] if not np.isnan(latest['Signal_Line']) else None
    bb_upper     = latest['BB_Upper'] if not np.isnan(latest['BB_Upper']) else None
    bb_lower     = latest['BB_Lower'] if not np.isnan(latest['BB_Lower']) else None

    # ── Signal Logic ──────────────────────
    score = 0
    reasons = []

    if latest_ma20 is not None and latest_price > latest_ma20:
        score += 1; reasons.append("Price above 20-day MA")
    if latest_ma50 is not None and latest_price > latest_ma50:
        score += 1; reasons.append("Price above 50-day MA")
    if latest_rsi is not None and latest_rsi < 70:
        score += 1
    if latest_rsi is not None and latest_rsi < 40:
        score += 1; reasons.append("RSI oversold — potential bounce")
    if latest_macd is not None and latest_sig is not None and latest_macd > latest_sig:
        score += 1; reasons.append("MACD bullish crossover")
    if bb_lower is not None and latest_price <= bb_lower:
        score += 1; reasons.append("Price at lower Bollinger Band — reversal zone")

    if latest_rsi is not None and latest_rsi > 75:
        score -= 2; reasons.append("RSI overbought — caution")
    if bb_upper is not None and latest_price >= bb_upper:
        score -= 1; reasons.append("Price at upper Bollinger Band")
    if latest_macd is not None and latest_sig is not None and latest_macd < latest_sig:
        score -= 1; reasons.append("MACD bearish signal")

    if score >= 3:
        signal = "STRONG BUY 🚀"
        trend = "Bullish"
    elif score >= 1:
        signal = "BUY 📈"
        trend = "Mildly Bullish"
    elif score <= -2:
        signal = "HIGH RISK ⚠️"
        trend = "Overbought/Bearish"
    else:
        signal = "WAIT ⏳"
        trend = "Neutral"

    # ── Chart Pattern Detection ───────────
    patterns = detect_patterns(df)

    # ── Volume Spike ──────────────────────
    avg_vol = df['Volume'].rolling(window=20).mean().iloc[-1]
    vol_spike = int(latest['Volume'] > 1.5 * avg_vol) if not np.isnan(avg_vol) else 0

    return {
        "price":     round(float(latest_price), 2),
        "ma20":      round(float(latest_ma20),  2) if latest_ma20 is not None else None,
        "ma50":      round(float(latest_ma50),  2) if latest_ma50 is not None else None,
        "rsi":       round(float(latest_rsi),   2) if latest_rsi is not None else None,
        "macd":      round(float(latest_macd),  4) if latest_macd is not None else None,
        "signal":    signal,
        "trend":     trend,
        "score":     score,
        "reasons":   reasons,
        "patterns":  patterns,
        "vol_spike": vol_spike,
        "bb_upper":  round(float(bb_upper), 2) if bb_upper is not None else None,
        "bb_lower":  round(float(bb_lower), 2) if bb_lower is not None else None,
        "dates":     df.index.strftime('%Y-%m-%d').tolist()[-60:],
        "prices":    [round(float(p), 2) for p in df['Close'].tolist()[-60:]],
        "volumes":   [int(v) for v in df['Volume'].tolist()[-60:]],
        "ma20_line": [round(float(v), 2) if not np.isnan(v) else None
                      for v in df['MA20'].tolist()[-60:]],
        "bb_upper_line": [round(float(v), 2) if not np.isnan(v) else None
                          for v in df['BB_Upper'].tolist()[-60:]],
        "bb_lower_line": [round(float(v), 2) if not np.isnan(v) else None
                          for v in df['BB_Lower'].tolist()[-60:]],
    }


# ─────────────────────────────────────────
# 🔍 Chart Pattern Detection
# ─────────────────────────────────────────
def detect_patterns(df):
    patterns = []
    closes = df['Close'].values

    # Golden Cross / Death Cross (MA20 vs MA50)
    ma20 = df['MA20'].values
    ma50 = df['MA50'].values
    if len(ma20) >= 2 and len(ma50) >= 2:
        if not (np.isnan(ma20[-1]) or np.isnan(ma20[-2]) or
                np.isnan(ma50[-1]) or np.isnan(ma50[-2])):
            if ma20[-2] < ma50[-2] and ma20[-1] > ma50[-1]:
                patterns.append({"name": "Golden Cross 🌟", "type": "bullish",
                                  "desc": "20-day MA crossed above 50-day MA — strong buy signal"})
            elif ma20[-2] > ma50[-2] and ma20[-1] < ma50[-1]:
                patterns.append({"name": "Death Cross ☠️", "type": "bearish",
                                  "desc": "20-day MA crossed below 50-day MA — sell signal"})

    # RSI Divergence (price up, RSI down = bearish divergence)
    rsi = df['RSI'].values
    window = 10
    if len(closes) > window:
        price_trend = closes[-1] - closes[-window]
        rsi_trend   = rsi[-1]   - rsi[-window]
        if price_trend > 0 and rsi_trend < -5:
            patterns.append({"name": "Bearish RSI Divergence 📉", "type": "bearish",
                              "desc": "Price rising but RSI falling — momentum weakening"})
        elif price_trend < 0 and rsi_trend > 5:
            patterns.append({"name": "Bullish RSI Divergence 📈", "type": "bullish",
                              "desc": "Price falling but RSI rising — reversal likely"})

    # Support / Resistance breakout
    recent = closes[-20:]
    resistance = np.max(recent[:-3])
    support    = np.min(recent[:-3])
    curr_price = closes[-1]

    if curr_price > resistance * 1.01:
        patterns.append({"name": "Resistance Breakout 🔥", "type": "bullish",
                          "desc": f"Price broke above resistance ₹{round(float(resistance),2)}"})
    elif curr_price < support * 0.99:
        patterns.append({"name": "Support Breakdown ⛔", "type": "bearish",
                          "desc": f"Price broke below support ₹{round(float(support),2)}"})

    # Consecutive up/down days
    if len(closes) >= 5:
        last5 = closes[-5:]
        if all(last5[i] < last5[i+1] for i in range(4)):
            patterns.append({"name": "5-Day Uptrend 🔼", "type": "bullish",
                              "desc": "5 consecutive green days — strong momentum"})
        elif all(last5[i] > last5[i+1] for i in range(4)):
            patterns.append({"name": "5-Day Downtrend 🔽", "type": "bearish",
                              "desc": "5 consecutive red days — selling pressure"})

    return patterns


# ─────────────────────────────────────────
# 🤖 AI Explanation (Groq/LLaMA)
# ─────────────────────────────────────────
def generate_explanation(data, symbol):
    try:
        patterns_text = ", ".join([p['name'] for p in data.get('patterns', [])]) or "None detected"
        reasons_text  = "; ".join(data.get('reasons', [])) or "General analysis"

        ma20_text = f"MA20: ₹{data['ma20']}" if data['ma20'] is not None else "MA20: N/A"
        ma50_text = f"MA50: ₹{data['ma50']}" if data['ma50'] is not None else "MA50: N/A"
        rsi_text = f"RSI: {data['rsi']}" if data['rsi'] is not None else "RSI: N/A"
        macd_text = f"MACD: {data['macd']}" if data['macd'] is not None else "MACD: N/A"

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system",
                 "content": ("You are a professional Indian stock market analyst. "
                             "Give concise, actionable advice in 2-3 sentences. "
                             "Mention specific numbers. Be direct.")},
                {"role": "user",
                 "content": (f"Stock: {symbol}\n"
                             f"Price: ₹{data['price']} | {ma20_text} | {ma50_text}\n"
                             f"{rsi_text} | {macd_text}\n"
                             f"Signal: {data['signal']} | Trend: {data['trend']}\n"
                             f"Patterns: {patterns_text}\n"
                             f"Key reasons: {reasons_text}\n\n"
                             "Explain what this means for a retail investor in 2-3 clear lines.")}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"AI insight unavailable: {str(e)}"


# ─────────────────────────────────────────
# 🏠 Routes
# ─────────────────────────────────────────
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    data   = request.json
    symbol = data.get("symbol", "").upper().strip()

    if not symbol:
        return jsonify({"error": "No symbol provided"})

    if not symbol.endswith(".NS"):
        symbol += ".NS"

    try:
        result = analyze_stock(symbol)

        if "error" not in result:
            result["explanation"] = generate_explanation(result, symbol.replace(".NS", ""))

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/portfolio", methods=["POST"])
def portfolio():
    data    = request.json
    symbols = data.get("symbols", [])
    results = []

    for sym in symbols[:8]:   # cap at 8 stocks
        s = sym.upper().strip()
        if not s.endswith(".NS"):
            s += ".NS"
        r = analyze_stock(s)
        if "error" not in r:
            r["symbol"] = sym.upper()
            results.append(r)

    return jsonify({"portfolio": results})


@app.route("/chat", methods=["POST"])
def chat():
    data    = request.json
    message = data.get("message", "")

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system",
                 "content": ("You are SmartInvestor AI — India's smartest stock market assistant. "
                             "You understand NSE/BSE stocks, mutual funds, SIPs, FII/DII flows, "
                             "technical analysis, and fundamental analysis. "
                             "Give specific, actionable answers. Mention Indian market context. "
                             "Keep answers under 4 sentences unless asked for more.")},
                {"role": "user", "content": message}
            ]
        )
        reply = response.choices[0].message.content
    except Exception as e:
        reply = f"Error: {str(e)}"

    return jsonify({"reply": reply})


if __name__ == "__main__":
    app.run(debug=True)
