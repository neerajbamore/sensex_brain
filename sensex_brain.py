import os
import time
import json
import math
import signal
import requests as rq
import pandas as pd
# PyOTP abhi zaroori hai login ke liye, isliye import karein:
import pyotp 
from datetime import datetime, timedelta, timezone
from logzero import logger
from typing import Dict, List, Tuple, Optional
from SmartApi import SmartConnect # SmartConnect ko direct import karein

# ==========================
# Env / Config
# ==========================
# Credentials, jo Render Env Vars se aayenge
API_KEY        = os.environ.get("api_key")
CLIENT_CODE    = os.environ.get("client_code")
PIN            = os.environ.get("pin")
TOTP_SECRET    = os.environ.get("totp_secret")

TG_BOT_TOKEN   = os.environ.get("telegram_bot_token")
TG_CHAT_ID     = os.environ.get("telegram_chat_id")

# Poll interval (seconds)
POLL_SECS      = int(os.environ.get("poll_secs", "180"))  # 3 minutes

# Movement thresholds for alerts (tune as needed)
THRESH = {
    "oi_pct":   float(os.environ.get("thr_oi_pct",   "10")),  # 10% change
    "iv_abs":   float(os.environ.get("thr_iv_abs",   "1.0")), # +/-1 IV point
    "vol_pct":  float(os.environ.get("thr_vol_pct",  "20")),  # 20% change
    "ltp_pct":  float(os.environ.get("thr_ltp_pct",  "1.5")), # 1.5% change
}

# How many CE/PE each side from ATM
N_STRIKES_EACH_SIDE = int(os.environ.get("n_strikes_each_side", "7"))

# Timezone
IST = timezone(timedelta(hours=5, minutes=30))

# Exchanges / segments used by Angel
EXCH_BSE_CASH = "BSE"
EXCH_BFO_OPT  = "BFO"

INDEX_NAME_FILTER = "SENSEX"
INSTR_OPT   = "OPTIDX"

SCRIP_URL = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"

# ==========================
# Telegram helper
# ==========================
def tg_send(msg: str):
    if not (TG_BOT_TOKEN and TG_CHAT_ID):
        logger.warning("Telegram creds missing; skipping alert")
        return
    try:
        # Request ki encoding ko ensure karne ke liye
        rq.post(
            f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage",
            json={"chat_id": TG_CHAT_ID, "text": msg[:4000], "parse_mode": "Markdown"},
            timeout=10,
        )
    except Exception as e:
        logger.error("Telegram send failed: %s", e)

# ==========================
# SmartAPI login
# ==========================
def login_smartapi():
    if not all([API_KEY, CLIENT_CODE, PIN, TOTP_SECRET]):
        raise RuntimeError("Missing API creds in environment.")

    smart = SmartConnect(API_KEY)
    try:
        # FIX: pyotp abhi bhi yahan use ho raha hai, isliye requirements.txt mein zaroori hai.
        totp = pyotp.TOTP(TOTP_SECRET).now() 
        data = smart.generateSession(CLIENT_CODE, PIN, totp)
        if not data or data.get("status") is False:
            raise RuntimeError(f"Login failed: {data}")
        logger.info("Logged in successfully")
        tg_send("✅ Sensex Brain: SmartAPI login successful.")
        return smart
    except Exception as e:
        logger.exception("Login error")
        raise

# ==========================
# Scrip master helpers
# ==========================
_SCRIP_CACHE: Optional[pd.DataFrame] = None

def load_scrip() -> pd.DataFrame:
    global _SCRIP_CACHE
    if _SCRIP_CACHE is not None:
        return _SCRIP_CACHE
    r = rq.get(SCRIP_URL, timeout=30)
    r.raise_for_status()
    df = pd.DataFrame(r.json())
    # normalize
    df.columns = [c.strip().lower() for c in df.columns]
    _SCRIP_CACHE = df
    return df

def nearest_expiry_for_sensex_options() -> pd.Timestamp:
    df = load_scrip()
    # SENSEX options ke liye robust filtering
    odf = df[
        (df["exch_seg"] == EXCH_BFO_OPT) &
        (df["instrumenttype"] == INSTR_OPT) 
    ].copy()
    
    # FIX: FILTERING BY NAME/SYMBOL (Enhanced Search)
    sensex_filter = (
        odf["name"].str.upper().str.contains(INDEX_NAME_FILTER, na=False) |
        odf["symbol"].str.upper().str.contains(INDEX_NAME_FILTER, na=False) # FIX: Symbol column mein bhi check karein
    )
    odf = odf[sensex_filter].copy()

    if odf.empty:
        raise RuntimeError("No SENSEX option scrip found in master after filtering.")
        
    # to datetime
    odf["expiry"] = pd.to_datetime(odf["expiry"], errors='coerce') 
    odf.dropna(subset=['expiry'], inplace=True)
    
    now = pd.Timestamp.now(tz=IST).normalize()
    # future expiries (>= today 00:00 IST)
    future = odf[odf["expiry"] >= pd.Timestamp(now).tz_localize(None)]
    if future.empty:
        raise RuntimeError("No upcoming SENSEX option expiry found in scrip master")
    # earliest expiry
    exp = future["expiry"].sort_values().unique()[0]
    return exp

def get_sensex_spot_token() -> Tuple[str, str]:
    """Return (token, tradingsymbol) for SENSEX cash index on BSE"""
    df = load_scrip()
    cdf = df[(df["exch_seg"] == EXCH_BSE_CASH) & (df["name"].str.fullmatch("SENSEX", na=False))]
    if cdf.empty:
        # fallback contains
        cdf = df[(df["exch_seg"] == EXCH_BSE_CASH) & (df["name"].str.contains("SENSEX", na=False))]
    if cdf.empty:
        raise RuntimeError("SENSEX cash token not found in scrip master")
    row = cdf.iloc[0]
    return str(row["token"]), row["symbol"]

def get_spot_ltp(smart) -> float:
    token, tsym = get_sensex_spot_token()
    try:
        q = smart.getLtpData(exchange=EXCH_BSE_CASH, tradingsymbol=tsym, symboltoken=str(token))
        ltp = float(q["data"]["ltp"])
        return ltp
    except Exception as e:
        logger.error("Spot LTP fetch failed: %s", e)
        raise # <-- FIX: Agar Spot LTP nahi milta toh fail karo
        
def round_to_100(x: float) -> float:
    return round(x / 100.0) * 100.0

def find_atm_strike(smart) -> float:
    try:
        ltp = get_spot_ltp(smart)
        atm = round_to_100(ltp)
        return atm
    except Exception:
        logger.exception("Could not find current ATM. Script will terminate.")
        raise # <-- FIX: Agar Spot nahi mila toh yahan bhi fail karein

def build_chain_tokens(atm: float, expiry: pd.Timestamp) -> List[Dict]:
    """Return [{'exch':..., 'token':..., 'tsym':..., 'strike':..., 'side': 'CE/PE'}] for N CE + N PE."""
    df = load_scrip()
    odf = df[
        (df["exch_seg"] == EXCH_BFO_OPT) &
        (df["instrumenttype"] == INSTR_OPT)
    ].copy()
    
    # FIX: FILTERING BY NAME/SYMBOL (Enhanced Search)
    sensex_filter = (
        odf["name"].str.upper().str.contains(INDEX_NAME_FILTER, na=False) |
        odf["symbol"].str.upper().str.contains(INDEX_NAME_FILTER, na=False)
    )
    odf = odf[sensex_filter].copy()
    
    odf["expiry"] = pd.to_datetime(odf["expiry"], errors='coerce')
    odf.dropna(subset=['expiry'], inplace=True)
    odf = odf[odf["expiry"] == expiry]

    # strikes we want
    steps = [i * 100 for i in range(0, N_STRIKES_EACH_SIDE + 1)]
    ce_strikes = [atm + s for s in steps]                 # ATM + 0..N*100
    pe_strikes = [atm - s for s in steps if s != 0]       # ATM -100..-N*100 (avoid duplicate ATM)
    targets = set([(st, "CE") for st in ce_strikes] + [(st, "PE") for st in pe_strikes])

    rows = []
    # match by numeric strike + optiontype
    for _, r in odf.iterrows():
        try:
            # Check for valid strike value
            strike_value = r.get("strike")
            if strike_value is None or pd.isna(strike_value):
                continue
            
            strike = float(strike_value)
            opt = str(r.get("symbol", "")).strip().upper()  # contains CE/PE at end usually
            side = "CE" if opt.endswith("CE") else ("PE" if opt.endswith("PE") else None)
            if side is None:
                continue
            key = (strike, side)
            if key in targets:
                rows.append({
                    "exch": r["exch_seg"],
                    "token": str(r["token"]),
                    "tsym": r["symbol"],
                    "strike": strike,
                    "side": side,
                })
        except Exception:
            continue

    # sort nicely: CE low->high, then PE high->low (closer to ATM first)
    ces = sorted([x for x in rows if x["side"] == "CE"], key=lambda z: z["strike"])
    pes = sorted([x for x in rows if x["side"] == "PE"], key=lambda z: -z["strike"])
    chain = ces + pes
    
    if not chain:
        # FIX: Agar is point par chain empty hai toh error do
        logger.error("Found %d SENSEX options for expiry %s, but 0 matched strikes (ATM: %.0f).", 
                     len(odf[odf["expiry"] == expiry]), expiry.date(), atm)
        raise RuntimeError("Option chain tokens not found for requested strikes/expiry.") # <-- FIX: Runtime Error
        
    return chain

# ==========================
# Quote parsing helpers
# ==========================
# (No changes needed in this section)
def _first_present(d: dict, *keys):
    for k in keys:
        if d is None:
            break
        if "." in k:
            cur = d
            ok = True
            for part in k.split("."):
                if isinstance(cur, dict) and part in cur:
                    cur = cur[part]
                else:
                    ok = False
                    break
            if ok and cur is not None:
                return cur
        else:
            if k in d and d[k] is not None:
                return d[k]
    return None

def parse_quote_payload(qdata: dict) -> Dict[str, Optional[float]]:
    """
    Try to extract ltp, oi, iv, volume from Angel REST quote.
    """
    if isinstance(qdata, dict) and "data" in qdata:
        data = qdata["data"]
        if isinstance(data, list) and data:
            payload = data[0]
        elif isinstance(data, dict):
            payload = data
        else:
            payload = qdata
    else:
        payload = qdata

    ltp    = _first_present(payload, "ltp", "last_price", "lastprice", "last_traded_price")
    oi     = _first_present(payload, "oi", "open_interest", "openinterest", "o_i")
    iv     = _first_present(payload, "iv", "implied_volatility", "impliedvol", "greeks.iv")
    volume = _first_present(payload, "volume", "total_traded_volume", "tottrdvol")
    try:
        return {
            "ltp":    float(ltp)    if ltp    is not None else None,
            "oi":     float(oi)     if oi     is not None else None,
            "iv":     float(iv)     if iv     is not None else None,
            "volume": float(volume) if volume is not None else None,
        }
    except Exception:
        out = {}
        for k, v in [("ltp", ltp), ("oi", oi), ("iv", iv), ("volume", volume)]:
            try:
                out[k] = float(v) if v is not None else None
            except Exception:
                out[k] = None
        return out

# ==========================
# Technicals on SENSEX spot
# ==========================
# (No changes needed in this section)
class SpotTech:
    def __init__(self, max_len=200):
        self.ts: List[pd.Timestamp] = []
        self.pr: List[float] = []
        self.max_len = max_len

    def push(self, price: float):
        self.ts.append(pd.Timestamp.now(tz=IST))
        self.pr.append(float(price))
        if len(self.pr) > self.max_len:
            self.ts = self.ts[-self.max_len:]
            self.pr = self.pr[-self.max_len:]

    def indicators(self) -> Dict[str, float]:
        if len(self.pr) < 35:
            return {}
        s = pd.Series(self.pr)
        ema_fast = s.ewm(span=12, adjust=False).mean().iloc[-1]
        ema_slow = s.ewm(span=26, adjust=False).mean().iloc[-1]
        macd = ema_fast - ema_slow
        signal = pd.Series([ema_fast - ema_slow for ema_fast, ema_slow in zip(
            s.ewm(span=12, adjust=False).mean(),
            s.ewm(span=26, adjust=False).mean()
        )]).ewm(span=9, adjust=False).mean().iloc[-1]
        # RSI(14)
        delta = s.diff()
        up = delta.clip(lower=0).rolling(14).mean()
        down = (-delta.clip(upper=0)).rolling(14).mean()
        rs = (up / (down + 1e-9)).iloc[-1]
        rsi = 100 - (100 / (1 + rs))
        return {"ema12": float(ema_fast), "ema26": float(ema_slow), "macd": float(macd), "signal": float(signal), "rsi": float(rsi)}

# ==========================
# Main loop
# ==========================
def main_loop():
    smart = login_smartapi()

    expiry = nearest_expiry_for_sensex_options()
    logger.info("Nearest SENSEX options expiry: %s", expiry.strftime("%Y-%m-%d"))

    atm = find_atm_strike(smart)
    logger.info("Calculated ATM strike: %.0f", atm)

    chain = build_chain_tokens(atm, expiry)
    logger.info("Tracking %d option instruments (total)", len(chain))

    tg_send(f"🧠 Sensex Brain started\nExpiry: {expiry.date()}\nATM: {atm:.0f}\nTracking {len(chain)} options (±{N_STRIKES_EACH_SIDE} strikes).")

    prev: Dict[str, Dict[str, Optional[float]]] = {}
    spot_hist = SpotTech()

    while True:
        cycle_started = datetime.now(tz=IST)
        try:
            # Spot LTP + indicators
            spot = get_spot_ltp(smart)
            spot_hist.push(spot)
            tech = spot_hist.indicators()

            # Pull quotes for each option (REST)
            alerts = []
            for scrip in chain:
                try:
                    quote_payload = None
                    try:
                        quote_payload = smart.getQuotes({"exchange": scrip["exch"], "symboltoken": scrip["token"]})
                    except Exception:
                        quote_payload = smart.getLtpData(exchange=scrip["exch"], tradingsymbol=scrip["tsym"], symboltoken=scrip["token"])

                    parsed = parse_quote_payload(quote_payload)
                    key = f'{scrip["side"]}:{int(scrip["strike"])}'

                    # Compare with previous snapshot
                    if key in prev:
                        p = prev[key]
                        # OI %
                        if parsed.get("oi") is not None and p.get("oi"):
                            chg = (parsed["oi"] - p["oi"]) / max(p["oi"], 1e-9) * 100.0
                            if abs(chg) >= THRESH["oi_pct"]:
                                alerts.append(f'⚠️ OI {chg:+.1f}% | {key} | OI {p["oi"]:.0f} → {parsed["oi"]:.0f}')
                        # IV abs
                        if parsed.get("iv") is not None and p.get("iv") is not None:
                            diff = parsed["iv"] - p["iv"]
                            if abs(diff) >= THRESH["iv_abs"]:
                                alerts.append(f'⚠️ IV {diff:+.2f} | {key} | IV {p["iv"]:.2f} → {parsed["iv"]:.2f}')
                        # Volume %
                        if parsed.get("volume") is not None and p.get("volume"):
                            chg = (parsed["volume"] - p["volume"]) / max(p["volume"], 1e-9) * 100.0
                            if abs(chg) >= THRESH["vol_pct"]:
                                alerts.append(f'⚠️ VOL {chg:+.1f}% | {key} | {p["volume"]:.0f} → {parsed["volume"]:.0f}')
                        # LTP %
                        if parsed.get("ltp") is not None and p.get("ltp"):
                            chg = (parsed["ltp"] - p["ltp"]) / max(p["ltp"], 1e-9) * 100.0
                            if abs(chg) >= THRESH["ltp_pct"]:
                                alerts.append(f'⚠️ LTP {chg:+.2f}% | {key} | {p["ltp"]:.2f} → {parsed["ltp"]:.2f}')
                    # save snapshot
                    prev[key] = parsed
                except Exception as e:
                    logger.warning("Quote fetch/parsing error for %s: %s", scrip["tsym"], e)

            # If any alerts, send one combined Telegram message
            if alerts:
                techline = ""
                if tech:
                    techline = f"\nSpot *{spot:.2f}* | EMA *{tech['ema12']:.0f}* / *{tech['ema26']:.0f}* | MACD *{tech['macd']:.2f}*/*{tech['signal']:.2f}* | RSI *{tech['rsi']:.1f}*"
                msg = "🚨 Sensex Options Alert(s):\n• " + "\n• ".join(alerts) + techline
                tg_send(msg)
                logger.info("Alerts sent: %d", len(alerts))
            else:
                logger.info("No significant changes this cycle. Spot=%.2f", spot)

        except Exception as e:
            logger.exception("Cycle error: %s", e)
            tg_send(f"❌Sensex Brain cycle error: {e}")

        # sleep until next 3-minute boundary from cycle start
        elapsed = (datetime.now(tz=IST) - cycle_started).total_seconds()
        sleep_for = max(5, POLL_SECS - int(elapsed))
        time.sleep(sleep_for)

# ==========================
# Entrypoint
# ==========================
def _graceful_exit(signum, frame):
    logger.info("Shutting down (signal %s)...", signum)
    tg_send("🛑 Sensex Brain stopping.")
    raise SystemExit

if __name__ == "__main__":
    signal.signal(signal.SIGINT, _graceful_exit)
    signal.signal(signal.SIGTERM, _graceful_exit)
    main_loop()
