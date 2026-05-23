"""
Market Monitor Dashboard
Data source: Stockbee Market Monitor Google Sheet (live fetch on every run).

Usage:  python market_monitor.py   →  market_monitor.html
"""

from pathlib import Path
import argparse
import io
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import pandas as pd

# Load .env if present (keeps secrets out of source control)
_env = Path(__file__).parent / ".env"
if _env.exists():
    for _line in _env.read_text().splitlines():
        if "=" in _line and not _line.startswith("#"):
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

OUT_HTML     = Path(__file__).parent / "market_monitor.html"
STOCKBEE_URL = (
    "https://docs.google.com/spreadsheets/d/"
    "1O6OhS7ciA8zwfycBfGPbP2fWJnR0pn2UUvFZVDP9jpE/export?format=csv"
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _s(series):
    return [None if pd.isna(v) else round(float(v), 6) for v in series]

def _d(dates):
    return [d.strftime("%Y-%m-%d") for d in dates]

def _badge_cls(val, *, higher_good=True, neutral_at=None):
    if pd.isna(val):
        return "neu"
    if neutral_at is not None and abs(val - neutral_at) < 0.03:
        return "neu"
    return "up" if (val > (neutral_at or 0)) == higher_good else "dn"

def _get_plotly_js():
    import plotly
    for name in ("plotly.min.js", "plotly.js"):
        p = Path(plotly.__file__).parent / "package_data" / name
        if p.exists():
            return p.read_text(encoding="utf-8")
    return None


# ── data loading ──────────────────────────────────────────────────────────────

def load() -> pd.DataFrame:
    """Fetch all data from Stockbee Google Sheet."""
    try:
        r = requests.get(STOCKBEE_URL, timeout=30)
        r.raise_for_status()
    except Exception as e:
        raise RuntimeError(f"Stockbee fetch failed: {e}")

    raw = pd.read_csv(io.StringIO(r.text), header=1)
    mask = raw.iloc[:, 0].astype(str).str.match(r"^\d{1,2}/\d{1,2}/\d{4}$", na=False)
    raw = raw[mask].copy()
    if raw.empty:
        raise RuntimeError("No data rows found in Stockbee sheet")

    def _num(col_idx):
        return pd.to_numeric(
            raw.iloc[:, col_idx].astype(str).str.replace(",", ""), errors="coerce"
        )

    universe = _num(13)

    df = pd.DataFrame()
    df["date"]               = pd.to_datetime(raw.iloc[:, 0], format="%m/%d/%Y", errors="coerce")
    df["up_4pct"]            = _num(1)
    df["dn_4pct"]            = _num(2)
    df["ratio_5d"]           = _num(3)
    df["ratio_10d"]          = _num(4)
    df["up_25pct_quarter"]   = _num(5)  / universe
    df["down_25pct_quarter"] = _num(6)  / universe
    df["up_25pct_month"]     = _num(7)  / universe
    df["down_25pct_month"]   = _num(8)  / universe
    df["up_50pct_month"]     = _num(9)  / universe
    df["down_50pct_month"]   = _num(10) / universe
    df["up_13pct_34day"]     = _num(11) / universe
    df["down_13pct_34day"]   = _num(12) / universe
    df["t2108"]              = _num(14) / 100
    df["sp500"]              = _num(15)

    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    # Compute SMA on full history before cutting to 6 months
    df["sp500_sma50"] = df["sp500"].rolling(50, min_periods=1).mean()

    cutoff = df["date"].max() - pd.DateOffset(months=6)
    df = df[df["date"] >= cutoff].reset_index(drop=True)

    print(f"  Loaded {len(df)} rows  {df['date'].iloc[0].date()} → {df['date'].iloc[-1].date()}")
    return df


# ── Claude analysis ───────────────────────────────────────────────────────────

def _md_to_html(text: str) -> str:
    import re
    html, in_ul = [], False
    for line in text.splitlines():
        # Section headers: **1. Title**
        m = re.match(r'^\*\*(.+)\*\*\s*$', line.strip())
        if m:
            if in_ul:
                html.append("</ul>"); in_ul = False
            html.append(f'<p class="ai-section">{m.group(1)}</p>')
            continue
        # Bullet points
        if re.match(r'^[-•*] ', line.strip()):
            if not in_ul:
                html.append("<ul>"); in_ul = True
            content = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', line.strip()[2:])
            html.append(f"<li>{content}</li>")
            continue
        # Close list on blank or non-bullet line
        if in_ul:
            html.append("</ul>"); in_ul = False
        if line.strip():
            content = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', line.strip())
            html.append(f"<p>{content}</p>")
    if in_ul:
        html.append("</ul>")
    return "\n".join(html)


def get_claude_analysis(df: pd.DataFrame) -> str:
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        return "<p style='color:var(--red)'>GEMINI_API_KEY not set — add it to .env</p>"

    prompt = """You are an experienced professional swing momentum trader with years of experience. Focus on technical momentum, price action, volume, and market breadth for 3–15 day holding periods.

Provide a clear, concise market status update for swing trading right now. Use the latest price data and sentiment.

Structure your response exactly like this:

**1. Overall Market Bias**
(Bullish / Bearish / Neutral / Cautious) + one-sentence justification.

**2. Market Momentum & Quality**
- Key indices (SPX): trend, position relative to 20/50/200 DMAs, volume profile.
- Market breadth.
- VIX level and trend.

**3. Sector Rotation & Leadership**
Top 2–3 leading sectors, top 2–3 leading industries and top 2–3 leading themes right now. Highlight any strong momentum rotation.

**4. Key Levels & Risk**
Major volatility outlook, and primary risks for the next 1–2 weeks.

**5. Headlines & News**
Summarize today's top market headlines and their likely impact on the stock market over the next 1–2 weeks.

**6. Final Trade Thesis**
One-paragraph summary of the highest-probability swing opportunity right now and key catalysts to watch.

Be objective, data-driven, and honest about weak conditions. Highlight any signs of exhaustion or reversal. Prioritize high relative strength setups with volume confirmation."""

    try:
        resp = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}",
            headers={"content-type": "application/json"},
            json={"contents": [{"parts": [{"text": prompt}]}]},
            timeout=60,
        )
        if not resp.ok:
            return f"<p style='color:var(--red)'>Gemini error {resp.status_code}: {resp.text[:400]}</p>"
        data          = resp.json()
        candidate     = data["candidates"][0]
        finish_reason = candidate.get("finishReason", "")
        text          = candidate["content"]["parts"][0]["text"].strip()
        html          = _md_to_html(text)
        if finish_reason not in ("STOP", ""):
            html += f"<p style='color:var(--yel)'>[Cut off — reason: {finish_reason}]</p>"
        return html
    except Exception as e:
        return f"<p style='color:var(--red)'>{e}</p>"


# ── HTML builder ──────────────────────────────────────────────────────────────

def build_html(df: pd.DataFrame, analysis: str = "", pjs: str = "") -> str:
    L   = df.iloc[-1]
    dt  = df["date"]

    sp500_price = f"{L.sp500:,.2f}" if pd.notna(L.sp500) else "—"
    sp500_chg, sp500_cls = "", "neu"
    if len(df) > 1 and pd.notna(L.sp500) and pd.notna(df.iloc[-2]["sp500"]):
        chg       = (L.sp500 / df.iloc[-2]["sp500"] - 1) * 100
        sp500_chg = f"{chg:+.2f}%"
        sp500_cls = "up" if chg >= 0 else "dn"

    last_date = df["date"].iloc[-1].strftime("%b %d, %Y")

    # ── JSON payload ─────────────────────────────────────────────────────────
    D = json.dumps({
        "dates":   _d(dt),
        "sp500":   _s(df["sp500"]),
        "sma50":   _s(df["sp500_sma50"]),
        "t2108":   _s(df["t2108"] * 100),
        "up4":     _s(df["up_4pct"]),
        "dn4":     _s(-df["dn_4pct"]),          # negative so bars go below zero
        "r5":      _s(df["ratio_5d"]),
        "r10":     _s(df["ratio_10d"]),
        "up_q":    _s(df["up_25pct_quarter"]  * 100),
        "dn_q":    _s(df["down_25pct_quarter"] * 100),
        "up_m25":  _s(df["up_25pct_month"]    * 100),
        "dn_m25":  _s(df["down_25pct_month"]  * 100),
        "up_m50":  _s(df["up_50pct_month"]    * 100),
        "dn_m50":  _s(df["down_50pct_month"]  * 100),
        "up_34":   _s(df["up_13pct_34day"]    * 100),
        "dn_34":   _s(df["down_13pct_34day"]  * 100),
    }, separators=(",", ":"))

    # ── badges ───────────────────────────────────────────────────────────────
    def badge(label, val, cls="neu"):
        return f'<span class="badge {cls}"><span class="blabel">{label}</span>{val}</span>'

    b_t2108  = badge("T2108",  f"{L.t2108*100:.1f}%",              _badge_cls(L.t2108, neutral_at=0.4))
    b_up4    = badge("UP 4%",  f"{int(L.up_4pct):,}",              "up")
    b_dn4    = badge("DN 4%",  f"{int(L.dn_4pct):,}",              "dn")
    b_r5     = badge("5D",     f"{L.ratio_5d:.2f}",                _badge_cls(L.ratio_5d,  neutral_at=1))
    b_r10    = badge("10D",    f"{L.ratio_10d:.2f}",               _badge_cls(L.ratio_10d, neutral_at=1))
    b_upq    = badge("UP",     f"{L.up_25pct_quarter*100:.1f}%",   "up")
    b_dnq    = badge("DOWN",   f"{L.down_25pct_quarter*100:.1f}%", "dn")
    b_upm    = badge("UP25",   f"{L.up_25pct_month*100:.1f}%",     "up")
    b_dnm    = badge("DN25",   f"{L.down_25pct_month*100:.1f}%",   "dn")
    b_upm50  = badge("UP50",   f"{L.up_50pct_month*100:.1f}%",     "up")
    b_dnm50  = badge("DN50",   f"{L.down_50pct_month*100:.1f}%",   "dn")
    b_up34   = badge("UP13",   f"{L.up_13pct_34day*100:.1f}%",     "up")
    b_dn34   = badge("DN13",   f"{L.down_13pct_34day*100:.1f}%",   "dn")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Market Monitor</title>
<style>
:root{{
  --bg:#07090e;
  --surf:#0c1018;
  --bdr:#1a2333;
  --hdr:#09111c;
  --grid:#101820;
  --txt:#7a8fa8;
  --lit:#bfd0e8;
  --grn:#00e5a0;
  --red:#ff3a55;
  --blu:#3d9eff;
  --yel:#f5c940;
  --pur:#9f7aff;
  --org:#ff8c40;
  --tel:#00cce0;
  --r:4px;
}}
*{{box-sizing:border-box;margin:0;padding:0}}
html,body{{
  background:var(--bg);
  color:var(--lit);
  font-family:'Consolas','Cascadia Code','JetBrains Mono','Courier New',monospace;
  font-size:13px;
  min-height:100vh;
}}
#topbar{{
  display:flex;align-items:center;justify-content:space-between;
  padding:12px 28px;background:var(--hdr);
  border-bottom:1px solid var(--bdr);position:sticky;top:0;z-index:100;
}}
#logo{{font-size:15px;font-weight:700;letter-spacing:5px;text-transform:uppercase;color:var(--lit)}}
#logo em{{font-style:normal;color:var(--grn)}}
#topstats{{display:flex;gap:32px;align-items:center}}
.tstat{{text-align:right}}
.tstat .tl{{font-size:9px;letter-spacing:2px;text-transform:uppercase;color:var(--txt);display:block;margin-bottom:2px}}
.tstat .tv{{font-size:17px;font-weight:700;color:var(--lit)}}
.tstat .tv.up{{color:var(--grn)}}
.tstat .tv.dn{{color:var(--red)}}
.tstat .tv.neu{{color:var(--yel)}}
#updated{{font-size:10px;color:var(--txt);letter-spacing:1px}}
#refresh-btn{{
  display:inline-flex;align-items:center;gap:6px;
  padding:5px 13px;border-radius:var(--r);border:1px solid var(--bdr);
  background:rgba(61,158,255,0.08);color:var(--blu);
  font-family:inherit;font-size:10px;font-weight:700;letter-spacing:2px;
  text-transform:uppercase;cursor:pointer;transition:background .15s,border-color .15s;
}}
#refresh-btn:hover{{background:rgba(61,158,255,0.18);border-color:var(--blu)}}
#refresh-btn:disabled{{opacity:.4;cursor:not-allowed}}
#refresh-btn .spin{{display:none;animation:spin .7s linear infinite}}
#refresh-btn.running .spin{{display:inline}}
#refresh-btn.running .idle{{display:none}}
@keyframes spin{{to{{transform:rotate(360deg)}}}}
#panels{{padding:16px 28px 32px;display:flex;flex-direction:column;gap:10px}}
.panel{{background:var(--surf);border:1px solid var(--bdr);border-radius:var(--r);overflow:hidden}}
.phdr{{
  display:flex;align-items:center;justify-content:space-between;
  padding:7px 14px;background:var(--hdr);border-bottom:1px solid var(--bdr);
}}
.ptitle{{font-size:9px;letter-spacing:2.5px;text-transform:uppercase;color:var(--txt)}}
.pbadges{{display:flex;gap:6px;flex-wrap:wrap}}
.badge{{
  display:inline-flex;align-items:center;gap:5px;font-size:11px;font-weight:700;
  padding:2px 8px 2px 6px;border-radius:3px;background:rgba(255,255,255,0.05);
}}
.badge.up{{color:var(--grn);border:1px solid rgba(0,229,160,0.25);background:rgba(0,229,160,0.07)}}
.badge.dn{{color:var(--red);border:1px solid rgba(255,58,85,0.25);background:rgba(255,58,85,0.07)}}
.badge.neu{{color:var(--yel);border:1px solid rgba(245,201,64,0.25);background:rgba(245,201,64,0.07)}}
.blabel{{font-size:8px;font-weight:400;letter-spacing:1px;opacity:0.7;text-transform:uppercase}}
.chart{{width:100%}}
.reflegend{{display:flex;gap:14px;padding:4px 14px 8px;flex-wrap:wrap}}
.rl{{font-size:10px;color:var(--txt);display:flex;align-items:center;gap:5px}}
.rl span{{display:inline-block;width:22px;height:1px;border-top:1.5px dashed}}
#ai-panel{{background:var(--surf);border:1px solid var(--bdr);border-radius:var(--r);overflow:hidden;margin:16px 28px 0}}
#ai-panel .phdr{{display:flex;align-items:center;gap:10px;padding:7px 14px;background:var(--hdr);border-bottom:1px solid var(--bdr)}}
#ai-panel .ai-tag{{font-size:8px;font-weight:700;letter-spacing:2px;text-transform:uppercase;padding:2px 7px;border-radius:3px;background:rgba(159,122,255,0.15);color:var(--pur);border:1px solid rgba(159,122,255,0.3)}}
#ai-body{{padding:14px 18px;line-height:1.7;color:var(--lit);font-size:12px;display:flex;flex-direction:column;gap:4px}}
#ai-body ul{{list-style:none;padding:0;margin:2px 0 4px 0;display:flex;flex-direction:column;gap:4px}}
#ai-body li{{padding-left:14px;position:relative;color:var(--lit)}}
#ai-body li::before{{content:'›';position:absolute;left:0;color:var(--pur);font-weight:700}}
#ai-body p{{margin:0}}
#ai-body .ai-section{{font-size:10px;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:var(--pur);margin-top:10px}}
</style>
</head>
<body>

<div id="topbar">
  <div id="logo"><em>MARKET</em> MONITOR</div>
  <button id="refresh-btn" onclick="doRefresh()">
    <span class="idle">&#8635; UPDATE</span>
    <svg class="spin" width="12" height="12" viewBox="0 0 12 12"><circle cx="6" cy="6" r="5" fill="none" stroke="currentColor" stroke-width="1.5" stroke-dasharray="20 10"/></svg>
    <span class="running-lbl" style="display:none">UPDATING…</span>
  </button>
  <div id="topstats">
    <div class="tstat">
      <span class="tl">S&amp;P 500</span>
      <span class="tv">{sp500_price}</span>
    </div>
    <div class="tstat">
      <span class="tl">1D CHG</span>
      <span class="tv {sp500_cls}">{sp500_chg}</span>
    </div>
    <div class="tstat">
      <span class="tl">T2108</span>
      <span class="tv {'up' if L.t2108>0.4 else ('neu' if L.t2108>0.2 else 'dn')}">{L.t2108*100:.1f}%</span>
    </div>
    <div class="tstat">
      <span class="tl">Updated</span>
      <span class="tv neu" style="font-size:13px">{last_date}</span>
    </div>
  </div>
</div>

<div id="ai-panel">
  <div class="phdr">
    <span class="ptitle">MARKET ANALYSIS</span>
    <span class="ai-tag">GEMINI 2.5 FLASH</span>
  </div>
  <div id="ai-body">{analysis}</div>
</div>

<div id="panels">

  <!-- 1. S&P 500 -->
  <div class="panel">
    <div class="phdr">
      <span class="ptitle">S&amp;P 500</span>
      <div class="pbadges">
        <span class="badge neu"><span class="blabel">PRICE</span>{sp500_price}</span>
        <span class="badge {sp500_cls}"><span class="blabel">1D</span>{sp500_chg}</span>
      </div>
    </div>
    <div id="c-spy" class="chart"></div>
  </div>

  <!-- 2. T2108 -->
  <div class="panel">
    <div class="phdr">
      <span class="ptitle">T2108 — % Stocks Above 40-Day MA</span>
      <div class="pbadges">{b_t2108}</div>
    </div>
    <div class="reflegend">
      <div class="rl"><span style="border-color:#ff3a55"></span>Oversold 20%</div>
      <div class="rl"><span style="border-color:#00e5a0"></span>Overbought 80%</div>
    </div>
    <div id="c-t2108" class="chart"></div>
  </div>

  <!-- 3. Daily 4% Moves -->
  <div class="panel">
    <div class="phdr">
      <span class="ptitle">Daily Breadth — Stocks Up / Down 4%</span>
      <div class="pbadges">{b_up4}{b_dn4}</div>
    </div>
    <div id="c-4pct" class="chart"></div>
  </div>

  <!-- 4. 5d / 10d Ratio -->
  <div class="panel">
    <div class="phdr">
      <span class="ptitle">4% Up/Down Ratio — Rolling 5d &amp; 10d</span>
      <div class="pbadges">{b_r5}{b_r10}</div>
    </div>
    <div id="c-ratio" class="chart"></div>
  </div>

  <!-- 5. Quarterly Momentum -->
  <div class="panel">
    <div class="phdr">
      <span class="ptitle">Quarterly Momentum — ±25% in 63 Days</span>
      <div class="pbadges">{b_upq}{b_dnq}</div>
    </div>
    <div id="c-qtr" class="chart"></div>
  </div>

  <!-- 6. Monthly Momentum ±25% -->
  <div class="panel">
    <div class="phdr">
      <span class="ptitle">Monthly Momentum — ±25% in 21 Days</span>
      <div class="pbadges">{b_upm}{b_dnm}</div>
    </div>
    <div id="c-month25" class="chart"></div>
  </div>

  <!-- 7. Monthly Momentum ±50% -->
  <div class="panel">
    <div class="phdr">
      <span class="ptitle">Monthly Momentum — ±50% in 21 Days</span>
      <div class="pbadges">{b_upm50}{b_dnm50}</div>
    </div>
    <div id="c-month50" class="chart"></div>
  </div>

  <!-- 8. 34-Day Momentum -->
  <div class="panel">
    <div class="phdr">
      <span class="ptitle">34-Day Momentum — ±13% in 34 Days</span>
      <div class="pbadges">{b_up34}{b_dn34}</div>
    </div>
    <div id="c-d34" class="chart"></div>
  </div>

</div>

<script>{pjs}</script>
<script>
const D = {D};

const BG   = '#07090e';
const SURF = '#0c1018';
const GRID = '#121a26';
const TXT  = '#7a8fa8';
const LIT  = '#bfd0e8';
const GRN  = '#00e5a0';
const RED  = '#ff3a55';
const BLU  = '#3d9eff';
const YEL  = '#f5c940';
const ORG  = '#ff8c40';
const TEL  = '#00cce0';

function baseLayout(h, showX) {{
  return {{
    height: h,
    paper_bgcolor: SURF, plot_bgcolor: SURF,
    margin: {{l:58, r:18, t:8, b: showX ? 36 : 6}},
    font: {{family:"'Consolas','Cascadia Code','JetBrains Mono',monospace", size:10, color:TXT}},
    xaxis: {{
      showgrid:true, gridcolor:GRID, gridwidth:1, zeroline:false,
      tickfont:{{size:9,color:TXT}}, showticklabels:showX,
      tickformat:'%b %y', rangeslider:{{visible:false}}, linecolor:GRID,
    }},
    yaxis: {{
      showgrid:true, gridcolor:GRID, gridwidth:1, zeroline:false,
      tickfont:{{size:9,color:TXT}}, linecolor:GRID, tickcolor:GRID,
    }},
    legend: {{
      orientation:'h', x:0, y:1.0, xanchor:'left', yanchor:'bottom',
      font:{{size:10,color:TXT}}, bgcolor:'rgba(0,0,0,0)', traceorder:'normal',
    }},
    hovermode:'x unified',
    hoverlabel:{{
      bgcolor:'#0c1018', bordercolor:GRID,
      font:{{family:"'Consolas',monospace", size:11, color:LIT}},
    }},
  }};
}}

function hline(y, color, dash='dot') {{
  return {{
    type:'line', x0:0, x1:1, xref:'paper', y0:y, y1:y, yref:'y',
    line:{{color:color, width:1, dash:dash}}, layer:'below traces',
  }};
}}

function scatter(x, y, name, color, dash, width) {{
  return {{
    type:'scatter', mode:'lines', x, y, name,
    line:{{color, width:width||1.5, dash:dash||'solid'}},
    hovertemplate:`<b>${{name}}</b>: %{{y:.2f}}<extra></extra>`,
  }};
}}

function area(x, y, name, color, fillcolor) {{
  return {{
    type:'scatter', mode:'lines', x, y, name,
    fill:'tozeroy', line:{{color, width:1.5}},
    fillcolor: fillcolor,
    hovertemplate:`%{{y:.1f}}%<extra></extra>`,
  }};
}}

const CONFIG = {{
  scrollZoom:true, displayModeBar:true, displaylogo:false,
  modeBarButtonsToRemove:['select2d','lasso2d','autoScale2d'],
  responsive:true,
}};

/* ── 1. S&P 500 ─────────────────────────────────────────────────── */
Plotly.newPlot('c-spy', [
  scatter(D.dates, D.sp500, 'S&P 500', BLU, 'solid', 1.8),
  scatter(D.dates, D.sma50, 'SMA 50',  YEL, 'dot',   1.2),
], Object.assign(baseLayout(300,false), {{
  yaxis:{{showgrid:true,gridcolor:GRID,gridwidth:1,zeroline:false,
         tickfont:{{size:9,color:TXT}},linecolor:GRID,tickformat:',.0f'}},
}}), CONFIG);

/* ── 2. T2108 ───────────────────────────────────────────────────── */
Plotly.newPlot('c-t2108', [
  area(D.dates, D.t2108, 'T2108', TEL, 'rgba(0,204,224,0.10)'),
], Object.assign(baseLayout(300,false), {{
  shapes:[hline(20, RED), hline(80, GRN)],
  yaxis:{{showgrid:true,gridcolor:GRID,gridwidth:1,zeroline:false,
         tickfont:{{size:9,color:TXT}},linecolor:GRID,ticksuffix:'%',range:[0,100]}},
}}), CONFIG);

/* ── 3. Daily 4% Moves ──────────────────────────────────────────── */
(function(){{
  Plotly.newPlot('c-4pct', [
    {{type:'bar', x:D.dates, y:D.up4,  name:'↑4%', marker:{{color:'rgba(0,229,160,0.75)'}},
      hovertemplate:'↑4%%: %{{y:,d}}<extra></extra>'}},
    {{type:'bar', x:D.dates, y:D.dn4,  name:'↓4%', marker:{{color:'rgba(255,58,85,0.75)'}},
      hovertemplate:'↓4%%: %{{customdata:,d}}<extra></extra>',
      customdata:D.dn4.map(v => v === null ? null : -v)}},
  ], Object.assign(baseLayout(300,false), {{
    barmode:'overlay',
    shapes:[hline(0, TXT, 'solid')],
    yaxis:{{showgrid:true,gridcolor:GRID,gridwidth:1,zeroline:false,
           tickfont:{{size:9,color:TXT}},linecolor:GRID}},
  }}), CONFIG);
}})();

/* ── 4. 5d / 10d Ratio ──────────────────────────────────────────── */
Plotly.newPlot('c-ratio', [
  scatter(D.dates, D.r5,  'Ratio 5d',  GRN, 'solid', 1.8),
  scatter(D.dates, D.r10, 'Ratio 10d', YEL, 'dot',   1.5),
], Object.assign(baseLayout(300,false), {{
  shapes:[hline(1, TXT, 'solid')],
  yaxis:{{showgrid:true,gridcolor:GRID,gridwidth:1,zeroline:false,
         tickfont:{{size:9,color:TXT}},linecolor:GRID}},
}}), CONFIG);

/* ── 5. Quarterly Momentum ──────────────────────────────────────── */
Plotly.newPlot('c-qtr', [
  scatter(D.dates, D.up_q, '↑25% / 63d', GRN, 'solid', 1.8),
  scatter(D.dates, D.dn_q, '↓25% / 63d', RED, 'solid', 1.8),
], Object.assign(baseLayout(300,false), {{
  yaxis:{{showgrid:true,gridcolor:GRID,gridwidth:1,zeroline:false,
         tickfont:{{size:9,color:TXT}},linecolor:GRID,ticksuffix:'%'}},
}}), CONFIG);

/* ── 6. Monthly Momentum ±25% ───────────────────────────────────── */
Plotly.newPlot('c-month25', [
  scatter(D.dates, D.up_m25, '↑25% / 21d', GRN, 'solid', 1.8),
  scatter(D.dates, D.dn_m25, '↓25% / 21d', RED, 'solid', 1.8),
], Object.assign(baseLayout(300,false), {{
  yaxis:{{showgrid:true,gridcolor:GRID,gridwidth:1,zeroline:false,
         tickfont:{{size:9,color:TXT}},linecolor:GRID,ticksuffix:'%'}},
}}), CONFIG);

/* ── 7. Monthly Momentum ±50% ───────────────────────────────────── */
Plotly.newPlot('c-month50', [
  scatter(D.dates, D.up_m50, '↑50% / 21d', TEL, 'solid', 1.8),
  scatter(D.dates, D.dn_m50, '↓50% / 21d', ORG, 'solid', 1.8),
], Object.assign(baseLayout(300,false), {{
  yaxis:{{showgrid:true,gridcolor:GRID,gridwidth:1,zeroline:false,
         tickfont:{{size:9,color:TXT}},linecolor:GRID,ticksuffix:'%'}},
}}), CONFIG);

/* ── 8. 34-Day Momentum ─────────────────────────────────────────── */
Plotly.newPlot('c-d34', [
  scatter(D.dates, D.up_34, '↑13% / 34d', GRN, 'solid', 1.8),
  scatter(D.dates, D.dn_34, '↓13% / 34d', RED, 'solid', 1.8),
], Object.assign(baseLayout(300, true), {{
  yaxis:{{showgrid:true,gridcolor:GRID,gridwidth:1,zeroline:false,
         tickfont:{{size:9,color:TXT}},linecolor:GRID,ticksuffix:'%'}},
}}), CONFIG);

/* ── linked x-axis zoom/pan ─────────────────────────────────────── */
const CHARTS = ['c-spy','c-t2108','c-4pct','c-ratio','c-qtr','c-month25','c-month50','c-d34'];
let syncing = false;

CHARTS.forEach(id => {{
  document.getElementById(id).on('plotly_relayout', evt => {{
    if (syncing) return;
    let upd = null;
    if (evt['xaxis.range[0]'] !== undefined) {{
      upd = {{'xaxis.range[0]': evt['xaxis.range[0]'], 'xaxis.range[1]': evt['xaxis.range[1]']}};
    }} else if (evt['xaxis.autorange'] === true) {{
      upd = {{'xaxis.autorange': true}};
    }}
    if (!upd) return;
    syncing = true;
    Promise.all(CHARTS.filter(c => c !== id).map(c => Plotly.relayout(c, upd)))
      .then(() => {{ requestAnimationFrame(() => {{ syncing = false; }}); }});
  }});
}});

/* ── refresh button ─────────────────────────────────────────────── */
function doRefresh() {{
  const btn = document.getElementById('refresh-btn');
  btn.disabled = true;
  btn.classList.add('running');
  const lbl = btn.querySelector('.running-lbl');
  lbl.style.display = 'inline';
  window.location.href = 'marketmonitor:';
  let s = 60;
  const tick = () => {{
    lbl.textContent = 'UPDATING… ' + s + 's';
    if (s-- <= 0) {{ location.reload(); return; }}
    setTimeout(tick, 1000);
  }};
  tick();
}}
</script>
</body>
</html>"""

    return html


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    # load() and _get_plotly_js() are independent — run them in parallel
    with ThreadPoolExecutor(max_workers=2) as pool:
        f_df  = pool.submit(load)
        f_pjs = pool.submit(_get_plotly_js)
        df    = f_df.result()
        pjs   = f_pjs.result() or ""

    # Gemini needs df, so it starts as soon as load() finishes
    print("  Fetching Gemini analysis …")
    analysis = get_claude_analysis(df)
    html     = build_html(df, analysis, pjs)

    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"Saved → {OUT_HTML}  ({OUT_HTML.stat().st_size // 1024} KB)")

    if not args.refresh:
        chrome_paths = [
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        ]
        for chrome in chrome_paths:
            if Path(chrome).exists():
                subprocess.Popen([chrome, OUT_HTML.as_uri()])
                break
        else:
            try:
                import os; os.startfile(str(OUT_HTML))
            except Exception:
                pass


if __name__ == "__main__":
    main()
