# -*- coding: utf-8 -*-
"""Track broker branch trading performance and next-day profit.

追蹤特定券商分點的交易績效，計算其持倉隔日的獲利情況。

主要功能：
- calculate_next_day_profit: 計算分點的隔日獲利
- aggregate_broker_performance: 彙總分點績效
- track_target_brokers: 追蹤特定目標分點
- export_broker_ranking: 匯出全市場券商排行
"""

import os
import json
import time
from datetime import date, datetime, timedelta
from typing import Optional, List, Dict, Any

import pandas as pd

# Target broker branches to track (重點監控券商)
TARGET_BROKERS = [
    "凱基-台北",
    "凱基-虎尾",
    "美林",
    "摩根士丹利",  # Morgan Stanley
    "摩根大通",    # JP Morgan
    "高盛",        # Goldman Sachs
    "瑞銀",        # UBS
    "元大-台北",
    "富邦-台北",
    "港商野村",
    "美商高盛"
]

# Data directories
DATA_DIR = "data"
BROKER_DATA_DIR = os.path.join(DATA_DIR, "broker")
DOCS_DIR = os.path.join("docs", "data")
TIMESERIES_DIR = os.path.join(DOCS_DIR, "timeseries")

# Global cache for stock prices to avoid repeated loading
_PRICE_CACHE = {}

def ensure_dirs():
    """Ensure required directories exist."""
    os.makedirs(BROKER_DATA_DIR, exist_ok=True)
    os.makedirs(DOCS_DIR, exist_ok=True)

def load_stock_prices(stock_code: str) -> pd.DataFrame:
    """
    Load stock price data for calculating next-day returns.
    Reads from docs/data/timeseries/{code}.json
    """
    if stock_code in _PRICE_CACHE:
        return _PRICE_CACHE[stock_code]

    path = os.path.join(TIMESERIES_DIR, f"{stock_code}.json")
    if not os.path.exists(path):
        # 如果找不到檔案，返回空 DataFrame
        return pd.DataFrame()
    
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # === FIX: Handle both List and Dict formats ===
        # 修正：部分 JSON 是 list 格式，部分可能是 dict 格式
        records = []
        if isinstance(data, list):
            records = data
        elif isinstance(data, dict):
            records = data.get("data", [])
            
        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)
        
        # 確保有日期欄位
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date")
        
        # 快取結果
        _PRICE_CACHE[stock_code] = df
        return df
        
    except Exception as e:
        print(f"Error loading prices for {stock_code}: {e}")
        return pd.DataFrame()

def get_next_day_price(stock_code: str, trade_date: str) -> Optional[float]:
    """Get the closing price of the stock on the next trading day."""
    df = load_stock_prices(stock_code)
    if df.empty:
        return None
        
    # Convert trade_date to timestamp for comparison
    try:
        ts_date = pd.to_datetime(trade_date)
    except:
        return None
    
    # Find records after trade date
    if "date" not in df.columns:
        return None
        
    future_data = df[df["date"] > ts_date]
    
    if future_data.empty:
        return None
        
    # Get the immediate next record
    next_record = future_data.iloc[0]
    
    # 嘗試獲取價格 (優先順序: close > price > adj_close)
    for col in ['close', 'price', 'adj_close']:
        if col in next_record:
            val = next_record[col]
            # 確保數值有效
            if pd.notnull(val) and val != 0:
                return float(val)
            
    return None

def calculate_next_day_profit(trades: List[Dict]) -> pd.DataFrame:
    """
    Calculate profit/loss for trades based on next day's price action.
    """
    results = []
    print(f"Calculating profit for {len(trades)} trades...")
    
    for trade in trades:
        stock_code = trade.get("stock_code")
        date_str = trade.get("date")
        
        # 轉換日期格式 (處理 12/19 這種格式)
        try:
            date_str_s = str(date_str)
            if "/" in date_str_s and len(date_str_s) <= 5:
                current_year = datetime.now().year
                dt = datetime.strptime(f"{current_year}/{date_str_s}", "%Y/%m/%d")
                date_iso = dt.strftime("%Y-%m-%d")
            else:
                date_iso = date_str_s
        except:
            date_iso = str(date_str)

        try:
            trade_price = float(trade.get("price", 0) or 0)
            net_vol = float(trade.get("net_vol", 0) or 0)
        except ValueError:
            trade_price = 0
            net_vol = 0
        
        # 如果沒有交易價格或量，跳過
        if trade_price == 0 or net_vol == 0:
            trade_data = trade.copy()
            trade_data.update({
                "profit": 0,
                "profit_pct": 0,
                "has_profit_data": False
            })
            results.append(trade_data)
            continue

        # 取得隔日收盤價
        next_price = get_next_day_price(stock_code, date_iso)
        
        profit = 0.0
        profit_pct = 0.0
        
        if next_price:
            # 損益計算邏輯：
            # 買超 (net_vol > 0): (隔日價 - 買入價) * 張數 * 1000
            # 賣超 (net_vol < 0): (賣出價 - 隔日價) * 張數 * 1000 (假設回補)
            if net_vol > 0:
                profit = (next_price - trade_price) * net_vol * 1000
                profit_pct = (next_price - trade_price) / trade_price * 100
            else:
                profit = (trade_price - next_price) * abs(net_vol) * 1000
                profit_pct = (trade_price - next_price) / trade_price * 100
        
        # 將計算結果加入字典
        trade_data = trade.copy()
        trade_data.update({
            "date_iso": date_iso,
            "next_day_price": next_price,
            "profit": profit,
            "profit_pct": profit_pct,
            "has_profit_data": next_price is not None
        })
        
        results.append(trade_data)

    return pd.DataFrame(results)

def aggregate_broker_performance(profit_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate performance stats by broker.
    """
    if profit_df.empty:
        return pd.DataFrame()
        
    # 只統計有成功計算出損益的交易
    valid_df = profit_df[profit_df["has_profit_data"] == True].copy()
    
    if valid_df.empty:
        return pd.DataFrame()

    stats = valid_df.groupby("broker_name").agg({
        "profit": "sum",
        "net_vol": "sum",
        "stock_code": "nunique", # 操作幾檔股票
        "profit_pct": "mean"     # 平均報酬率
    }).reset_index()
    
    # 計算勝率 (獲利 > 0 的次數 / 總次數)
    win_rates = valid_df.groupby("broker_name").apply(
        lambda x: (x["profit"] > 0).sum() / len(x) * 100
    ).reset_index(name="win_rate")
    
    stats = pd.merge(stats, win_rates, on="broker_name")
    return stats.sort_values("profit", ascending=False)

def export_target_broker_trades(all_trades: List[Dict]):
    """
    Export trades grouped by target brokers.
    """
    output_path = os.path.join(DOCS_DIR, "target_broker_trades.json")
    
    broker_groups = {}
    
    for trade in all_trades:
        b_name = trade.get("broker_name", "")
        # Check if this broker is interesting
        if not any(target in b_name for target in TARGET_BROKERS):
            continue
            
        if b_name not in broker_groups:
            broker_groups[b_name] = []
            
        broker_groups[b_name].append(trade)
    
    # Sort trades within each broker by absolute net volume
    for b in broker_groups:
        broker_groups[b].sort(key=lambda x: abs(float(x.get("net_vol", 0))), reverse=True)
        # Keep top 10 per broker
        broker_groups[b] = broker_groups[b][:10]
    
    result = {
        "updated": datetime.now().isoformat(),
        "brokers": broker_groups
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Saved target broker trades to {output_path}")

def export_broker_ranking(all_trades: List[Dict], output_path: str = None):
    """
    Export total broker ranking (buy/sell volume) to JSON.
    """
    if output_path is None:
        output_path = os.path.join(DOCS_DIR, "broker_ranking.json")
    
    if not all_trades:
        print("No trades to rank.")
        return

    df = pd.DataFrame(all_trades)
    
    # 確保數值型態正確
    for col in ['net_vol', 'buy_vol', 'sell_vol']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Group by broker and aggregate
    broker_stats = df.groupby("broker_name").agg({
        "net_vol": "sum",
        "buy_vol": "sum",
        "sell_vol": "sum",
        "stock_code": "nunique",
    }).reset_index()
    
    broker_stats.columns = ["broker_name", "total_net_vol", "total_buy_vol", 
                            "total_sell_vol", "stocks_count"]
    
    # Sort by absolute net volume
    broker_stats["abs_net_vol"] = broker_stats["total_net_vol"].abs()
    broker_stats = broker_stats.sort_values("abs_net_vol", ascending=False)
    
    # Save to JSON
    result = {
        "updated": datetime.now().isoformat(),
        "data": broker_stats.drop(columns=["abs_net_vol"]).to_dict(orient="records")
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"Saved broker ranking to {output_path}")

def track_target_brokers(hot_stocks: List[str]):
    """
    Track specific target brokers' activities in hot stocks.
    """
    print("=== Starting track_broker_profit.py (v2.1) ===")
    print(f"Tracking target broker performance...")
    ensure_dirs()
    
    # 讀取今日最新的交易資料
    latest_trades_path = os.path.join(DOCS_DIR, "broker_trades_latest.json")
    if not os.path.exists(latest_trades_path):
        print("No latest broker trades found.")
        return [], {}
        
    with open(latest_trades_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        # 這裡也要防禦，有些 JSON 可能是 list
        if isinstance(data, dict):
            all_trades = data.get("data", [])
        elif isinstance(data, list):
            all_trades = data
        else:
            all_trades = []
        
    print(f"Got {len(all_trades)} total trades")
    
    if not all_trades:
        print("Warning: No trade data found in broker_trades_latest.json")
        return [], pd.DataFrame()

    # 1. 匯出全市場券商排行 (Top Buyers/Sellers)
    export_broker_ranking(all_trades)
    
    # 2. 篩選目標券商交易
    target_trades = [
        t for t in all_trades 
        if any(b in t.get("broker_name", "") for b in TARGET_BROKERS)
    ]
    
    print(f"Found {len(target_trades)} trades from target brokers")
    
    # 3. 計算隔日損益 (如果有歷史股價資料的話)
    try:
        profit_df = calculate_next_day_profit(target_trades)
        
        # 4. 統計分點績效
        if not profit_df.empty:
            performance = aggregate_broker_performance(profit_df)
            if not performance.empty:
                print("\n=== Broker Performance Estimate (Next Day) ===")
                print(performance.head())
        else:
            performance = pd.DataFrame()
            
    except Exception as e:
        print(f"Warning: Profit calculation skipped due to error: {e}")
        import traceback
        traceback.print_exc()
        performance = pd.DataFrame()
    
    # 5. 匯出目標券商交易明細供前端使用
    export_target_broker_trades(all_trades)
    
    return target_trades, performance

if __name__ == "__main__":
    # 執行主流程
    track_target_brokers([])