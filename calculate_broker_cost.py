# -*- coding: utf-8 -*-
"""
calculate_broker_cost.py

計算「重點券商」在特定股票上的「預估持有成本」。

功能特點：
1. 可指定股票代碼進行計算 (python calculate_broker_cost.py --codes 2330 2317)
2. 若未指定，則自動從 broker_trades_latest.json 讀取今日有交易的熱門股。
3. 使用「加權平均法 (VWAP)」計算累積持倉成本。
4. 輸出 JSON (docs/data/broker_cost_analytics.json) 供前端繪圖。
"""

import os
import json
import time
import argparse
import pandas as pd
from datetime import datetime

# 引用現有的抓取模組
# 注意：必須確保 fetch_broker_data.py 在同一目錄下
try:
    from fetch_broker_data import fetch_broker_history, close_browser
except ImportError:
    print("Warning: fetch_broker_data module not found. Crawling capabilities disabled.")
    def fetch_broker_history(*args, **kwargs): return pd.DataFrame()
    def close_browser(): pass

# 設定路徑
DATA_DIR = "data"
BROKER_DIR = os.path.join(DATA_DIR, "broker")
DOCS_DATA_DIR = os.path.join("docs", "data")
HISTORY_CSV = os.path.join(BROKER_DIR, "broker_history.csv")
OUTPUT_JSON = os.path.join(DOCS_DATA_DIR, "broker_cost_analytics.json")
LATEST_TRADES_JSON = os.path.join(DOCS_DATA_DIR, "broker_trades_latest.json")

# 設定要計算成本的重點券商目標
# 這些是市場上公認具備影響力的分點
TARGET_BROKERS = [
    "凱基-台北",
    "凱基-虎尾",
    "美林",
    "摩根士丹利",
    "摩根大通",
    "高盛",
    "瑞銀",
    "元大-台北",
    "富邦-台北",
    "港商野村",
    "美商高盛",
    "台灣摩根士丹利",
    "新加坡商瑞銀"
]

def load_target_stocks_from_latest():
    """從最新的交易紀錄中找出重點券商有操作的股票代碼"""
    if not os.path.exists(LATEST_TRADES_JSON):
        return []
    
    try:
        with open(LATEST_TRADES_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        trades = data.get("data", []) if isinstance(data, dict) else data
        
        # 找出重點券商今日有交易的股票
        target_stocks = set()
        for t in trades:
            b_name = t.get("broker_name", "")
            if any(target in b_name for target in TARGET_BROKERS):
                target_stocks.add(t.get("stock_code"))
                
        return list(target_stocks)
    except Exception as e:
        print(f"Error loading latest trades: {e}")
        return []

def calculate_vwap_cost(df_detail):
    """
    計算加權平均成本 (VWAP) 趨勢
    df_detail 必須包含: date, buy_vol, sell_vol, price (或成交均價)
    """
    # 確保資料按日期排序
    df = df_detail.sort_values("date").copy()
    
    # 累積變數
    cumulative_inventory = 0  # 庫存張數
    cumulative_cost_money = 0 # 總成本金額 (千元)
    
    trend_data = []
    
    for _, row in df.iterrows():
        buy_vol = float(row.get("buy_vol", 0))
        sell_vol = float(row.get("sell_vol", 0))
        net_vol = buy_vol - sell_vol
        
        # 假設當日買賣都發生在「收盤價」或「均價」
        # 如果有分開的買入均價/賣出均價會更準，但目前只有一個 price
        price = float(row.get("price", 0))
        if price == 0:
            continue
            
        # 簡單成本模型：
        # 買入增加成本，賣出依比例減少成本 (FIFO 概念的簡化版 -> 平均成本法)
        
        # 1. 處理買入
        if buy_vol > 0:
            cost_increase = buy_vol * price
            cumulative_cost_money += cost_increase
            cumulative_inventory += buy_vol
            
        # 2. 處理賣出 (假設賣出的是庫存的平均成本部分)
        if sell_vol > 0:
            # 如果還有庫存，賣出部分依比例扣除總成本
            if cumulative_inventory > 0:
                avg_cost = cumulative_cost_money / cumulative_inventory
                cost_decrease = sell_vol * avg_cost
                
                # 更新
                cumulative_cost_money -= cost_decrease
                cumulative_inventory -= sell_vol
            else:
                # 若已無庫存還賣超 (放空?)，暫時視為負庫存，成本以當日價計算
                cumulative_inventory -= sell_vol
                cumulative_cost_money -= (sell_vol * price)

        # 計算當日結束後的平均成本
        current_avg_cost = 0
        if cumulative_inventory != 0:
            current_avg_cost = cumulative_cost_money / cumulative_inventory
            
        trend_data.append({
            "date": row["date"],
            "price": price,
            "inventory": int(cumulative_inventory),
            "cost": round(current_avg_cost, 2) if cumulative_inventory > 0 else 0
        })
        
    return trend_data

def get_broker_id_map():
    """建立 券商名稱 -> ID 的對照表 (從歷史資料)"""
    id_map = {}
    
    # 1. 嘗試從 broker_history.csv 讀取
    if os.path.exists(HISTORY_CSV):
        try:
            df = pd.read_csv(HISTORY_CSV)
            if "broker_name" in df.columns and "broker_id" in df.columns:
                # 建立字典: { "凱基-台北": "9268", ... }
                mapping = df.drop_duplicates("broker_name")[["broker_name", "broker_id"]]
                id_map = dict(zip(mapping.broker_name, mapping.broker_id))
        except Exception as e:
            print(f"Error reading history CSV: {e}")

    # 2. 補充一些常見的靜態 ID (避免歷史檔沒有)
    static_map = {
        "凱基-台北": "9268",
        "凱基-虎尾": "9200", # 注意：總公司或分點需確認
        "美林": "1440",
        "摩根士丹利": "1470",
        "台灣摩根士丹利": "1470",
        "摩根大通": "8440",
        "美商高盛": "1480",
        "高盛": "1480",
        "瑞銀": "1650",
        "新加坡商瑞銀": "1650",
        "元大-台北": "9800", # 元大總公司
        "富邦-台北": "9600",
        "富邦": "9600",
        "港商野村": "1560",
        "港商麥格理": "1360"
    }
    id_map.update(static_map)
    return id_map

def main():
    # 解析命令列參數
    parser = argparse.ArgumentParser(description="Calculate Broker Cost Analysis")
    parser.add_argument("--codes", nargs="+", help="Specific stock codes to analyze (e.g., 2330 2317)")
    args = parser.parse_args()

    # 決定要分析的股票清單
    target_stock_codes = []
    if args.codes:
        print(f"Using specified stock codes: {args.codes}")
        target_stock_codes = args.codes
    else:
        print("No codes specified, auto-detecting from latest trades...")
        target_stock_codes = load_target_stocks_from_latest()
        # 限制數量避免跑太久 (取前 5 檔熱門的就好)
        target_stock_codes = target_stock_codes[:5]
    
    if not target_stock_codes:
        print("No target stocks found. Exiting.")
        return

    print(f"Target stocks for analysis: {target_stock_codes}")
    
    # 載入券商 ID 對照表
    broker_map = get_broker_id_map()
    
    analysis_results = {}
    
    # 開始分析：針對每一檔股票，找出買賣超最大的重點券商，並計算成本
    # 這裡簡化邏輯：直接針對 TARGET_BROKERS 去抓取這些股票的歷史
    
    for stock_code in target_stock_codes:
        stock_code = str(stock_code)
        
        # 為了節省時間，我們只針對這檔股票「累積買賣超」最大的 1~2 家重點券商進行分析
        # 這裡需要一個邏輯來決定「誰是這檔股票的主力」
        # 暫時簡單化：遍歷所有重點券商，抓取資料 (實務上這樣會很久，建議結合 update_broker 的排名結果)
        
        # 優化：只挑選 3 家重點券商進行爬取 (避免 Timeout)
        selected_brokers_for_stock = []
        for b_name in TARGET_BROKERS:
            b_id = broker_map.get(b_name)
            if b_id:
                selected_brokers_for_stock.append((b_name, b_id))
        
        # 隨機或依序挑選 (正式版應該依據 broker_ranking.json 的名次)
        # 這裡為了展示，我們先只抓前 2 個匹配到的券商
        run_count = 0
        
        for broker_name, broker_id in selected_brokers_for_stock:
            if run_count >= 2: break # 每檔股票只看 2 家主力
            
            print(f"Fetching history: {broker_name} ({broker_id}) -> Stock {stock_code}")
            
            try:
                # 呼叫 fetch_broker_data.py 裡的功能抓取詳細歷史 (含股價)
                # 抓取過去 60 天數據
                df_detail = fetch_broker_history(stock_code, broker_id, days=60)
                
                if df_detail.empty:
                    print(f"  -> No data found.")
                    continue
                
                # 計算成本曲線
                cost_trend = calculate_vwap_cost(df_detail)
                
                if not cost_trend:
                    continue

                # 存入結果
                # key 格式: "券商名稱_股票代碼"
                key = f"{broker_name}_{stock_code}"
                analysis_results[key] = {
                    "broker": broker_name,
                    "stock": stock_code,
                    "data": cost_trend
                }
                
                run_count += 1
                
                # 避免過度頻繁請求
                time.sleep(2)
                
            except Exception as e:
                print(f"  -> Error: {e}")

    # 3. 匯出 JSON
    close_browser()
    
    if analysis_results:
        os.makedirs(DOCS_DATA_DIR, exist_ok=True)
        # 讀取舊資料合併 (選擇性)
        if os.path.exists(OUTPUT_JSON):
            try:
                with open(OUTPUT_JSON, "r", encoding="utf-8") as f:
                    old_data = json.load(f)
                    if isinstance(old_data, dict): # Handle new format check
                         # 如果舊資料是 list (舊版格式)，就忽略或轉換
                         pass
                    elif isinstance(old_data, list):
                         # 舊版可能是 list，這裡我們現在統一用 dict 結構比較好擴充，但為了相容前端...
                         # 前端 index.html 預期的是一個 Object (key 是 broker_stock)
                         pass
            except:
                pass

        # 直接覆蓋 (保持最新狀態)
        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(analysis_results, f, ensure_ascii=False, indent=2)
        
        print(f"Saved cost analysis to {OUTPUT_JSON}")
    else:
        print("No analysis results generated.")

if __name__ == "__main__":
    main()