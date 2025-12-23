# -*- coding: utf-8 -*-
"""
calculate_broker_cost.py

計算「重點券商」在特定股票上的「預估持有成本」。

原理：
1. 從 broker_history.csv (或當日資料) 建立「券商名稱 -> 代碼」的對照表。
2. 針對 TARGET_BROKERS (重點券商) 與其交易量大的股票，使用 fetch_broker_history 抓取詳細歷史(含股價)。
3. 使用「加權平均法 (VWAP)」計算累積持倉成本。
4. 輸出 JSON 供前端繪圖。
"""

import os
import json
import time
import pandas as pd
from datetime import datetime

# 引用現有的抓取模組
from fetch_broker_data import fetch_broker_history, close_browser

# 設定路徑
DATA_DIR = "data"
BROKER_DIR = os.path.join(DATA_DIR, "broker")
DOCS_DATA_DIR = os.path.join("docs", "data")
HISTORY_CSV = os.path.join(BROKER_DIR, "broker_history.csv")
OUTPUT_JSON = os.path.join(DOCS_DATA_DIR, "broker_cost_analytics.json")

# 設定要計算成本的目標 (範例：只針對這些券商與他們主要操作的股票)
# 實際使用時，可以從 update_broker.py 的 TARGET_BROKERS 讀取
TARGET_BROKERS = [
    "凱基-台北", "摩根大通", "台灣摩根士丹利", "高盛", "美林", "元大"
]

def load_broker_map():
    """從歷史 CSV 中建立 券商名稱 -> Broker ID 的對照表"""
    if not os.path.exists(HISTORY_CSV):
        print(f"[WARN] {HISTORY_CSV} not found. Run update_broker.py first.")
        return {}
    
    df = pd.read_csv(HISTORY_CSV)
    # 建立 broker_name : broker_id 的字典
    # 注意：需過濾掉 broker_id 為空或 NaN 的資料
    valid = df.dropna(subset=["broker_id"])
    broker_map = dict(zip(valid["broker_name"], valid["broker_id"]))
    return broker_map

def calculate_vwap_cost(df: pd.DataFrame):
    """
    計算加權平均成本 (VWAP)
    df 必須包含: date, net_vol, price
    df 必須按日期排序 (由舊到新)
    """
    # 確保按日期排序
    df = df.sort_values("date", ascending=True).reset_index(drop=True)
    
    inventory = 0      # 庫存張數
    avg_cost = 0.0     # 平均成本
    
    results = []
    
    for _, row in df.iterrows():
        net = row["net_vol"]
        price = row["price"]
        date_str = row["date"]
        
        # 簡單邏輯：
        # 買進 -> 更新成本 (加權平均)
        # 賣出 -> 成本不變 (視為減少庫存)，若庫存歸零則重置成本
        
        if net > 0:
            # 買進
            total_value = (inventory * avg_cost) + (net * price)
            inventory += net
            avg_cost = total_value / inventory if inventory > 0 else 0
            
        elif net < 0:
            # 賣出
            inventory += net
            # 若賣到空倉或變負 (融券/放空)，這裡簡化處理：
            # 1. 若還有庫存，成本價通常維持不變 (先進先出或平均成本法在賣出時不影響剩餘單位成本)
            # 2. 若庫存 <= 0，重置成本為 0 (或當日市價，視為重新開始)
            if inventory <= 0:
                inventory = 0
                avg_cost = 0 # 歸零重置
        
        results.append({
            "date": date_str,
            "price": price,          # 當日收盤/均價
            "cost": round(avg_cost, 2), # 預估成本
            "inventory": inventory,   # 預估庫存
            "net_vol": net
        })
        
    return results

def main():
    print("=== Start Analyzing Broker Cost ===")
    
    # 1. 取得 Broker ID 對照表
    broker_map = load_broker_map()
    if not broker_map:
        print("No broker history found to map IDs.")
        return

    # 2. 找出每個重點券商「最近交易量最大」的 1 檔股票來示範
    # (為了節省時間，不抓全部。實務上可改為讀取使用者關注清單)
    df_hist = pd.read_csv(HISTORY_CSV)
    
    analysis_results = {}
    
    for broker_name in TARGET_BROKERS:
        if broker_name not in broker_map:
            continue
            
        broker_id = str(broker_map[broker_name]).replace(".0", "") # 清理 ID
        
        # 找出該券商最近交易最頻繁或量最大的股票
        broker_trades = df_hist[df_hist["broker_name"] == broker_name]
        if broker_trades.empty:
            continue
            
        # 簡單取交易量絕對值加總最大的前 2 檔股票
        top_stocks = broker_trades.groupby("stock_code")["net_vol"].apply(lambda x: x.abs().sum()).sort_values(ascending=False).head(2).index.tolist()
        
        for stock_code in top_stocks:
            stock_code = str(stock_code)
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
                
                # 存入結果
                key = f"{broker_name}_{stock_code}"
                analysis_results[key] = {
                    "broker": broker_name,
                    "stock": stock_code,
                    "data": cost_trend
                }
                
                # 避免過度頻繁請求
                time.sleep(2)
                
            except Exception as e:
                print(f"  -> Error: {e}")

    # 3. 匯出 JSON
    close_browser()
    
    if analysis_results:
        os.makedirs(DOCS_DATA_DIR, exist_ok=True)
        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(analysis_results, f, ensure_ascii=False, indent=2)
        print(f"Saved cost analysis to {OUTPUT_JSON}")
    else:
        print("No analysis results generated.")

if __name__ == "__main__":
    main()