import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore') # 隱藏一些不必要的警告

# 🌟 1. 積分表與校準邏輯 (與你的 app.py 完全一致)
score_map = {
    '建國中學': 33.8, '成淵高中': 27.6, '和平高中': 25.8, '南湖高中': 21.6, '復興高中': 11.6
}

def expert_calibration(ml_pred, school_name, subject, avg_input_score, mode='高一二'):
    pt = score_map.get(school_name, 18.0)
    base = (pt * 0.4) + 0.5 
    slope = 10.0 - (pt / 10.0) 
    expert_val = base + (avg_input_score - 72) / slope
    
    if subject in ['學測_數A', '學測_自然']: 
        expert_val -= (35 - pt) * 0.04 
    if subject in ['學測_社會', '學測_國文']: 
        expert_val += 0.8 
        
    return (ml_pred * 0.35) + (expert_val * 0.65)

# 🌟 2. 設定測試情境
test_schools = ['建國中學', '成淵高中', '和平高中', '南湖高中', '復興高中']
test_scores = [60, 75, 90]
test_subjects = ['學測_國文', '學測_數A', '學測_自然', '學測_社會']

print("🔍 啟動系統合理性自動檢驗...")
print("-" * 50)

results = []

# 🌟 3. 跑迴圈自動推演
for school in test_schools:
    for score in test_scores:
        row = {'代表學校': school, '段考分數': score}
        for sub in test_subjects:
            try:
                # 載入大腦
                model = joblib.load(f"model_{sub}.joblib")
                features = joblib.load(f"features_{sub}.joblib")
                
                # 偽造輸入特徵 (假設該生是理組，所有該科相關段考皆為目前設定的分數)
                in_df = {}
                for f in features:
                    if f == '會考積分': in_df[f] = score_map.get(school, 18.0)
                    elif '類組_理組' in f: in_df[f] = 1
                    elif '類組_' in f: in_df[f] = 0
                    else: in_df[f] = score
                
                # 預測與校準
                ml_pred = model.predict(pd.DataFrame([in_df])[features])[0]
                final = expert_calibration(ml_pred, school, sub, score)
                
                # 換算區間
                center = round(final)
                low = max(1, center - 1)
                high = min(15, center + 1)
                
                row[sub.replace('學測_', '')] = f"{low}~{high} ({center})"
            except Exception as e:
                row[sub.replace('學測_', '')] = "尚未訓練"
                
        results.append(row)

# 🌟 4. 輸出成漂亮的表格並存檔
df_res = pd.DataFrame(results)
print(df_res.to_string(index=False))

# 存成 Excel 方便你慢慢看
df_res.to_csv("系統合理性檢驗報告.csv", index=False, encoding='utf-8-sig')
print("-" * 50)
print("✅ 檢驗完畢！已將結果儲存為 '系統合理性檢驗報告.csv'")