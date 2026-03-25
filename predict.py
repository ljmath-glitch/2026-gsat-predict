import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
import joblib

# 🌟 北北基會考錄取積分映射表 (用於 AI 數值特徵)
score_map = {
    '建國中學': 33.8, '北一女中': 33.6, '師大附中(男)': 32.8, '師大附中(女)': 32.8, '師大附中': 32.8,
    '成功高中': 31.6, '中山女中': 30.8, '松山高中(男)': 30.6, '松山高中(女)': 29.8, '松山高中': 30.2,
    '大同高中': 29.6, '中崙高中': 28.6, '政大附中': 28.6, '板橋高中': 28.2, '成淵高中': 27.6, 
    '麗山高中': 26.8, '海山高中': 26.6, '大直高中': 26.4, '和平高中': 25.8, '西松高中': 24.6,
    '北大高中': 24.6, '明倫高中': 24.6, '內湖高中': 23.8, '百齡高中': 23.6, '景美女中': 22.8,
    '中正高中': 22.6, '中和高中': 21.8, '陽明高中': 21.6, '南湖高中': 21.6, '新莊高中': 20.6,
    '南港高中': 20.6, '永平高中': 20.6, '華江高中': 19.6, '新北高中': 19.6, '永春高中': 18.6,
    '丹鳳高中': 18.6, '華僑高中': 18.6, '新店高中': 17.6, '錦和高中': 17.6, '三民高中': 17.6,
    '大理高中': 16.6, '萬芳高中': 16.6, '光復高中': 16.6, '三重高中': 15.6, '竹圍高中': 15.6,
    '育成高中': 15.6, '林口高中': 15.4, '清水高中': 13.8, '樹林高中': 12.8, '安康高中': 12.6,
    '泰山高中': 12.6, '復興高中': 11.6, '秀峰高中': 10.6, '明德高中': 9.6, '石碇高中': 5.4
}

print("--- 🛡️ 啟動數據防護與 AI 積分特徵學習 ---")
try:
    df = pd.read_excel('學測級分預測分析.xlsx')
except FileNotFoundError:
    print("❌ 找不到 '學測級分預測分析.xlsx'，請確認檔案位置。")
    exit()

# 1. 數據清洗：強制轉數值、剔除 0 與文字
exclude_cols = ['姓名', '學校', '類組']
for col in df.columns:
    if col not in exclude_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].replace(0, np.nan) # 🌟 核心：把 0 視為空值，不參與平均

# 2. 注入積分特徵
df['會考積分'] = df['學校'].map(score_map).fillna(15.0)

# 3. 類組與平均特徵生成
df['類組'] = df['類組'].astype(str).replace({'理科': '理組', '文科': '文組', '自然組': '理組', '社會組': '文組'})
df = pd.get_dummies(df, columns=['類組'], dtype=int)

subjects_core = ['國文', '英文', '數學', '物理', '化學', '生物', '地科', '歷史', '地理', '公民']
for sub in subjects_core:
    cols = [c for c in df.columns if sub in c and '學測' not in c and '模考' not in c and c not in exclude_cols]
    if cols: 
        df[f'{sub}_段考平均'] = df[cols].mean(axis=1, skipna=True)

# 4. 訓練配置
subject_config = {
    '學測_國文': ['國文_段考平均', '會考積分'],
    '學測_英文': ['英文_段考平均', '會考積分'],
    '學測_數A': ['數學_段考平均', '會考積分'],
    '學測_數B': ['數學_段考平均', '會考積分'],
    '學測_自然': ['物理_段考平均', '化學_段考平均', '生物_段考平均', '地科_段考平均', '會考積分'],
    '學測_社會': ['歷史_段考平均', '地理_段考平均', '公民_段考平均', '會考積分']
}

for target, feats in subject_config.items():
    if target not in df.columns: continue
    train_df = df.dropna(subset=[target])
    
    current_feats = [f for f in feats if f in train_df.columns] + [c for c in train_df.columns if '類組_' in c]
    
    X = train_df[current_feats].fillna(train_df[current_feats].median())
    y = train_df[target]
    
    if len(X) < 3: continue
    
    # 使用具備處理遺漏值能力的模型
    model = HistGradientBoostingRegressor(max_iter=100, random_state=42)
    model.fit(X, y)
    
    joblib.dump(model, f"model_{target}.joblib")
    joblib.dump(current_feats, f"features_{target}.joblib")
    print(f"✅ {target} 訓練完成 (已整合積分特徵)")

print("\n🎉 大腦升級完畢！現在請執行 app.py。")