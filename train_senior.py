import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

print("🚀 開始訓練【高三單次】與【高三兩次】專屬 AI 模型...")

# 1. 讀取並清洗數據
try:
    df = pd.read_excel('senior_data.xlsx')
except FileNotFoundError:
    print("❌ 找不到 senior_data.xlsx，請確認檔名與路徑是否正確！")
    exit()

df = df.replace(['-', '－', ' '], 0)
df = df.apply(pd.to_numeric, errors='coerce').fillna(0)

targets = ['學測_國文', '學測_英文', '學測_數A', '學測_數B', '學測_自然', '學測_社會']

# 定義兩組不同的特徵 (武器)
features_single = ['模考1_國文', '模考1_英文', '模考1_數學', '模考1_自然', '模考1_社會']
features_double = [
    '模考1_國文', '模考1_英文', '模考1_數學', '模考1_自然', '模考1_社會',
    '模考2_國文', '模考2_英文', '模考2_數A', '模考2_數B', '模考2_自然', '模考2_社會'
]

# 2. 開始訓練
for target in targets:
    train_df = df[df[target] > 0]
    if len(train_df) < 5: continue

    y = train_df[target]

    # --- 訓練【單次模考】專屬模型 ---
    X_single = train_df[features_single]
    model_single = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=5)
    model_single.fit(X_single, y)
    joblib.dump(model_single, f"model_高三單次_{target}.joblib")
    joblib.dump(features_single, f"features_高三單次_{target}.joblib")
    
    # --- 訓練【兩次模考】專屬模型 ---
    X_double = train_df[features_double]
    model_double = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=5)
    model_double.fit(X_double, y)
    joblib.dump(model_double, f"model_高三兩次_{target}.joblib")
    joblib.dump(features_double, f"features_高三兩次_{target}.joblib")

print("🎉 單次與兩次專屬模型訓練完畢！總共產出 24 個檔案！")