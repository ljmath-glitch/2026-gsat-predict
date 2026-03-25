import requests
import json

# 👇 貼上你最新部署的 Google Webhook 網址 👇
url = "https://script.google.com/macros/s/AKfycbwun6ZOROBtYlbVLm7Wp08ul1wE2dxclJTdlQO7DaMjmRISTjoErvpRISeAVHtp3VFo/exec"

data = {
    "stage": "測試階段",
    "group": "測試類組",
    "school": "測試學校",
    "target": "測試目標",
    "inputs": "測試輸入",
    "predictions": "測試預測"
}

print("🚀 正在嘗試發送資料給 Google...")
try:
    response = requests.post(url, json=data)
    print(f"✅ 狀態碼: {response.status_code}")
    print(f"📄 回傳內容: {response.text}")
    
    if response.status_code == 200 and "Success" in response.text:
        print("🎉 恭喜！資料發送完美成功，趕快去檢查表格！")
    else:
        print("⚠️ 連線成功，但 Google 沒有回傳正確的成功訊息，請檢查 Apps Script 程式碼。")
        
except Exception as e:
    print(f"❌ 發送失敗！錯誤原因：{e}")