import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time

# ==========================================
# 🌟 全域三引擎快取 
# ==========================================
@st.cache_resource
def load_all_models():
    models = {'高一二': {}, '高三單次': {}, '高三兩次': {}}
    features = {'高一二': {}, '高三單次': {}, '高三兩次': {}}
    targets = ['學測_國文', '學測_英文', '學測_數A', '學測_數B', '學測_自然', '學測_社會']
    
    for target in targets:
        try:
            models['高一二'][target] = joblib.load(f"model_{target}.joblib")
            features['高一二'][target] = joblib.load(f"features_{target}.joblib")
        except: pass
        try:
            models['高三單次'][target] = joblib.load(f"model_高三單次_{target}.joblib")
            features['高三單次'][target] = joblib.load(f"features_高三單次_{target}.joblib")
        except: pass
        try:
            models['高三兩次'][target] = joblib.load(f"model_高三兩次_{target}.joblib")
            features['高三兩次'][target] = joblib.load(f"features_高三兩次_{target}.joblib")
        except: pass
        
    return models, features

GLOBAL_MODELS, GLOBAL_FEATURES = load_all_models()

# --- 🌟 1. 核心邏輯與資料庫區 ---
score_map = {
    '建國中學': 33.8, '北一女中': 33.6, '師大附中': 32.8, '成功高中': 31.6, '中山女中': 30.8, 
    '松山高中': 30.2, '大同高中': 29.6, '中崙高中': 28.6, '政大附中': 28.6, '板橋高中': 28.2, 
    '成淵高中': 27.6, '麗山高中': 26.8, '海山高中': 26.6, '大直高中': 26.4, '和平高中': 25.8, 
    '西松高中': 24.6, '北大高中': 24.6, '明倫高中': 24.6, '內湖高中': 23.8, '百齡高中': 23.6, 
    '景美女中': 22.8, '中正高中': 22.6, '中和高中': 21.8, '陽明高中': 21.6, '南湖高中': 21.6, 
    '新莊高中': 20.6, '南港高中': 20.6, '永平高中': 20.6, '華江高中': 19.6, '新北高中': 19.6, 
    '永春高中': 18.6, '丹鳳高中': 18.6, '華僑高中': 18.6, '新店高中': 17.6, '錦和高中': 17.6, 
    '三民高中': 17.6, '大理高中': 16.6, '萬芳高中': 16.6, '光復高中': 16.6, '三重高中': 15.6, 
    '竹圍高中': 15.6, '育成高中': 15.6, '林口高中': 15.4, '清水高中': 13.8, '樹林高中': 12.8, 
    '安康高中': 12.6, '泰山高中': 12.6, '復興高中': 11.6, '秀峰高中': 10.6, '明德高中': 9.6, '石碇高中': 5.4
}

gsat_standards = {
    '國文': {'頂標':13, '前標':12, '均標':10, '後標':9, '底標':7},
    '英文': {'頂標':13, '前標':11, '均標':8, '後標':4, '底標':3},
    '數A': {'頂標':11, '前標':9, '均標':6, '後標':4, '底標':3},
    '數B': {'頂標':12, '前標':10, '均標':6, '後標':4, '底標':3},
    '自然': {'頂標':13, '前標':12, '均標':9, '後標':7, '底標':5},
    '社會': {'頂標':13, '前標':12, '均標':10, '後標':8, '底標':7}
}

target_goals = {
    "🏥 醫牙中醫 (自費)": {"國文": 13, "英文": 14, "數學": 14, "自然": 14},
    "🏆 頂大電資 (台清交成)": {"英文": 13, "數學": 14, "自然": 14},
    "💊 藥學系": {"國文": 13, "英文": 13, "數學": 12, "自然": 13},
    "⚡ 中字輩電資 (央興山正)": {"英文": 12, "數學": 12, "自然": 13},
    "⚙️ 中字輩理工": {"英文": 11, "數學": 10, "自然": 12},
    "⚖️ 中字輩法商": {"國文": 12, "英文": 12, "數學": 9, "社會": 12},
    "🔬 中字輩冷門理科": {"英文": 10, "數學": 9, "自然": 11},
    "📜 中字輩冷門文科": {"國文": 12, "英文": 11, "社會": 11},
    "💼 頂大商管 (台政)": {"國文": 13, "英文": 14, "數學": 12},
    "🏛️ 頂大文法 (台政)": {"國文": 14, "英文": 14, "社會": 14},
    "📈 北大商管": {"國文": 12, "英文": 13, "數學": 10},
    "✒️ 北大文法": {"國文": 13, "英文": 13, "社會": 13},
    "🎓 師範大學 (綜合)": {"國文": 13, "英文": 12, "社會": 12},
    "🏗️ 前段私立理工 (輔淡東中元逢長)": {"英文": 8, "數學": 7, "自然": 8},
    "📊 前段私立文法商 (輔淡東)": {"國文": 11, "英文": 10, "社會": 10},
    "🗺️ 地名國立大學 (嘉大/屏大等)": {"國文": 11, "英文": 9, "數學": 8},
    "🏫 一般私立大學": {"國文": 8, "英文": 6}
}

def expert_calibration(ml_pred, school_name, subject, avg_input_score):
    pt = score_map.get(school_name, 18.0)
    base = (pt * 0.4) + 0.5 
    slope = 10.0 - (pt/10.0) 
    expert_val = base + (avg_input_score - 72) / slope
    if subject in ['學測_數A', '學測_數B', '學測_自然']: expert_val -= (35 - pt) * 0.04 
    if subject in ['學測_社會', '學測_國文']: expert_val += 0.8 
    return (ml_pred * 0.35) + (expert_val * 0.65)

def senior_calibration(ml_pred, mock_vals):
    if not mock_vals: return ml_pred
    mock_avg = np.mean(mock_vals)
    return (ml_pred * 0.4) + (mock_avg * 0.6) + 0.3

def get_standard_info(sub_name, score):
    stds = gsat_standards.get(sub_name)
    if not stds: return "未知", "#eee", "#555", 0
    if score >= stds['頂標']: return "🏆 頂標", "#e8f0fe", "#1967d2", 5
    if score >= stds['前標']: return "🥇 前標", "#e6f4ea", "#137333", 4
    if score >= stds['均標']: return "🥈 均標", "#fef7e0", "#b06000", 3
    if score >= stds['後標']: return "🥉 後標", "#fce8e6", "#c5221f", 2
    if score >= stds['底標']: return "⚠️ 底標", "#fce8e6", "#c5221f", 1
    return "💔 未達底", "#f1f3f4", "#5f6368", 0

def clear_all_scores():
    for key in list(st.session_state.keys()):
        if key.startswith("editor_"):
            del st.session_state[key]

def get_val(df, row_name, col_name):
    if row_name in df.index and col_name in df.columns:
        val = df.loc[row_name, col_name]
        return float(val) if pd.notna(val) else 0.0
    return 0.0

def get_mock_vals(target_sub, in_df_senior, is_single):
    sub = target_sub.replace("學測_", "")
    vals = []
    if sub in ['數A', '數B']:
        if in_df_senior.get('模考1_數學', 0) > 0: vals.append(in_df_senior['模考1_數學'])
    else:
        if in_df_senior.get(f'模考1_{sub}', 0) > 0: vals.append(in_df_senior[f'模考1_{sub}'])
    if not is_single:
        if in_df_senior.get(f'模考2_{sub}', 0) > 0: vals.append(in_df_senior[f'模考2_{sub}'])
    return vals

# --- 🎨 2. 頁面設定與自訂 CSS ---
st.set_page_config(page_title="2026 學測戰略預測系統", page_icon="🎯", layout="wide")

st.markdown("""
<style>
    .result-card { background-color: var(--background-color); border-radius: 12px; padding: 20px; text-align: left; border: 1px solid var(--secondary-background-color); margin-bottom: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
    .score-title { font-size: 1.2rem; font-weight: 600; color: var(--text-color); margin-bottom: 5px; }
    .score-num { font-size: 1.8rem; font-weight: 800; color: #ff4b4b; float: right; margin-top: -30px;}
    .progress-bg { width: 100%; background-color: var(--secondary-background-color); border-radius: 8px; height: 16px; position: relative; margin-top: 25px; margin-bottom: 5px;}
    .progress-fill { background-color: #4b8bff; height: 100%; border-radius: 8px; transition: width 0.5s ease-in-out;}
    .marker { position: absolute; top: -5px; height: 26px; border-left: 2px dashed #ff9800; }
    .marker-text { position: absolute; top: -20px; font-size: 0.7rem; color: var(--text-color); opacity: 0.7; transform: translateX(-50%); white-space: nowrap;}
    .badge-safe { background-color: #e6f4ea; color: #137333; padding: 6px 14px; border-radius: 20px; font-weight: bold; margin: 5px; display: inline-block; border: 1px solid #ceead6;}
    .badge-reach { background-color: #fef7e0; color: #b06000; padding: 6px 14px; border-radius: 20px; font-weight: bold; margin: 5px; display: inline-block; border: 1px solid #feefc3;}
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { border-radius: 8px 8px 0px 0px; padding: 10px 20px; font-size: 1.1rem; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 側邊欄 (Sidebar)
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=60)
    st.title("戰略參數設定")
    st.caption("調整學生的基本背景與目標")
    
    st.markdown("### 👤 學生檔案")
    grade_mode = st.radio("目前階段", ["🌱 高一、高二 (潛力)", "🔥 高三 (模考實戰)"], help="高一二將啟用機器學習常模引擎；高三則啟用純實戰校準引擎。")
    is_senior = "高三" in grade_mode
    
    group = st.selectbox("所屬類組", ["理組", "文組"], help="理組將對齊校系的數A門檻；文組則自動對齊數B門檻。")
    
    if not is_senior:
        school = st.selectbox("就讀學校", list(score_map.keys()), help="高中校名底蘊會影響高一二的潛力預估爆發力。")
    else:
        st.info("💡 高三已啟用純實戰校準，為求公平不計校名。")
        school = "高三無須填寫"
        
    st.markdown("### 🎯 終極夢想")
    target_choice = st.selectbox("設定首選目標學群", list(target_goals.keys()), help="系統將針對此首選目標，為您產出專屬的教練診斷評語。")

# ==========================================
# 主畫面：資料輸入區
# ==========================================
st.title("🎯 2026 學測落點 AI 戰略分析")
st.caption("結合大數據常模與機器學習，精準預測學測落點，並自動比對全台 17 大目標學群。")

col_title, col_btn = st.columns([5, 1])
with col_title:
    st.markdown("#### 📊 學習成績輸入區")
with col_btn:
    st.button("🧹 清空表格", on_click=clear_all_scores, use_container_width=True)

with st.container(border=True):
    if not is_senior:
        subjects_list = ['國文', '英文', '數學', '物理', '化學', '生物', '地科', '歷史', '地理', '公民']
        input_type = st.radio("填寫模式", ["🎯 單次平均 (0~100分)", "📈 四學期成績 (0~100分)"], horizontal=True)
        idx_names = ["單次平均"] if "單次" in input_type else ["高一上", "高一下", "高二上", "高二下"]
        max_score = 100
        st.caption("📝 點擊表格輸入 **原始分數**。未考科目請填 0。")
    else:
        subjects_list = ['國文', '英文', '數A', '數B', '自然', '社會']
        input_type = st.radio("填寫模式", ["🥇 僅考過第一次模考 (1~15級分)", "🥈 已考過第一與第二次模考 (1~15級分)"], horizontal=True)
        idx_names = ["第一次模考"] if "僅考過" in input_type else ["第一次模考", "第二次模考"]
        max_score = 15
        st.caption("📝 點擊表格輸入 **模考級分**。未考科目請填 0。")
    
    df_init = pd.DataFrame(0, index=idx_names, columns=subjects_list)
    editor_key = f"editor_{is_senior}_{len(idx_names)}"
    config = {col: st.column_config.NumberColumn(min_value=0, max_value=max_score, step=1) for col in subjects_list}
    edited_df = st.data_editor(df_init, key=editor_key, use_container_width=True, column_config=config)

# ==========================================
# 執行預測邏輯
# ==========================================
if st.button("🚀 產出全域落點與戰略分析報告", type="primary", use_container_width=True):
    results = {}
    
    # 防呆：成績不能全 0
    if edited_df.sum().sum() == 0:
        st.error("⚠️ 偵測到成績為全 0，請至少填寫一個科目的成績再進行預測！")
        st.stop()
    
    with st.status("🚀 啟動 AI 戰略預測引擎...", expanded=True) as status:
        st.write("🔍 正在解析成績數據並分配底層模型...")
        time.sleep(0.3)
        
        if is_senior:
            is_single = "僅考過" in input_type
            engine_name = '高三單次' if is_single else '高三兩次'
            st.write(f"🧠 啟動【{engine_name}】專屬 AI 模型與防低估演算法...")
            
            math1_col = "數A" if group == "理組" else "數B"
            in_df_senior = {
                '模考1_國文': get_val(edited_df, "第一次模考", "國文"), '模考1_英文': get_val(edited_df, "第一次模考", "英文"),
                '模考1_數學': get_val(edited_df, "第一次模考", math1_col), '模考1_自然': get_val(edited_df, "第一次模考", "自然"),
                '模考1_社會': get_val(edited_df, "第一次模考", "社會"),
            }
            if not is_single:
                in_df_senior.update({
                    '模考2_國文': get_val(edited_df, "第二次模考", "國文"), '模考2_英文': get_val(edited_df, "第二次模考", "英文"),
                    '模考2_數A': get_val(edited_df, "第二次模考", "數A"), '模考2_數B': get_val(edited_df, "第二次模考", "數B"),
                    '模考2_自然': get_val(edited_df, "第二次模考", "自然"), '模考2_社會': get_val(edited_df, "第二次模考", "社會"),
                })
            
            for target in ['學測_國文', '學測_英文', '學測_數A', '學測_數B', '學測_自然', '學測_社會']:
                if target not in GLOBAL_MODELS[engine_name]: continue
                
                mock_vals = get_mock_vals(target, in_df_senior, is_single)
                if not mock_vals or sum(mock_vals) == 0: continue 

                model = GLOBAL_MODELS[engine_name][target]
                features_list = GLOBAL_FEATURES[engine_name][target]
                in_df = {f: in_df_senior.get(f, 0) for f in features_list}
                ml_pred = model.predict(pd.DataFrame([in_df])[features_list])[0]
                final_pred = senior_calibration(ml_pred, mock_vals)
                
                center = max(1, min(15, int(round(final_pred))))
                results[target.replace("學測_","")] = {"low": max(1, center - 1), "high": min(15, center + 1), "center": center}
                
        else:
            st.write("🧠 啟動高一二常模底蘊 AI 模型...")
            inputs = {}
            for col in subjects_list:
                vals = edited_df[col].replace(0, np.nan).dropna().tolist()
                if not vals: inputs[col] = 0
                elif len(vals) == 1: inputs[col] = vals[0]
                else:
                    weights = np.linspace(1, len(vals), len(vals))
                    weights = weights / weights.sum()
                    inputs[col] = np.average(vals, weights=weights) + ((vals[-1] - vals[0]) * 0.15)

            sub_map = {'學測_國文': ['國文'], '學測_英文': ['英文'], '學測_數A': ['數學'], '學測_數B': ['數學'], '學測_自然': ['物理', '化學', '生物', '地科'], '學測_社會': ['歷史', '地理', '公民']}
            for target, src in sub_map.items():
                if target not in GLOBAL_MODELS['高一二']: continue
                model = GLOBAL_MODELS['高一二'][target]
                features = GLOBAL_FEATURES['高一二'][target]
                
                valid_scores = [inputs[s] for s in src if inputs.get(s, 0) > 0]
                if not valid_scores: continue
                
                avg_score = np.mean(valid_scores)
                in_df = {f: score_map[school] if f == '會考積分' else (1 if group in f else (inputs.get(f.replace("_段考平均", ""), 0) if inputs.get(f.replace("_段考平均", ""), 0) > 0 else np.nan)) for f in features}
                ml_pred = model.predict(pd.DataFrame([in_df])[features])[0]
                final = expert_calibration(ml_pred, school, target, avg_score)
                center = max(1, min(15, int(round(final))))
                results[target.replace("學測_","")] = {"low": max(1, center - 1), "high": min(15, center + 1), "center": center}
            
        st.write("🎯 正在掃描全台 17 大學群與目標落差...")
        time.sleep(0.3)
        status.update(label="✨ 戰略診斷報告生成完畢！", state="complete", expanded=False)

    if results:
        # ==========================================
        # 報告輸出渲染
        # ==========================================
        st.markdown("---")
        safe_zones, reach_zones = [], []
        
        for goal_name, reqs in target_goals.items():
            is_qualified = True
            total_shortfall = 0
            missing_details = []
            
            for req_sub, req_score in reqs.items():
                actual_sub = "數A" if (req_sub == "數學" and group == "理組") else ("數B" if (req_sub == "數學" and group == "文組") else req_sub)
                    
                if actual_sub in results:
                    diff = results[actual_sub]['center'] - req_score
                    if diff < 0:
                        is_qualified = False
                        total_shortfall += abs(diff)
                        missing_details.append(f"{actual_sub}差{-diff}級")
                else:
                    is_qualified = False
                    total_shortfall += 99
                    missing_details.append(f"未測驗{actual_sub}")
            
            if is_qualified: safe_zones.append(goal_name)
            elif total_shortfall <= 2: reach_zones.append(f"{goal_name} ({', '.join(missing_details)})") 

        tab1, tab2, tab3 = st.tabs(["🎯 戰略總覽與學群", "📊 各科戰力卡片", "📋 匯出與分享"])
        
        with tab1:
            st.subheader(f"🎯 專屬夢想診斷：【{target_choice}】")
            with st.container(border=True):
                reqs = target_goals[target_choice]
                missing_for_dream = []
                is_dream_safe = True
                
                for req_sub, req_score in reqs.items():
                    actual_sub = "數A" if (req_sub == "數學" and group == "理組") else ("數B" if (req_sub == "數學" and group == "文組") else req_sub)
                    if actual_sub in results:
                        diff = results[actual_sub]['center'] - req_score
                        if diff < 0:
                            is_dream_safe = False
                            missing_for_dream.append(f"**{actual_sub}** 尚差 {-diff} 級分")
                    else:
                        is_dream_safe = False
                        missing_for_dream.append(f"未輸入 **{actual_sub}** 影響評估")
                        
                if is_dream_safe:
                    st.success(f"🎉 **戰略綠燈**：太棒了！你目前的預估戰力已經穩穩達標【{target_choice}】！請繼續保持現有的讀書節奏！")
                    st.balloons() 
                else:
                    st.error(f"🚀 **戰略衝刺**：距離夢想【{target_choice}】還差一點點火侯！")
                    st.markdown(f"💡 **教練建議**：接下來的複習計畫，請務必將火力集中在救援短板： {', '.join(missing_for_dream)}。")

            st.markdown("<br>", unsafe_allow_html=True)
            st.subheader("💡 AI 全域學群配對雷達")
            with st.container(border=True):
                if safe_zones:
                    st.markdown("**✅ 穩健錄取區 (已達門檻)：**")
                    st.markdown("".join([f"<div class='badge-safe'>{zone}</div>" for zone in safe_zones]), unsafe_allow_html=True)
                else:
                    st.warning("目前預測分數尚無穩健錄取的頂段學群，請參考下方的衝刺區！")
                st.markdown("<br>", unsafe_allow_html=True)
                if reach_zones:
                    st.markdown("**🔥 衝刺挑戰區 (只差 1~2 級分)：**")
                    st.markdown("".join([f"<div class='badge-reach'>{zone.split(' (')[0]} <span style='font-size:0.8rem; font-weight:normal;'>({zone.split(' (')[1]}</span></div>" for zone in reach_zones]), unsafe_allow_html=True)

        with tab2:
            st.subheader("📊 各科預估戰力詳情")
            col_res1, col_res2, col_res3 = st.columns(3) 
            
            for i, (sub, d) in enumerate(results.items()):
                label, bg_color, text_color, _ = get_standard_info(sub, d['center'])
                std_dict = gsat_standards[sub]
                pct, front_pct, avg_pct = (d['center'] / 15.0) * 100, (std_dict['前標'] / 15.0) * 100, (std_dict['均標'] / 15.0) * 100
                
                card_html = (
                    f'<div class="result-card">'
                    f'<div class="score-title">{sub} <span style="font-size:0.8rem; font-weight:normal; opacity:0.6;">區間: {d["low"]}~{d["high"]}</span></div>'
                    f'<div class="score-num">{d["center"]}</div>'
                    f'<div style="display:inline-block; background-color:{bg_color}; color:{text_color}; padding:2px 8px; border-radius:10px; font-size:0.75rem; font-weight:bold; margin-bottom:15px;">{label}</div>'
                    f'<div class="progress-bg"><div class="progress-fill" style="width: {pct}%;"></div><div class="marker" style="left: {avg_pct}%;"></div><div class="marker-text" style="left: {avg_pct}%;">均標</div><div class="marker" style="left: {front_pct}%; border-left: 2px dashed #4caf50;"></div><div class="marker-text" style="left: {front_pct}%;">前標</div></div></div>'
                )
                with [col_res1, col_res2, col_res3][i % 3]:
                    st.markdown(card_html, unsafe_allow_html=True)

        with tab3:
            st.subheader("📋 匯出戰略報告")
            st.caption("點擊下方代碼框右上角的「複製圖示」，即可將完整報告傳送給老師或家長。")
            
            report_text = f"🎯 【2026 學測 AI 戰略預測報告】\n"
            report_text += f"🎓 階段：{grade_mode.split(' ')[1]} | 📚 類組：{group}\n"
            if not is_senior: report_text += f"🏫 學校：{school}\n"
            report_text += f"\n🎯 專屬目標診斷：【{target_choice}】\n"
            report_text += ("✅ 戰略綠燈：目前實力已達標！\n" if is_dream_safe else f"🚀 戰略衝刺：需要補足 {', '.join(missing_for_dream).replace('**', '')}\n")
            report_text += f"\n📊 預估級分：\n"
            for sub, d in results.items():
                label, _, _, _ = get_standard_info(sub, d['center'])
                report_text += f"- {sub}：{d['center']} 級分 ({label})\n"
                
            report_text += f"\n💡 AI 智能配對：\n"
            report_text += f"✅ 穩健錄取區：\n" + ("無\n" if not safe_zones else "\n".join([f"- {z}" for z in safe_zones]) + "\n")
            report_text += f"\n🔥 衝刺挑戰區：\n" + ("無\n" if not reach_zones else "\n".join([f"- {z}" for z in reach_zones]))
            report_text += f"\n\n-- 由 2026 學測 AI 戰略系統自動生成 --"
            
            st.code(report_text, language="markdown")