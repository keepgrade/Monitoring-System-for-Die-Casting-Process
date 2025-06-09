###########################################
# mold_code별 Isolation Forest 
###########################################
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import shap

df_train = pd.read_csv('./df_final.csv')
df_test = pd.read_csv('./streaming_df.csv')

# 2. 전처리는 copy로
df_train_copy = df_train.copy()
df_test_copy = df_test.copy()

drop_cols = ['count', 'passorfail', 'registration_time', 'is_anomaly', 'anomaly_score', 'anomaly_level']
df_train_copy = df_train_copy.drop(columns=[col for col in drop_cols if col in df_train_copy.columns])
df_test_copy = df_test_copy.drop(columns=[col for col in drop_cols if col in df_test_copy.columns])

# 3. 공통 mold_code만 추출
train_molds = set(df_train_copy['mold_code'].unique())
test_molds = set(df_test_copy['mold_code'].unique())
common_molds = train_molds & test_molds

# 4. 수치형 변수 추출
num_cols = df_train_copy.select_dtypes(include='number').columns.tolist()
if 'mold_code' in num_cols:
    num_cols.remove('mold_code')

# 5. 원본에 컬럼 초기화
for df in [df_train, df_test]:
    df['is_anomaly'] = 0
    df['anomaly_score'] = np.nan
    df['anomaly_level'] = '정상'

# 6. mold_code별 IsolationForest 적용
for mold in common_molds:
    idx_train = df_train_copy['mold_code'] == mold
    idx_test = df_test_copy['mold_code'] == mold

    X_train = df_train_copy.loc[idx_train, num_cols].fillna(df_train_copy.loc[idx_train, num_cols].mean())
    X_test = df_test_copy.loc[idx_test, num_cols].fillna(df_test_copy.loc[idx_test, num_cols].mean())

    model = IsolationForest(contamination=0.05, random_state=42)
    model.fit(X_train)
    pred_train = model.predict(X_train)
    score_train = model.decision_function(X_train)
    pred_test = model.predict(X_test)
    score_test = model.decision_function(X_test)

    # 각 mold_code별 행에 예측 결과 할당
    df_train.loc[idx_train, 'is_anomaly'] = pred_train
    df_train.loc[idx_train, 'anomaly_score'] = score_train
    df_test.loc[idx_test, 'is_anomaly'] = pred_test
    df_test.loc[idx_test, 'anomaly_score'] = score_test

# 7. anomaly_level 분류 함수 (NaN은 '새제품')
def assign_anomaly_level(df):
    score = df['anomaly_score']
    cut_2 = np.nanpercentile(score, 2)
    cut_5 = np.nanpercentile(score, 5)
    print(f"✅ 2% 기준 (심각): {cut_2:.5f}")
    print(f"✅ 5% 기준 (경도): {cut_5:.5f}")
    cond_serious = score <= cut_2
    cond_mild = (score > cut_2) & (score <= cut_5)
    anomaly_level = np.where(cond_serious, '심각', np.where(cond_mild, '경도', '정상'))
    anomaly_level = np.where(score.isna(), '데이터수집중', anomaly_level)
    return anomaly_level

# 8. 원본에 적용
df_train['anomaly_level'] = assign_anomaly_level(df_train)
df_test['anomaly_level'] = assign_anomaly_level(df_test)

# 9. 저장
# df_train.to_csv('df_train_with_anomaly.csv', index=False)
# df_test.to_csv('df_test_with_anomaly.csv', index=False)


for mold in common_molds:
    idx_train = df_train_copy['mold_code'] == mold
    idx_test = df_test_copy['mold_code'] == mold

    X_train = df_train_copy.loc[idx_train, num_cols].fillna(df_train_copy.loc[idx_train, num_cols].mean())
    X_test = df_test_copy.loc[idx_test, num_cols].fillna(df_test_copy.loc[idx_test, num_cols].mean())

    model = IsolationForest(contamination=0.05, random_state=42)
    model.fit(X_train)
    pred_test = model.predict(X_test)
    score_test = model.decision_function(X_test)

    df_test.loc[idx_test, 'is_anomaly'] = pred_test
    df_test.loc[idx_test, 'anomaly_score'] = score_test

    # Top1~Top3 컬럼 초기화(없으면)
    for col in ['top1', 'top2', 'top3', 'top1_val', 'top2_val', 'top3_val']:
        if col not in df_test.columns:
            df_test[col] = np.nan


# SHAP 계산: 모델, 해당 그룹 X_test, 피처 이름
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
    
        # is_anomaly가 -1인 행(이상치)만 반복
        mask_anom = (df_test.loc[idx_test, 'is_anomaly'] == -1).values  # local 인덱스 기준
        anom_indices = np.where(mask_anom)[0]
    
        for i in anom_indices:
            shap_row = shap_values[i]
            # SHAP 절댓값 내림차순 정렬(변수 중요도)
            top_idx = np.argsort(np.abs(shap_row))[::-1][:3]
            top_names = [num_cols[j] for j in top_idx]  # 피처명
            top_vals = [shap_row[j] for j in top_idx]   # SHAP값
    
            # 실제 df_test에서 해당 행의 전역 인덱스
            global_idx = df_test.loc[idx_test].index[i]
            # 변수명, 절댓값 모두 기록
            for k, (col, valcol) in enumerate(zip(['top1', 'top2', 'top3'], ['top1_val', 'top2_val', 'top3_val'])):
                if k < len(top_names):
                    df_test.at[global_idx, col] = top_names[k]
                    df_test.at[global_idx, valcol] = abs(top_vals[k])
    except Exception as e:
        print(f"SHAP error for mold_code={mold}: {e}")
# df_train.to_csv('df_train_with_anomaly.csv', index=False)
# df_test.to_csv('df_test_with_anomaly.csv', index=False)


import pickle
pickle.dump(model, open("model_iso.pkl", "wb"))