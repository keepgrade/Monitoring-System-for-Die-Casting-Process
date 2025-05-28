import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from xgboost import XGBClassifier
app_dir = Path(__file__).parent
df = pd.read_csv(app_dir / "./data/df_final.csv",index_col=0)



from sklearn.ensemble import IsolationForest

# target 제거 후 수치형 피처만 사용
X = df.drop(columns=["passorfail"], errors='ignore').select_dtypes(include='number')

# 스케일링 (SHAP 안정성 위해 추천)
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)


model = IsolationForest(contamination=0.05, random_state=42)
model.fit(X_scaled)

# 예측 결과: -1은 이상치, 1은 정상

# 예측을 X_scaled로 하면 학습이랑 예측을 같은 데이터로?
# 수정 필요!!!!!!!!!!!!!!!!!!!!!!!!
df["is_anomaly"] = pd.Series(model.predict(X_scaled), index=df.index).map({1: 0, -1: 1})


# decision_function은 0을 기준으로 이상 여부를 나눔
scores = model.decision_function(X_scaled)



# shap로 이상원인 분석
import shap

# Tree 기반 모델용 explainer 생성
explainer = shap.Explainer(model, X_scaled)

# SHAP 값 계산 (전체 샘플 또는 일부만)
shap_values = explainer(X_scaled)


# 예: 이상치로 분류된 첫 번째 샘플 찾기
first_anomaly_idx = df[df['is_anomaly'] == 1].index[0]
df[df['is_anomaly'] == 1].head()
len(df)
len(df[df['passorfail'] == 1])
len(df[df['is_anomaly_custom'] == 1])
len(df[(df['is_anomaly_custom'] == 1) & (df['passorfail'] == 1)])
# SHAP force plot (Jupyter 환경 추천)
shap.initjs()
shap.force_plot(
    explainer.expected_value,
    shap_values[first_anomaly_idx].values,
    X_scaled.loc[first_anomaly_idx],
    matplotlib=True
)




# 기여도 높은 순서대로 정렬
shap_df = pd.DataFrame({
    'feature': X.columns,
    'shap_value': shap_values[first_anomaly_idx].values
}).sort_values(by='shap_value', key=abs, ascending=False)

print("이 이상치에 가장 기여한 변수:")
print(shap_df.head(5))