import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
app_dir = Path(__file__).parent
df = pd.read_csv(app_dir / "./data/df_final.csv",index_col=0)

X = df.drop(columns=['passorfail'])
y = df['passorfail']

# 4. 훈련/테스트 데이터 분할
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 5. 수치형, 범주형 컬럼 나누기
num_columns = train_X.select_dtypes(include='number').columns.tolist()
cat_columns = train_X.select_dtypes(include='object').columns.tolist()

# 6. 수치형 파이프라인: StandardScaler + PCA
num_preprocess = make_pipeline(
    StandardScaler()
)

# 7. 범주형 파이프라인: OneHotEncoder
cat_preprocess = make_pipeline(
    OneHotEncoder(handle_unknown='ignore')
)
preprocess = ColumnTransformer([
    ("num", num_preprocess, num_columns),
    ("cat", cat_preprocess, cat_columns)
])

from sklearn.ensemble import GradientBoostingClassifier
full_pipe = Pipeline([("preprocess", preprocess),("classifier", GradientBoostingClassifier())])
full_pipe_xgb = Pipeline([
    ("preprocess", preprocess),
    ("classifier", XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
])

GradientBoosting_param = {'classifier__learning_rate': np.arange(0.1, 0.3, 0.05)}

XGB_param = {
    'classifier__learning_rate': np.arange(0.1,0.4,0.05),
    'classifier__max_depth': [3, 4, 5],  # 추가로 추천
    'classifier__n_estimators': [42]   # 기본값 고정 또는 튜닝
}

from sklearn.model_selection import GridSearchCV
GradientBoosting_search = GridSearchCV(
    estimator = full_pipe,
    param_grid = GradientBoosting_param,
    cv = 5,
    scoring = 'f1_macro')
XGB_search = GridSearchCV(
    estimator=full_pipe,
    param_grid=XGB_param,
    cv=5,
    scoring='f1_macro',
    n_jobs=-1,
    verbose=1
)

model = GradientBoosting_search.fit(train_X, train_y)

print('best 파라미터 조합 :', GradientBoosting_search.best_params_)

importances = GradientBoosting_search.best_estimator_.named_steps["classifier"].feature_importances_
# 피처 이름과 함께 DataFrame으로 정리
feature_names = GradientBoosting_search.best_estimator_ \
    .named_steps["preprocess"] \
    .get_feature_names_out()
    
    
feature_importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values(by='importance', ascending=False)
feature_importance_df.head(10)
# 시각화
plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance_df)
plt.title("Feature Importance (Isolation Forest)")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()



model_xgb = XGB_search.fit(train_X,train_y)

print('best 파라미터 조합 :', GradientBoosting_search.best_params_)
print('교차검증 f1 score :', GradientBoosting_search.best_score_)


print('best 파라미터 조합 :', XGB_search.best_params_)
print('교차검증 f1 score :', XGB_search.best_score_)

from sklearn.metrics import f1_score
gb_pred = GradientBoosting_search.predict(test_X)
xgb_pred = XGB_search.predict(test_X)
print('테스트 f1 score :', f1_score(test_y, gb_pred))
print('테스트 f1 score :', f1_score(test_y, xgb_pred))

import pickle
pickle.dump(model, open("model.pkl", "wb"))

model_loaded = pickle.load(open("model.pkl", "rb"))
# 예측 확인
print(model_loaded.predict(X[:200]))