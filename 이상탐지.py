# 기본 라이브러리 및 시각화 패키지 임포트
import pandas as pd                 # 데이터프레임 구조 처리
import numpy as np                 # 수치 계산 및 배열 처리
import janitor                     # 데이터 전처리 지원 (clean_names 등)
import matplotlib.pyplot as plt    # 시각화 도구
import seaborn as sns              # 고급 시각화 도구
import time                        # 시간 측정용
from sklearn.preprocessing import StandardScaler  # 데이터 정규화 도구

# 학습 데이터 불러오기
train = pd.read_csv("train.csv")     # 훈련용 CSV 파일 로드
train.info()                         # 데이터 구조(결측치, 타입 등) 확인

# 검증 데이터 불러오기
valid = pd.read_csv("val.csv")       # 검증용 CSV 파일 로드
valid.info()                         # 검증용 데이터 구조 확인 (class 열이 있음)

# 변수 이름 정리 (소문자 + 밑줄 형식으로 자동 정리)
train = train.clean_names()          # ex: "Credit Amount" → "credit_amount"
valid = valid.clean_names()

# 불필요한 'id' 열 제거
train = train.drop(['id'], axis=1)   # 학습 데이터의 id 제거 (예측에 필요 없음)
valid_x = valid.drop(['id', 'class'], axis=1)  # 검증용 feature만 남김
valid_y = valid['class']            # 정답(label)만 따로 추출

# ============================
# LOF(Local Outlier Factor) 모델 정의 및 학습
# ============================

from sklearn.neighbors import LocalOutlierFactor

# 전체 학습 데이터 수(n)에 대해 log(n)를 취한 값을 k값(n_neighbors)로 설정
# 이는 밀도 기반 이상치 탐지에서 일반적인 경험적 기준임
minpts = np.round(np.log(train.shape[0])).astype(int)  # 예: 10,000개면 약 9~10

# LOF 모델 객체 생성
clf = LocalOutlierFactor(
    n_neighbors=minpts,           # 🔹 이상 탐지 시 고려할 이웃의 수 (k)
    contamination=0.001,          # 🔹 전체 데이터 중 이상치 비율 설정 (0.1%로 가정)
    novelty=True                  # 🔹 훈련 데이터 외 새로운 데이터(valid)에 대해 predict 허용
)
clf.fit(train)                    # 학습용 데이터로 모델 학습 진행

# (참고) 위와 같은 파라미터 설정의 예시
# lof = LocalOutlierFactor(
#     n_neighbors=12,
#     contamination=0.001,
#     novelty=True
# )

# ============================
# 모델 예측 및 평가
# ============================

from sklearn.metrics import confusion_matrix, classification_report
from sklearn import set_config  # (사용하지 않았지만 scikit-learn 설정용)

# 검증 데이터에 대한 예측 수행 (1:정상, -1:이상치)
pred_val = clf.predict(valid_x)

# sklearn LOF는 1(정상), -1(이상치) 형식으로 반환함
# 이와 맞추기 위해 검증용 y값을 동일하게 재매핑
valid_y.replace(1, -1, inplace=True)  # 실제 사기건(class==1)을 -1(이상치)로 변경
valid_y.replace(0, 1, inplace=True)   # 정상 거래(class==0)를 1로 변경

# 예측 결과와 실제 결과를 하나의 데이터프레임으로 묶기
result = pd.DataFrame({'real': valid_y, 'pred': pred_val})

# 혼동행렬(Confusion Matrix) 계산
# 예: [[TN, FP], [FN, TP]] 구조
confusion = confusion_matrix(result.real, result.pred)
print(confusion)

# 정밀도(Precision), 재현율(Recall), F1 Score, Accuracy 등 평가 지표 출력
print(classification_report(result.real, result.pred))

# ===========================
# Isolation Forest
# ===========================
import pandas as pd
import numpy as np
import janitor  # clean_names 함수 등 활용
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, classification_report

# ===========================
# 📁 1. 데이터 불러오기 및 전처리
# ===========================

# 학습 및 검증용 데이터 로딩
train = pd.read_csv("train.csv")
valid = pd.read_csv("val.csv")

# 변수 이름을 깔끔한 소문자 + 밑줄로 변환
train = train.clean_names()
valid = valid.clean_names()

# id는 예측에 불필요하므로 제거
train = train.drop(['id'], axis=1)
valid_x = valid.drop(['id', 'class'], axis=1)
valid_y = valid['class']  # 정답 라벨

# ===========================
# 🌲 2. Isolation Forest 모델 정의 및 학습
# ===========================

clf = IsolationForest(
    contamination=0.001,  # 🔹 전체 샘플 중 이상치 비율 추정 (0.1%)
    random_state=0        # 🔹 결과 재현을 위한 랜덤 시드 고정
)
clf.fit(train)            # 모델 훈련

# ===========================
# 🔍 3. 예측 및 결과 분석
# ===========================

# (1: 정상, -1: 이상치)의 형태로 예측 결과 반환됨
pred_val = clf.predict(valid_x)

# 💡 정답 라벨도 동일한 형식으로 변환
valid_y.replace(1, -1, inplace=True)  # 실제 사기 → 이상치(-1)
valid_y.replace(0, 1, inplace=True)   # 실제 정상 → 정상(1)

# 결과 데이터프레임 생성
result = pd.DataFrame({'real': valid_y, 'pred': pred_val})

# 🔢 혼동 행렬 확인
print(confusion_matrix(result.real, result.pred))

# 📝 정밀도, 재현율, F1 점수 등 상세 지표 출력
print(classification_report(result.real, result.pred))




import pandas as pd
import numpy as np
import janitor
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# ===========================
# 📁 1. 데이터 불러오기 및 전처리
# ===========================

train = pd.read_csv("train.csv")
valid = pd.read_csv("val.csv")

train = train.clean_names()
valid = valid.clean_names()

train = train.drop(['id'], axis=1)
valid_x = valid.drop(['id', 'class'], axis=1)
valid_y = valid['class'].copy()

# ===========================
# 🌲 2. Isolation Forest 모델 학습
# ===========================

# contamination 지정하지 않음
clf = IsolationForest(random_state=0)
clf.fit(train)

# ===========================
# 🧮 3. Anomaly Score 계산
# ===========================

# decision_function: 이상치일수록 score ↓ (정상은 score ↑)
scores = -clf.decision_function(valid_x)  # 부호 반전하여 "클수록 이상치"로 설정

# 히스토그램으로 이상치 분포 시각화
plt.figure(figsize=(8, 4))
plt.hist(scores, bins=50, color='skyblue')
plt.title("Anomaly Score Distribution")
plt.xlabel("Score (Higher = More Anomalous)")
plt.ylabel("Frequency")
plt.axvline(np.percentile(scores, 99), color='red', linestyle='--', label='Top 1% Threshold')
plt.legend()
plt.show()

# ===========================
# ✂️ 4. Threshold 기반 이상치 판단
# ===========================

# 이상치 기준 threshold 설정 (예: 상위 1%)
threshold = np.percentile(scores, 99)  # 상위 1%를 이상치로 간주
pred_val = np.where(scores >= threshold, -1, 1)  # score가 크면 이상치 (-1), 아니면 정상 (1)

# ===========================
# 📊 5. 성능 평가
# ===========================

from sklearn.metrics import confusion_matrix, classification_report

# 라벨 포맷 맞춤 (1: 정상, -1: 사기)
valid_y.replace({1: -1, 0: 1}, inplace=True)

result = pd.DataFrame({'real': valid_y, 'pred': pred_val})

print("📌 Confusion Matrix")
print(confusion_matrix(result.real, result.pred))

print("\n📌 Classification Report")
print(classification_report(result.real, result.pred))









class RealTimeStreamer:
    def __init__(self):
        self.test_df = streaming_df.copy()
        self.pointer = 0
        self.current_data = pd.DataFrame(columns=selected_cols)

        # ✅ 통합된 누적 데이터프레임 (초기값 = static_df의 공통 컬럼만)
        self.total_df = static_df[self._common_columns()].copy()

    def get_next_batch(self, n=1):
        if self.pointer >= len(self.test_df):
            return None

        end = min(self.pointer + n, len(self.test_df))
        batch = self.test_df.iloc[self.pointer:end]

        # 필요한 컬럼만 추출 및 전처리
        batch = self._preprocess(batch)

        # 누적 저장
        self.current_data = pd.concat([self.current_data, batch], ignore_index=True)
        self.total_df = pd.concat([self.total_df, batch], ignore_index=True)

        self.pointer = end
        return batch

    def get_current_data(self):
        """현재까지 스트리밍된 데이터 (선택된 컬럼 기준)"""
        return self.current_data

    def get_total_data(self):
        """static_df + streaming_df 누적된 전체 데이터"""
        return self.total_df

    def reset_stream(self):
        """스트리밍 상태 초기화"""
        self.pointer = 0
        self.current_data = pd.DataFrame(columns=selected_cols)
        self.total_df = static_df[self._common_columns()].copy()

    def get_stream_info(self):
        """진행률 정보 반환"""
        progress = 100 * self.pointer / len(self.test_df)
        return {
            "progress": progress,
            "total": len(self.test_df),
            "current": self.pointer
        }

    def _preprocess(self, df):
        """필요한 컬럼만 추출 (향후 전처리 확장 가능)"""
        return df[self._common_columns()].copy()

    def _common_columns(self):
        """static_df와 streaming_df 간 공통 컬럼 반환"""
        return sorted(set(static_df.columns).intersection(set(streaming_df.columns)))
