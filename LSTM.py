# ========================================
# 📦 패키지 임포트
# ========================================
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from time import time

# ========================================
# 📁 데이터 불러오기
# ========================================
df_final = pd.read_csv("dashboard/data/df_final.csv", parse_dates=["registration_time"])
streaming_df = pd.read_csv("dashboard/data/streaming_df.csv", parse_dates=["registration_time"])

# ========================================
# ⚙️ 전처리 함수
# ========================================
feature_cols = [
    'molten_temp', 'cast_pressure', 'biscuit_thickness',
    'upper_mold_temp1', 'upper_mold_temp2', 'lower_mold_temp1', 'lower_mold_temp2',
    'sleeve_temperature', 'physical_strength', 'Coolant_temperature',
    'is_anomaly'
]
target_col = 'passorfail'

def preprocess(df):
    df = df.dropna(subset=feature_cols + [target_col]).reset_index(drop=True)
    scaler = MinMaxScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    return df

df_final = preprocess(df_final)
streaming_df = preprocess(streaming_df)

# ========================================
# 🧱 슬라이딩 윈도우 시퀀스 생성
# ========================================
def create_sequences(df, window_size=10):
    X, y = [], []
    for i in range(len(df) - window_size):
        X.append(df[feature_cols].iloc[i:i+window_size].values)
        y.append(df[target_col].iloc[i + window_size])
    return np.array(X), np.array(y)

window_size = 10
X_train, y_train = create_sequences(df_final, window_size)
X_test, y_test = create_sequences(streaming_df, window_size)

print("✅ 시퀀스 생성 완료")
print("X_train:", X_train.shape, "y_train:", y_train.shape)

# ========================================
# 🧠 LSTM 모델 구성
# ========================================
model = Sequential()
model.add(LSTM(32, input_shape=(window_size, len(feature_cols))))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# ========================================
# 🎯 모델 학습 (에포크 별 진행률 출력)
# ========================================
epochs = 10
print("🚀 모델 학습 시작 (총 {} 에포크)".format(epochs))

start_time = time()
for i in range(1, epochs + 1):
    epoch_start = time()
    history = model.fit(X_train, y_train, epochs=1, batch_size=16, verbose=0)
    epoch_time = time() - epoch_start
    percent = (i / epochs) * 100
    print(f"⏳ Epoch {i}/{epochs} 완료 | 진행률: {percent:.0f}% | 소요 시간: {epoch_time:.2f}초 | 손실: {history.history['loss'][0]:.4f}")

total_time = time() - start_time
print(f"✅ 전체 학습 완료! 총 소요 시간: {total_time:.2f}초")

# ========================================
# 💾 모델 및 메타데이터 저장
# ========================================
os.makedirs("model", exist_ok=True)
model.save("model/lstm_model.h5")

metadata = {
    "feature_cols": feature_cols,
    "window_size": window_size
}
with open("model/model_metadata.pkl", "wb") as f:
    pickle.dump(metadata, f)

print("📁 모델 및 메타데이터 저장 완료 (model/ 폴더)")

# ========================================
# 📊 학습 결과 시각화
# ========================================
plt.figure(figsize=(8, 4))
plt.plot(history.history['loss'], marker='o')
plt.title("훈련 손실 (Loss)")
plt.xlabel("Epoch")
plt.ylabel("Binary Cross-Entropy Loss")
plt.grid(True)
plt.tight_layout()
plt.savefig("model/training_loss.png")
print("📈 훈련 손실 시각화 저장 완료")

# ========================================
# 🔍 예측 vs 실제 시각화
# ========================================
y_pred_prob = model.predict(X_test).flatten()

plt.figure(figsize=(10, 4))
plt.plot(y_pred_prob, label="예측 불량 확률")
plt.plot(y_test[:len(y_pred_prob)], label="실제 불량 여부", linestyle="--")
plt.title("예측 vs 실제 불량 (Streaming Test)")
plt.xlabel("시퀀스 인덱스")
plt.ylabel("불량 확률 / 여부")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("model/prediction_vs_actual.png")
print("📉 예측 결과 시각화 저장 완료")
