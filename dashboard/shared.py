# ================================
# shared.py
# ✅ 실시간 스트리밍 기반 공정 모니터링 대시보드용 공통 모듈
# - 데이터 로딩
# - 센서 이름 정의
# - 실시간 시뮬레이션 클래스 정의
# ================================

import pandas as pd
from pathlib import Path
import joblib

# ================================
# 📁 데이터 로딩
# ================================
app_dir = Path(__file__).parent

# ✅ 정적 데이터 (누적 데이터 분석용)
try:
    static_df = pd.read_csv(app_dir / "./data/df_final.csv", index_col=0, encoding="utf-8")
except UnicodeDecodeError:
    static_df = pd.read_csv(app_dir / "./data/df_final.csv", index_col=0, encoding="ISO-8859-1")

# ✅ 스트리밍 데이터 (실시간 시각화용)
try:
    streaming_df = pd.read_csv(app_dir / "./data/streaming_df.csv", index_col=0, encoding="utf-8")
except UnicodeDecodeError:
    streaming_df = pd.read_csv(app_dir / "./data/streaming_df.csv", index_col=0, encoding="cp949")


# ✅ 센서 데이터의 사람이 읽기 쉬운 한글 이름과 단위 정의
# UI 카드나 그래프 라벨링 시 활용
sensor_labels = {
    "molten_temp": ("용탕온도", "°C"),
    "cast_pressure": ("주조압력", "bar"),
    "high_section_speed": ("고속구간속도", "mm/s"),
    "low_section_speed": ("저속구간속도", "mm/s"),
    # 필요 시 더 추가
}

# ================================
# 🔧 실시간 스트리밍 클래스 정의
# ================================
class RealTimeStreamer:
    def __init__(self):
        self.test_df = streaming_df.copy()
        self.pointer = 0
        # 지금까지 스트리밍된 센서 데이터 (그래프 시각화용 누적 프레임)
        self.current_data = pd.DataFrame(columns=sensor_labels)
        # static_df에서 streaming_df와 공통된 컬럼만 추출하여 초기화 (누적용)
        self.total_df = static_df[self._common_columns()].copy()

    def get_next_batch(self, n=1):

        end = min(self.pointer + n, len(self.test_df))
        batch = self.test_df.iloc[self.pointer:end]

        try:
            # ✅ 전처리 시 에러 방지
            batch = self._preprocess(batch)

            # 누적 저장
            self.current_data = pd.concat([self.current_data, batch], ignore_index=True)
            self.total_df = pd.concat([self.total_df, batch], ignore_index=True)

        except Exception as e:
            print(f"[⚠️ 컬럼 오류 무시] {e}")
            # 전처리 실패 시 현재 batch는 무시하고 넘어감
            batch = pd.DataFrame()  # 빈 DF 반환

        self.pointer = end
        return batch

    def get_current_data(self):
        # 현재까지 스트리밍된 데이터 (선택된 컬럼 기준)
        return self.current_data

    def get_total_data(self):
        # static_df + streaming_df 누적된 전체 데이터
        return self.total_df

    def reset_stream(self):
        # 스트리밍 상태 초기화
        self.pointer = 0
        self.current_data = pd.DataFrame(columns=sensor_labels)
        self.total_df = static_df[self._common_columns()].copy()

    def get_stream_info(self):
        # 진행률 정보 반환
        progress = 100 * self.pointer / len(self.test_df)
        return {
            "progress": progress,
            "total": len(self.test_df),
            "current": self.pointer
        }

    def _preprocess(self, df):
        # 필요한 컬럼만 추출 (향후 전처리 확장 가능)
        return df[self._common_columns()].copy()

    def _common_columns(self):
        # static_df와 streaming_df 간 공통 컬럼 반환
        return list(set(static_df.columns).intersection(set(streaming_df.columns)))
