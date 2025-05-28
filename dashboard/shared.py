# ================================
# shared.py
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
    
# 시각화에 사용할 센서 컬럼
selected_cols = ['molten_temp', 'cast_pressure', 'high_section_speed']

# ================================
# 🔧 실시간 스트리밍 클래스 정의
# ================================
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
        return list(set(static_df.columns).intersection(set(streaming_df.columns)))
