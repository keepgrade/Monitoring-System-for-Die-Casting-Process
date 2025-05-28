# ================================
# shared.py
# ================================

import pandas as pd
from pathlib import Path

# ================================
# 📁 데이터 로딩
# ================================
app_dir = Path(__file__).parent

# 정적 데이터 (누적 데이터 분석용)
static_df = pd.read_csv(app_dir / "./data/df_final.csv", index_col=0)

# 스트리밍 데이터 (실시간 시각화용)
streaming_df = pd.read_csv(app_dir / "./data/streaming_df.csv", index_col=0)

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

    def get_next_batch(self, n=1):
        if self.pointer >= len(self.test_df):
            return None

        end = min(self.pointer + n, len(self.test_df))
        batch = self.test_df.iloc[self.pointer:end]

        # 필요한 컬럼만 추출
        batch = self._preprocess(batch)

        # 누적 저장
        self.current_data = pd.concat([self.current_data, batch], ignore_index=True)
        self.pointer = end
        return batch

    def get_current_data(self):
        return self.current_data

    def reset_stream(self):
        self.pointer = 0
        self.current_data = pd.DataFrame(columns=selected_cols)

    def get_stream_info(self):
        progress = 100 * self.pointer / len(self.test_df)
        return {
            "progress": progress,
            "total": len(self.test_df),
            "current": self.pointer
        }

    def _preprocess(self, df):
        # 필요한 컬럼만 선택 (향후 전처리 로직 확장 가능)
        return df[selected_cols].copy()
