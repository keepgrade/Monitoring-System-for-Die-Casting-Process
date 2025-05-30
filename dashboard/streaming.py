# ================================
# 📦 1. Import
# ================================
from shiny import App, ui, render, reactive
from shinyswatch import theme
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from shared import streaming_df, app_dir
from shiny import App


# 사용할 센서 컬럼 선택
selected_cols = ['molten_temp', 'cast_pressure', 'high_section_speed']
df_selected = streaming_df[selected_cols].reset_index(drop=True)

# ================================
# 🔧 2. 실시간 스트리밍 데이터 클래스
# ================================
class RealTimeStreamer:
    def __init__(self):
        self.full_data = df_selected.copy()
        self.current_index = 0

    def get_next_batch(self, batch_size=1):
        if self.current_index >= len(self.full_data):
            return None
        end_index = min(self.current_index + batch_size, len(self.full_data))
        batch = self.full_data.iloc[self.current_index:end_index].copy()
        self.current_index = end_index
        return batch

    def get_current_data(self):
        if self.current_index == 0:
            return pd.DataFrame()
        return self.full_data.iloc[:self.current_index].copy()

    def reset_stream(self):
        self.current_index = 0

    def get_stream_info(self):
        return {
            'total_rows': len(self.full_data),
            'current_index': self.current_index,
            'progress': (self.current_index / len(self.full_data)) * 100 if len(self.full_data) > 0 else 0
        }

# ================================
# 🖼️ 3. UI 정의
# ================================
app_ui = ui.page_fluid(
    ui.h2("🚀 실시간 스트리밍 대시보드"),
    ui.row(
        ui.column(4,
            ui.input_action_button("start", "▶ 시작", class_="btn-success"),
            ui.input_action_button("pause", "⏸ 일시정지", class_="btn-warning"),
            ui.input_action_button("reset", "🔄 리셋", class_="btn-secondary"),
            ui.output_ui("stream_status"),
            ui.output_ui("progress_bar")
        ),
        ui.column(8,
            ui.output_plot("stream_plot", height="400px"),
            ui.output_table("recent_data_table")
        )
    )
)

# ================================
# ⚙️ 4. 서버 로직
# ================================
def server(input, output, session):
    streamer = reactive.Value(RealTimeStreamer())
    current_data = reactive.Value(pd.DataFrame())
    is_streaming = reactive.Value(False)

    @reactive.effect
    @reactive.event(input.start)
    def on_start():
        is_streaming.set(True)

    @reactive.effect
    @reactive.event(input.pause)
    def on_pause():
        is_streaming.set(False)

    @reactive.effect
    @reactive.event(input.reset)
    def on_reset():
        streamer.get().reset_stream()
        current_data.set(pd.DataFrame())
        is_streaming.set(False)

    @reactive.effect
    def stream_data():
        try:
            if not is_streaming.get():
                return
            reactive.invalidate_later(1)
            s = streamer.get()
            next_batch = s.get_next_batch(1)
            if next_batch is not None:
                current_data.set(s.get_current_data())
            else:
                is_streaming.set(False)
        except Exception as e:
            print("⛔ 오류 발생:", e)
            is_streaming.set(False)

    @output
    @render.ui
    def stream_status():
        try:
            status = "🟢 스트리밍 중" if is_streaming.get() else "🔴 정지됨"
            return ui.div(status)
        except Exception as e:
            return ui.div(f"에러: {str(e)}")

    @output
    @render.ui
    def progress_bar():
        try:
            info = streamer.get().get_stream_info()
            progress = info['progress']
            return ui.div(f"진행률: {progress:.1f}%")
        except Exception as e:
            return ui.div(f"에러: {str(e)}")

    @output
    @render.plot
    def stream_plot():
        try:
            df = current_data.get()
            if df.empty:
                fig, ax = plt.subplots()
                ax.text(0.5, 0.5, "스트리밍을 시작하세요", ha='center', va='center')
                ax.set_xticks([])
                ax.set_yticks([])
                return fig
            fig, ax = plt.subplots(figsize=(10, 4))
            for col in selected_cols:
                ax.plot(df[col].values, label=col)
            ax.legend()
            ax.set_title("실시간 센서 데이터")
            ax.grid(True)
            return fig
        except Exception as e:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, f"에러: {str(e)}", ha='center', va='center')
            return fig

    @output
    @render.table
    def recent_data_table():
        try:
            df = current_data.get()
            if df.empty:
                return pd.DataFrame({"상태": ["데이터 없음"]})
            return df.tail(10).round(2)
        except Exception as e:
            return pd.DataFrame({"에러": [str(e)]})

# ================================
# 🚀 5. 앱 실행
# ================================
app = App(app_ui, server)