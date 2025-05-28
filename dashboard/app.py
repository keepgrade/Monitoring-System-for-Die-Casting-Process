# ================================
# 📦 1. Import
# ================================
from shiny import App, ui, render, reactive
import pandas as pd
import matplotlib.pyplot as plt
from shared import RealTimeStreamer, selected_cols , static_df, streaming_df # 필요 시 추가
import numpy as np
from datetime import datetime
import matplotlib as mpl
import joblib
plt.rcParams['font.family'] = 'Malgun Gothic'  # 윈도우
mpl.rcParams['axes.unicode_minus'] = False  # 마이너스 부호 깨짐 방지

combined_df = reactive.Value(static_df.copy())

# ================================
# 🖼️ 2. UI 정의
# ================================

app_ui = ui.page_fluid(
            ui.tags.head(
                ui.tags.link(
                    href="https://cdn.jsdelivr.net/npm/bootswatch@5.3.2/dist/journal/bootstrap.min.css",
                    rel="stylesheet"
                    )
                ), 
                ui.page_navbar(
                    ui.nav_panel("공정 over view",
                        ui.row(
                            ui.column(4,
                                ui.input_action_button( "start", "▶ 시작", class_="btn-success"
                                ),
                                ui.input_action_button("pause", "⏸ 일시정지", class_="btn-warning"),
                                ui.input_action_button("reset", "🔄 리셋", class_="btn-secondary"),
                                ui.output_ui("stream_status"),
                                ui.output_ui("progress_bar")
                            ),
                            ui.layout_columns(
                                ui.card(
                                    ui.card_header("[A]실시간 대시보드"),
                                    ui.output_plot("stream_plot", height="400px"),
                                    ui.div(
                                        ui.output_table("recent_data_table"),
                                        style="max-height: 200px; overflow-y: auto;"
                                    )
                                ),
                                ui.card(
                                    ui.card_header("[B]")
                                ),
                            ),
                            ui.layout_columns(
                                ui.card(
                                    ui.card_header("[C]")
                                ),
                                ui.card(
                                    ui.card_header("[D]")
                                ),
                            )    
                        )
                    ),
                    ui.nav_panel("공정 이상 탐지",
                        ui.layout_columns(
                            ui.card(
                                ui.card_header("[A]"),
                            ),
                            ui.card(
                                ui.card_header("[B]"),
                            )
                        ),
                        ui.layout_columns(
                            ui.card(
                                ui.card_header("[C]"),
                            ),
                            ui.card(
                                ui.card_header("[D]"),
                            )
                        )
                    ),
                    ui.nav_panel("품질 이상 탐지지",
                        ui.layout_columns(
                            ui.card(
                                ui.card_header("[A]"),
                                ui.input_select(
                                    "grouping_unit", 
                                    "📅 기간 단위 선택", 
                                    choices=["일", "주", "월"], 
                                    selected="일"
                                ),
                                ui.output_plot("defect_rate_plot", height="300px"),
                                
                            ),
                            ui.card(
                                ui.card_header("[B]"),
                            )
                        ),
                        ui.layout_columns(
                            ui.card(
                                ui.card_header("[C]"),
                            ),
                            ui.card(
                                ui.card_header("[D]"),
                            )
                        )
                    ),
                    title = "🚀실시간 스트리밍 대시보드"
                )
            )


# ================================
# ⚙️ 3. 서버 로직
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
            reactive.invalidate_later(1)  # 1초 간격으로 새 데이터 불러오기
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
            ax.set_xticks([])
            ax.set_yticks([])
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


    
    @output
    @render.plot
    def defect_rate_plot():
        try:
            unit = input.grouping_unit()  # "일", "주", "월"

            #df_vis = static_df.copy()
            df_vis = streamer.get().get_total_data()

            # 문자열 날짜를 datetime으로 변환
            df_vis['datetime'] = pd.to_datetime(
                df_vis['date'] + " " + df_vis['time'], 
                format="%H:%M:%S %Y-%m-%d",
                errors="coerce"
            )

            # 그룹핑 기준 추가
            if unit == "일":
                df_vis['group'] = df_vis['datetime'].dt.strftime('%Y-%m-%d')
            elif unit == "주":
                df_vis['group'] = df_vis['datetime'].dt.to_period('W').astype(str)
            elif unit == "월":
                df_vis['group'] = df_vis['datetime'].dt.to_period('M').astype(str)

            # 각 그룹별 불량률 계산
            group_result = df_vis.groupby(['group', 'passorfail']).size().unstack(fill_value=0)
    
            # 가장 최근 group 선택 (예: 마지막 날짜)
            latest_group = group_result.index[-1]
            counts = group_result.loc[latest_group]
    
            # 시각화
            fig, ax = plt.subplots()
            labels = ['양품', '불량']
            sizes = [counts.get(0, 0), counts.get(1, 0)]
            colors = ['#4CAF50', '#F44336']
    
            wedges, _, _ = ax.pie(
                sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90
            )
            ax.axis('equal')
            ax.set_title(f"{latest_group} ({unit} 기준) 불량률")
            ax.legend(wedges, labels, title="예측 결과", loc="upper right", bbox_to_anchor=(1.1, 1))
    
            return fig
    
        except Exception as e:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, f"에러: {str(e)}", ha='center', va='center')
            return fig
# ================================
# 🚀 4. 앱 실행
# ================================
app = App(app_ui, server)
