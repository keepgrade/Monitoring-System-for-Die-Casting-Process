# ================================
# 📦 1. Import
# ================================
from shiny import App, ui, render, reactive
import pandas as pd
import matplotlib.pyplot as plt
from shared import RealTimeStreamer, sensor_labels, static_df, streaming_df
import numpy as np
from datetime import datetime, timedelta
import matplotlib as mpl
import joblib
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'Malgun Gothic'  # 윈도우
mpl.rcParams['axes.unicode_minus'] = False  # 마이너스 부호 깨짐 방지

# ================================
# 🖼️ 2. UI 정의
# ================================

app_ui = ui.page_fluid(
    ui.tags.head(
        ui.tags.link(
            href="https://cdn.jsdelivr.net/npm/bootswatch@5.3.2/dist/journal/bootstrap.min.css",
            rel="stylesheet"
        ),
        ui.tags.style("""
            .alert-card { border-radius: 10px; margin: 10px 0; }
            .normal-card { background-color: #d4edda; border-color: #c3e6cb; }
            .anomaly-card { background-color: #f8d7da; border-color: #f5c6cb; }
            .log-container { max-height: 300px; overflow-y: auto; }
            .status-good { color: #28a745; font-weight: bold; }
            .status-bad { color: #dc3545; font-weight: bold; }
        """)
    ), 
    ui.page_navbar(
        # ================================
        # TAB 1: 공정 모니터링 overview
        # ================================
        ui.nav_panel("공정 모니터링 Overview",
            ui.row(
                ui.column(12,
                    ui.div(
                        ui.input_action_button("start", "▶ 시작", class_="btn-success me-2"),
                        ui.input_action_button("pause", "⏸ 일시정지", class_="btn-warning me-2"),
                        ui.input_action_button("reset", "🔄 리셋", class_="btn-secondary me-2"),
                        ui.output_ui("stream_status"),
                        ui.output_ui("progress_bar"),
                        class_="mb-3"
                    )
                )
            ),
            ui.layout_columns(
                # [A] 실시간 그래프
                ui.card(
                    ui.card_header("📊 [A] 실시간 그래프"),
                    ui.div(
                        ui.input_checkbox_group(
                            "sensor_filter",
                            "센서 선택",
                            choices=sensor_labels,
                            selected=sensor_labels
                        ),
                        class_="mb-3"
                    ),
                    ui.output_plot("real_time_graph", height="350px")
                ),
                # [B] 실시간 값
                ui.card(
                    ui.card_header("📈 [B] 실시간 값"),
                    ui.output_ui("real_time_values")
                ),
                col_widths=[8, 4]
            ),
            ui.layout_columns(
                # [C] 실시간 로그
                ui.card(
                    ui.card_header("📝 [C] 실시간 로그"),
                    ui.div(
                        ui.input_text("log_filter", "키워드 검색", placeholder="검색어 입력..."),
                        class_="mb-2"
                    ),
                    ui.div(
                        ui.output_ui("real_time_logs"),
                        class_="log-container"
                    )
                ),
                # [D] 이상 불량 알림 탭
                ui.card(
                    ui.card_header("🚨 [D] 이상 불량 알림"),
                    ui.output_ui("anomaly_alerts")
                ),
                col_widths=[6, 6]
            )    
        ),
        
        # ================================
        # TAB 2: 이상 예측
        # ================================
        ui.nav_panel("이상 예측",
            ui.layout_columns(
                # [A] 주요 변수의 이상 발생 횟수
                ui.card(
                    ui.card_header("📊 [A] 주요 변수의 이상 발생 횟수"),
                    ui.output_plot("anomaly_variable_count", height="300px")
                ),
                # [B] 이상 탐지 알림
                ui.card(
                    ui.card_header("🔔 [B] 이상 탐지 알림"),
                    ui.output_ui("anomaly_notifications")
                ),
                col_widths=[6, 6]
            ),
            ui.layout_columns(
                # [C] 시간에 따른 이상 분석
                ui.card(
                    ui.card_header("📈 [C] 시간에 따른 이상 분석"),
                    ui.div(
                        ui.input_select(
                            "anomaly_time_unit", 
                            "시간 단위 선택", 
                            choices=["1시간", "3시간", "일", "주", "월"], 
                            selected="일"
                        ),
                        class_="mb-3"
                    ),
                    ui.output_plot("anomaly_time_analysis", height="300px")
                ),
                # [D] SHAP 해석, 변수 기여도 분석
                ui.card(
                    ui.card_header("🔍 [D] SHAP 변수 기여도 분석"),
                    ui.output_table("shap_analysis_table")
                ),
                col_widths=[6, 6]
            )
        ),
        
        # ================================
        # TAB 3: 품질
        # ================================

            ui.nav_panel("품질 이상 탐지",
                                    ui.layout_columns(
                                        ui.card(
                                            ui.card_header("[A]"),
                                            ui.input_select(
                                                "grouping_unit", 
                                                "📅 기간 단위 선택", 
                                                choices=["일", "주", "월"], 
                                                selected="일"
                                            ),
                                            ui.output_ui("group_choice"),
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
    # 초기 상태
    streamer = reactive.Value(RealTimeStreamer())
    current_data = reactive.Value(pd.DataFrame())
    is_streaming = reactive.Value(False)
    logs = reactive.Value([])
    prediction_logs = reactive.Value([])

    # ================================
    # 스트리밍 제어
    # ================================
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
        logs.set([])
        prediction_logs.set([])

    # 스트리밍 실패 카운터 (최대 5회 허용)
    fail_count = reactive.Value(0)

    @reactive.effect
    def stream_data():
        reactive.invalidate_later(1)  # 항상 재호출 예약

        if not is_streaming.get():
            return

        try:
            s = streamer.get()
            next_batch = s.get_next_batch(1)

            # 데이터가 비어있지 않으면 정상 처리
            if next_batch is not None and not next_batch.empty:
                fail_count.set(0)
                current_data.set(s.get_current_data())

                new_logs = logs.get()
                timestamp = datetime.now().strftime("%H:%M:%S")
                row = next_batch.iloc[-1]

                for sensor, (label, unit) in sensor_labels.items():
                    if sensor in row and pd.notna(row[sensor]):
                        new_logs.append(f"[{timestamp}] {label}: {row[sensor]:.1f}{unit}")

                if len(new_logs) > 1000000:
                    new_logs = new_logs[-100000:]

                logs.set(new_logs)

                # 예측 로그
                pred_logs = prediction_logs.get()
                if 'passorfail' in row:
                    prob = np.random.uniform(0, 1)
                    result = "불량" if prob > 0.5 else "양품"
                    pred_logs.append(f"[{timestamp}] 확률: {prob:.3f}, 판정: {result}")
                    if len(pred_logs) > 20:
                        pred_logs = pred_logs[-20:]
                    prediction_logs.set(pred_logs)

            else:
                # fail_count 누적, 5회 이상이면 종료
                fc = fail_count.get() + 1
                fail_count.set(fc)
                if fc >= 100:
                    print("[⛔️ 종료] 데이터 스트림이 5회 이상 비었습니다.")
                    is_streaming.set(False)

        except Exception as e:
            print(f"[⚠️ 스트리밍 예외] {e}")
            # 스트리밍을 끄지 말고 예외만 로깅

    # ================================
    # TAB 1: 공정 모니터링 Overview
    # ================================

    # ▶ 데이터 스트리밍 진행률을 퍼센트로 표시합니다.
    @output
    @render.ui
    def stream_status():
        status = "🟢 스트리밍 중" if is_streaming.get() else "🔴 정지됨"
        return ui.div(status, class_="badge bg-info")

    @output
    @render.ui
    def progress_bar():
        try:
            progress = streamer.get().get_stream_info()
            return ui.div(f"진행률: {progress:.1f}%", class_="text-muted small")
        except:
            return ui.div("진행률: 0%", class_="text-muted small")
    # ================================
    # TAP 1 [A] - 스트리밍 표시 
    # ================================
    @output
    @render.plot
    def real_time_graph():
        try:
            df = current_data.get()

            # 1. 데이터가 아예 없을 경우
            if df.empty:
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.text(0.5, 0.5, "스트리밍을 시작하세요", ha='center', va='center', fontsize=16)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                return fig

            # 2. 사용자가 선택한 센서
            selected_sensors = input.sensor_filter() or []

            # 3. 그래프 준비
            fig, ax = plt.subplots(figsize=(12, 6))

            for col in selected_sensors:
                if col in df.columns:
                    col_data = df[col]

                    # 숫자형이고 NaN이 아닌 데이터가 있는 경우만 그림
                    if pd.api.types.is_numeric_dtype(col_data) and not col_data.dropna().empty:
                        ax.plot(range(len(col_data)), col_data, label=col, linewidth=2)

            ax.legend()
            ax.set_title("실시간 센서 데이터")
            ax.grid(True, alpha=0.3)
            ax.set_xlabel("시간 (인덱스)")

            return fig

        except Exception as e:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(0.5, 0.5, f"오류 발생: {str(e)}", ha='center', va='center', fontsize=12)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            return fig



    @output
    @render.ui
    def real_time_values():
        try:
            df = current_data.get()
            if df.empty:
                return ui.div("데이터 없음", class_="text-muted")

            latest = df.iloc[-1] if len(df) > 0 else None
            prev = df.iloc[-2] if len(df) > 1 else latest

            cards = []
            for col in sensor_labels:
                if col in df.columns:
                    current_val = latest[col]
                    prev_val = prev[col] if prev is not None else current_val
                    
                    # 증감 화살표
                    if current_val > prev_val:
                        arrow = "⬆️"
                        color_class = "text-success"
                    elif current_val < prev_val:
                        arrow = "⬇️"
                        color_class = "text-danger"
                    else:
                        arrow = "➡️"
                        color_class = "text-muted"
                    
                    # 임계값 체크 (예시)
                    warning_class = ""
                    if col == 'molten_temp' and current_val > 850:
                        warning_class = "border-danger"
                    elif col == 'cast_pressure' and current_val > 200:
                        warning_class = "border-danger"
                    
                    cards.append(
                        ui.div(
                            ui.h6(col.replace('_', ' ').title()),
                            ui.h4(f"{current_val:.1f} {arrow}", class_=color_class),
                            class_=f"card p-3 mb-2 {warning_class}"
                        )
                    )
            
            return ui.div(*cards)
            
        except Exception as e:
            return ui.div(f"오류: {str(e)}", class_="text-danger")

    @output
    @render.ui
    def real_time_logs():
        try:
            all_logs = logs.get()
            filter_keyword = input.log_filter()
            
            if filter_keyword:
                filtered_logs = [log for log in all_logs if filter_keyword.lower() in log.lower()]
            else:
                filtered_logs = all_logs
            
            # 최근 로그가 위로 올라오도록 역순 정렬
            filtered_logs = list(reversed(filtered_logs[-20:]))
            
            if not filtered_logs:
                return ui.div("로그 없음", class_="text-muted")
            
            log_items = []
            for log in filtered_logs:
                # 키워드에 따른 색상 구분
                if "온도" in log:
                    badge_class = "bg-warning"
                elif "압력" in log:
                    badge_class = "bg-info"
                else:
                    badge_class = "bg-secondary"
                
                log_items.append(
                    ui.div(
                        ui.span(log, class_=f"badge {badge_class} mb-1 d-block text-start")
                    )
                )
            
            return ui.div(*log_items)
            
        except Exception as e:
            return ui.div(f"오류: {str(e)}", class_="text-danger")

    @output
    @render.ui
    def anomaly_alerts():
        try:
            df = streamer.get().get_total_data()
            if df.empty:
                return ui.div("데이터 없음", class_="text-muted")

            # 최신 데이터로 이상/불량 상태 확인
            latest = df.iloc[-1] if len(df) > 0 else None
            
            # 이상 탐지 카드
            anomaly_status = "이상" if hasattr(latest, 'is_anomaly') and latest.get('is_anomaly', 0) == 1 else "정상"
            anomaly_score = latest.get('anomaly_score', 0) if latest is not None else 0
            anomaly_icon = "❌" if anomaly_status == "이상" else "✅"
            anomaly_class = "anomaly-card alert alert-danger" if anomaly_status == "이상" else "normal-card alert alert-success"
            
            # 불량 예측 카드
            defect_status = "불량" if hasattr(latest, 'predicted_label') and latest.get('predicted_label', 0) == 1 else "양품"
            defect_prob = latest.get('predict_proba', 0) if latest is not None else 0
            defect_icon = "❌" if defect_status == "불량" else "✅"
            defect_class = "anomaly-card alert alert-danger" if defect_status == "불량" else "normal-card alert alert-success"
            
            return ui.div(
                # 이상 탐지 카드
                ui.div(
                    ui.h6(f"{anomaly_icon} 이상 탐지"),
                    ui.p(f"상태: {anomaly_status}"),
                    ui.p(f"점수: {anomaly_score:.3f}"),
                    ui.p(f"시각: {datetime.now().strftime('%H:%M:%S')}"),
                    ui.input_action_button("goto_anomaly", "이상탐지 확인하기", class_="btn btn-sm btn-outline-primary"),
                    class_=anomaly_class
                ),
                # 불량 예측 카드
                ui.div(
                    ui.h6(f"{defect_icon} 불량 예측"),
                    ui.p(f"상태: {defect_status}"),
                    ui.p(f"확률: {defect_prob:.3f}"),
                    ui.p(f"시각: {datetime.now().strftime('%H:%M:%S')}"),
                    ui.input_action_button("goto_quality", "불량탐지 확인하기", class_="btn btn-sm btn-outline-primary"),
                    class_=defect_class
                )
            )
            
        except Exception as e:
            return ui.div(f"오류: {str(e)}", class_="text-danger")

    # ================================
    # TAB 2: 이상 예측
    # ================================
    @output
    @render.plot
    def anomaly_variable_count():
        try:
            df = streamer.get().get_total_data()
            if df.empty:
                fig, ax = plt.subplots()
                ax.text(0.5, 0.5, "데이터 없음", ha='center', va='center')
                return fig

            # 이상 데이터만 필터링
            if 'is_anomaly' in df.columns:
                anomaly_df = df[df['is_anomaly'] == 1]
            else:
                # 임시로 상위 20% 데이터를 이상으로 간주
                threshold = df['anomaly_score'].quantile(0.8) if 'anomaly_score' in df.columns else 0.8
                anomaly_df = df[df.get('anomaly_score', 0) > threshold]

            if anomaly_df.empty:
                fig, ax = plt.subplots()
                ax.text(0.5, 0.5, "이상 데이터 없음", ha='center', va='center')
                return fig

            # SHAP 기준 변수별 영향도 집계 (시뮬레이션)
            variables = sensor_labels
            counts = {}
            
            for var in variables:
                # 각 이상 샘플에서 해당 변수가 가장 큰 영향을 준 횟수 계산
                # 실제로는 SHAP 값을 사용하지만, 여기서는 시뮬레이션
                counts[var] = np.random.randint(1, len(anomaly_df)//2)

            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(counts.keys(), counts.values(), color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'])
            ax.set_title("주요 변수의 이상 발생 횟수 (SHAP 기반)")
            ax.set_xlabel("변수명")
            ax.set_ylabel("이상 발생 횟수")
            
            # 막대 위에 값 표시
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                       f'{int(height)}', ha='center', va='bottom')
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            return fig
            
        except Exception as e:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, f"오류: {str(e)}", ha='center', va='center')
            return fig

    @output
    @render.ui
    def anomaly_notifications():
        try:
            df = streamer.get().get_total_data()
            if df.empty:
                return ui.div("데이터 없음", class_="text-muted")

            # 최근 이상 발생 건 조회
            if 'is_anomaly' in df.columns:
                recent_anomalies = df[df['is_anomaly'] == 1].tail(5)
            else:
                threshold = df['anomaly_score'].quantile(0.8) if 'anomaly_score' in df.columns else 0.8
                recent_anomalies = df[df.get('anomaly_score', 0) > threshold].tail(5)

            if recent_anomalies.empty:
                return ui.div("최근 이상 없음", class_="text-success")

            notifications = []
            risk_counts = {"위험": 0, "경고": 0, "주의": 0}
            
            for idx, row in recent_anomalies.iterrows():
                score = row.get('anomaly_score', 0)
                
                # 위험도 분류
                if score > 0.9:
                    risk_level = "위험"
                    icon = "🔴"
                    risk_counts["위험"] += 1
                elif score > 0.7:
                    risk_level = "경고"
                    icon = "🟡"
                    risk_counts["경고"] += 1
                else:
                    risk_level = "주의"
                    icon = "🟠"
                    risk_counts["주의"] += 1
                
                # 주요 원인 (시뮬레이션)
                main_cause = np.random.choice(sensor_labels)
                time_str = datetime.now().strftime('%H:%M:%S')
                
                notifications.append(
                    ui.div(
                        ui.p(f"{icon} [{risk_level}] {time_str}"),
                        ui.p(f"주요 원인: {main_cause}"),
                        ui.p(f"이상 점수: {score:.3f}"),
                        class_="border p-2 mb-2 rounded"
                    )
                )

            # 위험도별 누적 건수
            summary = ui.div(
                ui.h6("위험도별 누적 건수"),
                ui.p(f"🔴 위험: {risk_counts['위험']}건"),
                ui.p(f"🟡 경고: {risk_counts['경고']}건"),
                ui.p(f"🟠 주의: {risk_counts['주의']}건"),
                class_="bg-light p-2 mb-3 rounded"
            )

            return ui.div(summary, *notifications)
            
        except Exception as e:
            return ui.div(f"오류: {str(e)}", class_="text-danger")


    @output
    @render.plot
    def anomaly_time_analysis():
        try:
            df = streamer.get().get_total_data()
            if df.empty or 'datetime' not in df.columns:
                fig, ax = plt.subplots()
                ax.text(0.5, 0.5, "시간 데이터 없음", ha='center', va='center')
                return fig

            time_unit = input.anomaly_time_unit()

            # datetime 컬럼 생성/변환
            if 'datetime' not in df.columns:
                if 'date' in df.columns and 'time' in df.columns:
                    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], errors="coerce")
                else:
                    df['datetime'] = pd.date_range(start='2024-01-01', periods=len(df), freq='H')

            # 시간 단위별 그룹핑
            if time_unit == "1시간":
                df['time_group'] = df['datetime'].dt.floor('H')
            elif time_unit == "3시간":
                df['time_group'] = df['datetime'].dt.floor('3H')
            elif time_unit == "일":
                df['time_group'] = df['datetime'].dt.date
            elif time_unit == "주":
                df['time_group'] = df['datetime'].dt.to_period('W')
            elif time_unit == "월":
                df['time_group'] = df['datetime'].dt.to_period('M')

            # 이상 건수 집계
            if 'is_anomaly' in df.columns:
                anomaly_counts = df[df['is_anomaly'] == 1].groupby('time_group').size()
            else:
                threshold = df['anomaly_score'].quantile(0.8) if 'anomaly_score' in df.columns else 0.8
                anomaly_counts = df[df.get('anomaly_score', 0) > threshold].groupby('time_group').size()

            if anomaly_counts.empty:
                fig, ax = plt.subplots()
                ax.text(0.5, 0.5, "이상 데이터 없음", ha='center', va='center')
                return fig

            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(range(len(anomaly_counts)), anomaly_counts.values, marker='o', linewidth=2, markersize=6)
            ax.set_title(f"시간에 따른 이상 발생량 ({time_unit} 단위)")
            ax.set_xlabel("시간")
            ax.set_ylabel("이상 건수")
            ax.grid(True, alpha=0.3)

            # x축 라벨 설정
            if len(anomaly_counts) > 10:
                step = len(anomaly_counts) // 10
                tick_positions = range(0, len(anomaly_counts), step)
                tick_labels = [str(anomaly_counts.index[i]) for i in tick_positions]
                ax.set_xticks(tick_positions)
                ax.set_xticklabels(tick_labels, rotation=45)

            plt.tight_layout()
            return fig

        except Exception as e:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, f"에러 발생: {str(e)}", ha='center', va='center')
            return fig
    # ================================
    # TAB 3: 품질 분석
    # ================================
    @output
    @render.plot
    def defect_rate_plot():
        try:
            unit = input.grouping_unit()  # "일", "주", "월"

            #df_vis = static_df.copy()
            df_vis = streamer.get().get_total_data()

            # 문자열 날짜를 datetime으로 변환
            df_vis['datetime'] = pd.to_datetime(df_vis['registration_time'], errors="coerce")

            # 그룹핑 기준 추가
            if unit == "일":
                df_vis['group'] = df_vis['datetime'].dt.strftime('%Y-%m-%d')
            elif unit == "주":
                df_vis['group'] = df_vis['datetime'].dt.to_period('W').astype(str)
            elif unit == "월":
                df_vis['group'] = df_vis['datetime'].dt.to_period('M').astype(str)

            # 각 그룹별 불량률 계산
            group_result = df_vis.groupby(['group', 'passorfail']).size().unstack(fill_value=0)
    
            selected_group = input.selected_group()
            if selected_group not in group_result.index:
                raise ValueError("선택한 그룹에 대한 데이터가 없습니다.")
            counts = group_result.loc[selected_group]
    
            # 시각화
            fig, ax = plt.subplots()
            labels = ['양품', '불량']
            sizes = [counts.get(0, 0), counts.get(1, 0)]
            colors = ['#4CAF50', '#F44336']
    
            wedges, _, _ = ax.pie(
                sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90
            )
            ax.axis('equal')
            ax.set_title(f"{selected_group} ({unit} 기준) 불량률")
            ax.legend(wedges, labels, title="예측 결과", loc="upper right", bbox_to_anchor=(1.1, 1))
    
            return fig
    
        except Exception as e:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, f"에러: {str(e)}", ha='center', va='center')
            return fig
        
    @output
    @render.ui
    def group_choice():
        try:
            unit = input.grouping_unit()
            df_vis = streamer.get().get_total_data()
            df_vis['datetime'] = pd.to_datetime(df_vis['registration_time'], errors="coerce")

            if unit == "일":
                df_vis['group'] = df_vis['datetime'].dt.strftime('%Y-%m-%d')
            elif unit == "주":
                df_vis['group'] = df_vis['datetime'].dt.to_period('W').astype(str)
            elif unit == "월":
                df_vis['group'] = df_vis['datetime'].dt.to_period('M').astype(str)

            unique_groups = sorted(df_vis['group'].dropna().unique())
            return ui.input_select("selected_group", "📆 조회할 기간 선택", choices=unique_groups, selected=unique_groups[-1] if unique_groups else None)
        except:
            return ui.input_select("selected_group", "📆 조회할 기간 선택", choices=["선택 불가"], selected=None)

# ================================
# 🚀 4. 앱 실행
# ================================
app = App(app_ui, server)
