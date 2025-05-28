# 🏭 Monitoring System for Die Casting Process

센서 기반 주조(Die Casting) 공정의 실시간 모니터링 및 품질 예측 시스템
공정 중 발생할 수 있는 이상 징후를 실시간으로 탐지하고,
불량률을 사전에 예측하는 스마트 팩토리 솔루션입니다.

---

## 📌 프로젝트 개요

다이캐스팅 제조 공정에서 수집되는 센서 데이터를 기반으로
이상 탐지 및 불량 예측 기능을 탑재한 **실시간 스트리밍 대시보드**를 구축했습니다.
Shiny for Python을 통해 직관적인 시각화와 사용자 인터페이스를 제공합니다.

---

## 🚀 주요 기능

* 📈 **실시간 공정 데이터 스트리밍 및 시각화**
* ⚠️ **이상 탐지 (Z-Score, Isolation Forest, PCA 등)**
* 🧠 **불량 예측 모델 (RandomForest, XGBoost 등)**
* 🧾 **SHAP 기반 변수 중요도 분석 및 해석 지원**
* 🖥️ **Shiny 대시보드 3개 탭 구성 (모니터링 / 이상 / 품질)**

---

## 🧰 기술 스택

| 분야      | 사용 기술                                          |
| ------- | ---------------------------------------------- |
| 언어      | Python                                         |
| 프레임워크   | [Shiny for Python](https://shiny.posit.co/py/) |
| 데이터베이스  | SQLite                                         |
| 시각화     | Plotly, Matplotlib                             |
| 머신러닝    | Scikit-learn, SHAP                             |
| 백엔드 API | FastAPI                                        |

---

## 📂 프로젝트 구조

```
📁 Monitoring-System-for-Die-Casting-Process/
├── app.py                 # Shiny 메인 앱
├── streaming.py           # SQLite 기반 실시간 스트리밍 처리
├── shared.py              # 공용 함수/데이터
├── model/                 # 모델 파일 저장소
├── data/                  # 센서 데이터 및 DB
├── requirements.txt       # 패키지 의존성 목록
└── README.md              # 프로젝트 설명 문서
```

---

## 🧪 실행 방법

```bash
# 패키지 설치
pip install -r requirements.txt

# 앱 실행
shiny run --reload app.py
```

※ SQLite 기반 센서 스트리밍 기능이 탑재되어 있으므로 `sensor_stream` DB를 미리 생성해야 합니다.

---

## 📊 대시보드 구성

* **모니터링 탭**: 실시간 그래프, 로그, 이상 알림
* **이상 예측 탭**: 이상 탐지 결과, 발생 히트맵, SHAP 해석
* **품질 탭**: 불량률 분석, 품질 예측 결과, 변수 영향도

---

## 🌱 향후 계획

* MQTT 기반 센서 연동
* 이상 탐지 자동 알림 (예: Slack 연동)
* 사용자 정의 경고 임계값 설정 기능

---

## 🙋‍♂️ 제작자

* **이름**: 신태선, 박수현, 한규민, 이주연
* **이메일**: [sts1373@naver.com](sts1373@naver.com)
* 기여 문의 및 협업 제안 환영!

---

