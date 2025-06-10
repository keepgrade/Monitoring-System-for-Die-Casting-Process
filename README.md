
# 🏭 Monitoring System for Die Casting Process

센서 기반 주조(Die Casting) 공정의 **실시간 모니터링 및 품질 예측 시스템**입니다.
공정 중 발생 가능한 이상 징후를 탐지하고, **불량을 사전에 예측**함으로써 **자원 낭비를 줄이고 생산 효율을 극대화**하는 스마트 팩토리 솔루션을 구현하였습니다.

---

## 📌 프로젝트 개요

본 프로젝트는 **다이캐스팅 제조 공정의 센서 데이터를 활용하여**,

* 실시간 공정 상태 스트리밍
* 이상 탐지 및 원인 해석
* 불량률 예측 및 주요 변수 분석
  기능을 통합한 **실시간 대시보드 시스템**입니다.

사용자는 **생산 라인 책임자 및 관리자**로, 공정 효율성 제고 및 불량률 저감이라는 목적을 달성할 수 있습니다.

---

## 🚀 주요 기능

| 기능 영역       | 설명                                    |
| ----------- | ------------------------------------- |
| 📈 실시간 스트리밍 | 센서 데이터 기반 공정 상태 시각화                   |
| ⚠️ 이상 탐지    | Isolation Forest 기반 이상 판별, SHAP 해석 지원 |
| 🧠 불량 예측    | XGBoost 기반 이진 분류 모델, 주요 변수 시각화        |
| 🧾 설명 가능 AI | SHAP, Feature Importance 등 해석 가능성 확보  |
| 🖥️ 대시보드 구성 | ① 실시간 모니터링 ② 이상 탐지 ③ 품질 예측 탭으로 구성     |

---

## 🧰 기술 스택

| 분야    | 사용 기술                                                        |
| ----- | ------------------------------------------------------------ |
| 언어    | Python                                                       |
| 프레임워크 | [Shiny for Python](https://shiny.posit.co/py/)               |
| 시각화   | Plotly, Matplotlib                                           |
| 머신러닝  | Scikit-learn, XGBoost, SHAP                                  |
| 배포    | [shinyapps.io](https://www.shinyapps.io) (Shiny 공식 클라우드 서비스) |

---

## 📂 프로젝트 구조

```plaintext
Monitoring-System-for-Die-Casting-Process/
├── app.py                 # Shiny 메인 앱
├── shared.py              # 공용 함수 및 데이터 관리
├── model/                 # 학습된 모델 파일 저장소
├── data/                  # 센서 데이터 및 전처리 결과
├── requirements.txt       # Python 의존 패키지 목록
└── README.md              # 프로젝트 설명 문서
```

---

## 🧪 실행 방법

```bash
# 패키지 설치
pip install -r requirements.txt

# 앱 실행 (Shiny for Python)
shiny run --reload app.py
```

> 💡 *실행 전, `data/` 폴더 내 센서 CSV 파일이 존재해야 정상 실행됩니다.*

---

## 📊 대시보드 구성

### ① 모니터링 탭

* 실시간 센서 값 스트리밍 및 그래프 표시
* 실시간 로그 확인, 이상 여부 알림 제공

### ② 이상 탐지 탭

* 공정별 Isolation Forest 적용 결과 표시
* SHAP 기반 이상 원인 Top 3 변수 시각화
* 시간/공정별 이상 발생 히트맵

### ③ 품질 예측 탭

* XGBoost 기반 불량 예측 결과
* Confusion Matrix, F1 Score 등 성능지표
* Feature Importance 시각화로 주요 원인 분석

> 📂 부록 안내
대시보드 구성에 사용된 전처리 로직, 센서 데이터 구조 정의, 파생 변수 생성 과정 등은 부록에 상세히 정리되어 있습니다.
이는 프로젝트 이해도 향상과 향후 유지보수/확장성 확보를 위한 기술 명세 문서로 활용 가능합니다.
---

## 🔧 데이터 전처리 및 이슈 해결

* 결측 및 이상값 보정: `molten_temp`, `molten_volume`, `cast_pressure`, `EMS_operation_time` 등
* 불필요 칼럼 제거 및 오기입 처리
* 불균형 데이터 대응: `scale_pos_weight` 조정
* 전처리 기준: 전문가 판단 및 통계 기반 이상 탐지

---

## 🤖 모델 구성 및 해석

### 📍 이상 탐지 모델: **Isolation Forest**

* 공정별 학습 → 실시간 이상 판별
* SHAP 기반 주요 변수 영향도 해석

### 📍 불량 예측 모델: **XGBoost**

* Gradient Boosting 기반 이진 분류
* 불균형 데이터 대응 및 GridSearch 기반 하이퍼파라미터 튜닝

### 📍 해석 도구

* SHAP 값 시각화 (이상 탐지)
* Feature Importance (불량 예측)
* Confusion Matrix 등 성능 평가 지표

---

## 🔍 성능 평가 및 검증

* Confusion Matrix, Precision, Recall, F1-score 기반 모델 평가
* 이상 탐지에 대해서는 SHAP 기반 전문가 해석 병행
* 실제 불량률과의 비교를 통한 정량 검증 가능

---

## ⚠️ 한계 및 개선 방향

* 불량/이상 라벨 부족으로 인한 한계 존재
* 도메인 해석이 어려운 변수 일부 존재
* 추후 개선 방향:

  * MQTT 기반 실시간 센서 연동
  * 이상 탐지 알림 기능 (Slack 등)
  * 사용자 맞춤 경고 임계값 설정 기능
  * 다양한 탐지 및 분류 모델 실험 지속

---

## 🙋‍♂️ 제작자 및 문의

* **제작자**: 신태선, 박수현, 한규민, 이주연
* **문의**: [sts1373@naver.com](mailto:sts1373@naver.com)
* **라이선스**: [MIT License](./LICENSE)

> 💡 프로젝트 또는 협업 제안이 있다면 언제든지 연락 주세요!

