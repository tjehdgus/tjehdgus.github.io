---
title: "원가율 예측 시뮬레이터: 사용한 기술 스택과 구성"
description: "이전 회사에서 만든 견적 전 원가율 시뮬레이터의 기술 구성을 정리했다. 회사 데이터와 내부 수치는 다루지 않는다."
date: 2026-04-07
lang: ko
key: cost-simulator
tags: [RandomForest, 회귀, 피처 엔지니어링, FastAPI, Azure, LLM]
tistory: https://mynews76154.tistory.com/6
---

## 개요

이전 회사에서 영업 담당자가 견적을 내기 전에 원가율과 적자 여부를 확인하는 웹 시뮬레이터를 만들었다. 이 글은 어떤 기술을 어디에 썼는지만 정리한다. 데이터, 모델 성능, 업무 기준값, 내부 시스템 구성은 다루지 않는다.

## 기술 스택

| 영역 | 기술 |
|---|---|
| 모델 | scikit-learn RandomForest 회귀 |
| 데이터 처리 | pandas, NumPy |
| 백엔드 | Python, FastAPI |
| 프론트엔드 | 단일 HTML 페이지 (JavaScript) |
| LLM | Azure OpenAI (결과 해설 생성) |
| 인증·저장소 | bcrypt 비밀번호 해시, Azure Cosmos DB |
| 배포 | Docker, Azure Container Apps, Azure File 볼륨 |

## 구성

```text
판매 실적 데이터
  → 전처리 / 파생 컬럼
  → 그룹 통계 · 시계열 통계 테이블 (서버 시작 시 1회)

예측 요청
  → 피처 조립
  → RandomForest 회귀
  → 최근 실적 기반 보정
  → 원가율 계산 / 시뮬레이션
  → LLM 해설
```

## 구성 요소별 기술

- **문제 정의.** 사용자가 단가를 바꿔 보는 시뮬레이터라서 단가를 모델 입력에서 뺐다. 모델은 단가와 독립적인 값을 회귀하고, 원가율은 입력 단가로 나눠 계산한다. 단가 변경 시뮬레이션은 모델을 다시 호출하지 않는다.
- **피처 엔지니어링.** 범주형 라벨 인코딩, 로그 변환, 주기형 시간 피처(sin/cos), 그룹별 통계, lag, 이동평균·이동표준편차를 조합했다. 결측은 상위 그룹 통계로 계층적으로 채웠다.
- **누수 방지.** 그룹 평균 피처를 만들 때 예측 대상 거래가 통계에 섞이지 않도록 처리했다.
- **앙상블 보정.** 모델 예측과 최근 실적의 가중 이동평균을 기하평균으로 결합했다.
- **콜드 스타트.** 이력이 없는 신규 입력은 피처를 줄인 별도 모델로 처리했다.
- **LLM 해설.** 예측 결과와 과거 이력 요약을 프롬프트에 넣어 해설을 생성했다. 판단은 사용자에게 남기도록 지시형 표현을 막았다.
- **서빙.** FastAPI로 예측·시뮬레이션 API를 만들고 Docker 이미지로 Azure Container Apps에 배포했다. 모델과 데이터는 볼륨으로 마운트해 이미지 재빌드 없이 교체한다.
- **사용자 관리.** bcrypt 해시, 비밀번호 정책, 로그인 시도 제한, 역할 기반 권한, 감사 로그를 붙였다.

## 참고

- 티스토리 개발 기록: https://mynews76154.tistory.com/6
- Random Forests (Breiman, 2001)
- FastAPI, Azure Container Apps, Azure Cosmos DB, Azure OpenAI
