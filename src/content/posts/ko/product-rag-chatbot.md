---
title: "제품 문의 RAG 챗봇: 사용한 기술 스택과 구성"
description: "이전 회사에서 만든 고객 제품 문의 RAG 챗봇의 기술 구성을 정리했다. 회사 데이터와 내부 수치는 다루지 않는다."
date: 2026-10-16
lang: ko
key: product-rag-chatbot
tags: [RAG, ChromaDB, bge-m3, PII 마스킹, FastAPI, WebSocket]
tistory: https://mynews76154.tistory.com/13
---

## 개요

이전 회사에서 고객의 제품 문의에 사내 문서를 근거로 1차 응대하고, 사람이 처리할 요청은 상담 신청으로 넘기는 RAG 챗봇을 만들었다. 이 글은 어떤 기술을 어디에 썼는지만 정리한다. 데이터, 평가 결과, 프롬프트 내용, 내부 시스템 구성은 다루지 않는다.

## 기술 스택

| 영역 | 기술 |
|---|---|
| 수집·파싱 | Selenium, BeautifulSoup, python-docx |
| 임베딩 | BAAI/bge-m3 (비교: multilingual-e5-large, ko-sroberta-multitask) |
| 벡터 DB | ChromaDB |
| 오케스트레이션 | LangChain |
| LLM | Azure OpenAI (교체 가능한 LLM 인터페이스) |
| 개인정보 보호 | Microsoft Presidio + 한국어 패턴 인식기 |
| 백엔드 | Python, FastAPI, WebSocket |
| 세션 | Redis / DB / 메모리 폴백 |
| 평가 | Hit@k, MRR@k, RAGAS |

## 구성

```text
[인덱싱]
웹 크롤링 / DOCX / JSON
  → 공통 섹션 스키마로 파싱
  → 청킹 → 메타데이터·맥락 enrichment
  → 임베딩 → 소스별 벡터 컬렉션

[응답]
질문 → 입력 검사 → 세션 이력
  ├─ 벡터 검색 → 출처 전송
  └─ PII 마스킹 → 프롬프트 → LLM 스트리밍 → PII 복원 → 전송

[후속]
상담 신청 → 리드 저장 → 관리자 화면 요약·분류
```

## 구성 요소별 기술

- **데이터 정규화.** 형식이 다른 소스(웹 표, DOCX, JSON)를 하나의 섹션 스키마로 맞췄다. 웹 표는 병합 셀을 그리드로 펼쳐 경로형 열 이름을 만들었다.
- **소스별 컬렉션.** 소스마다 벡터 컬렉션을 분리해 한 소스만 다시 색인할 수 있게 하고, 검색 결과는 거리 기준으로 합쳤다.
- **청킹·enrichment.** 섹션 단위와 재귀 분할을 비교하고, 청크 앞에 메타데이터 태그나 LLM이 쓴 맥락 문장(Contextual Retrieval)을 붙이는 방식을 실험했다. 맥락 생성은 해시 캐시로 재호출을 막았다.
- **검색 실험 파이프라인.** 임베딩 모델 × 청킹 × enrichment 조합을 자동으로 돌려 Hit@k, MRR@k를 비교하는 실험 코드를 만들었다.
- **PII 마스킹.** Presidio로 개인정보를 자리표시자로 바꿔 LLM에 보내고, 스트리밍 응답에서 태그가 잘리지 않도록 버퍼링하며 복원했다.
- **멀티턴.** Redis → DB → 메모리 순으로 폴백하는 세션 매니저로 최근 대화를 프롬프트에 넣었다.
- **스트리밍 서빙.** FastAPI WebSocket으로 출처, 토큰, 종료 이벤트를 순서대로 보낸다. 입력 길이 제한과 rate limit을 두었다.
- **상담 연결.** 근거가 없거나 담당자가 처리할 요청은 상담 신청 리드로 저장하고, 관리자 화면에서 LLM으로 대화를 요약·분류했다.

## 참고

- 티스토리 개발 기록: https://mynews76154.tistory.com/13
- BAAI/bge-m3, intfloat/multilingual-e5-large, jhgan/ko-sroberta-multitask
- ChromaDB, LangChain, FastAPI, Redis, Microsoft Presidio, RAGAS
- Anthropic, "Contextual Retrieval"
