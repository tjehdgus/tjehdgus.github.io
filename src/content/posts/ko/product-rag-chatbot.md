---
title: "제조 기업 제품 문의 RAG 챗봇: 소스별 컬렉션 인덱싱, PII 마스킹 스트리밍, 청킹·enrichment 탐색"
description: "제품 카탈로그(DOCX)·스펙(JSON)·홈페이지 크롤링 데이터를 소스별 ChromaDB 컬렉션으로 색인하고, PII를 마스킹한 질문으로 LLM 답변을 WebSocket 스트리밍한 뒤 복원하는 제품 문의 챗봇의 구조와 검색 실험 설계를 정리했다."
date: 2026-10-16
lang: ko
key: product-rag-chatbot
tags: [RAG, ChromaDB, bge-m3, PII 마스킹, FastAPI, Contextual Retrieval]
tistory: https://mynews76154.tistory.com/13
---

## 초록

이전 회사(제조 기업)에서 고객 제품 문의에 1차 응대하는 RAG 챗봇을 만들었다. 제품 카탈로그(DOCX), 제품 수치 스펙(JSON), 홈페이지 크롤링 데이터를 섹션 단위로 파싱해 bge-m3로 임베딩하고 ChromaDB에 색인했다. FastAPI WebSocket으로 답변을 스트리밍하며, 질문 속 개인정보는 Presidio로 자리표시자로 바꿔 LLM에 보내고 스트림에서 복원한다. 대화 이력은 Redis → DB → 메모리 순으로 내려가는 세션 매니저가 관리하고, 근거가 없거나 견적·기술 자료 요청이면 상담 신청 리드로 넘긴다. 검색 설정 선택을 위해 임베딩 모델 3종 × 청킹 122종 × enrichment 3종을 Hit@5, MRR@5로 전수 탐색하는 실험 파이프라인을 만들었다. 보고된 값으로 문의 1차 응대 시간이 평균 15분에서 10초 이내로 줄었다.

회사명, 제품명, 데이터 샘플, 경로는 일반화했고 코드 인용의 파일명과 식별 정보도 지웠다.

## 1. 문제 정의

**과제.** 고객이 웹 채팅으로 제품 용도, 적합 제품, 물성 수치, 사업장·대리점 연락처를 물으면 사내 문서를 근거로 답한다. 담당자가 처리해야 하는 요청(단가, 안전·기술 자료, 자사와 무관한 문의)은 상담 신청으로 연결한다.

| 항목 | 내용 |
|---|---|
| 입력 | 고객 질문(한국어/영어, 최대 2,000자), 세션 내 이전 대화 |
| 출력 | 스트리밍 답변, 근거 문서 출처 목록, 필요 시 상담 신청 폼 |
| 부가 출력 | 상담 신청 리드, 대화 요약과 문의 유형(5분류) |

**평가 지표.** 테스트 질문마다 정답 제품명을 지정하고 상위 k개 검색 결과로 측정했다. 생성 품질용으로 RAGAS 4종(context precision/recall, faithfulness, answer relevancy)을 선택 실행할 수 있게 했다.

```text
hit(q) = 1  if 상위 k개 중 정답 제품명을 포함한 문서가 있으면, else 0
Hit@k  = sum(hit(q)) / N
MRR@k  = sum(1 / rank(q)) / N     (미적중이면 0)
k = 5
```

**제약 조건.** 외부 LLM API로 고객 개인정보 원문을 보내지 않는다. 문서에 없는 정보(가짜 URL, 이메일, 전화번호 포함)는 만들지 않는다. 데이터 소스마다 갱신 주기가 달라 한 소스만 바뀌면 그 소스만 다시 색인할 수 있어야 한다.

## 2. 데이터

세 소스를 공통 데이터 클래스 `ParsedSection(source, category, product_name, sub_section, content, section_type, extra_metadata)`로 정규화했다. 아래 수치는 레포에 포함된 카탈로그·스펙 파일에 로더를 실행해 확인했다. 회사 정보 JSON은 레포에서 제외되어 있어 확인하지 못했다.

| 소스 | 형식 | 파싱 단위 | 평균 길이(자) | 범위(자) |
|---|---|---|---|---|
| 제품 카탈로그 | DOCX 서술형 | Heading 1/2/3 | 329 | 85 ~ 981 |
| 제품 수치 스펙 | JSON (표 크롤링) | 제품 1개 | 291 | 162 ~ 388 |
| 회사·사업장·대리점 | JSON (크롤링) | 인사말, 조직, 연혁 시기, 사업장, 대리점 지역 | 미확인 | 미확인 |

카탈로그 섹션은 제품 설명, 트러블 대응, FAQ, 카테고리 개요로 구성되며 대부분이 제품 설명이다. 섹션 수는 회사 규모를 드러내므로 싣지 않는다.

- **크롤링.** Selenium과 BeautifulSoup으로 표를 읽고, `rowspan`/`colspan` 병합 헤더를 그리드로 펼쳐 `"물성 > pH"` 같은 경로형 열 이름을 만들었다.
- **스펙 텍스트화.** 제품 한 개를 `[카테고리] 제품명 - 제품 스펙` 헤더와 `• 속성: 값` 목록으로 바꿨다. 경로형 키는 마지막 요소만 남기고 빈 값과 `-`는 버렸다.
- **DOCX 파싱.** 단락 스타일로 계층을 추적해 Heading 2(제품) 단위로 잘랐다. Heading 3 제목으로 트러블 대응과 FAQ를 구분하고, 구분선 단락은 건너뛰었다.

## 3. 방법

### 3.1 전체 파이프라인

```text
[인덱싱]
홈페이지 크롤링 / 카탈로그 DOCX / 스펙 JSON
  → 섹션 파싱(ParsedSection)
  → 청킹 (by_product | recursive_N)
  → enrichment (none | enriched | contextual)
  → bge-m3 임베딩 → ChromaDB (컬렉션: docx / product_spec / company)

[응답, WebSocket 1턴]
질문 → 길이·rate limit 검사 → 세션 이력 추가
  ├─ 검색(top-k) → 출처 목록 전송
  └─ PII 마스킹 → 프롬프트(규칙 + 이전 대화 + 문서 + 질문)
     → LLM 스트리밍 → PII 복원(버퍼링) → 토큰 전송 → 이력·DB 저장

[후속]
상담 신청 → 리드 생성 + 대화 연결 → 관리자 화면 요약·유형 분류
```

- 섹션 파싱: 형식이 다른 소스를 같은 스키마로 맞춘다.
- 청킹·enrichment: 섹션을 나누고 앞에 메타데이터나 맥락 문장을 붙인다.
- 응답: 검색 결과와 최근 대화를 규칙형 프롬프트에 넣어 스트리밍한다.
- 후속: 답하지 않은 요청을 리드로 담당자에게 넘긴다.

### 3.2 소스별 컬렉션과 검색 병합

한 영속 디렉터리 안에 소스별 컬렉션 3개를 두었다. 스펙만 바뀌면 `product_spec` 컬렉션만 삭제 후 재생성하고, 운영 스크립트는 `embed_single.py product_spec | docx | all`로 소스 단위 재색인을 지원한다. 검색은 컬렉션마다 `similarity_search_with_score(query, k)`로 k개씩 뽑고 거리 기준으로 합쳐 상위 k개를 쓴다. 세 컬렉션이 같은 모델과 정규화 설정을 써서 거리를 그대로 비교할 수 있다.

### 3.3 청킹과 enrichment

| 전략 | 설명 |
|---|---|
| `by_product` | 섹션 1개 = 문서 1개 |
| `recursive_N` | RecursiveCharacterTextSplitter, N = 300 ~ 1,500(10 간격, 121종), overlap = max(10, 0.1N). N 이하 섹션은 분할하지 않음 |

| enrichment | 청크 앞에 붙는 내용 | 비용 |
|---|---|---|
| `none` | `[카테고리] 제품명` 수준 헤더 | 없음 |
| `enriched` | `[데이터출처]`, `[카테고리]`, `[제품명]`, `[섹션]`, `[문서유형: 설명]` 태그 | 없음 |
| `contextual` | enriched 태그 + LLM이 쓴 2~3문장 `[맥락]` | 청크당 LLM 1회 |

`contextual`은 Anthropic의 Contextual Retrieval을 따랐다. 같은 소스·카테고리·제품의 섹션을 모아(섹션당 500자, 전체 2,000자 상한) 문서 맥락으로 주고, 청크의 내용과 문서 내 역할, 검색 키워드를 쓰게 했다. gpt-4o-mini(temperature 0, max_tokens 200)를 5개 스레드로 병렬 호출하고, `md5(문서 맥락 + 청크)` 키로 파일 캐시해 같은 청크는 다시 호출하지 않는다. 실패한 청크는 원문을 유지한다.

### 3.4 PII 마스킹과 스트리밍 복원

Presidio `AnalyzerEngine`에 한국어 패턴 인식기를 추가했다. 전화번호(구분자 있는 형식 score 0.9, 10~11자리 숫자 0.7), 이메일(0.9), 이름(한글 2~4자 + 뒤따르는 호칭·직함 lookahead, 0.7)이다. 탐지 구간을 `<ENTITY_i>`로 바꾸고 복원 매핑을 반환하며, 오프셋이 어긋나지 않도록 시작 위치 역순으로 치환했다.

`fordocker/pii_handler.py`
```python
    results = detect_pii(text)
    
    if not results:
        return text, {}
    
    # 복원용 매핑 생성
    mapping = {}
    masked_text = text
    
    # 위치 역순으로 정렬 (뒤에서부터 치환해야 위치 안 꼬임)
    sorted_results = sorted(results, key=lambda x: x.start, reverse=True)
    
    for i, result in enumerate(sorted_results):
        original = text[result.start:result.end]
        placeholder = f"<{result.entity_type}_{i}>"
        
        mapping[placeholder] = original
        masked_text = masked_text[:result.start] + placeholder + masked_text[result.end:]
    
    return masked_text, mapping
```

LLM은 답변에 자리표시자를 그대로 쓰고 서버가 원래 값으로 되돌린다. 스트리밍에서는 `<PHONE_`까지만 도착한 토큰을 내보내면 복원할 수 없으므로, 버퍼 끝에 닫히지 않은 태그가 있으면 계속 모으고 아니면 버퍼를 복원해 내보냈다. 매핑이 비어 있으면 버퍼링하지 않는다.

`fordocker/server.py`
```python
    pii_pattern = re.compile(r'<[A-Z_]+_\d+>')
    partial_pattern = re.compile(r'<[A-Z_]*$')  # 불완전한 태그 감지
    
    buffer = ""
    
    for chunk in llm.stream(prompt):
        buffer += chunk
        
        # 불완전한 PII 태그가 버퍼 끝에 있는지 확인
        if partial_pattern.search(buffer):
            # 버퍼에 불완전한 태그가 있으면 계속 축적
            continue
        
        # 완전한 태그들을 복원하고 출력
        restored = unmask_pii(buffer, pii_mapping)
        yield restored
        buffer = ""
```

검색에는 원문 질문을 썼다. 임베딩 모델이 서버 안에서 로컬로 돌아 외부로 나가지 않기 때문이다.

### 3.5 프롬프트 규칙

한국어·영어 프롬프트를 따로 두고 세션 언어 설정에 따라 고른다.

| 규칙 | 동작 |
|---|---|
| 근거 우선 | 문서에 있으면 답하고, 없으면 모른다고 말한 뒤 상담 신청 안내 |
| 유형 되묻기 | 사용 방식에 따라 두 유형으로 갈리는 제품군은 유형이 없으면 먼저 묻고, 모르면 차이를 한 줄로 설명 |
| 넘길 요청 | 단가, 안전·기술 자료, 무관한 문의는 상담 신청으로 연결 |
| 타사 브랜드 | 자사 제품명이나 "대체·호환" 표현 없이 사용 환경을 묻는다. 일반 기업명 오인 방지용 예외 예시 포함 |

출력 형식으로는 마크다운 금지(UI가 렌더링하지 않음), 제품명 대괄호, 속성별 줄바꿈, 약어 풀어 쓰기를 지정했다.

### 3.6 멀티턴 세션 매니저

`BaseSessionManager` 아래 Redis(키당 JSON 리스트, `setex`로 TTL 24시간 갱신), DB(대화 테이블 조회), 메모리(dict, 세션 생성·집계 시 만료 정리) 구현을 두었다. `SESSION_TYPE=auto`면 Redis `ping`을 먼저 시도하고 실패하면 DB, 그다음 메모리로 내려간다. 모두 최근 10쌍(메시지 20개)만 남겨 `고객: ... / 상담봇: ...` 텍스트로 `[이전 대화]`에 넣는다.

### 3.7 서빙과 리드 저장

- **LLM.** `BaseLLM`(invoke, stream) 인터페이스 뒤에 Azure OpenAI gpt-4o-mini(temperature 0.1, max_tokens 1024)를 두어 로컬 모델로 교체할 수 있게 했다.
- **WebSocket.** 서버는 `session` → `sources` → `start` → `chunk`* → `end`를 보내고, 클라이언트는 질문 외에 `set_language`, `clear`, `submit_lead` 액션을 보낸다.
- **보호 장치.** 질문 2,000자 제한, 세션별 분당 30회 rate limit. 운영 모드에서 관리자 비밀번호와 CORS origin이 없으면 기동을 막는다.
- **리드.** 폼 값을 리드로 저장하고 세션 대화를 연결한다. 상태는 `new → contacted → converted → closed` 순서로만 바뀐다. 관리자 화면에서는 LLM이 대화를 2~3문장으로 요약하고 문의 유형을 A~E 중 하나로 분류한다.

## 4. 실험

### 4.1 실험 설정

| 항목 | 설정 |
|---|---|
| 임베딩 모델 | bge-m3(1024d), multilingual-e5-large(1024d), ko-sroberta-multitask(768d) |
| 청킹 | by_product + recursive 121종 = 122종 |
| enrichment | none, enriched(기본), contextual(명시 지정 시) |
| 조합 수 | 기본 3 × 122 × 2 = 732, contextual 포함 1,098 |
| 임베딩 실행 | CPU, 정규화, batch 32 |
| 평가셋 | 20문항(질문, 정답 문장, 정답 제품명): 제품 설명·추천 11, 물성 수치 5, 회사·대리점 4 |

### 4.2 결과

조합별 지표 파일과 인덱스는 레포에서 제외되어 있어 Hit@5, MRR@5 값은 싣지 않는다. 아래는 레포의 청킹 코드를 카탈로그·스펙 전체 섹션에 실행한 청크 통계다. 절대 청크 수 대신 by_product 대비 배율로 표기한다.

| 청킹 | none 청크 수(배율) | none 평균(자) | enriched 청크 수(배율) | enriched 평균(자) |
|---|---|---|---|---|
| by_product | ×1.00 | 329 | ×1.00 | 419 |
| recursive_300 | ×2.18 | 153 | ×2.46 | 172 |
| recursive_500 | ×1.02 | 322 | ×1.08 | 387 |
| recursive_800 | ×1.01 | 326 | ×1.01 | 414 |
| recursive_1000 | ×1.00 | 328 | ×1.00 | 417 |
| recursive_1500 | ×1.00 | 329 | ×1.00 | 419 |

enriched 태그가 청크당 평균 약 90자를 더해 같은 N에서 더 많이 분할된다.

### 4.3 비교 실험

| 비교 축 | 확인한 내용 | 판정 |
|---|---|---|
| recursive N의 실효 범위 | 121개 N 중 서로 다른 청크 집합은 none 44, enriched 47. N ≥ 1,010(none), N ≥ 1,090(enriched)부터 by_product와 동일 | 상한 쪽 35~41% 조합이 중복 |
| 평가셋 해상도 | 20문항, 한 문항 = Hit@5 5%p | 122종 중 최고점 선택은 노이즈와 구분 어려움 |
| 적중 판정 | 정답 제품명이 본문에 부분 문자열로만 있어도 적중 | 여러 제품명이 함께 나오는 청크가 과대 적중 |
| RAGAS 생성 지표 | `answer` 자리에 챗봇 답변 대신 정답 문장 입력 | faithfulness, answer relevancy 무의미 |
| 실험과 서빙의 k | 실험 top-5, 서빙 k=3 | 실험 점수가 서빙 품질을 대변하지 않음 |
| contextual 비용 | 청크 수만큼 LLM 호출, 해시 캐시로 재실행 비용 제거 | 기본 실행에서 제외 |

카탈로그 섹션 중 400자를 넘는 것은 약 11%이고 스펙은 모두 400자 이하다. 청크 크기는 300~1,000 구간의 몇 개 값만 두고 평가 문항을 늘리는 편이 낫다. 회사 정보 소스는 이 계산에 포함하지 못했다.

## 5. 한계와 향후 과제

1. **대화 이력의 PII.** 질문은 마스킹해 보내지만 이력에는 원문을 저장하고 그 이력이 프롬프트에 들어간다. WebSocket 경로는 생성 전에 사용자 메시지를 이력에 추가하므로 같은 턴에서도 원문이 전달된다. 관리자 요약도 원문 대화를 LLM에 보낸다. 이력과 요약 입력을 마스킹 상태로 저장해야 한다.
2. **스트리밍 복원 정규식.** `<[A-Z_]*$`는 `<PHONE_`처럼 이름 부분에서 끊긴 경우만 잡는다. 청크 경계가 `<PHONE_NUMBER_0`과 `>` 사이에 오면 미완성 태그가 그대로 출력된다. `<[A-Z_]*\d*$`로 숫자 꼬리까지 포함해야 한다.
3. **실험과 서빙 인덱스 불일치.** 서빙 코드는 1차 크롤링 파이프라인의 단일 컬렉션을 k=3으로 읽고, 소스별 컬렉션·enrichment·거리 병합은 실험 파이프라인에만 있다. 같은 빌드 코드로 연결해야 한다.
4. **평가셋.** 실제 상담 로그에서 질문을 뽑아 정답 청크 ID를 지정하고 100문항 이상으로 늘려야 한다. 템플릿 생성 1,000문항은 표현이 단조로워 보조 용도에 그친다.
5. **프롬프트 길이와 중복 검색.** 형식 규칙까지 프롬프트에 넣어 규칙 충돌 여지가 있으므로 마크다운 제거, 약어 풀기는 후처리로 옮길 수 있다. WebSocket 한 턴에서 출처용과 생성용으로 검색을 두 번 하므로 결과를 재사용하면 임베딩 호출이 줄어든다.

## 참고

- 티스토리 개발 기록: https://mynews76154.tistory.com/13
- 임베딩: BAAI/bge-m3, intfloat/multilingual-e5-large, jhgan/ko-sroberta-multitask
- LLM: Azure OpenAI gpt-4o-mini
- 프레임워크: ChromaDB, LangChain, FastAPI, Redis
- PII: Microsoft Presidio
- 평가: RAGAS
- 기법: Anthropic, "Contextual Retrieval"
