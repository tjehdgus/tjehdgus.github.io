---
title: "Postgres 하나로 다 할 수 있을까? 큐, 벡터 검색, 분석, 객체 스토리지로 나눠 본 경계선"
description: "\"그냥 Postgres 쓰면 된다\"는 말을 메시지 큐, 벡터 검색, 분석(OLAP), 객체 스토리지 기반 DB 네 영역으로 나눠 벤치마크와 사례로 따져 봤다. 대체는 되지만, 어디까지 충분한지는 워크로드별 경계선이 있다."
date: 2026-10-16
lang: ko
key: pg-everything
kind: review
tags: [PostgreSQL, pgvector, Kafka, OLAP, DuckDB, 데이터엔지니어링]
tistory: https://mynews76154.tistory.com/28
---

<p>요즘 "그냥 Postgres 쓰면 된다"는 말이 자주 보인다. 검색은 <code>tsvector</code>/<code>tsquery</code>로, 문서 저장은 <code>jsonb</code>로, 캐시는 <code>UNLOGGED TABLE</code>로, 벡터 검색은 <code>pgvector</code>로, 분석은 <code>pg_lake</code>, <code>pg_duckdb</code>, <code>pg_mooncake</code> 같은 확장으로 대신한다는 식이다. 최근에는 Kafka 같은 메시지 시스템까지 Postgres로 대신하는 벤치마크가 화제가 됐다.</p>

<p>RAG 서비스를 만들다 보면 이 질문을 자주 만난다. 벡터 DB를 따로 띄울지, 이미 쓰는 Postgres에 pgvector를 붙일지. 그래서 이번에 영역을 넷으로 나눠 자료를 찾아봤다. 메시지 큐, 벡터 검색, 분석(OLAP), 그리고 아예 반대 방향인 객체 스토리지 기반 DB다. 기준은 하나다. 대체는 할 수 있어도 성능과 운영이 맞아야 한다.</p>

<p><img src="/img/reviews/pg-everything-1-ko.png" alt="운영할 시스템을 줄이는 두 가지 방향" /></p>

<h2>1. 메시지 큐: 대부분의 서비스에는 충분하다</h2>

<h3>벤치마크가 보여준 것</h3>

<p>가장 많이 인용되는 자료는 Kafka 쪽에 오래 몸담은 Stanislav Kozlovski가 쓴 "Kafka is fast -- I'll use Postgres"(2025년 10월)다. Postgres 기본 기능만으로 두 가지를 만들어 쟀다.</p>

<ul>
<li><strong>Kafka 방식 Pub/Sub</strong>: 파티션마다 테이블을 두고, 카운터 테이블로 오프셋을 원자적으로 매기고, 컨슈머 그룹별 읽은 위치를 따로 기록한다.</li>
<li><strong>작업 큐</strong>: <code>SELECT ... FOR UPDATE SKIP LOCKED</code>로 작업을 하나씩 가져가고, 끝나면 아카이브 테이블로 옮긴다.</li>
</ul>

<p>메시지 1KiB, fsync를 켜서 내구성을 유지한 조건에서 결과는 이렇다.</p>

<table>
<thead><tr><th>구성</th><th>쓰기</th><th>읽기(5개 컨슈머 그룹)</th><th>p99 지연</th></tr></thead>
<tbody>
<tr><td>4 vCPU 1대</td><td>약 5,000 msg/s</td><td>약 25,000 msg/s</td><td>60ms</td></tr>
<tr><td>4 vCPU 3대 복제</td><td>약 5,000 msg/s</td><td>약 25,000 msg/s</td><td>186ms</td></tr>
<tr><td>96 vCPU 1대</td><td>약 243,000 msg/s</td><td>약 120만 msg/s</td><td>853ms</td></tr>
</tbody>
</table>

<p>작업 큐는 4 vCPU 1대에서 초당 약 2,900건, p99 17.7ms였다. 결론은 "한계에 부딪히기 전까지는 Postgres로 버텨라"다. 다만 테스트가 2~15분으로 짧았고, 같은 일을 해도 전용 시스템보다 CPU를 훨씬 많이 쓰며, Schema Registry나 Kafka Connect 같은 생태계가 없다는 점은 저자도 인정했다. Kafka를 Rust로 다시 만들었다는 이야기로 퍼지기도 했는데, Rust는 Postgres나 S3를 저장소로 쓰면서 Kafka API를 제공하는 별도 오픈소스 프로젝트 Tansu를 각주에서 소개할 때만 나온다.</p>

<h3>오래 돌리면 생기는 문제: bloat</h3>

<p>짧은 벤치마크에서 안 보이는 문제가 있다. <code>SKIP LOCKED</code> 방식 큐는 처리한 행을 계속 <code>DELETE</code>하거나 <code>UPDATE</code>한다. Postgres는 행을 지워도 바로 공간을 반납하지 않고 dead tuple로 남겨 두기 때문에, 부하가 계속되면 VACUUM 부담과 인덱스 bloat가 쌓이고 성능이 서서히 떨어진다.</p>

<p>이 문제를 정면으로 다룬 프로젝트가 2026년 5월에 나온 PgQue다. Skype가 쓰던 PgQ 구조를 가져와서, 행 단위 삭제 대신 이벤트 테이블을 통째로 돌려 쓰는(rotation) 방식으로 bloat를 없앴다. C 확장 없이 SQL 파일 하나로 설치되어 RDS나 Supabase 같은 관리형 Postgres에서도 쓸 수 있다. 대신 배치 단위로 처리하기 때문에 전달 지연이 보통 1~2초다.</p>

<p>정리하면 초당 수천 건 수준의 작업 큐나 이벤트라면 Postgres로 충분하다. 다만 서비스 DB와 같은 인스턴스에 올리면 장애가 번질 수 있으니 분리하는 게 좋고, 오래 돌릴 거라면 bloat 대책이 있는 구조를 고르는 게 맞다. 밀리초 단위 지연과 대규모 팬아웃이 필요해지면 그때 전용 시스템을 볼 시점이다.</p>

<h2>2. 벡터 검색: 숫자보다 필터와 꼬리 지연이 경계선</h2>

<h3>처리량만 보면 Postgres가 이기기도 한다</h3>

<p>"벡터 DB는 그래도 전용 엔진이 빠르지 않나"에 대해 자주 인용되는 숫자가 있다. Postgres 호스팅 회사 Tiger Data(구 Timescale)가 2025년 4월에 공개한 벤치마크로, 5,000만 개 벡터(768차원), 재현율 99% 조건이다.</p>

<table>
<thead><tr><th>지표</th><th>Postgres + pgvector + pgvectorscale</th><th>Qdrant</th></tr></thead>
<tbody>
<tr><td>처리량(QPS)</td><td>471</td><td>41</td></tr>
<tr><td>p50 지연</td><td>31ms</td><td>31ms</td></tr>
<tr><td>p99 지연</td><td>75ms</td><td>39ms</td></tr>
<tr><td>인덱스 빌드</td><td>약 11시간</td><td>약 3.3시간</td></tr>
</tbody>
</table>

<p>처리량은 Postgres가 11배 높지만, 꼬리 지연과 인덱스 빌드 시간은 Qdrant가 낫다. 그리고 조건을 봐야 한다. 기본 pgvector가 아니라 디스크 기반 인덱스(StreamingDiskANN)를 추가하는 pgvectorscale을 쓴 결과이고, 필터 없이 순수 벡터 검색만 쟀으며, Postgres를 파는 회사가 낸 자료다.</p>

<h3>실제 RAG에서 먼저 걸리는 건 필터</h3>

<p>실서비스 RAG는 거의 항상 필터가 붙는다. 사용자별, 문서 종류별, 기간별로 걸러서 검색한다. 근사 인덱스(HNSW)에서는 인덱스로 후보를 먼저 뽑고 필터를 나중에 적용한다. 기본 설정(<code>hnsw.ef_search = 40</code>)에서 전체의 10%만 남기는 필터를 걸면 후보 40개 중 평균 4개만 살아남는다. 상위 10개를 달라고 했는데 4개만 돌아오는 상황이 생긴다.</p>

<p><img src="/img/reviews/pg-everything-2-ko.png" alt="HNSW 인덱스에서 필터를 걸면 결과가 모자라는 이유" /></p>

<p>pgvector는 0.8.0부터 반복 인덱스 스캔(iterative index scan)으로 이 문제를 완화했다. 결과가 부족하면 인덱스를 더 읽어 채우는 방식이다. 그래도 필터가 많은 서비스라면 필터를 건 상태로 재현율과 지연을 따로 재 봐야 한다.</p>

<p>찾아본 자료들의 경험칙을 모으면 대략 이렇다.</p>

<ul>
<li>100만 개 미만: pgvector HNSW로 충분하다.</li>
<li>100만~1,000만 개: 튜닝한 pgvector로 대부분 된다.</li>
<li>1,000만~1억 개: 필터가 적고 처리량이 중요하면 pgvectorscale, 필터가 많거나 p99 40ms 미만이 필요하면 전용 DB를 본다.</li>
<li>1억 개 이상이거나 한 대에 안 들어가면: 샤딩이 되는 전용 DB가 맞다.</li>
</ul>

<h3>전용 벡터 DB도 "벡터 중심"을 버리고 있다</h3>

<p>흥미로운 건 전용 벡터 DB 쪽 움직임이다. 객체 스토리지 기반 검색 엔진 turbopuffer는 2026년 9월 "RIP, vector database"라는 글에서 저장 구조를 바꾼다고 밝혔다. 지금까지는 모든 데이터를 "어느 벡터 클러스터의 몇 번째인가"라는 주소를 키로 저장했다. 벡터 인덱스가 곧 기본(primary) 인덱스였다. 그런데 필터, 전문 검색, 집계 같은 기능을 얹을수록 문제가 커졌다.</p>

<ul>
<li>문서 하나에 벡터가 여러 개면 문서 내용이 벡터마다 중복 저장된다.</li>
<li>클러스터를 재조정할 때마다 문서 내용과 다른 인덱스까지 같이 옮겨야 한다.</li>
<li>최신 쿼리 엔진은 수천 행 단위로 묶어 처리하는데(DuckDB 2,048행, ClickHouse 약 65,000행), 벡터 클러스터는 100~200개 단위라 모든 쿼리가 이 크기에 묶인다. 전문 검색 인덱스를 클러스터에서 떼어 256개 고정 블록으로 다시 만들었더니 인덱스는 약 10배 작아지고 쿼리는 최대 약 20배 빨라졌다고 한다.</li>
</ul>

<p><img src="/img/reviews/pg-everything-3-ko.png" alt="turbopuffer v2와 v3의 저장 구조 비교" /></p>

<p>그래서 v3에서는 벡터 인덱스를 여러 보조 인덱스 중 하나로 내린다. 벡터 검색을 버린다는 뜻이 아니라, 일반 검색 엔진에 가까운 구조로 간다는 뜻이다. 아직 성능 벤치마크는 공개 전이다. Postgres는 벡터를 받아들이고, 벡터 DB는 일반 DB처럼 변하고 있다. 양쪽이 같은 지점으로 모이는 셈이다.</p>

<h2>3. 분석(OLAP): 사실상 Postgres 안에 다른 엔진을 넣는 것</h2>

<p>Postgres는 행 단위로 저장하는 OLTP DB라서 수억 행을 집계하는 분석 쿼리에는 약하다. 그래서 나온 확장들은 모두 Postgres 안이나 옆에 열 지향 엔진을 붙이는 방식이다. 흥미롭게도 세 확장 모두 큰 회사와 엮여 있다.</p>

<table>
<thead><tr><th>확장</th><th>누가</th><th>방식</th></tr></thead>
<tbody>
<tr><td>pg_duckdb</td><td>DuckDB Labs, Hydra, MotherDuck</td><td>Postgres 프로세스 안에 DuckDB를 넣어 분석 쿼리를 실행</td></tr>
<tr><td>pg_mooncake</td><td>Mooncake Labs (2025년 Databricks 인수)</td><td>Postgres 테이블의 열 지향 사본을 만들어 DuckDB로 조회</td></tr>
<tr><td>pg_lake</td><td>Crunchy Data (2025년 Snowflake 인수, 같은 해 11월 오픈소스)</td><td>Postgres에서 Iceberg 테이블을 트랜잭션으로 읽고 쓰기</td></tr>
</tbody>
</table>

<p>Snowflake와 Databricks가 Postgres 쪽 회사를 사들였다는 점이 흥미롭다. 운영 DB인 Postgres와 분석 플랫폼 사이의 경계를 자기 쪽으로 끌어오려는 움직임으로 보인다.</p>

<p>성능은 아직 전용 엔진과 차이가 있다. ClickHouse 기반 서비스를 하는 Tinybird의 비교 글에 따르면, 1억 행 규모 ClickBench 데이터에서 pg_mooncake가 쿼리 유형에 따라 ClickHouse보다 1.2~4배 느렸다(벤더 글이고 표의 출처가 명확하지 않아 참고만 했다). 여러 자료가 공통으로 제시하는 경계는 대략 이렇다.</p>

<ul>
<li>1TB 미만, 동시 분석 사용자 20명 미만: Postgres와 확장으로 충분하다.</li>
<li>1~5TB 또는 동시 사용자 20~50명: 단일 노드로 버티지만 두 번째 엔진을 고민할 시점이다.</li>
<li>단일 테이블 5TB 이상, 동시 사용자 50명 이상, 고빈도 스트리밍 적재: ClickHouse나 Snowflake 같은 분산 엔진이 맞다.</li>
</ul>

<p>결국 OLAP은 "Postgres로 대체"라기보다 "Postgres를 창구로 두고 분석 엔진을 안에 넣는다"에 가깝다. 실제로 대체가 되는지는 같은 데이터와 쿼리로 나란히 돌려 봐야 안다. 나도 기회가 되면 ClickHouse와 같이 놓고 재 보고 싶은 부분이다.</p>

<h2>4. 반대 방향: 객체 스토리지만으로 DB를 만든다</h2>

<p>한쪽에서 Postgres 하나에 기능을 몰아넣는 동안, 다른 쪽에서는 S3 같은 객체 스토리지를 원본 저장소로 두고 그 위에 엔진을 올리는 시도가 늘고 있다.</p>

<h3>왜 지금 가능해졌나</h3>

<p>계기 중 하나는 2024년 8월 S3에 추가된 조건부 쓰기다. "이 객체가 없을 때만 쓰기"가 가능해지면서, 001, 002처럼 순서 번호가 붙은 파일을 쓰는 것만으로 DB 로그처럼 동시 쓰기 충돌을 막을 수 있게 됐다. 별도 합의 알고리즘이나 메타데이터 DB 없이도 객체 스토리지 하나로 정합성을 다룰 수 있게 된 것이다. 객체 스토리지는 싸고, 가용 영역 간 복제가 기본이고, 사실상 무한히 늘어난다. 대신 지연이 크고 객체를 부분 수정할 수 없다.</p>

<h3>어떤 제품들이 있나</h3>

<ul>
<li><strong>WarpStream</strong>: S3 위에 만든 Kafka 호환 시스템. 2024년 Confluent가 인수했다.</li>
<li><strong>SlateDB</strong>: 객체 스토리지 위의 키-값 저장소(LSM 트리). 쓰기 지연은 크지만 내구성은 S3가 보장한다.</li>
<li><strong>turbopuffer</strong>: 앞에서 본 검색 엔진. 원본은 S3에 두고 NVMe SSD와 메모리를 캐시로 쓴다.</li>
<li><strong>duckdb_object_storage</strong>: 2026년 10월에 공개된 DuckDB 커뮤니티 확장. SlateDB 위에 DuckDB 데이터베이스를 그대로 올린다.</li>
</ul>

<p>마지막 확장이 특히 재미있다. <code>ATTACH 'duckdb_objfs://analytics.db'</code> 한 줄로 S3 버킷의 DB를 붙이면 기본 키, 외래 키, 인덱스, 트랜잭션이 일반 DuckDB처럼 동작한다. TPC-H 기준으로 메모리 백엔드에서는 네이티브 DuckDB보다 오히려 빨랐고(0.92배 시간), S3에서 처음 읽을 때는 기존 방식(HTTPFS)보다 약 20% 느렸다. 제약도 분명하다.</p>

<ul>
<li>쓰기는 한 프로세스만 할 수 있다.</li>
<li>커밋마다 S3 왕복이 생겨 수십 ms가 걸리니 작은 쓰기는 묶어야 한다.</li>
<li>읽는 쪽에는 최대 약 10초 늦게 반영된다.</li>
</ul>

<p>DuckLake나 Iceberg는 여러 엔진(Spark, Trino 등)이 같은 데이터를 공유하기 위한 포맷이고, 이 확장은 DuckDB 하나로 충분할 때 버킷 하나로 끝내기 위한 도구라서 용도가 다르다.</p>

<h2>정리: 워크로드별로 보면</h2>

<table>
<thead><tr><th>영역</th><th>Postgres로 충분한 구간</th><th>전용 시스템을 볼 신호</th></tr></thead>
<tbody>
<tr><td>메시지 큐</td><td>초당 수천 건, 지연 수십 ms~수 초 허용</td><td>밀리초 지연, 대규모 팬아웃, 커넥터 생태계 필요</td></tr>
<tr><td>벡터 검색</td><td>1,000만 개 이하, 필터가 단순함</td><td>필터 많은 검색에서 재현율 저하, p99 40ms 미만 요구, 1억 개 이상</td></tr>
<tr><td>분석</td><td>1TB 미만, 동시 사용자 적음</td><td>수 TB 이상, 동시 쿼리 많음, 스트리밍 적재</td></tr>
</tbody>
</table>

<p>자료를 모아 보니 두 흐름이 같은 방향을 보고 있다. Postgres 쪽은 익숙한 저장소 하나에 기능을 붙이고, 객체 스토리지 쪽은 저장은 싸게 맡기고 엔진과 캐시로 성능을 낸다. 둘 다 운영할 시스템 수를 줄이는 쪽이다. 전용 DB가 사라진다기보다, 전용 DB를 따로 띄워야 하는 경계선이 점점 뒤로 밀리고 있다. 그래서 내 답은 이렇다. 대체는 되지만 성능이 맞아야 하고, 그건 내 데이터로 재 봐야 안다.</p>

<h2>내 경우에 대입해 보면</h2>

<p>개인 프로젝트로 만들고 있는 일기 앱도 Supabase Postgres에 pgvector를 붙여 과거 일기를 의미 검색한다. 사용자 한 명당 일기는 많아야 수천 개이고, 검색은 항상 "이 사용자의 일기 중에서"라는 필터가 붙는다. 이 정도 규모면 별도 벡터 DB를 띄우는 것보다 일기 테이블 옆에 벡터 컬럼을 두는 쪽이 훨씬 단순하다. 다만 이번에 필터 문제를 다시 보면서, 사용자 수가 늘어나면 필터를 건 상태의 재현율을 따로 재 봐야겠다는 생각이 들었다. 반대로 문서 수십만 건 이상을 다루는 RAG라면 필터와 전문 검색을 섞는 하이브리드 검색에서 어디가 먼저 병목이 되는지부터 확인할 것 같다. turbopuffer가 구조를 바꾼 이유가 정확히 그 지점이라서다.</p>

<p>다른 경험이나 의견이 있으면 댓글로 남겨 주시면 좋겠다.</p>

<h2>참고 자료</h2>

<ul>
<li><a href="https://topicpartition.io/blog/postgres-pubsub-queue-benchmarks">Kafka is fast -- I'll use Postgres (TopicPartition, 2025.10)</a></li>
<li><a href="https://www.postgresql.org/about/news/pgque-v01-postgres-3284/">PgQue v0.1 - Zero-bloat Postgres queue (PostgreSQL News, 2026.05)</a></li>
<li><a href="https://ecorpit.com/pgvector-vs-dedicated-vector-database-rag-decision-2026/">pgvector or a dedicated vector database? A RAG decision guide (2026.07)</a></li>
<li><a href="https://turbopuffer.com/blog/rip-vector-database">RIP, vector database (turbopuffer, 2026.09)</a></li>
<li><a href="https://datalakehousehub.com/blog/postgres-meets-the-lakehouse">Postgres Meets the Lakehouse: pg_lake, pg_duckdb, and When Postgres Is Enough</a></li>
<li><a href="https://www.tinybird.co/blog/clickhouse-vs-postgresql-pg-mooncake">ClickHouse vs PostgreSQL with pg_mooncake (Tinybird, 2025.10)</a></li>
<li><a href="https://thenewstack.io/object-store-apps-cloud-natives-freshest-architecture/">Object store apps: cloud native's freshest architecture (The New Stack, 2025.02)</a></li>
<li><a href="https://dentiny.github.io/posts/duckdb-on-object-storage/">DuckDB on object storage (2026.10)</a></li>
</ul>
