---
title: "Can Postgres do everything? Drawing the line across queues, vector search, analytics and object storage"
description: "\"Just use Postgres\" put to the test in four areas: message queues, vector search, analytics (OLAP) and object-storage-native databases. Postgres can stand in for all of them, but each workload has a point where it stops being enough."
date: 2026-10-16
lang: en
key: pg-everything
kind: review
tags: [PostgreSQL, pgvector, Kafka, OLAP, DuckDB, Data Engineering]
tistory: https://mynews76154.tistory.com/28
---

"Just use Postgres" is everywhere lately. Search with `tsvector`/`tsquery`, documents with `jsonb`, caching with `UNLOGGED TABLE`, vector search with `pgvector`, and analytics with extensions like `pg_lake`, `pg_duckdb` and `pg_mooncake`. Recently a benchmark that replaced a messaging system like Kafka with Postgres made the rounds as well.

Anyone building a RAG service runs into this question sooner or later: stand up a separate vector database, or add pgvector to the Postgres you already run? So I split the question into four areas and went through the material: message queues, vector search, analytics (OLAP), and the opposite approach, databases built on object storage. The criterion is simple. Postgres can replace a lot of things, but the performance and the operations have to work out.

![Two ways to run fewer systems](/img/reviews/pg-everything-1-en.png)

## 1. Message queues: enough for most services

### What the benchmark showed

The most cited source is "Kafka is fast -- I'll use Postgres" (October 2025) by Stanislav Kozlovski, who has spent years in the Kafka world. He built two things using only core Postgres features and measured them:

- **Kafka-style pub/sub**: one table per partition, a counter table to assign offsets atomically, and a separate record of each consumer group's read position.
- **A work queue**: jobs are claimed one at a time with `SELECT ... FOR UPDATE SKIP LOCKED` and moved to an archive table when done.

With 1 KiB messages and fsync on for durability, the results were:

| Setup | Writes | Reads (5 consumer groups) | p99 latency |
|---|---|---|---|
| 1 × 4 vCPU | ~5,000 msg/s | ~25,000 msg/s | 60ms |
| 3 × 4 vCPU, replicated | ~5,000 msg/s | ~25,000 msg/s | 186ms |
| 1 × 96 vCPU | ~243,000 msg/s | ~1.2M msg/s | 853ms |

The work queue reached about 2,900 jobs per second on a single 4 vCPU instance, with p99 at 17.7ms. The conclusion: "stick with Postgres until you hit its limits." The author also acknowledged the caveats: the tests ran only 2 to 15 minutes, Postgres uses far more CPU than a dedicated system for the same work, and there is no ecosystem like Schema Registry or Kafka Connect. The post also spread as a story about rewriting Kafka in Rust, but Rust only comes up in a footnote about Tansu, a separate open-source project that exposes the Kafka API on top of Postgres or S3 storage.

### The long-run problem: bloat

There is a problem short benchmarks don't show. A `SKIP LOCKED` queue keeps running `DELETE` or `UPDATE` on processed rows. Postgres doesn't reclaim the space immediately; it leaves dead tuples behind. Under sustained load, VACUUM pressure and index bloat build up and performance degrades slowly.

PgQue, released in May 2026, tackles this head-on. It borrows the PgQ design Skype used and removes bloat by rotating whole event tables instead of deleting individual rows. It installs as a single SQL file with no C extension, so it works on managed Postgres such as RDS or Supabase. The trade-off is batch processing, which puts typical delivery latency at 1 to 2 seconds.

In short, Postgres is enough for work queues or events in the thousands per second. It is better kept off the instance that serves your application database, so failures don't spread, and for long-running use you should pick a design that handles bloat. Once you need millisecond latency or large fan-out, that is the time to look at a dedicated system.

## 2. Vector search: the line is filters and tail latency, not headline numbers

### On throughput alone, Postgres can win

There is a frequently cited answer to "aren't dedicated vector databases still faster?" It comes from a benchmark published in April 2025 by Tiger Data (formerly Timescale), a Postgres hosting company: 50 million 768-dimensional vectors at 99% recall.

| Metric | Postgres + pgvector + pgvectorscale | Qdrant |
|---|---|---|
| Throughput (QPS) | 471 | 41 |
| p50 latency | 31ms | 31ms |
| p99 latency | 75ms | 39ms |
| Index build | ~11 hours | ~3.3 hours |

Postgres delivers 11x the throughput, but Qdrant wins on tail latency and index build time. And the conditions matter: this is not stock pgvector but pgvectorscale, which adds a disk-based index (StreamingDiskANN); it measures pure vector search with no filters; and it was published by a company that sells Postgres.

### In real RAG, filters bite first

Production RAG almost always filters: by user, by document type, by date range. With an approximate index (HNSW), the index picks candidates first and the filter is applied afterwards. With the default `hnsw.ef_search = 40`, a filter that keeps only 10% of rows leaves on average 4 of the 40 candidates. You ask for the top 10 and get 4 back.

![Why filtering an HNSW index can return too few results](/img/reviews/pg-everything-2-en.png)

pgvector 0.8.0 mitigates this with iterative index scans: when there aren't enough results, it reads further into the index to fill them. Even so, a filter-heavy service should measure recall and latency separately with filters applied.

Pulling together the rules of thumb from the sources:

- Under 1M vectors: pgvector HNSW is enough.
- 1M to 10M: tuned pgvector handles most cases.
- 10M to 100M: pgvectorscale if filters are light and throughput matters; a dedicated database if filters are heavy or you need p99 under 40ms.
- Over 100M, or more than fits on one machine: a dedicated database with sharding.

### Dedicated vector databases are moving away from "vector-first" too

The interesting movement is on the dedicated side. In its September 2026 post "RIP, vector database," turbopuffer, a search engine built on object storage, announced a change to its storage layout. Until now, every piece of data was stored under the key "which vector cluster, which position in it." The vector index was the primary index. The more features they layered on top, such as filters, full-text search and aggregations, the worse this got:

- A document with several vectors had its contents stored once per vector.
- Every cluster rebalance had to move document contents and the other indexes along with it.
- Modern query engines process data in batches of thousands of rows (2,048 in DuckDB, about 65,000 in ClickHouse), but vector clusters hold 100 to 200 entries, so every query was tied to that size. When they detached the full-text index from the clusters and rebuilt it in fixed 256-entry blocks, the index got about 10x smaller and queries up to about 20x faster.

![turbopuffer v2 vs v3 storage layout](/img/reviews/pg-everything-3-en.png)

So v3 demotes the vector index to one secondary index among several. This doesn't mean dropping vector search; it means moving toward the structure of a general search engine. Performance benchmarks haven't been published yet. Postgres is absorbing vectors while vector databases are turning into general databases. Both sides are converging on the same point.

## 3. Analytics (OLAP): really, putting another engine inside Postgres

Postgres is a row-oriented OLTP database, so it struggles with analytical queries that aggregate hundreds of millions of rows. The extensions that address this all attach a columnar engine inside or next to Postgres. Notably, all three are tied to big companies.

| Extension | Who | Approach |
|---|---|---|
| pg_duckdb | DuckDB Labs, Hydra, MotherDuck | Embeds DuckDB in the Postgres process to run analytical queries |
| pg_mooncake | Mooncake Labs (acquired by Databricks in 2025) | Keeps a columnar copy of Postgres tables and queries it with DuckDB |
| pg_lake | Crunchy Data (acquired by Snowflake in 2025, open-sourced that November) | Transactional reads and writes of Iceberg tables from Postgres |

Snowflake and Databricks both buying Postgres-side companies stands out. It looks like an effort to pull the boundary between Postgres as the operational database and the analytics platform toward their own side.

Performance still trails dedicated engines. According to a comparison by Tinybird, which runs a ClickHouse-based service, pg_mooncake was 1.2 to 4x slower than ClickHouse on 100-million-row ClickBench data, depending on the query type (a vendor post, and the table's source isn't clear, so I treat it as a reference only). The thresholds that several sources agree on look roughly like this:

- Under 1 TB and fewer than 20 concurrent analytics users: Postgres plus extensions is enough.
- 1 to 5 TB, or 20 to 50 concurrent users: a single node holds up, but it's time to consider a second engine.
- A single table over 5 TB, more than 50 concurrent users, or high-frequency streaming ingestion: a distributed engine like ClickHouse or Snowflake.

So OLAP is less "replace it with Postgres" and more "keep Postgres as the front door and put an analytics engine inside." Whether it actually replaces a dedicated engine can only be known by running the same data and queries side by side. That's something I'd like to measure against ClickHouse myself when I get the chance.

## 4. The opposite direction: databases built purely on object storage

While one camp packs features into a single Postgres, another is increasingly putting engines on top of object storage like S3, with the object store as the source of truth.

### Why now

One trigger was conditional writes, added to S3 in August 2024. Once "write only if this object doesn't exist" became possible, writing sequentially numbered files (001, 002, ...) was enough to prevent concurrent write conflicts, just like a database log. Consistency could now be handled with object storage alone, without a separate consensus algorithm or metadata database. Object storage is cheap, replicated across availability zones by default and effectively unlimited. In exchange, latency is high and objects can't be partially updated.

### What's out there

- **WarpStream**: a Kafka-compatible system built on S3. Acquired by Confluent in 2024.
- **SlateDB**: a key-value store (LSM tree) on object storage. Write latency is high, but S3 guarantees durability.
- **turbopuffer**: the search engine from earlier. The source of truth lives in S3, with NVMe SSD and memory as cache.
- **duckdb_object_storage**: a DuckDB community extension released in October 2026 that runs a full DuckDB database on top of SlateDB.

The last one is especially interesting. A single `ATTACH 'duckdb_objfs://analytics.db'` attaches a database in an S3 bucket, and primary keys, foreign keys, indexes and transactions behave as in regular DuckDB. On TPC-H, the in-memory backend was actually faster than native DuckDB (0.92x the time), and cold reads from S3 were about 20% slower than the existing approach (HTTPFS). The limits are clear:

- Only one process can write.
- Every commit costs an S3 round trip of tens of milliseconds, so small writes need to be batched.
- Readers see changes up to about 10 seconds late.

DuckLake and Iceberg are formats for sharing the same data across multiple engines (Spark, Trino and so on). This extension is for when DuckDB alone is enough and you want everything in one bucket, so they serve different purposes.

## Summary by workload

| Area | Where Postgres is enough | Signals to look at a dedicated system |
|---|---|---|
| Message queue | Thousands per second, tens of ms to a few seconds of latency acceptable | Millisecond latency, large fan-out, need for a connector ecosystem |
| Vector search | Up to 10M vectors, simple filters | Recall drops under heavy filtering, p99 under 40ms required, 100M+ vectors |
| Analytics | Under 1 TB, few concurrent users | Several TB or more, many concurrent queries, streaming ingestion |

Putting the material together, the two trends point the same way. The Postgres side adds features to one familiar store; the object storage side keeps storage cheap and gets performance from engines and caches. Both reduce the number of systems you have to run. Dedicated databases aren't going away, but the point at which you need to stand one up keeps moving further out. So my answer is: Postgres can replace them, but only if the performance fits, and the only way to know is to measure with your own data.

## Applying it to my own case

The diary app I'm building as a personal project also uses pgvector on Supabase Postgres for semantic search over past entries. Each user has at most a few thousand entries, and every search carries a "within this user's diary" filter. At this scale, a vector column next to the diary table is far simpler than running a separate vector database. Revisiting the filter problem, though, I realised I should measure filtered recall separately as the user base grows. For a RAG system over hundreds of thousands of documents or more, on the other hand, I'd start by checking which part becomes the bottleneck first in hybrid search that mixes filters and full-text search. That is exactly why turbopuffer changed its architecture.

If you have other experiences or opinions, I'd love to hear them in the comments.

## References

- [Kafka is fast -- I'll use Postgres (TopicPartition, 2025.10)](https://topicpartition.io/blog/postgres-pubsub-queue-benchmarks)
- [PgQue v0.1 - Zero-bloat Postgres queue (PostgreSQL News, 2026.05)](https://www.postgresql.org/about/news/pgque-v01-postgres-3284/)
- [pgvector or a dedicated vector database? A RAG decision guide (2026.07)](https://ecorpit.com/pgvector-vs-dedicated-vector-database-rag-decision-2026/)
- [RIP, vector database (turbopuffer, 2026.09)](https://turbopuffer.com/blog/rip-vector-database)
- [Postgres Meets the Lakehouse: pg_lake, pg_duckdb, and When Postgres Is Enough](https://datalakehousehub.com/blog/postgres-meets-the-lakehouse)
- [ClickHouse vs PostgreSQL with pg_mooncake (Tinybird, 2025.10)](https://www.tinybird.co/blog/clickhouse-vs-postgresql-pg-mooncake)
- [Object store apps: cloud native's freshest architecture (The New Stack, 2025.02)](https://thenewstack.io/object-store-apps-cloud-natives-freshest-architecture/)
- [DuckDB on object storage (2026.10)](https://dentiny.github.io/posts/duckdb-on-object-storage/)
