---
title: "Session Flow: hook 기반 Claude Code 멀티 세션 기록, 이름 기반 하네스, 스냅샷 diff 설계"
description: "여러 Claude Code 세션의 메시지 흐름과 파일 수정을 hook으로 로컬 JSONL에 기록하고 VS Code 확장에서 그래프와 diff로 보여주는 서버 없는 도구의 설계 문서. 하네스 안내·차단 로직과 검증 구성을 정리했다."
date: 2026-10-16
lang: ko
key: session-flow
tags: [Claude Code, hooks, VS Code Extension, Multi-agent, Observability, Diff]
tistory: https://mynews76154.tistory.com/26
---

## 초록

한 작업 폴더에서 여러 Claude Code 세션이 `SendMessage`로 일을 주고받으면 메시지 방향과 세션별 파일 변경을 추적하기 어렵다. Session Flow는 Claude Code 플러그인이 hook 9종으로 세션 활동을 `~/.session-flow/`의 JSONL과 파일 전/후 스냅샷으로 남기고, VS Code 확장이 이를 감시해 세션 그래프, 이벤트 목록, git 스타일 diff를 그린다. 서버와 런타임 의존성은 없다. 하네스 파일(멤버·역할·메인·메시지 방향)이 있으면 hook이 각 세션에 역할을 안내하고, 선택적으로 정한 방향 밖 메시지를 `PreToolUse`에서 거부한다. v0.6.0에서 멤버 식별을 주소 추정에서 `/rename` 이름 기준으로 바꾸며 그래프 모듈 842줄을 제거했다. 테스트 6개 파일, assert 246개가 모두 통과한다.

## 1. 문제 정의

메인 세션이 작업을 나누고 다른 세션들이 `SendMessage`로 지시와 결과를 주고받는 구성에서, 다음 세 가지를 IDE 안에서 확인하는 것이 목표다.

- 어느 세션이 어느 세션에 메시지를 보냈는가 (방향, 횟수, 본문)
- 어느 세션이 어떤 파일을 어떻게 고쳤는가 (처음 수정 전 ~ 마지막 수정 후)
- 정한 분업 구조(메인 경유 등)를 따르고 있는가

| 구분 | 내용 |
| --- | --- |
| 입력 | hook 이벤트(stdin JSON), 세션 transcript(JSONL), `.claude/session-flow.json` |
| 출력 | `events.jsonl`, 스냅샷, 세션에 주입되는 역할 안내·차단 응답, IDE 그래프·diff |

정량 지표 대신 다음을 테스트로 고정했다. 기록 이벤트의 종류·순서, 발신 주소 → 세션 연결, 차단 판정(정한 방향·답장·하네스 밖·미확인 주소), diff 추가/삭제 줄 수.

## 2. 요구사항과 설계 제약

| 요구사항 | 설계 결정 |
| --- | --- |
| 관찰 도구가 본 작업을 막지 않음 | 예외는 `errors.log`에만 남기고 항상 exit 0 |
| 원격 SSH에서도 동작 | 서버 없음. 같은 머신의 로컬 파일만 공유 |
| 확장에 npm 런타임 의존성 없음 | diff 직접 구현(78줄), `vsce package --no-dependencies` |
| 변경분을 주지 않는 `Write`도 diff | Pre/Post 시점 파일 전체 스냅샷 |
| 재시작·다른 PC에서 멤버 유지 | 멤버 = `/rename` 이름 = `SendMessage` 수신 주소 |
| 안내가 컨텍스트를 오염시키지 않음 | 시작 시 1회, 이후 설정 해시가 바뀔 때만 |
| 차단은 오탐보다 미탐 | 미확인 주소 통과, 답장 허용 |
## 3. 아키텍처

### 3.1 전체 파이프라인

```text
Claude Code 세션들 ──hook stdin JSON──▶ plugin/scripts/record.js
  ├─ events.jsonl append
  ├─ 수정 툴: snapshots/<tool_use_id>.before / .after
  ├─ 자기 소켓 주소 → addr.json, /rename 이름 → titles.json
  └─ 하네스 파일 있음 → harness.js: 역할 안내(additionalContext) / 차단(deny)
        ▼
~/.session-flow/ ──fs.watchFile 1s + 200ms debounce──▶ extension.js
  ├─ store.js        이벤트 → 세션·에이전트 레인·턴, 메시지 → 그래프
  ├─ transcripts.js  /rename 이름 이어 읽기 (5s 폴링)
  ├─ harness.js      하네스 로드/저장, 멤버 해석
  ├─ diff.js         스냅샷 전/후 → hunk
  └─ panel.js / timeline.js  webview
```

### 3.2 플러그인과 hook

`hooks.json`은 `SessionStart`, `SessionEnd`, `UserPromptSubmit`, `PreToolUse`, `PostToolUse`, `PostToolUseFailure`, `SubagentStart`, `SubagentStop`, `Stop`에 같은 `record.js`를 건다. `PreToolUse`만 matcher를 `Edit|Write|MultiEdit|NotebookEdit|Agent|Task|SendMessage`로 좁혀, 스냅샷·위임·송신·차단이 필요한 툴만 실행 전 hook을 거친다.

다른 세션의 메시지는 받는 쪽 `UserPromptSubmit`에 `<cross-session-message from="...">` 또는 `Message from @handle:` 형태로 들어오므로 정규식으로 `message_in`을 분리한다. 세션의 소켓 주소(`CLAUDE_CODE_MESSAGING_SOCKET`)는 메인 이벤트에 `self_addr`로 붙이고 `addr.json`에 갱신한다.

### 3.3 기록 포맷

| 파일 | 내용 |
| --- | --- |
| `events.jsonl` | 이벤트 한 줄 = JSON 하나, append only |
| `snapshots/<id>.before/.after` | 수정 직전·직후 파일 전체. 새 파일은 빈 before, 1MB 초과는 생략 |
| `addr.json` | 소켓 주소 → 세션 ID |
| `titles.json` | 세션 ID → `{path, offset, title}` 이어 읽기 캐시 |
| `harness-state.json` | 세션별 마지막 안내 해시 |

공통 필드는 `v`, `ts`, `session_id`, `cwd`, `agent_id`, `agent_type`이다.

| kind | hook | 추가 필드 |
| --- | --- | --- |
| `prompt` / `message_in` | UserPromptSubmit | `summary`(120자), `detail`(4,000자), `target`(발신 주소) |
| `message` | PreToolUse(SendMessage) | `target`, `detail`, `blocked`, `block_reason` |
| `delegate` | PreToolUse(Agent/Task) | `target`(subagent_type), `detail` |
| `tool` / `tool_error` | PostToolUse(/Failure) | `tool_use_id`, `file`, `has_before`, `has_after` |
| `harness_brief` | SessionStart / UserPromptSubmit | `detail`(주입한 안내문) |

그 밖에 `session_start/end`, `stop`, `subagent_start/stop`이 있다. 스냅샷 이름이 `tool_use_id`라 이벤트 하나로 전/후 파일을 찾는다.

하네스 파일은 `version: 2`, `name`, `enforce`, `members[{name, role, main, session?}]`, `edges[{from, to}]`, `layout`이다. v1(세션 ID 기반)은 읽을 때 이름 기반으로 변환하고, 저장 시 멤버 밖·자기 자신·중복 방향을 지우고 `main`을 하나로 제한한다.

### 3.4 VS Code 확장

`store.js`는 vscode 의존성이 없는 순수 함수다. `buildSessions`는 마지막 이벤트가 10분 이내이고 `session_end`가 없으면 live로 본다. `sessionTurns`는 질문 또는 받은 메시지 하나를 턴으로 묶고, `buildGraph`는 최근 90초 내 엣지를 active로 표시한다. `changedFiles`는 파일별 첫 수정과 마지막 수정을 짝지어 누적 diff 범위를 만든다.

`buildGraph`에는 v0.6.0 이전의 주소 추정 경로(소켓 기록, `ListAgents` 출력, 본문 일치)가 남아 있다. 지금은 멤버 이름과 `/rename` 이름 주소록이 추정보다 우선하므로, 추정은 멤버 밖 메시지 집계에만 쓰인다.

`/rename`은 hook을 남기지 않아 transcript의 `custom-title`을 5초마다 4MB 청크 이어 읽기로 확인한다. 지시 보내기는 Claude Code URI로 대상 세션 입력창을 채우고, 실패하면 클립보드로 대체한다.

### 3.5 diff 알고리즘

공통 접두·접미를 먼저 잘라 LCS 표를 줄인다. 코드 수정은 대부분 파일 일부만 바꾸기 때문이다.

`extension/src/diff.js`

```js
function script(a, b) {
  let pre = 0;
  while (pre < a.length && pre < b.length && a[pre] === b[pre]) pre++;
  let suf = 0;
  while (suf < a.length - pre && suf < b.length - pre && a[a.length - 1 - suf] === b[b.length - 1 - suf]) suf++;
  const A = a.slice(pre, a.length - suf);
  const B = b.slice(pre, b.length - suf);
  const out = [];
  for (let i = 0; i < pre; i++) out.push({ t: ' ', o: i + 1, n: i + 1, text: a[i] });

  if (A.length * B.length > MAX_CELLS) {
    A.forEach((x, i) => out.push({ t: '-', o: pre + i + 1, text: x }));
    B.forEach((x, i) => out.push({ t: '+', n: pre + i + 1, text: x }));
  } else {
```

가운데는 `Uint32Array` LCS 표(`O(m*n)`)를 뒤에서 채워 역추적하고, `m*n > 4,000,000`이면 전체 교체로 표시한다. 변경 주변 3줄을 남겨 hunk로 묶고, webview로는 최대 4,000줄만 보낸다. 누적 diff는 첫 수정의 `before`와 마지막 수정의 `after`를 비교해 커밋 하나처럼 보인다.

### 3.6 하네스: 멤버 식별과 역할 안내

hook은 `cwd`에서 위로 올라가며 하네스 파일을 찾는다. 세션의 멤버 여부는 설정의 세션 ID 또는 `/rename` 이름 일치(trim·`@` 제거·소문자)로 정한다. 이름은 `titles.json` offset부터 최대 64MB만 읽는다. 확장의 `resolveMembers`는 같은 이름이 여럿이면 최근 세션을, 저장된 ID의 이름이 다르면 이름 쪽을 따른다.

`plugin/scripts/record.js`

```js
    if ((ev === 'SessionStart' || ev === 'UserPromptSubmit') && !h.agent_id && me) {
      const ctx = H.contextFor(found, me, myName);
      const state = H.readJson(stateFile, {});
      const hash = H.configHash(found, me, myName);
      // 시작할 때는 항상, 그 뒤로는 설정이나 이름이 바뀌었을 때만 다시 알려준다
      if (ctx && (ev === 'SessionStart' || state[sid] !== hash)) {
        response = { hookSpecificOutput: { hookEventName: ev, additionalContext: ctx } };
        state[sid] = hash;
        try { fs.writeFileSync(stateFile, JSON.stringify(state)); } catch { /* ignore */ }
        out.push({ ...base(h), kind: 'harness_brief', summary: '하네스 역할 안내', detail: ctx });
      }
    }
```

해시는 `sha1([cfg, me.name, myName])` 앞 12자리다. 안내문에는 이름·역할, 메인, 보낼 대상, 메인을 거쳐야 할 멤버가 들어가고, 실제 이름이 다르면 `/rename` 요청이 붙는다. 하네스 파일 작성은 `/session-flow:harness` 스킬로 세션에 맡길 수 있다.

### 3.7 enforce 로직

`enforce: true`이고 방향이 있을 때 `PreToolUse(SendMessage)`에서 수신자를 해석한다. 멤버 이름 일치를 먼저 보고, 아니면 소켓 주소를 `addr.json`으로 세션에 대응시켜 멤버를 찾는다. 알려진 세션이 멤버가 아니면 `outside`, 모르면 `null`이다.

| 조건 | 결과 |
| --- | --- |
| 꺼짐 / 방향 없음 / 이 세션이 멤버 아님 | 허용 |
| 수신자 미확인 | 허용(`unverified`) |
| 하네스 밖 세션 | 거부, 허용 대상 안내 |
| `from→to` 또는 `to→from` 방향 존재 | 허용 (역방향 = 답장) |
| 그 외 멤버 | 거부, "메인을 거쳐 전달" 안내 |

거부 이유는 `permissionDecisionReason`으로 세션에 전달되고, 기록에는 `blocked`가 남아 그래프에 빨간 엣지로 표시된다.

## 4. 검증

### 4.1 테스트 구성

Node `assert`만 쓴다. hook 테스트는 실제 `record.js`를 자식 프로세스로 실행한다. 확장 테스트는 가짜 vscode API를, webview 테스트는 jsdom을 쓴다(없으면 문법만 확인). CI는 Node 20에서 6개 파일을 돌리고, Release 워크플로는 통과 후 `.vsix`를 첨부한다.

| 파일 | 커버하는 경우 |
| --- | --- |
| `smoke.js` | 이벤트 12개 종류·순서, Edit/Write 스냅샷, 깨진 입력 → `errors.log`, 레인·턴 |
| `store.js` | 본문 일치 학습, ghost 노드, 다른 폴더 제외, 구버전 기록 재분류, 누적 diff 줄 수, 주소 재사용, 4MB 경계 뒤 `/rename`, 이름 우선 연결 |
| `harness.js` | 안내·재안내 생략, `/rename` 안내, 차단(방향·답장·미확인·소켓 주소·하네스 밖), 캐시, v1 변환 |
| `extension.js` | 폴더 필터, 멤버 추가/제거, 훅 없는 이름 변경 감지, 규칙 편집, 입력창 채우기·클립보드 대체, 외부 하네스 파일 반영 |
| `panel.js` | 이벤트 클릭/더블클릭, 세션 추가 팝업, 엣지 상태, diff 렌더링, XSS 방지 |
| `timeline.js` | 턴 카드 순서, 필터, 접기, diff 열기 |

### 4.2 결과

v0.9.0(`6fdd40e`)에서 jsdom 24로 실행했다. assert 수는 정적 개수와 실행 계측이 같았다.

| 파일 | smoke | store | harness | extension | panel | timeline | 합계 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| assert | 20 | 57 | 29 | 50 | 74 | 16 | 246 |
| 결과 | 통과 | 통과 | 통과 | 통과 | 통과 | 통과 | 6/6 |

규모는 확장 `src` 2,346줄, 플러그인 스크립트 411줄, 테스트 921줄이다. 2026-10-07 하루 18개 커밋으로 확장 v0.9.0, 플러그인 v0.6.0까지 올렸다.

### 4.3 설계 비교

| 멤버·주소 식별 방식 | 도입 | 현재 상태 |
| --- | --- | --- |
| 받은/보낸 본문 정규화 비교 | v0.2.0 | 멤버 밖 집계용으로만 유지 |
| 수동 이름·핸들 연결(`names.json`) | v0.2.0 | 하네스 주소록으로 흡수 |
| 고른 세션 외를 "기타" 노드로 | v0.4.0 | 하네스 화면에서 "외부 n" 배지로 대체 |
| 멤버 = 세션 ID (하네스 v1) | v0.5.0 | 재시작·다른 PC에서 불일치, v2로 자동 변환 |
| 자기 소켓 주소 기록, `ListAgents` 학습 | v0.5.1 | 소켓 기록은 차단 판정에 유지 |
| 멤버 = `/rename` 이름 (하네스 v2) | v0.6.0 | 현재. 이름이 곧 수신 주소라 추정 불필요 |

v0.6.0 커밋은 19개 파일, 추가 1,556줄·삭제 1,757줄이었고 `graph.js`(842줄)가 삭제됐다.

## 5. 한계와 향후 과제

1. 스냅샷은 파일 전체 평문 복사다. 비밀값 마스킹과 보존 정책이 없고, 1MB 초과 파일과 `Bash`로 바꾼 파일은 diff가 없다.
2. 확장은 변경마다 `events.jsonl` 전체를 다시 읽는다. 기록이 커지면 비용이 선형으로 늘어 offset 기반 증분 로드나 로그 회전이 필요하다.
3. `addr.json`, `titles.json`, `harness-state.json`에 잠금이 없어 여러 세션의 hook이 동시에 쓰면 마지막 쓰기가 이긴다.
4. 차단은 fail-open이다. 미확인 주소는 통과하고, 플러그인이 로드되지 않은 세션은 판정이 없다.
5. `custom-title`, `<cross-session-message>`, `ListAgents` 출력, 세션 열기 URI는 공개 API가 아니라 Claude Code 버전에 따라 깨질 수 있다. diff는 줄 단위이며 줄 안 변경 강조가 없다.

## 참고

- 코드: https://github.com/tjehdgus/session-flow
- 개발 기록: https://mynews76154.tistory.com/26
- Claude Code hooks, Claude Code plugin marketplace
- VS Code Extension API (TreeDataProvider, Webview, `vscode.diff`)
- LCS 기반 줄 단위 diff, jsdom
