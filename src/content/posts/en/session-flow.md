---
title: "Session Flow: Hook-Based Recording of Multi-Session Claude Code Work, Name-Based Harness, and Snapshot Diffs"
description: "Design document for a serverless tool that records message flow and file edits across Claude Code sessions into local JSONL via hooks and renders them as a graph and diffs in a VS Code extension. Covers the harness briefing and enforcement logic and the test setup."
date: 2026-10-09
lang: en
key: session-flow
tags: [Claude Code, hooks, VS Code Extension, Multi-agent, Observability, Diff]
tistory: https://mynews76154.tistory.com/26
---

## Abstract

When several Claude Code sessions in one workspace exchange work through `SendMessage`, it is hard to follow message direction and per-session file changes. In Session Flow, a Claude Code plugin attaches nine hooks that write session activity to JSONL and before/after file snapshots under `~/.session-flow/`; a VS Code extension watches those files and renders a session graph, an event list, and git-style diffs. There is no server and no runtime dependency. If a harness file (members, roles, main, message directions) exists, the hooks brief each session on its role and can optionally deny out-of-policy messages in `PreToolUse`. In v0.6.0, member identity moved from address inference to the session's `/rename` name, which removed an 842-line graph module. All 6 test files with 246 assertions pass.

## 1. Problem

In a setup where a main session splits the work and other sessions exchange instructions and results via `SendMessage`, the goal is to answer three questions inside the IDE:

- Which session sent what to which session (direction, count, body)?
- Which session changed which file, and how (before the first edit to after the last)?
- Is the agreed division of labor (e.g. routing through main) actually followed?

| Item | Content |
| --- | --- |
| Input | Hook events (stdin JSON), session transcripts (JSONL), `.claude/session-flow.json` |
| Output | `events.jsonl`, snapshots, role briefings and deny responses injected into sessions, IDE graph and diffs |

Instead of a numeric metric, the following properties are pinned by tests: kinds and order of recorded events, mapping from sender address to session, the enforcement verdict (declared direction, reply, outside the harness, unknown address), and added/removed line counts of diffs.

## 2. Requirements and design constraints

| Requirement | Design decision |
| --- | --- |
| The observer must never block real work | Exceptions go to `errors.log`; the hook always exits 0 |
| Must work over remote SSH | No server; plugin and extension share local files on the same machine |
| No npm runtime dependency in the extension | Hand-written diff (78 lines), `vsce package --no-dependencies` |
| Diff even for `Write`, which carries no delta | Full-file snapshots at Pre/Post time |
| Members survive restarts and other machines | Member = `/rename` name = `SendMessage` recipient address |
| Briefings must not pollute context | Once at start, then only when the config hash changes |
| Prefer false negatives over false positives when blocking | Unknown addresses pass; replies are allowed |

## 3. Architecture

### 3.1 Pipeline

```text
Claude Code sessions ──hook stdin JSON──▶ plugin/scripts/record.js
  ├─ append to events.jsonl
  ├─ edit tools: snapshots/<tool_use_id>.before / .after
  ├─ own socket address → addr.json, /rename name → titles.json
  └─ harness file present → harness.js: briefing (additionalContext) / deny
        ▼
~/.session-flow/ ──fs.watchFile 1s + 200ms debounce──▶ extension.js
  ├─ store.js        events → sessions, agent lanes, turns; messages → graph
  ├─ transcripts.js  incremental read of /rename names (5s poll)
  ├─ harness.js      load/save harness, resolve members
  ├─ diff.js         snapshot before/after → hunks
  └─ panel.js / timeline.js  webviews
```

### 3.2 Plugin and hooks

`hooks.json` binds the same `record.js` to `SessionStart`, `SessionEnd`, `UserPromptSubmit`, `PreToolUse`, `PostToolUse`, `PostToolUseFailure`, `SubagentStart`, `SubagentStop`, and `Stop`. Only `PreToolUse` uses a narrowed matcher, `Edit|Write|MultiEdit|NotebookEdit|Agent|Task|SendMessage`, so that only tools needing a snapshot, delegation record, send record, or enforcement check go through a pre-execution hook.

Messages from other sessions arrive in the receiver's `UserPromptSubmit` as `<cross-session-message from="...">` or `Message from @handle:`, and a regex splits them out as `message_in`. The session's own socket address (`CLAUDE_CODE_MESSAGING_SOCKET`) is attached to main-agent events as `self_addr` and written to `addr.json`.

### 3.3 Record format

| File | Content |
| --- | --- |
| `events.jsonl` | One JSON object per line, append only |
| `snapshots/<id>.before/.after` | Whole file right before and after an edit. Empty `before` for new files; files over 1MB skipped |
| `addr.json` | Socket address → session ID |
| `titles.json` | Session ID → `{path, offset, title}` incremental-read cache |
| `harness-state.json` | Last briefing hash per session |

Common fields are `v`, `ts`, `session_id`, `cwd`, `agent_id`, `agent_type`.

| kind | Hook | Extra fields |
| --- | --- | --- |
| `prompt` / `message_in` | UserPromptSubmit | `summary` (120 chars), `detail` (4,000 chars), `target` (sender) |
| `message` | PreToolUse(SendMessage) | `target`, `detail`, `blocked`, `block_reason` |
| `delegate` | PreToolUse(Agent/Task) | `target` (subagent_type), `detail` |
| `tool` / `tool_error` | PostToolUse(/Failure) | `tool_use_id`, `file`, `has_before`, `has_after` |
| `harness_brief` | SessionStart / UserPromptSubmit | `detail` (injected briefing) |

There are also `session_start/end`, `stop`, and `subagent_start/stop`. Snapshots are named by `tool_use_id`, so one event is enough to locate its before/after files.

The harness file has `version: 2`, `name`, `enforce`, `members[{name, role, main, session?}]`, `edges[{from, to}]`, and `layout`. v1 files (session-ID based) are converted to names on read. On save, edges pointing to non-members, self-loops, and duplicates are dropped, and only one `main` is kept.

### 3.4 VS Code extension

`store.js` is a set of pure functions with no vscode dependency. `buildSessions` marks a session live if its last event is within 10 minutes and there is no `session_end`. `sessionTurns` groups work under each prompt or received message, `buildGraph` marks edges active if traffic occurred in the last 90 seconds, and `changedFiles` pairs the first and last edit per file to define a cumulative diff range.

`buildGraph` still contains the pre-v0.6.0 address inference paths (socket records, `ListAgents` output, body matching). Now the address book built from member names and `/rename` names takes precedence, so inference is used only to count traffic with non-members.

`/rename` does not fire a hook, so the transcript's `custom-title` is rechecked every 5 seconds with 4MB-chunk incremental reads. "Send instruction" fills the target session's input box through a Claude Code URI and falls back to the clipboard on failure.

### 3.5 Diff algorithm

The common prefix and suffix are trimmed first to shrink the LCS table, since most code edits touch only part of a file.

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

The middle section fills a `Uint32Array` LCS table (`O(m*n)`) from the end and backtracks from the start; if `m*n > 4,000,000` it is shown as a full replacement. Three context lines are kept around changes to form hunks, and at most 4,000 lines are sent to the webview. The cumulative diff compares the first edit's `before` with the last edit's `after`, so repeated edits to one file read like a single commit.

### 3.6 Harness: member identity and briefing

The hook walks up from `cwd` to find the harness file. A session is a member if its session ID is listed in the config or its `/rename` name matches a member name (trimmed, leading `@` removed, lowercased). The name is read from the `titles.json` offset, at most 64MB per call. On the extension side, `resolveMembers` picks the most recent session when several share a name, and follows the name when a stored ID's current name differs.

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

(The comment reads: always brief at start; afterwards only when the config or name changes.) The hash is the first 12 hex chars of `sha1([cfg, me.name, myName])`. The briefing lists the session's name and role, the main, allowed recipients, and members to reach via the main; if the actual name differs, it asks the user to run `/rename`. Writing the harness file can be delegated to a session with the `/session-flow:harness` skill.

### 3.7 Enforcement logic

With `enforce: true` and at least one edge, `PreToolUse(SendMessage)` resolves the recipient: member-name match first, otherwise the socket address is mapped to a session through `addr.json` and then to a member. A known session that is not a member resolves to `outside`; an unknown address resolves to `null`.

| Condition | Verdict |
| --- | --- |
| Disabled / no edges / sender not a member | Allow |
| Recipient unknown | Allow (`unverified`) |
| Session outside the harness | Deny, list allowed recipients |
| Edge `from→to` or `to→from` exists | Allow (reverse = reply) |
| Any other member | Deny, suggest routing through main |

The reason is returned to the session as `permissionDecisionReason`, and the record keeps `blocked`, which the graph draws as a red edge.

## 4. Verification

### 4.1 Test setup

Tests use plain Node `assert`. Hook tests spawn the real `record.js` as a child process. Extension tests run against a fake vscode API; webview tests use jsdom (syntax check only if absent). CI runs all six files on Node 20, and the release workflow attaches the `.vsix` after they pass.

| File | Cases covered |
| --- | --- |
| `smoke.js` | Kinds and order of 12 events, Edit/Write snapshots, malformed input → `errors.log`, lanes and turns |
| `store.js` | Body-match learning, ghost nodes, other-folder exclusion, reclassifying old records, cumulative diff line counts, address reuse, `/rename` past a 4MB boundary, name-first resolution |
| `harness.js` | Briefing and skipped re-briefing, `/rename` hint, enforcement (edge, reply, unknown, socket address, outside), caches, v1 migration |
| `extension.js` | Folder filter, add/remove members, hookless rename detection, rule editing, input fill and clipboard fallback, externally written harness file |
| `panel.js` | Event click/double-click, add-session popup, edge states, diff rendering, XSS safety |
| `timeline.js` | Turn card order, filters, collapse, opening diffs |

### 4.2 Results

Run at v0.9.0 (`6fdd40e`) with jsdom 24. The static assertion count matched the count instrumented at run time.

| File | smoke | store | harness | extension | panel | timeline | Total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Assertions | 20 | 57 | 29 | 50 | 74 | 16 | 246 |
| Result | pass | pass | pass | pass | pass | pass | 6/6 |

The codebase is 2,346 lines in the extension `src`, 411 lines of plugin scripts, and 921 lines of tests. It reached extension v0.9.0 and plugin v0.6.0 in 18 commits on 2026-10-07.

### 4.3 Design comparison

| Member / address identification | Introduced | Current status |
| --- | --- | --- |
| Normalized matching of received vs. sent bodies | v0.2.0 | Kept only for non-member traffic |
| Manual name and handle linking (`names.json`) | v0.2.0 | Absorbed into the harness address book |
| Grouping unselected sessions into an "other" node | v0.4.0 | Replaced by an "external n" badge in the harness view |
| Member = session ID (harness v1) | v0.5.0 | Broke on restart and across machines; auto-migrated to v2 |
| Own socket address records, `ListAgents` learning | v0.5.1 | Socket records kept for enforcement |
| Member = `/rename` name (harness v2) | v0.6.0 | Current; the name is the recipient address, so no inference |

The v0.6.0 commit touched 19 files with 1,556 insertions and 1,757 deletions, removing `graph.js` (842 lines).

## 5. Limitations and future work

1. Snapshots are plain full-file copies, with no secret masking or retention policy. Files over 1MB and files changed via `Bash` get no diff.
2. The extension rereads the whole `events.jsonl` on every change, so cost grows linearly; offset-based incremental loading or log rotation is needed.
3. `addr.json`, `titles.json`, and `harness-state.json` have no locking, so concurrent hook writes from several sessions are last-writer-wins.
4. Enforcement is fail-open: unknown addresses pass, and sessions without the plugin loaded are never checked.
5. `custom-title`, `<cross-session-message>`, `ListAgents` output, and the session-open URI are not public APIs and may break across Claude Code versions. The diff is line-based with no intra-line highlighting.

## References

- Code: https://github.com/tjehdgus/session-flow
- Development log: https://mynews76154.tistory.com/26
- Claude Code hooks, Claude Code plugin marketplace
- VS Code Extension API (TreeDataProvider, Webview, `vscode.diff`)
- LCS-based line diff, jsdom
