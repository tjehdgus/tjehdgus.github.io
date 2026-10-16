---
title: "Session Flow: making conversations and diffs between Claude Code sessions visible in the IDE"
description: "A personal tool that records inter-session messages and file edits with hooks and draws them as a graph and diffs in a VS Code extension, including why I threw out an address-guessing design for one based on names."
date: 2026-10-16
lang: en
key: session-flow
tags: [Claude Code, hooks, VS Code extension, multi-agent, tooling]
tistory: https://mynews76154.tistory.com/26
---

These days I often run several Claude Code sessions in one working folder and split the work between them. A main session hands out tasks, the others do their parts, and they exchange results with SendMessage. Once there are three or four sessions, it gets hard to follow who said what to whom and which session changed which file. So I built Session Flow, a tool that shows this flow as a picture inside the IDE and lets me step in.

It reached v0.9.0 in a single day, October 7, 2026, across 18 commits: about 2,350 lines of extension JavaScript, 410 lines of plugin code and 920 lines of tests. ([GitHub](https://github.com/tjehdgus/session-flow))

## Architecture: files only, no server

- **Claude Code plugin**: hooks record session activity (prompts, tool calls, sub-agent delegation, messages between sessions, file contents before and after edits) under `~/.session-flow/`.
- **VS Code extension**: watches those files and renders a session graph, an event list and file diffs.

A server for a single-user tool means something else to start and look after, and it is harder to make work unchanged over remote SSH. Plain files behave the same on whatever machine the sessions run on.

## One rule: an observer must never block the work

The same recording script is attached to hooks such as SessionStart, UserPromptSubmit, PreToolUse, PostToolUse and SubagentStart/Stop, and appends one line per event to `events.jsonl`. If it throws, it writes to `errors.log` and still exits 0. If an observability tool ever stops the real work, people stop using it.

Diffs need the file's previous contents. The Edit tool carries the changed fragment, but Write does not, so when an editing tool runs the plugin snapshots the file at PreToolUse ("before") and PostToolUse ("after"). I wrote the diff myself (78 lines) to keep npm dependencies out of the extension: trim the common prefix and suffix, run an LCS only on the middle, then group changes into git-style hunks. Most edits touch a small part of a file, so trimming alone shrinks the work a lot.

## The problem that took longest: which session is this address?

What I really wanted to see was the flow between sessions. The sending side was easy: record the recipient of SendMessage at PreToolUse. The receiving side was not. Incoming messages identified the sender by a socket address or handle, not a session ID, so drawing a single arrow meant working out which session an address belonged to.

That took five commits: matching message bodies to link senders automatically, letting the user name sessions, grouping unknown addresses into an "other" node, having each session record its own socket address, and learning handles from ListAgents output. Accuracy improved, but the code kept growing more guessing paths.

## Building the harness forced a redesign

In the afternoon I added a "harness": not just viewing the graph, but designing it. You group the sessions in a folder into members, roles, a main session and allowed message directions, and hooks tell each session what its role is and who it should only talk to through the main session. The context is re-injected only when the configuration changes, to keep prompts clean.

Listing members by session ID broke immediately: IDs change on every restart and do not match on another machine. So in v0.6.0 members became the session's `/rename` name. That name works directly as a SendMessage recipient and survives restarts. The commit message reads "the name is the message address; no more address guessing". That commit deleted 842 lines of graph code and rewrote the panel, with 1,556 lines added and 1,757 removed.

If the system already gives you a stable, human-assigned name, building on it is far simpler than inferring internal IDs and addresses. I should have started there.

## Enforce loosely

With enforcement on, the harness rejects SendMessage calls outside the allowed directions at PreToolUse. Two things are deliberately lenient.

- **Unknown addresses pass.** Blocking a legitimate message because address resolution was wrong is worse than missing a rule violation.
- **Replies are allowed.** If only main → member is defined and members cannot report back, the harness gets in the way.

A rejection explains why and lists who the session can message, so it can reroute through the main session on its own. Writing the harness file can also be delegated to a session through the `/session-flow:harness` skill, since sessions that already split the work know that structure best.

## What I took away

- Observability tools must fail without blocking the work they observe.
- No external dependencies means it runs unchanged in remote environments.
- When guessing paths start multiplying, check whether the system already offers a stable identifier.
- When a false block costs more than a missed violation, start enforcement loose.
