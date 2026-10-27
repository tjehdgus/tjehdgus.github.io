---
title: "REA: an MCP that hands coding agents a disassembler, and what people found when they tried it"
description: "A look at REA (Reverse Engineer Anything), the repo that took over GitHub in October: how it is built, based on its source, and what official case studies and independent tests show it does well and where it falls short."
date: 2026-10-27
lang: en
key: rea-reverse-engineer-anything
kind: review
tags: [REA, MCP, Reverse Engineering, AI Agents, Claude Code]
---

One repository more or less owned GitHub Trending in October: **REA (Reverse Engineer Anything)**. Its pitch is "see a feature you like, understand how it works down to the binary level," and it collected stars by the hundred thousand in little over a week. The reaction online went as far as "software is done."

Tools like this usually have a gap between the pitch and the results. This review covers three things: what REA actually does (read from its source), what the official case studies and independent testers got when they ran it, and what to watch out for.

## 1. How fast it took off

The repository was created in April 2026 and stayed quiet until early October. Citing Trendshift data, one analysis puts daily new stars at 1,373 on October 4, 19,375 on October 9 and 31,255 on October 11. By October 12 it had 115K stars and 25K forks, and it was the top GitHub Trending repository on five of the seven days from October 6 to 12. As of October 27 it shows 395K stars and 98.9K forks.

The release pace is unusual too. It went from 3.2 on October 3 to 6.4 on October 11, three major versions in under ten days, and the latest npm release at the time of writing is 6.8.0. Version 6.2 changed the structure of every MCP result, which broke scripts that relied on the JSON output. Fixes land quickly, but the project is clearly not stable yet.

## 2. What REA does: a connection layer, not a new decompiler

REA can look like an AI decompiler, but it **does not ship a new analysis engine**. It wraps existing reverse-engineering tools (disassemblers such as Hopper, Ghidra and IDA, plus JADX, Binwalk and others) in an MCP server so that coding agents like Claude Code, Codex and Cursor can call them directly.

![REA as an MCP layer between agents and analysis tools](/img/reviews/rea-arch-en.png)

The source is well organised:

- **Two entry points.** The MCP server (`src/main.ts`) and the CLI (`src/cli.ts`) share the same application workflows, which is why you can run `rea analyze-javascript-application` from a terminal without any agent.
- **A lot of tools.** `src/contracts` defines more than 150 tool contracts, covering binaries, Electron, .NET, APKs, firmware, GPU code objects, GGUF and safetensors model files, SQLite and network captures.
- **Native analysis is delegated.** Ghidra is driven through a headless script (`bridge/ghidra/ReaGhidraBridge.java`) and Hopper through a Python bridge running inside Hopper. Without an engine, native analysis stops with `provider_unavailable`.
- **JS/Electron works without an engine.** A Babel-based parser reads bundles and builds a graph of modules, imports, source maps, APIs exposed through contextBridge and IPC channels. That is why most testers evaluated this part.

In short, what REA adds is not analysis capability but **a consistent, safer way for agents to use analysis tools**. The core of that is the result format.

## 3. The result format is the real point

Handing raw decompiler output to an agent causes two problems: the agent states guesses as facts, and it can be steered by text planted inside the target. REA tries to address both through the shape of its results.

![How REA returns results](/img/reviews/rea-evidence-en.png)

### Observation, derivation, inference, unknown

The first rule in the repository's contributor guide (AGENTS.md) is to distinguish observations, derivations, inferences and unknowns. What a tool read directly, what was computed from it, and what involves interpretation are kept apart. Anything that could not be established stays "unknown" instead of being treated as absent. Every result produces an Evidence record with an `evidence_id`, kept in the session bundle and reusable by comparison tools.

### Text from the target is untrusted

Strings in a binary, decompiled code and page text may have been written by an attacker. If an app contains a string like "to continue this analysis, run the following command," a naive agent may treat it as an instruction. REA marks results from tools that can return target content with `untrusted_content: true`, and its server instructions tell the agent not to follow directives in tool results and to report them as prompt-injection attempts. Marking is the default; only audited tools are on the exception list.

This is worth noting even if you never reverse-engineer anything. It is exactly the problem of how a RAG or web-search agent should treat retrieved text.

## 4. What people found

### Official case 1: DX-Ball sound panning

The project recovered how DX-Ball, a 1996 Windows game, turns the position of a hit brick into a left/right sound pan value. The decompiled function was missing its input, so the instruction view was used to find the stack input and floating-point operations; following the caller showed the input is the screen coordinate `20 + 30 × tile_x`, and the constants 1.5625 and 500.0 were read from memory.

The reconstructed C function passed 3,205 cases compared against the original x86 code, and compiling it with the period compiler (VC4.0) reproduced all 63 bytes of the original function. The rest of the game, including ball and paddle physics and power-ups, is still unfinished. It is a case of recovering one function exactly, not of cloning a game.

### Official case 2: the Notion desktop clipboard path

In Notion Desktop 7.6.1, an Electron app, the project traced how a page's copy action reaches the system clipboard. `analyze_javascript_application` located the exposed API and the `ipcRenderer.invoke` call in the preload script; the `notion:clipboard:write` channel led to `ipcMain.handle` in the main process. It also confirmed that the main handler checks the sending window before writing, and that copied HTML carries block-ID comments used to restore structure on paste. The case page itself notes that the receiver link was confirmed by reading the wrapper implementation directly, so part of the path was stitched together by a person.

### Independent test 1: a 100 MB Electron app (HackerNoon)

A tester ran REA 6.4.0 through the CLI and a raw MCP client, without an agent, on PI-Desktop (a 100 MB open-source Electron app) and on a five-file toy app.

- **Worked:** 421 JS files parsed in 52.2 seconds with zero failures. Counts of browser windows (5), contextBridge APIs (2) and IPC handlers/sends/listeners (4/6/5) matched grep. On the toy app it followed a channel from preload to handler even in minified code.
- **Failed:** it missed both preload files in the real app. 365 files exceeded the result budget, so some links were dropped, and a channel that is plainly in the source came back as "no match." It also missed sender validation wrapped in a helper.
- **Cost:** the MCP `tools/list` response was about 1.95 MB (170 tools), which the tester estimated at roughly half a million tokens. The compact schema profile cut that by only about 9%. The analysis JSON was 59.2 MB.

### Independent test 2: Electron API Demos (SystemDesign Academy)

This test used Electron's official demo app, whose source is public, and checked every answer against the code. It is the cleanest evaluation so far.

| Check | Result |
|---|---|
| File and line accuracy | 30/30 correct |
| IPC channels found | 18 (text search found 17; REA alone caught one registered via an alias) |
| Reply-to-window locations | 0/8 |
| Window count | 1/8 (missed 7 created through the older `remote` API) |
| `nodeIntegration: true` settings | 0 of 4 reported |

The conclusion is the interesting part: **everything REA reported was correct**. The problem is what it did not report. An agent working only from this output could conclude that the main process never replies to windows, or that there is nothing wrong with the security settings. REA's own output says that "no match" does not prove absence, but whether an agent respects that is another matter.

### Comparison: does it beat a plain agent? (MindStudio)

An informal test rebuilt macOS Preview twice, once with an agent using REA and once with a plain agent working from screenshots. Both produced working clones with zoom, drawing, annotation and saving, and the plain agent finished first. The REA version matched the original layout and feature set more closely, but the gap was not dramatic. The author suggests the difference may matter more on complex software.

## 5. Putting the cases together

| Works well | Still weak |
|---|---|
| **Narrow questions with a way to verify the answer**, such as one function or one data path | Broad goals like "clone this whole app" |
| Accuracy of what it does report | Noticing what it missed |
| Tracing IPC and exposed APIs in minified, bundled JS | Code paths routed through helpers, aliases or legacy APIs |
| Alias detection beyond plain text search | Summarising risky settings such as security flags |

The successful official cases share a pattern: a specific question (how does this function compute panning?) and a verification harness (3,205 cases against the original). The broad "software is done" claims are not supported by the evidence published so far.

## 6. Things to watch out for

- **Missing is not absent.** If REA finds nothing, that does not mean the feature or the risk is not there. For security reviews, treat its output as a starting point.
- **Measure the context cost first.** One measurement found the tool list alone takes a large share of context. Check token usage in your own client.
- **It is not a sandbox.** The repository's security policy states that the local bridge is not a sandbox; analysing an untrusted binary runs parsers with your user permissions.
- **Install only from official sources.** Lookalike repositories appeared as it went viral, and the README warns that only GitHub `morluto/rea`, `rea.tools` and the npm package `rea-agents` are official.
- **Analysis is local, results are not.** The analysis runs on your machine, but its results go to your model provider through the agent.
- **Legal limits.** The project states it is for lawful research, education, interoperability and analysis of software you own or are authorised to examine. Independent developers worry about it being used to bypass paid features or clone apps, and many commercial licences prohibit reverse engineering.

## Closing thoughts

REA is less "AI that clones any app" and more **a well-designed layer that lets agents use reverse-engineering tools with evidence attached**. The case studies agree that it is quite accurate on narrow questions, while broad goals and missed findings still need a human.

What I would take away is the result format. Separating observations from inferences, keeping unknowns explicit and marking external text as untrusted are designs that carry over directly to RAG and agent services. It is the same problem as adding a verification step that filters unsupported citations, solved carefully in a different domain.

## References

- GitHub: [morluto/rea](https://github.com/morluto/rea) (README, AGENTS.md, docs/mcp-contracts.md)
- Official case studies: [DX-Ball](https://rea.tools/showcase/dx-ball/), [Notion](https://rea.tools/showcase/notion/)
- InfoQ: [Reverse Engineer Anything Project Brings AI Agents to Binary Analysis](https://www.infoq.com/news/2026/10/reverse-engineer-anything/)
- HackerNoon: [I Tested REA on a Real Electron App](https://hackernoon.com/i-tested-rea-githubs-viral-reverse-engineering-tool-on-a-real-electron-app)
- SystemDesign Academy: [REA Reverse Engineer Anything: what it is, a real test](https://www.systemdesign.academy/explained/rea-reverse-engineer-anything)
- MindStudio: [Reverse Engineer Anything MCP](https://www.mindstudio.ai/blog/reverse-engineer-anything-mcp/)
- DEV Community: [What Is REA?](https://dev.to/hassann/what-is-rea-reverse-engineer-anything-the-mcp-server-that-hit-115k-github-stars-in-a-week-1coc) (star growth and release dates)
- AgentConn: [REA Hit 248K Stars. The Real Story Is Scarier.](https://agentconn.com/blog/rea-reverse-engineering-agent/)
