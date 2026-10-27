---
title: "Jev, the AI that doesn't write: what decision models are and where to use them"
description: "A look at Jev, a decision model that returns probabilities over fixed options instead of text: how it works, the launch numbers, what independent tests found, and how to use it as a cheap first gate in front of an LLM."
date: 2026-10-16
lang: en
key: jev-decision-model
kind: review
tags: [Jev, Decision Models, LLM, LLM-as-judge, Calibration, Agents]
tistory: https://mynews76154.tistory.com/29
---

Once you build services on LLMs, you notice that a lot of the calls are not asking for writing but for a decision. Is this ticket a refund request? Did this answer stray from its sources? Does this agent step need a human to check it? To get those decisions we have the LLM generate text, ask it nicely for JSON, and write separate code to repair the JSON when it breaks.

Jev, released in mid-September, goes straight at that problem. It produces no text at all; it only returns probabilities over a fixed set of options. Within a month of its launch OpenAI, Databricks and Cloudflare shipped similar features one after another, and "decision models" is starting to look like a category of its own. Below is what the model is, how it did in independent tests, and where it is actually worth using.

## What Jev is

Jev was released on September 15, 2026 by TypeSafe, a San Francisco startup. Its founder, Diogo Almeida, is described as having worked on ChatGPT at OpenAI. The name comes from William Stanley Jevons, the economist behind the Jevons paradox: when something becomes more efficient, people end up consuming more of it. The implication seems to be that once decisions get cheap enough, we will make far more of them.

TypeSafe calls Jev a "System One model," borrowing Daniel Kahneman's split between fast intuition (System 1) and slow deliberation (System 2). If an LLM that thinks at length and writes is System 2, Jev is System 1: it just makes the call, quickly.

### Inputs and outputs

Instead of a prompt, you send Jev a **state** and a **list of questions**. There are only three question types:

- **Choice**: pick one of the given options. Example: is this ticket about a refund, shipping, or something else?
- **Score**: rate on a predefined scale. Example: how aggressive is this comment, from 1 to 5?
- **Noul**: the probability that a claim is true. Example: is this answer grounded in the provided document?

The response is a probability distribution for each question. Because Jev does not generate tokens one at a time and instead evaluates every question in a single parallel pass, off-schema answers and broken JSON are structurally impossible. The trade-off is that it gives no reasons or explanation at all.

![How an LLM and a decision model return output](/img/reviews/jev-1-en.png)

### Training and the numbers

TypeSafe calls its training method RLCD (Reinforcement Learning for Calibrated Decisions). It is reinforcement learning aimed not only at getting the answer right but at making the probabilities honest, so that when the model says 70%, it is right 70% of the time.

The company's published numbers:

- Latency of 70–500 ms
- $0.042 per million input tokens, with output free
- 68% average accuracy across four in-house workflows (security incident response, agent trace review, invoice processing, customer support). GPT-5.6 Terra also scored 68%, and Opus 5 scored 73%.
- On the same evaluation, cost per item was $0.0004 for Jev, $0.03 for Terra and $0.18 for Opus 5. Latency was 0.4 s, 10 s and 38 s respectively.

According to the Wall Street Journal, daily volume passed one trillion tokens within a week of launch, and the company is negotiating a funding round of more than $1 billion.

One caveat when reading these numbers: the "ground truth" in the in-house evaluation is not human labels but reference answers from another LLM. So 68% accuracy is closer to "how often it agreed with a frontier LLM." TypeSafe has also acknowledged that its own team built the evaluation tasks.

## How it did in independent tests

In the weeks after launch, a fair number of outside teams published hands-on results. The results are mixed, which actually makes the use cases clearer.

### Where it did well: simple, clear-cut classification

- **18,514 spam emails**: 98.3% zero-shot, with no training data. That is statistically indistinguishable from a TF-IDF logistic regression trained on about 14,800 examples (98.4%). The probabilities held up too: 99.9% of the emails it scored at 0.9 or above really were spam, and routing only the ambiguous band (0.3–0.7), 4.6% of emails, to a human left the rest 99.5% accurate.
- **Event listing review**: Jev 96% versus Gemini Flash-Lite 86%, at 58x lower cost per decision.

### Where it struggled: one question carrying the whole judgment

- **2,000 phishing emails**: asked "is this phishing?" in one shot, Jev scored 62.6% against 81.3% for Claude Haiku 4.5, a sizeable gap. But when the judgment was split into five questions (sender, links, urgency and so on) with a simple logistic regression on top trained on labeled data, the result flipped: Jev 95.0%, Haiku 93.2%. Cost per 1,000 emails was about $0.04 for Jev versus roughly $0.46–1 for Haiku.
- **Analytics anomaly detection**: it caught 100% of the meaningful anomalies (recall), but precision was 55.8%, so it flagged a lot of normal charts. If an expensive root-cause analysis step sits behind it, the cost of handling false positives can eat up what you saved on first-pass detection.

### Don't take the probabilities at face value

"Calibrated probabilities" is the core claim, and it drew the most pushback.

- On bank inquiry classification (BANKING77), mean confidence was 87.4% while actual accuracy was 75.3%, roughly 12 points of overconfidence.
- On questions designed to be unanswerable, it still answered with an average confidence of 0.74, while its actual accuracy was 44.7%.
- The bias differed by question type: yes/no questions were underconfident, while choice and score questions were overconfident.

Several tests converged on the same advice: recalibrate probabilities per question rather than per model, always include a "none of the above" option, and automate only the high-confidence band. The company's "no hallucinations" claim really means "no answers outside the schema." Picking the wrong option with high confidence is entirely possible.

## Big tech followed right away

In a little over two weeks after Jev's launch, similar products arrived one after another.

| Product | What it is |
| --- | --- |
| OpenAI Decisions API | Built on the Luna model. Accepts image input and returns classifications and next-action choices. |
| Databricks `ai_decide` | A decision function announced the day after OpenAI's, aimed at use directly inside the data platform. |
| Cloudflare Clef / Clef-flash | Qwen-based 27B and 9B models released under Apache 2.0. They use the same request format as Jev, so migrating is a matter of swapping the endpoint, and they accept images. |
| Amazon Strands Decider2B | Amazon had already shown its own decision model. |

Cloudflare Clef's architecture is a good illustration of how decision models are built. A regular LLM reads the input and questions in one pass; instead of generating text, an added output head scores each option, and a softmax turns those scores into probabilities. With no generation it is fast, and it cannot step outside the fixed options. Pricing is $0.24 per million input tokens for Clef and $0.09 for Clef-flash, more than Jev, but the weights are open so you can host it yourself.

## Where to use it

Putting the material together, a decision model is less a replacement for an LLM than a first gate in front of one. Roughly:

![A cascade with a decision model as the first gate](/img/reviews/jev-2-en.png)

- The decision model sees every request first. At around $0.0004 per item, you can afford to check everything.
- The high-confidence band is handled automatically.
- Only the ambiguous band goes to an LLM or a human. In the spam case that was 4.6%.

Here is how the good and bad fits break down:

| Good fit | Poor fit |
| --- | --- |
| Ticket and inquiry classification and routing | Summaries, writing replies, code generation |
| Content moderation, first-pass spam and phishing filters | Anything that needs an explanation of why (audits, regulatory work) |
| LLM output guardrails, full scans of agent traces | Open-ended questions where you can't fix the options in advance |
| Row-level labeling of large datasets | Complex judgments that need multi-step reasoning |

I think this matters most for teams using LLM-as-judge. Today, checking whether a RAG answer matches its sources or whether an agent did something odd means using an expensive LLM as the judge, so teams only sample. Scan everything with a decision model, have the LLM judge re-check only what looks like a failure, and you can cut cost substantially while actually widening coverage.

## How I would validate it before adopting it

My own version of the recommendations from the various tests:

1. Prepare 1,000–2,000 records where a human made the final call. Don't use another LLM's answers as ground truth.
2. Don't pack the judgment into one question; split it into several small ones. As the phishing case shows, accuracy can change dramatically.
3. Add "none of the above" to every choice question.
4. Recalibrate probabilities per question, then measure what share of traffic falls in the confidence band that meets your target accuracy. That share is your real savings.
5. Compare against the small LLM or fine-tuned classifier you use today on cost per correct answer, not per-token price.
6. Test prompt injection too: plant malicious instructions in the input and see whether the answer changes.

## Applying it to my own project

In the diary app I'm building as a side project, an LLM analyzes the emotions in each entry and writes a short action guide. Parts of that, like "what is the main emotion?" and "how intense is it?", are really decisions within a fixed set of options, which is exactly what a decision model is for. The guide itself is writing, so an LLM still has to produce it. Splitting it so that a decision model does the emotion classification cheaply and quickly, and the LLM only writes the guide from that result, looks like it would cut both call cost and response time. That said, labels with fuzzy boundaries like emotions will probably be harder to calibrate, so I would run the validation steps above first.

I don't know whether Jev itself will last. Big tech shipped similar features within two weeks, and open-weight models appeared right away. But the point it raises, that not every decision needs to come back as prose, is likely to stick. Even more so for agents that make dozens of decisions within a single task.

If you have tried it, I'd love to hear in the comments what you used it for and how it went.

## References

- [System One Models: Jev (DataCamp)](https://www.datacamp.com/zh/blog/system-one-models-jev)
- [Jev, un nouveau modèle d'IA conçu pour prendre des décisions rapides et structurées (BDM)](https://www.blogdumoderateur.com/jev-modele-ia/)
- [TypeSafe AI 决策模型 Jev 日处理量达 1 万亿 Token (2026.10.05, citing the WSJ report)](https://news.10jqka.com.cn/20261005/c680439311.shtml)
- [TypeSafe's Jev: Can decision models replace LLM judges? (Arize, 2026.09)](https://arize.com/blog/typesafe-jev-llm-judge/)
- [TypeSafe's Jev Scores 62.6% Asked Once and 95% Split Five Ways (2026.09)](https://www.beri.net/article/typesafe-jev-typed-decision-model-calibration-decomposition-shadow-eval)
- [An analysis of Jev: Faster and cheaper, but watch the accuracy (Amplitude, 2026.09)](https://amplitude.com/blog/jev-analysis)
- [A deep dive into Clef, Cloudflare's decision model (2026.10)](https://flaviocopes.com/clef/)
