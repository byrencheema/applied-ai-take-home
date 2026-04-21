# Agent Orange, Design Write-up

## Overview

Agent Orange is a Slack based Q&A agent built on LangGraph. It answers questions about Northstar's customers from a local SQLite database of support tickets, customer calls, internal docs, internal comms, and competitor research. It runs locally, talks to Slack over Socket Mode, and ships with a small CLI for iteration and evals.

This doc walks through what I built, how I built it, what I threw away, and what I'd do next.

## Starting point

I began by designing a Slack agent on top of LangGraph. The very first thing I checked was whether the database could actually answer the example queries in the take-home, because if retrieval was broken everything downstream would be broken too. The artifacts table has an FTS5 full text index and both example questions resolved to the right artifact in a single search, so retrieval was fine.

Once retrieval was proven I knew the agent would need tools, or ways to access the data. I created four of them: a way to search artifacts, get a single artifact by id, list customers by metadata filters, and get a single customer profile. I wrote the unit tests alongside them. The SQLite connection is read only and thread local, so parallel subagent calls don't collide on a shared cursor.

## First agent, and why I wiped it

LangGraph ships a prebuilt react agent, so naturally that's what I tried first. I used GPT 4.1 mini, gave it memory with the built in parameters, and pointed it at the four tools. It underperformed on the example queries given in the instructions, so I instantly just wiped it. The failures were not in the tools, they were in how the model searched and planned. I knew the core thing would be building my own state graph, so that is what I did.

## Custom state graph

I built an agent with its own graph: one node bound to the tools, with conditional edges into a ToolNode and back, looping until the model stops calling tools and emits a final answer. I bumped the model from 4.1 mini to 4.1. That was enough to pass the easy queries.

The hard queries exposed a different problem. FTS5 uses AND semantics by default, so verbose natural language questions under retrieve. Even when I widened retrieval, BM25 ranking was dominated by the most heavily documented customers, so the right answer was in the corpus but it was buried under accounts that just happened to mention the same keywords more often. The fix went into `search_artifacts` itself. First, multi term queries get OR rewritten, so a natural language question retrieves candidates instead of demanding every term match. Second, the result set gets capped at two hits per customer on top of an over fetched, BM25 ranked pool. The technical term for that second piece is result diversification. You over fetch, then diversify by grouping, so the top hits spread across customers instead of collapsing onto the loudest one.

## Plan, research, answer

Retrieval was fixed, but the planning problem was still there. I rebuilt the graph as three nodes with shared state. The shape looks like this:

```
           +------+        +----------+        +--------+
 START --> | plan | -----> | research | -----> | answer | --> END
           +------+        +----------+        +--------+
                                ^
                                |
                                v
                           +--------+
                           | tools  |
                           +--------+
```

The `plan` node is a plain LLM call with no tools. It reads the question and emits a short numbered plan for how to investigate it. The `research` node is the tool using loop. It sees the plan and the question framed into every turn, calls `search_artifacts`, `get_artifact`, `list_customers`, or `get_customer`, gets the result back from the `tools` node, and keeps going until it stops calling tools. When it stops, control goes to the `answer` node, which is another LLM call that writes the final Slack ready response. The full database schema and content reference is embedded in both the plan and research prompts so the model knows what tables exist, what the enums are, and what each artifact type tends to contain.

This split matters because the three tasks fail in different ways. A single node with tools tends to skip to an answer before it has weighed alternatives, especially on superlative questions like "which customer is most likely to defect", where surface features look similar across at risk accounts. A separate planner forces the model to decide the shape of the investigation before it sees any evidence, so it does not narrow too early onto the most documented account. A separate answer node can be tuned for Slack voice and verbatim reproduction of numeric specifics without cluttering the research prompt with formatting rules.

One more thing that lives in the research prompt. When a question asks about a pattern across accounts, like "which customers share this failure mode", FTS alone is not enough. Different customers describe the same underlying failure with different words, so one cohort wide query silently misses members whose vocabulary diverges. The research node is told explicitly to enumerate the cohort with `list_customers` and then query each member's artifacts individually, rather than trusting a single aggregate search. That one rule fixed the last stubborn hard case.

## Evals

Once I had the three node flow I needed a way to measure it, so I built a small eval harness. Each case is a question plus a list of `must_include` keywords, with tuple entries meaning any of. Unicode dashes are normalized to ASCII so the model can answer with either. A concurrent runner fires all the cases through the agent in parallel, prints a pass fail table, and exits non zero on any miss. I started at seven cases covering the example queries, and grew it to eighteen as I added harder scenarios like NordFryst renewal thresholds, Laurentia schema rejection, Helix canonical_id, Harbourline SCIM SLAs, and a NoiseGuard six customer cohort. There is also an out of scope case that asks about the weather and checks the agent refuses without leaking any artifact or customer ids.

## Prompt work

A lot of the middle of this project was prompt iteration. The moves that held up across runs were keeping numeric specifics verbatim, naming identifiers, and per cohort iteration. If the evidence says "7 to 10 business days" or "45 percent to 60 percent", the answer has to say the same thing, not "about a week" or "up to 60 percent". That kind of smoothing is silent data loss. If the question is about an error code or a product, the answer has to use the exact identifier, like `SI-SCHEMA-REG` or `Signal Ingest`. I wrote the rules as categories rather than listing literal example values, so I wasn't teaching to the test.

A few things that seemed promising but did not hold up. I tried `parallel_tool_calls=True` in the research node. Wall time dropped by almost half, but the suite regressed with different failures each run, so it was either noise or a real problem and I did not have enough A/B runs to tell which. I reverted. I also tried an Anthropic style refactor where all three prompts became prose driven general principles instead of bulleted rules with concrete counterexamples. It regressed several cases. The generalized prose said the same things, but the model followed bulleted rules much better. I reverted that too. I raised the temperature to around 0.3 to 0.5, non zero so the agent keeps a little creativity on phrasing and candidate weighing. Zero temperature was not better in this range.

## Agent Orange

I gave the agent an identity called Agent Orange, partly for fun and partly so there is a name to point at that is not "Northstar Q&A bot". Identity lives in the answer prompt only. I tried putting it in the planner too, with out of scope routing skipping research when the question was off topic, and it regressed a superlative case twice, so I pulled back. Now off topic questions still flow through plan and research, which may fire tools uselessly, but the answer node catches it and refuses in one short sentence pointing the user back at what the bot does cover.

Once evals were holding consistently I added the ten harder cases. I needed to pass those going forward as a higher bar.

## Model choice, and the deep agents side experiment

As a side experiment I tried the deep agents variant from the LangChain framework. A single `create_deep_agent` with the same four tools, built in `write_todos`, filesystem, and subagent scaffolding. I ran it against all eighteen eval cases under a few model and prompt combinations. GPT 4.1 scored 13 of 18. GPT 5 scored 17 of 18 but averaged around 26 seconds per case with worst case over 90 seconds, and tool calls exploded on cohort questions. The persistent failure across every deep agents variant was the "most likely to defect" superlative, because the single loop design misses the candidate weighing guard that the plan, research, answer split gives. So it was comparable at best, and much slower. I kept my original approach, which gives a lot more control over the workflow.

I kept the model at GPT 4.1 for the main agent. Some might argue for GPT 5 or GPT 5 mini, but I wanted to stay on a non reasoning model because it is faster and lighter, and 4.1 performed well enough that I did not see a real lead for another model even after testing. No ceiling I was pushing against.

## Slack surface

Once the brain was stable I spent real time on the Slack experience.

I added a thumbs up and thumbs down button on every answer, and clicks append a JSON line to `data/feedback.jsonl` with timestamp, user, thread key, and vote. It is a very basic feedback loop but it means users can say "that was wrong" and it gets stored somewhere. I trimmed the status chatter so the Assistants panel shimmer does not become a 15 line scroll. I added multi turn threading so follow up replies see the prior human turns and the final answer turns from the same thread, which lets questions like "and for NordFryst?" plan with context. I added native 1 to 1 DM handling. I also added a `:eyes:` reaction on channel messages while the agent is running, so the agent can acknowledge the query. 

Then I spent a while on the status UX. Slack has a lot of built in agent affordances. There are plan mode task cards where you can show a plan title and task cards that tick from pending to in progress to complete. There is `assistant.threads.setStatus`, which shows the shimmering "Agent is doing X" text. There is `chat.startStream` and `chat.appendStream` for streaming the answer token by token and attaching citation sources. I tried almost all of it. Per node `set_status` updates. Streaming the answer via `appendStream`. Emitting `plan_update` and `task_update` chunks for a live research timeline with three task cards for Planning, Researching, Writing, and tool outputs contributing artifact ids as citations.

The streaming path kept erroring with `streaming_mode_mismatch`. Slack seems to lock a stream into either text mode or task card mode based on the initial `chat.startStream` call, and later chunks of the other shape fail. I added fallbacks, hit the error again on a different chunk, and eventually ripped the whole streaming and task card path out. Honestly, I also found it kind of cluttered and cumbersome to code, and it did not really add a lot of new information to the user over a single message that updates continuously.

What I landed on is deliberately simpler. I post a placeholder message immediately that says something like "Agent Orange is thinking", then I `chat.update` that same message in place on every graph transition. First it says it is planning. Then it shows a single line per tool call with the key argument inlined, for example `Searching artifacts "BlueHarbor taxonomy", type=customer_call` or `Looking up customer BlueHarbor`. Then it says it is writing the answer. When the answer is ready, one final `chat.update` swaps the whole block for the real answer plus the feedback buttons. The answer goes through the `slackify-markdown` library so the agent can write standard markdown in its prompt and the library converts it to Slack mrkdwn. That way I never had to prompt the agent to write in Slack's specific format, I just wrote normal markdown and let the converter do it.

The nice thing about this pattern is that it works identically in channels, in DMs, and in the Assistants panel, with no surface specific code paths. It collapses to a single clean message at the end. There are no task cards left hanging around. I also added a fallback so if anything goes wrong with the in place update path, we still get a message posted to the user no matter what. The tradeoff is that I do not get token by token streaming of the answer, so it appears all at once at the end. 

I also synthesized the answer prompt a bit more so it spaces paragraphs out with blank lines, which reads easier in Slack. I tried rendering each paragraph as its own section block at one point, but Slack rendered consecutive section blocks so far apart that answers looked like two separate messages, so I reverted that.

## Packaging, CLI, and logs

Finally I split everything up into a package. The Slack surface is its own package at `src/slack`, with submodules for the app wiring, the runner that handles events, the in place status message, the feedback button block, and the slackify markdown wrapper. I moved the scripts into `src/cli` and wired `pyproject.toml` `[project.scripts]` entries so they are just `uv run slack-agent`, `uv run ask`, and `uv run trace`. The `ask` and `trace` entry points were quite helpful throughout the process. They let Claude Code test the agent end to end and see the full trace. You can give it a prompt and read exactly what tools fired with what arguments and what came back, which made eval regressions much faster to diagnose than rerunning the whole suite. I also did some formatting on the logs so the runner output has colored levels and dim timestamps, and it auto disables colors when stderr is not a TTY or `NO_COLOR` is set.

Then I wrote the README for take home submission and did the setup stuff.

## Security model

Everything runs locally. Local SQLite, local process, Socket Mode WebSocket from Slack straight into the agent. There is no public webhook, no inbound authentication surface, no server to harden, so there are not a lot of concerns there.

That said, if someone sets this up and invites the bot to a shared workspace, anyone in that workspace could start talking to it. As a bare minimum safety feature I added a whitelist in the environment variables, the `SLACK_ALLOWED_USERS` env var, which takes a space or comma separated list of Slack user ids. If set, the agent silently ignores events from anyone not on the list. If unset, everyone is allowed, which is fine for local development. It is not a real access control system, it is a minimum level safety feature so that no one can just join your server and start interacting with the agent.

## What I would do next

- Multi run eval stability. The suite holds 18 of 18 on good runs and 16 of 18 on bad ones with no code changes, which is temperature variance on the stubborn cases. If I ran each case several times and reported pass rate instead of pass or fail, I could actually tell real regressions from noise, and I could reopen the parallel tool call experiment with confidence.
- Retrying streaming. The `streaming_mode_mismatch` path was not fully debugged, and with more care around the initial `chat.startStream` shape it might be possible to get the task card UI back without the reliability hit. Token streaming alone would help perceived latency on the longer answers.
- Real access control. `SLACK_ALLOWED_USERS` is a band aid. For any real deployment I would want per user rate limits, an audit log, and probably a per workspace OAuth install so multiple teams could safely use the same hosted instance.
- Feedback driven eval growth. `data/feedback.jsonl` is already collecting thumbs, but nothing is reading it. The natural next step is a job that pulls the thumbs down threads, replays the question with trace, and suggests new eval cases from the failures.
- Running this against a real customer support dataset. The synthetic DB is templated, which is why verbatim preservation and per cohort iteration are the winning moves. Real, noisy, inconsistently formatted data might need different prompts and probably a re ranker on the FTS output.
