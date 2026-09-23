# decision architecture

[Documentation](README.md) / architecture

**Keel owns the options, state checks, and permissions. The selector returns a choice or abstains.**

A coding model's name in the picker does not give Keel control over that provider's internal loop.

![Keel architecture: prepare routes, choose one backend, validate the result, then use the embedded DeepSeek path or a provider-owned ACP loop.](diagrams/architecture.svg)

## 1. prepare an eligible route

For a fresh unpinned task, Keel builds candidates from installed, enabled providers and available models. It preserves explicit reasoning and provider options.

Pinned routes, live tasks, and resumed sessions keep their existing route. Provider authentication can still fail when a session starts.

Read [`jev_routing.rs`](../crates/engine/src/jev_routing.rs) for the route boundary.

## 2. ask the selected backend

| Backend | Behavior |
| --- | --- |
| Local Laya | Default when ready; uses the pinned Core ML worker and checkpoint. |
| Hosted Jev | Explicit opt-in; calls TypeSafe with an existing protected credential. |
| Normal | Bypasses decision selection. |

Laya and Jev use the host-authored `SelectionInput` contract: bounded current state and opaque eligible candidate IDs. An ID refers to an option the host already prepared.

They are different models. The host uses a smaller Laya shortlist and backend-specific abstention thresholds. Inference speed alone does not establish better coding outcomes.

Read [`decision_mode.rs`](../crates/engine/src/decision_mode.rs), [`laya-local`](../crates/laya-local), and [`jev-core`](../crates/jev-core).

## 3. check the result again

A selector result never grants permission.

Host-prepared actions include an ID, stored payload, task revision, read-set fingerprint, preconditions, expiry, and authorization result. The route path checks freshness before applying a choice.

| Result | Host response |
| --- | --- |
| Current, eligible route | Apply the checked route. |
| Abstention or unusable selection | Use the defined fallback. |
| Route no longer valid | Reject the stale choice and fall back. |
| Tool needs permission | Keep the existing approval path; selection does not bypass it. |

Only a verified execution outcome advances a dependency. ACP tool permissions and Cursor plan approvals still require explicit user answers.

## 4. use the correct execution boundary

### embedded DeepSeek

The selected backend can choose among four focus IDs: `inspect`, `implement`, `verify`, and `answer`.

1. The host builds candidates from registered tools.
2. The selector chooses a focus.
3. The host prepares and advertises that tool bundle against current schemas.
4. The coding model proposes output or a tool call.
5. The host checks the named tool again before dispatch.

`answer` has no tools. Calls outside the admitted bundle are rejected.

Read the embedded [`agent.rs`](../crates/dsh/agent-loop/src/agent.rs).

### external ACP agents

Codex, Claude Code, Cursor, Grok, Hermes, Pi, and other external agents retain their internal tool loops.

Keel inventories advertised commands, skills, modes, and tools. **An advertised capability is not an executable host action.**

For example, ACP `availableCommands` is metadata. Keel currently sends `/name` as prompt text; it does not have a generic command-execute RPC. A listed `/compact` command therefore does not become a selector action automatically.

Read [local connections](connections.md) for provider setup.

## 5. record evidence

Each completed decision gets a separate versioned chat record containing bounded information:

- Candidate summaries and the typed result.
- Confidence, probability, or fit when the backend supplies it.
- Host validation and fallback.
- The observed outcome, including tool dispatch, denial, or failure counts where available.

A temporary Laya/Jev activity label appears only during a live selector call. The record does not invent natural-language model reasoning.

## computer-use boundary

`keel computer-use decide` accepts a fresh, redacted observation and host-prepared, low-risk reversible candidates. Local Laya or optional direct TypeSafe Jev returns an ID or abstains.

**The command does not execute CUA actions.** Installed Codex Computer Use MCP tools may be attached to compatible ACP sessions, whose internal loops remain provider-owned.

## what remains outside this design

Keel does not run a selector for compaction, completion, or handoff when it has no verified executor for that operation.

Controlling an external loop end to end would require an interception point for every proposed action: inspect current state, select, validate, execute the stored action, then observe again. The current ACP adapters do not expose that complete boundary.

Automatic training is also absent. See the separately labeled [improvement-loop proposal](improvement-loop.md).

## model references

- [Laya model](https://github.com/NandhaKishorM/laya)
- [Laya Core ML runtime](https://github.com/mizorewww/laya-coreml)
- [TypeSafe Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev)
