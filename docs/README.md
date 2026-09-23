# documentation

Start with the page that matches what you need to do.

| I want to… | Read |
| --- | --- |
| Set up Keel and choose a mode | [User guide](guide.md) |
| Compile or package the app | [Build from source](build.md) |
| Understand who controls each action | [Decision architecture](decision-architecture.md) |
| Connect an installed coding agent | [Local connections](connections.md) |
| Follow the first-run flow | [Onboarding](onboarding.md) |
| Inspect the recorded build evidence | [Build report](build-report.md) |
| Understand the proposed evaluation loop | [Improvement loop](improvement-loop.md) |
| Find the article's source files | [Article sources](article-sources.md) |
| Check the source's origin | [Provenance](provenance.md) |

## diagrams

These are native, editable SVG files. They contain text and vector shapes, with no embedded screenshots or external fonts.

- [Architecture](diagrams/architecture.svg): selection, host checks, and the two execution paths.
- [Decision modes](diagrams/decision-modes.svg): local Laya, hosted Jev, and Normal mode.
- [Proposed improvement loop](diagrams/improvement-loop.svg): records, replay, comparison, and human review.

The architecture and mode diagrams describe this build. The improvement loop is **a proposal**, not an automatic learning feature.

Regenerate the SVG files after changing their source:

```sh
python3 scripts/generate-doc-diagrams.py
```

[Back to the readme](../README.md)
