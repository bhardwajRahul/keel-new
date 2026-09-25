# documentation

Start with the page that matches what you need to do.

| I want to… | Read |
| --- | --- |
| Set up Keel and choose a mode | [User guide](guide.md) |
| Compile or package the app | [Build from source](build.md) |
| Navigate the source and run checks | [Developer guide](development.md) |
| Understand who controls each action | [Decision architecture](decision-architecture.md) |
| Connect an installed coding agent | [Local connections](connections.md) |
| Follow the first-run flow | [Onboarding](onboarding.md) |
| Inspect the recorded build evidence | [Build report](archive/build-report-0.2.0.md) |
| Understand the proposed evaluation loop | [Improvement loop](proposals/improvement-loop.md) |
| Find the article's source files | [Article sources](archive/article-sources-0.2.0.md) |
| Check the source's origin | [Provenance](provenance.md) |

## diagrams

These are native, editable SVG files. They contain text and vector shapes, with no embedded screenshots or external fonts.

- [Readme hero](diagrams/hero.svg): the task flow in one banner. This file is hand-written; the script below does not generate it.
- [Architecture](diagrams/architecture.svg): selection, host checks, and the two execution paths.
- [Decision modes](diagrams/decision-modes.svg): local Laya, hosted Jev, and Normal mode.
- [Proposed improvement loop](proposals/improvement-loop.svg): records, replay, comparison, and human review.

The architecture and mode diagrams describe this build. The improvement loop is **a proposal**, not an automatic learning feature.

Current guides live directly in `docs/`. Dated publication and build records
live in `archive/`; unimplemented designs live in `proposals/`.

Regenerate the SVG files from the repository root after changing their source:

```sh
python3 tools/docs/generate-diagrams.py
```

[Back to the readme](../README.md)
