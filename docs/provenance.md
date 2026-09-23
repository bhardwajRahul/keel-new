# Source and architecture provenance

The supplied `keel.zip` was inspected as a prototype; it did not contain the full Avid coding interface. Keel uses the Avid-derived Rust/GPUI source as its working base, retaining the shell, composer, transcript, terminal, settings, and local agent adapters. No installed Avid app or data was changed.

The decision interface follows the prior host-owned architecture: versioned state, host-prepared candidate IDs, selector choice, fresh eligibility check, and existing permissions. Laya is the local default. Direct TypeSafe Jev is an opt-in selector for eligible new-task routes, bounded embedded DeepSeek steps, and explicit computer-use decision calls, using an existing protected credential; Keel has no key-entry UI. Neither model writes commands or patches. External ACP agents retain their own internal loops, so their tool use is not described as decision-first.
