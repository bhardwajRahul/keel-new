//! Durable projection state for dynamic runtime context, ported from
//! `packages/core/agent-loop/src/runtime-context.ts`: tracks the last
//! retained runtime-context snapshot in the log and proposes an uncommitted
//! replacement only when the rendered value actually changed.

use dsh_llm::{
    ContentBlock, ContextForm, ContextSnapshotSection, Message, MessageSource, create_user_message,
};
use dsh_session::{Session, SessionEventData, is_replacement_surface_event};
use std::cell::RefCell;
use std::rc::Rc;

const SOURCE: &str = "dsh-system-prompt";
const CLEARED: &str =
    "Current runtime context: none. Earlier runtime-context snapshots no longer apply.";

fn is_owned(message: &Message) -> bool {
    matches!(&message.source, MessageSource::Plugin { plugin, .. } if plugin == SOURCE)
}

fn text_of(message: &Message) -> Option<String> {
    match message.content.as_slice() {
        [ContentBlock::Text { text }] => Some(text.clone()),
        _ => None,
    }
}

#[derive(Clone)]
enum Retained {
    /// No snapshot ever existed.
    Never,
    /// A snapshot existed but none is currently retained on the surface.
    None,
    /// The retained snapshot: its seq and single-text content.
    Some { seq: u64, text: Option<String> },
}

/// Tracks the last retained runtime-context snapshot without owning its
/// commit; restored once from the log, then advanced per appended event.
pub struct RuntimeContextProjection {
    session: Rc<Session>,
    retained: RefCell<Retained>,
    /// Log position the projection has consumed.
    consumed: RefCell<usize>,
}

impl RuntimeContextProjection {
    pub fn new(session: Rc<Session>) -> RuntimeContextProjection {
        let mut retained = Retained::Never;
        let surface: std::collections::HashSet<u64> = session.surface_nodes().into_iter().collect();
        let events = session.events();
        for event in events.iter().rev() {
            let SessionEventData::UserMessage(message) = &event.data else {
                continue;
            };
            if !is_owned(message) {
                continue;
            }
            if matches!(retained, Retained::Never) {
                retained = Retained::None;
            }
            if surface.contains(&event.seq) {
                retained = Retained::Some {
                    seq: event.seq,
                    text: text_of(message),
                };
                break;
            }
        }
        let consumed = events.len();
        RuntimeContextProjection {
            session,
            retained: RefCell::new(retained),
            consumed: RefCell::new(consumed),
        }
    }

    /// Follow authoritative session events appended since the last call: an
    /// owned user message becomes the retained snapshot; a surface
    /// replacement shadowing it clears retention.
    fn catch_up(&self) {
        let events = self.session.events();
        let mut consumed = self.consumed.borrow_mut();
        for event in events.iter().skip(*consumed) {
            if let SessionEventData::UserMessage(message) = &event.data {
                if is_owned(message) {
                    *self.retained.borrow_mut() = Retained::Some {
                        seq: event.seq,
                        text: text_of(message),
                    };
                    continue;
                }
            }
            let retained_seq = match &*self.retained.borrow() {
                Retained::Some { seq, .. } => Some(*seq),
                _ => None,
            };
            if let Some(seq) = retained_seq {
                if is_replacement_surface_event(event)
                    && event
                        .source_event_seqs
                        .as_ref()
                        .map(|sources| sources.contains(&seq))
                        .unwrap_or(false)
                {
                    *self.retained.borrow_mut() = Retained::None;
                }
            }
        }
        *consumed = events.len();
    }

    /// Create an uncommitted snapshot message only when the retained value
    /// differs from `current`; an empty `current` after a previous snapshot
    /// proposes the explicit cleared marker.
    pub fn project(&self, current: &str, sections: &[ContextSnapshotSection]) -> Option<Message> {
        self.catch_up();
        let retained = self.retained.borrow().clone();
        if matches!(retained, Retained::Never) && current.is_empty() {
            return None;
        }
        let snapshot = if current.is_empty() { CLEARED } else { current };
        if let Retained::Some {
            text: Some(text), ..
        } = &retained
        {
            if text == snapshot {
                return None;
            }
        }
        let form = if sections.is_empty() {
            None
        } else {
            Some(ContextForm::Snapshot {
                sections: sections.to_vec(),
            })
        };
        Some(create_user_message(
            vec![ContentBlock::Text {
                text: snapshot.to_string(),
            }],
            MessageSource::Plugin {
                plugin: SOURCE.to_string(),
                form,
            },
        ))
    }
}
