//! Incremental chunk-to-message assembler, ported from
//! `packages/llm/llm/src/assembler.ts` — the single canonical assembly
//! algorithm the agent loop uses to build an assistant message from a chunk
//! stream while logging raw chunks for replay fidelity.

use crate::brand::CallId;
use crate::message::{Message, MessageSource, Role, create_message};
use crate::types::{BlockType, ContentBlock, FinishReason, StreamChunk, TokenUsage};
use std::collections::HashMap;

struct PartialBlock {
    block_type: BlockType,
    text: String,
    tool_call_id: Option<CallId>,
    tool_call_name: Option<String>,
    tool_call_arguments: String,
    /// Set by `BlockEnd` — authoritative, and freezes the partial.
    block: Option<ContentBlock>,
}

impl PartialBlock {
    fn new(block_type: BlockType) -> Self {
        PartialBlock {
            block_type,
            text: String::new(),
            tool_call_id: None,
            tool_call_name: None,
            tool_call_arguments: String::new(),
            block: None,
        }
    }
}

/// Incrementally assembles raw [`StreamChunk`]s into complete
/// [`ContentBlock`]s and a final assistant [`Message`].
///
/// Tolerant of delta-only protocols (no block-start/end); deltas arriving for
/// an index already closed by `BlockEnd` are ignored (malformed stream) so a
/// misbehaving adapter cannot grow memory or corrupt a completed block.
#[derive(Default)]
pub struct BlockAssembler {
    partials: HashMap<u64, PartialBlock>,
    order: Vec<u64>,
    usage: Option<TokenUsage>,
    finish: Option<FinishReason>,
    replay_state: Option<serde_json::Value>,
}

impl BlockAssembler {
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed one chunk into the assembly state, in stream order.
    pub fn push(&mut self, chunk: &StreamChunk) {
        match chunk {
            StreamChunk::BlockStart { index, block_type } => {
                if !self.partials.contains_key(index) {
                    self.order.push(*index);
                    self.partials.insert(*index, PartialBlock::new(*block_type));
                }
            }
            StreamChunk::TextDelta { index, text } => {
                let partial = self.ensure(*index, BlockType::Text);
                if partial.block.is_none() {
                    partial.text.push_str(text);
                }
            }
            StreamChunk::ReasoningDelta { index, text } => {
                let partial = self.ensure(*index, BlockType::Reasoning);
                if partial.block.is_none() {
                    partial.text.push_str(text);
                }
            }
            StreamChunk::ToolCallDelta {
                index,
                id,
                name,
                arguments_delta,
            } => {
                let partial = self.ensure(*index, BlockType::ToolCall);
                if partial.block.is_none() {
                    partial.tool_call_id = Some(id.clone());
                    if let Some(name) = name {
                        if !name.is_empty() {
                            partial.tool_call_name = Some(name.clone());
                        }
                    }
                    partial.tool_call_arguments.push_str(arguments_delta);
                }
            }
            StreamChunk::BlockEnd { index, block } => {
                let partial = self.ensure(*index, block.block_type());
                // First close wins; ignoring re-close stragglers keeps
                // streamed output and the assembled block in agreement.
                if partial.block.is_none() {
                    partial.block = Some(block.clone());
                }
            }
            StreamChunk::Usage { usage } => {
                self.usage = Some(*usage);
            }
            StreamChunk::Finish {
                reason,
                replay_state,
            } => {
                self.finish = Some(reason.clone());
                self.replay_state = replay_state.clone();
            }
        }
    }

    fn ensure(&mut self, index: u64, block_type: BlockType) -> &mut PartialBlock {
        if !self.partials.contains_key(&index) {
            self.order.push(index);
            self.partials.insert(index, PartialBlock::new(block_type));
        }
        self.partials.get_mut(&index).expect("partial just ensured")
    }

    fn assemble(partial: &PartialBlock, index: u64) -> ContentBlock {
        if let Some(block) = &partial.block {
            return block.clone();
        }
        match partial.block_type {
            BlockType::Text => ContentBlock::Text {
                text: partial.text.clone(),
            },
            BlockType::Reasoning => ContentBlock::Reasoning {
                text: partial.text.clone(),
            },
            BlockType::ToolCall => ContentBlock::ToolCall {
                id: partial
                    .tool_call_id
                    .clone()
                    .unwrap_or_else(|| CallId::new(format!("call-{index}"))),
                name: partial.tool_call_name.clone().unwrap_or_default(),
                arguments: partial.tool_call_arguments.clone(),
            },
            other => panic!("cannot assemble incomplete block of type {other:?}"),
        }
    }

    /// Assemble all blocks seen so far, in stream order. Max-token truncation
    /// drops tool calls that cannot be executed safely; an open block
    /// assembles from its accumulated deltas.
    pub fn blocks(&self) -> Vec<ContentBlock> {
        let blocks: Vec<ContentBlock> = self
            .order
            .iter()
            .map(|index| {
                let partial = self
                    .partials
                    .get(index)
                    .expect("ordered index has a partial");
                Self::assemble(partial, *index)
            })
            .collect();
        if matches!(self.finish(), FinishReason::MaxTokens) {
            blocks
                .into_iter()
                .filter(|block| !matches!(block, ContentBlock::ToolCall { .. }))
                .collect()
        } else {
            blocks
        }
    }

    /// Usage from the `Usage` chunk; `None` until one arrives.
    pub fn usage(&self) -> Option<TokenUsage> {
        self.usage
    }

    /// Finish reason from the `Finish` chunk; `Stop` when the stream ended
    /// without one.
    pub fn finish(&self) -> FinishReason {
        self.finish.clone().unwrap_or(FinishReason::Stop)
    }

    /// Adapter-private replay state from the terminal finish chunk, if any.
    pub fn replay_state(&self) -> Option<&serde_json::Value> {
        self.replay_state.as_ref()
    }

    /// The assembled assistant message over [`BlockAssembler::blocks`].
    pub fn message(&self, source: Option<MessageSource>) -> Message {
        create_message(
            Role::Assistant,
            self.blocks(),
            source.unwrap_or(MessageSource::Plugin {
                plugin: "dsh-llm/assembler".to_string(),
                form: None,
            }),
        )
    }
}
