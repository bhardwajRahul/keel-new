//! SSE decoding for the DeepSeek stream, ported from
//! `packages/llm/llm-deepseek/src/sse.ts`.
//!
//! Divergence: upstream delegates framing to the `eventsource-parser`
//! library; this port implements the SSE framing rules it relies on directly
//! (chunk reassembly, CRLF/BOM handling, comment and non-data field skipping,
//! multi-`data:` joining, dispatch only on the blank-line terminator — an
//! unterminated tail at EOF is truncation, not a flushable payload).
//! The literal `[DONE]` payload is yielded so the caller owns final flushing;
//! EOF before it is a truncated response the model call cannot trust.

/// The terminal payload DeepSeek (and OpenAI) send after the last chunk.
pub const DONE: &str = "[DONE]";

/// Incremental SSE frame decoder: feed raw byte chunks, collect dispatched
/// `data` payloads. Comments (`:`-prefixed lines) are reported through the
/// comment hook only — transport activity, never payload.
pub struct SseDecoder {
    buffer: String,
    /// Accumulated `data:` lines of the event under assembly.
    data_lines: Vec<String>,
    /// Carry for a byte chunk that ends mid-UTF-8 sequence.
    partial: Vec<u8>,
    /// BOM stripped once at stream start.
    stripped_bom: bool,
}

impl Default for SseDecoder {
    fn default() -> Self {
        SseDecoder {
            buffer: String::new(),
            data_lines: Vec::new(),
            partial: Vec::new(),
            stripped_bom: false,
        }
    }
}

impl SseDecoder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed one raw byte chunk; returns the data payloads of every event that
    /// completed inside it, in order. `on_comment` observes comment lines.
    pub fn feed(&mut self, bytes: &[u8], mut on_comment: impl FnMut(&str)) -> Vec<String> {
        // Reassemble UTF-8 across chunk boundaries: hold back an incomplete
        // trailing sequence.
        self.partial.extend_from_slice(bytes);
        let valid_up_to = match std::str::from_utf8(&self.partial) {
            Ok(_) => self.partial.len(),
            Err(error) => error.valid_up_to(),
        };
        let decoded: String = String::from_utf8_lossy(&self.partial[..valid_up_to]).into_owned();
        self.partial.drain(..valid_up_to);
        self.buffer.push_str(&decoded);
        if !self.stripped_bom {
            if let Some(stripped) = self.buffer.strip_prefix('\u{feff}') {
                self.buffer = stripped.to_string();
            }
            if !self.buffer.is_empty() {
                self.stripped_bom = true;
            }
        }

        let mut payloads = Vec::new();
        // Process complete lines; keep the unterminated tail buffered.
        while let Some(newline) = self.buffer.find('\n') {
            let mut line = self.buffer[..newline].to_string();
            self.buffer.drain(..=newline);
            if line.ends_with('\r') {
                line.pop();
            }
            if line.is_empty() {
                // Blank line: dispatch the assembled event, if it carried data.
                if !self.data_lines.is_empty() {
                    payloads.push(self.data_lines.join("\n"));
                    self.data_lines.clear();
                }
                continue;
            }
            if let Some(comment) = line.strip_prefix(':') {
                on_comment(comment.strip_prefix(' ').unwrap_or(comment));
                continue;
            }
            let (field, value) = match line.split_once(':') {
                Some((field, value)) => (field, value.strip_prefix(' ').unwrap_or(value)),
                None => (line.as_str(), ""),
            };
            if field == "data" {
                self.data_lines.push(value.to_string());
            }
            // Other fields (event, id, retry) are transport metadata this
            // protocol does not use.
        }
        payloads
    }
}
