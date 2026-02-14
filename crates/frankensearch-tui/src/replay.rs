//! Input recording and deterministic replay.
//!
//! Provides [`ReplayRecorder`] for capturing input events with timestamps
//! and [`ReplayPlayer`] for replaying them deterministically. This enables
//! automated testing, bug reproduction, and demo playback.

use std::time::Duration;

use crossterm::event::{KeyCode, KeyModifiers, MouseEventKind};
use serde::{Deserialize, Serialize};

use crate::input::InputEvent;

// ─── Replay State ────────────────────────────────────────────────────────────

/// State of the replay system.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplayState {
    /// Idle — not recording or playing.
    Idle,
    /// Recording input events.
    Recording,
    /// Playing back recorded events.
    Playing,
    /// Playback paused.
    Paused,
}

// ─── Input Record ────────────────────────────────────────────────────────────

/// A recorded input event with a timestamp offset from the start.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputRecord {
    /// Time offset from the start of recording.
    pub offset: Duration,
    /// The recorded event kind.
    pub event: RecordedEvent,
}

/// Serializable input event (subset of `InputEvent` that can be replayed).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RecordedEvent {
    /// A key press.
    Key {
        /// Key code as a string representation.
        key: String,
        /// Modifier bits.
        modifiers: u8,
    },
    /// A mouse event.
    Mouse {
        /// Mouse event kind description.
        kind: String,
        /// Column position.
        col: u16,
        /// Row position.
        row: u16,
    },
    /// Terminal resize.
    Resize {
        /// New width.
        width: u16,
        /// New height.
        height: u16,
    },
}

impl RecordedEvent {
    /// Create a key event record from crossterm types.
    #[must_use]
    pub fn from_key(key: KeyCode, modifiers: KeyModifiers) -> Self {
        Self::Key {
            key: format!("{key:?}"),
            modifiers: modifiers.bits(),
        }
    }

    /// Create a mouse event record.
    #[must_use]
    pub fn from_mouse(kind: MouseEventKind, col: u16, row: u16) -> Self {
        Self::Mouse {
            kind: format!("{kind:?}"),
            col,
            row,
        }
    }

    /// Create a resize event record.
    #[must_use]
    pub const fn from_resize(width: u16, height: u16) -> Self {
        Self::Resize { width, height }
    }
}

// ─── Replay Recorder ─────────────────────────────────────────────────────────

/// Records input events with timestamps for later replay.
pub struct ReplayRecorder {
    /// Recorded events.
    records: Vec<InputRecord>,
    /// When recording started.
    start: Option<std::time::Instant>,
    /// Current state.
    state: ReplayState,
}

impl ReplayRecorder {
    /// Create a new recorder.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            records: Vec::new(),
            start: None,
            state: ReplayState::Idle,
        }
    }

    /// Start recording.
    pub fn start(&mut self) {
        self.records.clear();
        self.start = Some(std::time::Instant::now());
        self.state = ReplayState::Recording;
    }

    /// Stop recording.
    pub const fn stop(&mut self) {
        self.state = ReplayState::Idle;
        self.start = None;
    }

    /// Record an input event.
    pub fn record(&mut self, event: &InputEvent) {
        if self.state != ReplayState::Recording {
            return;
        }

        let offset = self
            .start
            .map_or(Duration::ZERO, |s| s.elapsed());

        let recorded = match event {
            InputEvent::Key(key, mods) => RecordedEvent::from_key(*key, *mods),
            InputEvent::Mouse(kind, col, row) => {
                RecordedEvent::from_mouse(*kind, *col, *row)
            }
            InputEvent::Resize(w, h) => RecordedEvent::from_resize(*w, *h),
            InputEvent::Action(_) => return, // Don't record resolved actions.
        };

        self.records.push(InputRecord {
            offset,
            event: recorded,
        });
    }

    /// Get the recorded events.
    #[must_use]
    pub fn records(&self) -> &[InputRecord] {
        &self.records
    }

    /// Current state.
    #[must_use]
    pub const fn state(&self) -> ReplayState {
        self.state
    }

    /// Number of recorded events.
    #[must_use]
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Whether no events have been recorded.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Export records as JSON.
    ///
    /// # Errors
    ///
    /// Returns an error if serialization fails.
    pub fn export_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(&self.records)
    }
}

impl Default for ReplayRecorder {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Replay Player ───────────────────────────────────────────────────────────

/// Plays back recorded input events.
pub struct ReplayPlayer {
    /// Events to play.
    records: Vec<InputRecord>,
    /// Current position in the playback.
    position: usize,
    /// Current state.
    state: ReplayState,
}

impl ReplayPlayer {
    /// Create a new player with the given records.
    #[must_use]
    pub const fn new(records: Vec<InputRecord>) -> Self {
        Self {
            records,
            position: 0,
            state: ReplayState::Idle,
        }
    }

    /// Load records from JSON.
    ///
    /// # Errors
    ///
    /// Returns an error if deserialization fails.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        let records: Vec<InputRecord> = serde_json::from_str(json)?;
        Ok(Self::new(records))
    }

    /// Start playback from the beginning.
    pub const fn play(&mut self) {
        self.position = 0;
        self.state = ReplayState::Playing;
    }

    /// Pause playback.
    pub fn pause(&mut self) {
        if self.state == ReplayState::Playing {
            self.state = ReplayState::Paused;
        }
    }

    /// Resume playback.
    pub fn resume(&mut self) {
        if self.state == ReplayState::Paused {
            self.state = ReplayState::Playing;
        }
    }

    /// Stop playback.
    pub const fn stop(&mut self) {
        self.state = ReplayState::Idle;
        self.position = 0;
    }

    /// Get the next event to play, advancing the position.
    #[must_use]
    pub fn advance(&mut self) -> Option<&InputRecord> {
        if self.state != ReplayState::Playing {
            return None;
        }

        if self.position >= self.records.len() {
            self.state = ReplayState::Idle;
            return None;
        }

        let record = &self.records[self.position];
        self.position += 1;
        Some(record)
    }

    /// Current playback position.
    #[must_use]
    pub const fn position(&self) -> usize {
        self.position
    }

    /// Total number of events.
    #[must_use]
    pub fn total(&self) -> usize {
        self.records.len()
    }

    /// Current state.
    #[must_use]
    pub const fn state(&self) -> ReplayState {
        self.state
    }

    /// Whether playback is complete.
    #[must_use]
    pub fn is_done(&self) -> bool {
        self.position >= self.records.len()
    }

    /// Playback progress as a ratio (0.0 to 1.0).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn progress(&self) -> f64 {
        if self.records.is_empty() {
            return 1.0;
        }
        self.position as f64 / self.records.len() as f64
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use crossterm::event::{KeyCode, KeyModifiers};

    use super::*;

    #[test]
    fn recorder_starts_idle() {
        let recorder = ReplayRecorder::new();
        assert_eq!(recorder.state(), ReplayState::Idle);
        assert!(recorder.is_empty());
    }

    #[test]
    fn recorder_records_events() {
        let mut recorder = ReplayRecorder::new();
        recorder.start();

        let event = InputEvent::Key(KeyCode::Char('a'), KeyModifiers::NONE);
        recorder.record(&event);
        recorder.record(&event);

        assert_eq!(recorder.len(), 2);
        assert_eq!(recorder.state(), ReplayState::Recording);

        recorder.stop();
        assert_eq!(recorder.state(), ReplayState::Idle);
    }

    #[test]
    fn recorder_ignores_when_idle() {
        let mut recorder = ReplayRecorder::new();
        let event = InputEvent::Key(KeyCode::Char('a'), KeyModifiers::NONE);
        recorder.record(&event);
        assert!(recorder.is_empty());
    }

    #[test]
    fn recorder_ignores_action_events() {
        let mut recorder = ReplayRecorder::new();
        recorder.start();
        let event = InputEvent::Action(crate::input::KeyAction::Quit);
        recorder.record(&event);
        assert!(recorder.is_empty());
    }

    #[test]
    fn recorder_export_json() {
        let mut recorder = ReplayRecorder::new();
        recorder.start();
        let event = InputEvent::Key(KeyCode::Enter, KeyModifiers::NONE);
        recorder.record(&event);
        recorder.stop();

        let json = recorder.export_json().unwrap();
        assert!(json.contains("key"));
    }

    #[test]
    fn player_playback() {
        let records = vec![
            InputRecord {
                offset: Duration::from_millis(0),
                event: RecordedEvent::from_key(KeyCode::Char('a'), KeyModifiers::NONE),
            },
            InputRecord {
                offset: Duration::from_millis(100),
                event: RecordedEvent::from_key(KeyCode::Enter, KeyModifiers::NONE),
            },
        ];

        let mut player = ReplayPlayer::new(records);
        assert_eq!(player.state(), ReplayState::Idle);
        assert_eq!(player.total(), 2);

        player.play();
        assert_eq!(player.state(), ReplayState::Playing);

        let first = player.advance().unwrap();
        assert_eq!(first.offset, Duration::from_millis(0));
        assert_eq!(player.position(), 1);

        let second = player.advance().unwrap();
        assert_eq!(second.offset, Duration::from_millis(100));
        assert_eq!(player.position(), 2);

        // Exhausted.
        assert!(player.advance().is_none());
        assert!(player.is_done());
    }

    #[test]
    fn player_pause_resume() {
        let records = vec![InputRecord {
            offset: Duration::ZERO,
            event: RecordedEvent::from_key(KeyCode::Char('x'), KeyModifiers::NONE),
        }];

        let mut player = ReplayPlayer::new(records);
        player.play();
        player.pause();
        assert_eq!(player.state(), ReplayState::Paused);
        assert!(player.advance().is_none()); // Paused, no events.

        player.resume();
        assert_eq!(player.state(), ReplayState::Playing);
        assert!(player.advance().is_some());
    }

    #[test]
    fn player_progress() {
        let records = vec![
            InputRecord {
                offset: Duration::ZERO,
                event: RecordedEvent::from_resize(80, 24),
            },
            InputRecord {
                offset: Duration::from_millis(50),
                event: RecordedEvent::from_resize(120, 40),
            },
        ];

        let mut player = ReplayPlayer::new(records);
        assert!((player.progress() - 0.0).abs() < f64::EPSILON);

        player.play();
        let _ = player.advance();
        assert!((player.progress() - 0.5).abs() < f64::EPSILON);

        let _ = player.advance();
        assert!((player.progress() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn player_from_json_roundtrip() {
        let records = vec![InputRecord {
            offset: Duration::from_millis(42),
            event: RecordedEvent::from_key(KeyCode::Char('q'), KeyModifiers::CONTROL),
        }];

        let json = serde_json::to_string(&records).unwrap();
        let player = ReplayPlayer::from_json(&json).unwrap();
        assert_eq!(player.total(), 1);
    }

    #[test]
    fn recorded_event_serde() {
        let event = RecordedEvent::from_key(KeyCode::Enter, KeyModifiers::SHIFT);
        let json = serde_json::to_string(&event).unwrap();
        let decoded: RecordedEvent = serde_json::from_str(&json).unwrap();
        if let RecordedEvent::Key { modifiers, .. } = decoded {
            assert_eq!(modifiers, KeyModifiers::SHIFT.bits());
        } else {
            panic!("Expected Key variant");
        }
    }

    #[test]
    fn recorded_event_mouse() {
        let event = RecordedEvent::from_mouse(
            MouseEventKind::Down(crossterm::event::MouseButton::Left),
            10,
            20,
        );
        let json = serde_json::to_string(&event).unwrap();
        let decoded: RecordedEvent = serde_json::from_str(&json).unwrap();
        if let RecordedEvent::Mouse { col, row, .. } = decoded {
            assert_eq!(col, 10);
            assert_eq!(row, 20);
        } else {
            panic!("Expected Mouse variant");
        }
    }

    #[test]
    fn empty_player_progress() {
        let player = ReplayPlayer::new(vec![]);
        assert!((player.progress() - 1.0).abs() < f64::EPSILON);
    }
}
