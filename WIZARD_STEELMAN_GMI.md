# WIZARD_STEELMAN_GMI.md

## Steelman of Idea 1: Cross-Process Writer Lock + "Readers Never GC"

Fable's Idea 1 is the most critical contribution to the plan. I will steelman this idea by writing the strongest possible case for it as if it were my own, explicitly fixing the staleness heuristic weakness I identified in my reaction.

### The Strongest Case

The current Quill plan (§11) contains a fatal flaw that guarantees data destruction in our actual deployment environment. The plan mandates strict crash-only open with aggressive garbage collection of unreferenced files (§11.4) and limits writer serialization to an *in-process* `Mutex`. 

In our multi-agent swarms, it is standard practice to run a background `fsfs index --watch` daemon while concurrent agents run ad-hoc `fsfs search` commands. Under the current design, an `fsfs search` (reader) opening the index will trigger the crash-only GC. If the watch daemon (writer) has sealed a segment but not yet published the new manifest, the reader will violently delete the writer's in-flight segment. When the writer finally publishes, the index is permanently corrupted. This is a massive regression from `tantivy`, which natively uses a directory lockfile to prevent this.

Fable's solution is architecturally brilliant:
1. **Split the API:** Enforce a strict `QuillIndex::open()` (reader: touch nothing, never GC) and `QuillIndex::open_writer()` (writer: acquire lock, then run GC). This structurally eliminates the reader-destroying-writer hazard.
2. **Publish-Time CAS:** Before renaming the new `MANIFEST`, verify the generation matches expectations, turning a silent clobber into a typed `SearchError`.

### Fixing the Weakest Aspect: The False-Takeover Hazard

Fable proposed an advisory lockfile with an mtime-based heartbeat. In our environment, agents are frequently suspended (via `SIGSTOP`), meaning mtime staleness will inevitably cause false takeovers. A suspended writer will wake up, fail the CAS, and panic, leading to spurious task failures.

**The Fix:** Since `fsfs` indexes are local to a single node, we can leverage OS-level process liveness checks rather than relying purely on time heuristics. 
The `LOCK` file must store the owning process's PID. Before a writer attempts a staleness takeover, it performs a `kill(pid, 0)` syscall. 
- If the OS reports the process is alive (even if suspended or thrashing), **takeover is rejected**. The lock is genuinely held.
- If the OS reports the process is dead (ESRCH), the lock is abandoned and takeover is immediately safe, regardless of mtime.

By binding the lock takeover to OS-guaranteed process liveness rather than a tunable time heuristic, the mechanism becomes deterministic and impervious to `SIGSTOP` suspensions or CPU starvation.

**New Score: 1000.** With the liveness check replacing the mtime heuristic, this idea is structurally flawless, fixes a guaranteed data-loss regression, and requires zero new dependencies. It is mandatory for shipping.
