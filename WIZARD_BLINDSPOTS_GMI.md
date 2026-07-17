# WIZARD_BLINDSPOTS_GMI.md

## Blind-Spot Probe: What Both Wizards Missed

After this adversarial exchange, it is clear both CC and I focused heavily on inner-loop query execution and data-integrity mechanics. However, we both completely missed the systemic bottlenecks of the ingest pipeline and the overall `fsfs` interaction model. Here are two genuinely new ideas absent from both lists.

### 1. Differential Arena Allocators for Ingest Isolation
**What it is:** Explicitly pin `mimalloc` or `jemalloc` thread-local arenas to the specific rayon worker threads responsible for tokenization and AST parsing during heavy index builds.
**Why it wins:** Tokenization and text analysis are extraordinarily allocation-heavy. Even utilizing `CompactString` for small tokens, generating term dictionaries, posting lists, and forward indexes puts massive pressure on the global allocator. At the scale of 1M-doc nightly index builds, the `std::alloc` global lock becomes a hidden scalability ceiling. By routing ingest-path allocations to dedicated thread-local memory arenas, we bypass global lock contention entirely. This allows the multi-threaded fan-out to scale linearly with core count, directly attacking the tokenization bandwidth ceiling (QG-1).
**Implementation sketch:** Expose an initialization hook `QuillAllocator::init_worker_arena()` called at the start of every rayon task spawned by `LiveIngestPipeline`. The arena drops cleanly when the indexing batch is sealed.
**Risk/Cost:** Requires explicit allocator tuning which can be platform-dependent, though `mimalloc` is generally cross-platform safe. 
**Confidence:** 85%.

### 2. Query AST Canonicalization and Deduplication Layer
**What it is:** Introduce a logical planner phase *before* physical cursor evaluation that maps query syntax to a canonical normal form and deduplicates identical clauses (e.g., transforming `(A OR B) AND C AND A` to `(A OR B) AND C`).
**Why it wins:** Agents are notorious for pasting raw code fragments, error logs, or highly redundant natural language into `fsfs search`. Currently, the engine translates the AST directly into physical cursors. A redundant query forces the engine to perform duplicate Grimoire dictionary probes, allocate duplicate MaxScore heaps, and execute duplicate decoding loops. Canonicalization guarantees that any unique term is probed exactly once per query, capping the memory and CPU envelope for adversarial or generated agent queries.
**Implementation sketch:** Between parsing (e4.1) and execution, run a recursive visitor over the `QueryAST`. Sort children of boolean `AND`/`OR` nodes alphabetically by field/term, then `dedup()`. 
**Risk/Cost:** Trivial CPU overhead to sort/dedup the AST (microseconds), which pays off exponentially by saving dictionary I/O and heap allocations.
**Confidence:** 95%.
