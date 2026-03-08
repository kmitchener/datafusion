# Specialized Aggregate Exchange Plan

## Problem statement

We want to share group-key hashing work from `AggregateMode::Partial` to
`AggregateMode::FinalPartitioned` without widening the partial output schema.

The previous hidden-column prototype proved that reusing hashes can help, but it
also showed the cost of carrying an extra `UInt64` inside every `RecordBatch`:

- the final aggregate still spent most of its time in hash-table probe, equality,
  and insert work rather than in `create_hashes`
- partial output batches and repartition traffic grew materially on low-reduction
  queries such as ClickBench `Q8`, `Q12`, and `Q32`
- generic `RepartitionExec` still recomputed hashes anyway, because it hashes with
  `REPARTITION_RANDOM_STATE` while aggregation uses `AGGREGATION_HASH_SEED`

The right target is the internal handoff:

```text
AggregateExec(Partial)
  -> exchange on grouping key
  -> AggregateExec(FinalPartitioned)
```

We should preserve a sidecar hash buffer only across that handoff, not expose it
as part of the public row schema and not change `ExecutionPlan::execute` for the
rest of the engine.

## Goals

- Reuse one stable group-key hash across both:
  - exchange routing
  - `GroupValues` lookup in `FinalPartitioned`
- Avoid adding hidden hash columns to `RecordBatch`
- Keep public plan outputs unchanged
- Keep the normal `RecordBatch` path as the correctness fallback
- Land the work in phases so we can benchmark each step against ClickBench

## Non-goals

- No dictionary or surrogate-key sharing work
- No engine-wide change to `RecordBatchStream`
- No attempt to optimize `AggregateMode::Final` in the first iteration
- No spill support in the new exchange in v1
- No attempt to reduce the existing probe/equality/insert cost in `GroupValues`

## Current code boundaries

These are the concrete points this design builds on:

- `AggregateExec::try_new` and `create_schema` in
  `datafusion/physical-plan/src/aggregates/mod.rs`
- `AggregateExec::required_input_distribution` in
  `datafusion/physical-plan/src/aggregates/mod.rs`
- `GroupedHashAggregateStream` in
  `datafusion/physical-plan/src/aggregates/row_hash.rs`
- `GroupValues` and implementations in
  `datafusion/physical-plan/src/aggregates/group_values/`
- `BatchPartitioner` and `RepartitionExec` in
  `datafusion/physical-plan/src/repartition/mod.rs`
- `EnforceDistribution` in
  `datafusion/physical-optimizer/src/enforce_distribution.rs`
- the initial `Partial -> FinalPartitioned` aggregate shape created in
  `datafusion/core/src/physical_planner.rs`

Important constraints from the current implementation:

- `ExecutionPlan::execute` returns `SendableRecordBatchStream`, so a sidecar path
  must be explicit and local to aggregation
- `GroupedHashAggregateStream` always polls a `RecordBatch` input today
- `RepartitionExec` always hashes from expressions and always materializes output
  as `RecordBatch`
- `GroupValues::intern` computes hashes internally and most implementations do not
  retain a per-group hash vector that can later be emitted

## Proposed architecture

### 1. Add an aggregate-only payload stream

Add a new internal module:

`datafusion/physical-plan/src/aggregates/exchange.rs`

New internal payload types:

```rust
pub(crate) struct AggPayloadBatch {
    pub batch: RecordBatch,
    pub row_hashes: ScalarBuffer<u64>,
}

pub(crate) trait AggPayloadStream:
    Stream<Item = Result<AggPayloadBatch>> + Send
{
    fn schema(&self) -> SchemaRef;
}

pub(crate) type SendableAggPayloadStream =
    Pin<Box<dyn AggPayloadStream + Send>>;
```

This is intentionally not a replacement for `RecordBatchStream`. It is an
internal side channel used only between partial hash aggregation, the new
exchange, and final partitioned hash aggregation.

### 2. Add `AggregateExchangeExec`

Introduce a new execution plan node:

`datafusion/physical-plan/src/aggregates/exchange.rs`

Responsibilities:

- consume payload batches from a partial aggregate
- route rows to output partitions using `row_hashes[i] % target_partitions`
- preserve the same `row_hashes` sidecar in the output payload
- expose a normal `ExecutionPlan` surface for plan properties and fallback

Shape:

```text
AggregateExec(Partial)
  -> AggregateExchangeExec
  -> AggregateExec(FinalPartitioned)
```

`AggregateExchangeExec` should:

- implement `ExecutionPlan`
- report the same schema as its child
- report `Partitioning::Hash(group_exprs, target_partitions)` output
- not preserve ordering, matching `RepartitionExec`
- provide an inherent `execute_payload(...)` method for specialized consumers
- implement normal `execute(...)` by degrading payload back to `RecordBatch`
  and dropping hashes if some generic parent ever executes it

The fallback `execute(...)` path is not performance-sensitive; it exists to keep
the node valid in the wider physical-plan framework.

### 3. Teach `AggregateExec(Partial)` to produce payload

Add an inherent method on `AggregateExec`:

```rust
pub(crate) fn execute_payload(
    &self,
    partition: usize,
    context: &Arc<TaskContext>,
) -> Result<SendableAggPayloadStream>;
```

Only support this for:

- `AggregateMode::Partial`
- grouped hash aggregation
- no top-k/priority-queue execution path

Implementation approach:

- factor `GroupedHashAggregateStream` input/output plumbing so it can emit either:
  - `RecordBatch`
  - `AggPayloadBatch`
- keep one stream implementation if possible, but use a dedicated output enum so
  the payload path does not infect the rest of `ExecutionPlan`

Recommended shape:

```rust
enum GroupedHashEmit {
    RecordBatch(RecordBatch),
    Payload(AggPayloadBatch),
}
```

The payload variant is only enabled when the stream is constructed through
`AggregateExec::execute_payload`.

### 4. Extend `GroupValues` for reusable hashes

We need two separate capabilities:

- partial mode must emit one stored hash per emitted group row
- final mode must accept precomputed row hashes and skip `create_hashes`

Add internal extensions to `GroupValues`:

```rust
pub(crate) struct EmittedGroupValues {
    pub arrays: Vec<ArrayRef>,
    pub hashes: Option<ScalarBuffer<u64>>,
}

fn intern_with_hashes(
    &mut self,
    cols: &[ArrayRef],
    hashes: &[u64],
    groups: &mut Vec<usize>,
) -> Result<()> {
    self.intern(cols, groups)
}

fn emit_with_hashes(
    &mut self,
    emit_to: EmitTo,
) -> Result<EmittedGroupValues> {
    Ok(EmittedGroupValues {
        arrays: self.emit(emit_to)?,
        hashes: None,
    })
}
```

Implement the real fast path first in:

- `group_values/multi_group_by/mod.rs` (`GroupValuesColumn`)
- `group_values/row.rs` (`GroupValuesRows`)
- `group_values/single_group_by/primitive.rs` (`GroupValuesPrimitive`)

Those implementations should add a `group_hashes: Vec<u64>` aligned with group
id order. On every new-group insertion, store the already-computed hash into
`group_hashes`. On `EmitTo::First(n)`, shift or split `group_hashes` exactly the
same way group ids are shifted today.

This keeps the design compatible with early emit in partial mode.

Do not try to optimize all `GroupValues` implementations in the first patch. For
unsupported variants such as single-column byte maps, `emit_with_hashes` can
return `hashes: None` and the planner will fall back to the normal path.

### 5. Consume payload in `AggregateExec(FinalPartitioned)`

Extend `GroupedHashAggregateStream` so its input can be either:

```rust
enum AggregateInput {
    RecordBatch(SendableRecordBatchStream),
    Payload(SendableAggPayloadStream),
}
```

Wire this in `AggregateExec::execute_typed` or `GroupedHashAggregateStream::new`:

- if `self.mode == AggregateMode::FinalPartitioned`
- and `self.input` is `AggregateExchangeExec`
- then call `AggregateExchangeExec::execute_payload(...)`
- otherwise keep the existing `RecordBatch` input path

Split the current aggregation method into:

- `group_aggregate_record_batch(&RecordBatch)`
- `group_aggregate_payload_batch(&AggPayloadBatch)`

`group_aggregate_payload_batch` should:

- evaluate aggregate merge arguments from `batch`
- evaluate group columns from `batch`
- call `group_values.intern_with_hashes(&group_by_values, &row_hashes, ...)`

This removes the final-stage `create_hashes(...)` work without adding a visible
schema field.

### 6. Use one hash seed for both partial aggregation and exchange

The payload path only makes sense if the same hash value is valid for both
exchange routing and final-stage hash-table lookup.

Add a single constant in `aggregates/mod.rs`:

```rust
pub(crate) const AGGREGATE_EXCHANGE_HASH_SEED: ahash::RandomState = ...;
```

Then:

- use it in payload-capable `GroupValues` implementations
- use the same stored hash in `AggregateExchangeExec`
- stop relying on `REPARTITION_RANDOM_STATE` for this specialized path

The generic `RepartitionExec` seed remains unchanged.

### 7. Inject the specialized exchange from `EnforceDistribution`

Do not change the initial two-stage aggregate shape in the physical planner.

Instead, modify `EnforceDistribution` so that when it is about to insert
`RepartitionExec` below a `FinalPartitioned` aggregate, it can choose
`AggregateExchangeExec` instead.

Detection rule:

- parent is `AggregateExec` in `AggregateMode::FinalPartitioned`
- child is `AggregateExec` in `AggregateMode::Partial`
- child and parent group expressions still correspond to the same grouping keys
- no grouping sets
- child would execute via `GroupedHashAggregateStream`
- all group key datatypes are supported by payload-capable `GroupValues`

The planner should otherwise keep inserting ordinary `RepartitionExec`.

This keeps the optimization localized and preserves the normal physical planner
contract.

### 8. Treat `AggregateExchangeExec` as a distribution-changing barrier

Update optimizer utilities that currently special-case `RepartitionExec`:

- `enforce_distribution.rs`
- `enforce_sorting/*`
- `physical-optimizer/src/utils.rs`

`AggregateExchangeExec` should be treated like `RepartitionExec` for:

- clearing pushed ordering requirements
- disabling sort-preserving assumptions
- removal of redundant adjacent repartition-like operators

This prevents the new node from accidentally inheriting invalid ordering
metadata.

## Runtime behavior and fallbacks

### Supported in v1

- grouped `Partial -> FinalPartitioned`
- hash-aggregation path only
- `GroupValuesColumn`, `GroupValuesRows`, and `GroupValuesPrimitive`
- early emit in partial mode
- normal `FinalPartitioned` aggregate semantics

### Explicit fallback conditions in v1

Fallback to the existing `RecordBatch + RepartitionExec` path when:

- there is no `GROUP BY`
- grouping sets are present
- the partial aggregate uses the top-k/grouped-priority-queue path
- any group key type lands on a `GroupValues` implementation that does not yet
  implement `emit_with_hashes` and `intern_with_hashes`
- spill is required in the new exchange

### Skip-aggregation handling

`SkippingAggregation` must be supported in the specialized path, otherwise we
give up too many real queries.

Concrete handling:

- when partial aggregation switches to `ExecutionState::SkippingAggregation`,
  `transform_to_states(&batch)` already builds the partial-state `RecordBatch`
- compute group-key arrays from the same input batch
- hash those group-key arrays once with `AGGREGATE_EXCHANGE_HASH_SEED`
- emit `AggPayloadBatch { batch: states, row_hashes }`

This preserves the current partial semantics while still avoiding exchange-side
and final-side rehashing.

## Implementation steps

### Phase 1: scaffolding

1. Add `dev/aggregate-specialized-exchange-plan.md`
2. Add `aggregates/exchange.rs` with payload stream types
3. Export the new module from `aggregates/mod.rs`

### Phase 2: group-value hash retention

1. Extend `GroupValues` with `intern_with_hashes` and `emit_with_hashes`
2. Implement hash retention in:
   - `GroupValuesColumn`
   - `GroupValuesRows`
   - `GroupValuesPrimitive`
3. Add unit tests covering:
   - full emit
   - `EmitTo::First`
   - hash alignment after early emit
   - hash collision correctness

### Phase 3: partial payload emission

1. Add `AggregateExec::execute_payload`
2. Extend grouped hash aggregate execution to emit `AggPayloadBatch`
3. Keep existing `execute()` behavior unchanged
4. Add tests for:
   - partial grouped aggregation emits payload hashes aligned with output rows
   - skip-aggregation payload path

### Phase 4: specialized exchange

1. Implement `AggregateExchangeExec`
2. Copy the useful parts of `RepartitionExec` task/channel structure, but operate
   on payload batches
3. Partition rows with the provided `row_hashes`
4. Add a normal `execute()` fallback that drops hashes
5. Add metrics:
   - `payload_rows`
   - `payload_batches`
   - `payload_bytes`
   - `repartition_time`
   - `send_time`

### Phase 5: final aggregate payload consume path

1. Add `AggregateInput` enum to `GroupedHashAggregateStream`
2. Add `group_aggregate_payload_batch`
3. Route `FinalPartitioned` through payload only when the child is
   `AggregateExchangeExec`
4. Leave `Final`, `Single`, and `SinglePartitioned` unchanged

### Phase 6: optimizer insertion

1. Add an eligibility helper, probably in `enforce_distribution.rs`
2. Replace the injected `RepartitionExec` with `AggregateExchangeExec` when the
   aggregate pair is eligible
3. Update ordering/distribution utilities to recognize the new node
4. Add plan-shape assertions in optimizer tests

### Phase 7: measurement and tuning

1. Run targeted aggregate tests:
   - `cargo test -p datafusion-physical-plan aggregates`
   - `cargo test -p datafusion-physical-plan group_values`
   - `cargo test -p datafusion-physical-optimizer enforce_distribution`
2. Run the official benchmark comparison:
   - `./benchmarks/bench.sh compare main <new-branch>`
3. Inspect regressions with `EXPLAIN ANALYZE`, focusing on:
   - `Q8`
   - `Q12`
   - `Q32`
   - `Q34`
4. Confirm expected wins on higher-cardinality multi-stage group-bys:
   - `Q14`
   - `Q16`
   - `Q17`
   - `Q18`
   - `Q22`
   - `Q33`

## Testing plan

### Unit tests

- `GroupValuesColumn` emits matching arrays and hashes
- `GroupValuesRows` emits matching arrays and hashes
- `GroupValuesPrimitive` emits matching arrays and hashes
- `EmitTo::First` keeps remaining group ids and hashes aligned
- forced hash collisions still produce correct grouping

### Execution tests

- `Partial -> AggregateExchangeExec -> FinalPartitioned` matches current output
- partial skip-aggregation still matches current output
- unsupported group key types fall back to `RepartitionExec`
- `AggregateExchangeExec::execute()` fallback still yields the same rows when
  used as a plain `ExecutionPlan`

### Optimizer tests

- `EnforceDistribution` inserts `AggregateExchangeExec` only on eligible
  aggregate pairs
- `CombinePartialFinalAggregate` still collapses adjacent `Partial -> Final`
  pairs without repartition
- sort-enforcement rules treat `AggregateExchangeExec` as a distribution barrier

## Benchmark acceptance criteria

The new path is only worth keeping if it beats both:

- `main`
- the hidden-hash-column prototype

Minimum acceptance bar:

- no correctness failures on the official `clickbench_partitioned` suite
- no material regressions on `Q8` and `Q32`
- measurable improvement on the aggregate-heavy queries that previously
  benefited from hash reuse

If the full suite still shows mixed results, keep the optimization behind a
stricter eligibility gate rather than making it universal.

## Open questions

1. Should v1 support single-column byte groupings, or should those stay on the
   fallback path until `ArrowBytesMap` can expose/stash stable hashes cheaply?
2. Should `AggregateExchangeExec` support spill in its first landing, or should
   it immediately degrade to `RepartitionExec` when memory pressure appears?
3. Do we want a dedicated metric for hash-computation time inside `GroupValues`
   so the benefit is visible separately from the larger `time_calculating_group_ids`
   bucket?

## Recommended first patch series

Keep the first implementation series small and benchmarkable:

1. `GroupValues` hash retention for `GroupValuesColumn` and `GroupValuesRows`
2. payload-capable partial aggregate stream
3. `AggregateExchangeExec`
4. final payload consume path
5. optimizer insertion for eligible `Partial -> FinalPartitioned`
6. ClickBench compare against `main`

That sequence is enough to validate the design on the exact hot path we care
about without pulling in surrogate keys, dictionary output, or global stream
API churn.
