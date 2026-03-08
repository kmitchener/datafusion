// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use std::any::Any;
use std::fmt::Formatter;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use arrow::array::{AsArray, PrimitiveArray, RecordBatch};
use arrow::buffer::ScalarBuffer;
use arrow::compute::take_arrays;
use arrow::datatypes::{SchemaRef, UInt32Type, UInt64Type};
use arrow::record_batch::RecordBatchOptions;
use datafusion_common::config::ConfigOptions;
use datafusion_common::tree_node::TreeNodeRecursion;
use datafusion_common::{
    DataFusionError, HashMap as DFHashMap, Result, internal_err, not_impl_err,
};
use datafusion_common_runtime::SpawnedTask;
use datafusion_execution::TaskContext;
use datafusion_physical_expr::PhysicalExpr;
use futures::{FutureExt, Stream, StreamExt, TryStreamExt, ready};
use parking_lot::Mutex;
use tokio::task::yield_now;

use crate::aggregates::{AggregateExec, group_hash_column_index};
use crate::display::{DisplayAs, DisplayFormatType};
use crate::execution_plan::{EmissionType, EvaluationType, SchedulingType};
use crate::metrics::{ExecutionPlanMetricsSet, MetricBuilder, MetricsSet};
use crate::repartition::distributor_channels::{
    DistributionReceiver, DistributionSender, channels,
};
use crate::stream::RecordBatchStreamAdapter;
use crate::{
    Distribution, ExecutionPlan, ExecutionPlanProperties, Partitioning, PlanProperties,
    SendableRecordBatchStream, check_if_same_properties,
};

pub(crate) type SendableAggPayloadStream =
    Pin<Box<dyn Stream<Item = Result<AggPayloadBatch>> + Send>>;

#[derive(Debug)]
pub(crate) struct AggPayloadBatch {
    pub batch: RecordBatch,
    pub row_hashes: ScalarBuffer<u64>,
}

pub(crate) struct AggPayloadRecordBatchStreamAdapter {
    inner: SendableRecordBatchStream,
    schema: SchemaRef,
}

impl AggPayloadRecordBatchStreamAdapter {
    pub(crate) fn new(inner: SendableRecordBatchStream, schema: SchemaRef) -> Self {
        Self { inner, schema }
    }
}

impl Stream for AggPayloadRecordBatchStreamAdapter {
    type Item = Result<AggPayloadBatch>;

    fn poll_next(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        match ready!(self.inner.poll_next_unpin(cx)) {
            Some(Ok(batch)) => {
                let Some(hash_idx) = group_hash_column_index(batch.schema().as_ref())
                else {
                    return Poll::Ready(Some(Err(DataFusionError::Internal(
                        "aggregate payload batch missing hash column".to_string(),
                    ))));
                };
                let row_hashes = batch
                    .column(hash_idx)
                    .as_primitive::<UInt64Type>()
                    .values()
                    .clone();
                let columns = batch.columns()[..hash_idx].to_vec();
                match RecordBatch::try_new(Arc::clone(&self.schema), columns) {
                    Ok(batch) => {
                        Poll::Ready(Some(Ok(AggPayloadBatch { batch, row_hashes })))
                    }
                    Err(err) => Poll::Ready(Some(Err(err.into()))),
                }
            }
            Some(Err(err)) => Poll::Ready(Some(Err(err))),
            None => Poll::Ready(None),
        }
    }
}

type MaybePayload = Option<Result<AggPayloadBatch>>;

#[derive(Debug, Clone)]
struct AggregateExchangeMetrics {
    fetch_time: crate::metrics::Time,
    repartition_time: crate::metrics::Time,
    send_time: Vec<crate::metrics::Time>,
}

impl AggregateExchangeMetrics {
    fn new(
        input_partition: usize,
        num_output_partitions: usize,
        metrics: &ExecutionPlanMetricsSet,
    ) -> Self {
        let fetch_time =
            MetricBuilder::new(metrics).subset_time("fetch_time", input_partition);
        let repartition_time =
            MetricBuilder::new(metrics).subset_time("repartition_time", input_partition);
        let send_time = (0..num_output_partitions)
            .map(|output_partition| {
                let label = crate::metrics::Label::new(
                    "outputPartition",
                    output_partition.to_string(),
                );
                MetricBuilder::new(metrics)
                    .with_label(label)
                    .subset_time("send_time", input_partition)
            })
            .collect();
        Self {
            fetch_time,
            repartition_time,
            send_time,
        }
    }
}

#[derive(Debug)]
struct ConsumingInputStreamsState {
    receivers: DFHashMap<usize, DistributionReceiver<MaybePayload>>,
    abort_helper: Arc<Vec<SpawnedTask<()>>>,
}

#[derive(Default)]
enum AggregateExchangeState {
    #[default]
    NotInitialized,
    InputStreamsInitialized(Vec<(SendableAggPayloadStream, AggregateExchangeMetrics)>),
    ConsumingInputStreams(ConsumingInputStreamsState),
}

impl std::fmt::Debug for AggregateExchangeState {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            AggregateExchangeState::NotInitialized => write!(f, "NotInitialized"),
            AggregateExchangeState::InputStreamsInitialized(value) => {
                write!(f, "InputStreamsInitialized({})", value.len())
            }
            AggregateExchangeState::ConsumingInputStreams(_) => {
                write!(f, "ConsumingInputStreams")
            }
        }
    }
}

impl AggregateExchangeState {
    fn ensure_input_streams_initialized(
        &mut self,
        input: &Arc<dyn ExecutionPlan>,
        metrics: &ExecutionPlanMetricsSet,
        output_partitions: usize,
        ctx: Arc<TaskContext>,
    ) -> Result<()> {
        if !matches!(self, AggregateExchangeState::NotInitialized) {
            return Ok(());
        }

        let num_input_partitions = input.output_partitioning().partition_count();
        let mut streams_and_metrics = Vec::with_capacity(num_input_partitions);
        for i in 0..num_input_partitions {
            let exchange_metrics =
                AggregateExchangeMetrics::new(i, output_partitions, metrics);
            let timer = exchange_metrics.fetch_time.timer();
            let stream = execute_payload_input(input, i, Arc::clone(&ctx))?;
            timer.done();
            streams_and_metrics.push((stream, exchange_metrics));
        }
        *self = AggregateExchangeState::InputStreamsInitialized(streams_and_metrics);
        Ok(())
    }

    fn consume_input_streams(
        &mut self,
        input: &Arc<dyn ExecutionPlan>,
        metrics: &ExecutionPlanMetricsSet,
        output_partitions: usize,
        ctx: Arc<TaskContext>,
    ) -> Result<&mut ConsumingInputStreamsState> {
        let streams_and_metrics = match self {
            AggregateExchangeState::NotInitialized => {
                self.ensure_input_streams_initialized(
                    input,
                    metrics,
                    output_partitions,
                    Arc::clone(&ctx),
                )?;
                let AggregateExchangeState::InputStreamsInitialized(value) = self else {
                    return internal_err!(
                        "AggregateExchangeState must be initialized after ensure_input_streams_initialized"
                    );
                };
                value
            }
            AggregateExchangeState::ConsumingInputStreams(value) => return Ok(value),
            AggregateExchangeState::InputStreamsInitialized(value) => value,
        };

        let num_input_partitions = streams_and_metrics.len();
        let (base_txs, base_rxs) = channels(output_partitions);
        let txs = base_txs
            .into_iter()
            .map(|tx| vec![tx; num_input_partitions])
            .collect::<Vec<_>>();
        let mut receivers = base_rxs
            .into_iter()
            .enumerate()
            .collect::<DFHashMap<_, _>>();

        let mut spawned_tasks = Vec::with_capacity(num_input_partitions);
        for (input_partition, (stream, exchange_metrics)) in
            std::mem::take(streams_and_metrics).into_iter().enumerate()
        {
            let txs_for_input = txs
                .iter()
                .enumerate()
                .map(|(partition, senders)| (partition, senders[input_partition].clone()))
                .collect::<DFHashMap<_, _>>();
            let senders = txs_for_input.clone();

            let input_task = SpawnedTask::spawn(Self::pull_from_input(
                stream,
                txs_for_input,
                output_partitions,
                exchange_metrics,
            ));
            let wait_task = SpawnedTask::spawn(Self::wait_for_task(input_task, senders));
            spawned_tasks.push(wait_task);
        }

        *self =
            AggregateExchangeState::ConsumingInputStreams(ConsumingInputStreamsState {
                receivers: std::mem::take(&mut receivers),
                abort_helper: Arc::new(spawned_tasks),
            });

        match self {
            AggregateExchangeState::ConsumingInputStreams(value) => Ok(value),
            _ => unreachable!(),
        }
    }

    async fn pull_from_input(
        mut stream: SendableAggPayloadStream,
        mut output_channels: DFHashMap<usize, DistributionSender<MaybePayload>>,
        num_output_partitions: usize,
        metrics: AggregateExchangeMetrics,
    ) -> Result<()> {
        let mut indices = vec![vec![]; num_output_partitions];
        let mut batches_until_yield = num_output_partitions.max(1);

        while !output_channels.is_empty() {
            let timer = metrics.fetch_time.timer();
            let result = stream.next().await;
            timer.done();

            let payload = match result {
                Some(result) => result?,
                None => break,
            };

            if payload.batch.num_rows() == 0 {
                continue;
            }

            let repartition_timer = metrics.repartition_time.timer();
            let partitioned =
                partition_payload_batch(payload, num_output_partitions, &mut indices)?;
            repartition_timer.done();

            for (partition, payload) in partitioned {
                let timer = metrics.send_time[partition].timer();
                if let Some(sender) = output_channels.get(&partition)
                    && sender.send(Some(Ok(payload))).await.is_err()
                {
                    output_channels.remove(&partition);
                }
                timer.done();
            }

            if batches_until_yield == 0 {
                yield_now().await;
                batches_until_yield = num_output_partitions.max(1);
            } else {
                batches_until_yield -= 1;
            }
        }

        Ok(())
    }

    async fn wait_for_task(
        input_task: SpawnedTask<Result<()>>,
        txs: DFHashMap<usize, DistributionSender<MaybePayload>>,
    ) {
        match input_task.join().await {
            Err(err) => {
                let err = Arc::new(err);
                for (_, tx) in txs {
                    let err = Err(DataFusionError::Context(
                        "Join Error".to_string(),
                        Box::new(DataFusionError::External(Box::new(Arc::clone(&err)))),
                    ));
                    tx.send(Some(err)).await.ok();
                }
            }
            Ok(Err(err)) => {
                let err = Arc::new(err);
                for (_, tx) in txs {
                    tx.send(Some(Err(DataFusionError::from(&err)))).await.ok();
                }
            }
            Ok(Ok(())) => {
                for (_, tx) in txs {
                    tx.send(None).await.ok();
                }
            }
        }
    }
}

fn execute_payload_input(
    input: &Arc<dyn ExecutionPlan>,
    partition: usize,
    ctx: Arc<TaskContext>,
) -> Result<SendableAggPayloadStream> {
    if let Some(agg) = input.as_any().downcast_ref::<AggregateExec>() {
        return agg.execute_payload(partition, &ctx);
    }
    if let Some(exchange) = input.as_any().downcast_ref::<AggregateExchangeExec>() {
        return exchange.execute_payload(partition, ctx);
    }
    not_impl_err!(
        "aggregate payload execution is not supported for {}",
        input.name()
    )
}

fn partition_payload_batch(
    payload: AggPayloadBatch,
    num_output_partitions: usize,
    indices: &mut [Vec<u32>],
) -> Result<Vec<(usize, AggPayloadBatch)>> {
    let AggPayloadBatch { batch, row_hashes } = payload;

    indices.iter_mut().for_each(Vec::clear);
    for (row, hash) in row_hashes.iter().enumerate() {
        indices[(*hash % num_output_partitions as u64) as usize].push(row as u32);
    }

    let mut output = Vec::new();
    for (partition, rows) in indices.iter_mut().enumerate() {
        if rows.is_empty() {
            continue;
        }

        let taken_indices = std::mem::take(rows);
        let indices_array: PrimitiveArray<UInt32Type> = taken_indices.into();
        let columns = take_arrays(batch.columns(), &indices_array, None)?;
        let hashes = indices_array
            .values()
            .iter()
            .map(|row| row_hashes[*row as usize])
            .collect::<Vec<_>>();

        let mut options = RecordBatchOptions::new();
        options = options.with_row_count(Some(indices_array.len()));
        let batch = RecordBatch::try_new_with_options(batch.schema(), columns, &options)?;
        output.push((
            partition,
            AggPayloadBatch {
                batch,
                row_hashes: ScalarBuffer::from(hashes),
            },
        ));

        let (_, buffer, _) = indices_array.into_parts();
        let mut reuse = buffer.into_inner().into_vec::<u32>().map_err(|err| {
            DataFusionError::Internal(format!(
                "Could not convert partition indices buffer back to vec: {err:?}"
            ))
        })?;
        reuse.clear();
        *rows = reuse;
    }

    Ok(output)
}

struct PerPartitionPayloadStream {
    receiver: DistributionReceiver<MaybePayload>,
    _drop_helper: Arc<Vec<SpawnedTask<()>>>,
    remaining_partitions: usize,
}

impl PerPartitionPayloadStream {
    fn new(
        receiver: DistributionReceiver<MaybePayload>,
        drop_helper: Arc<Vec<SpawnedTask<()>>>,
        num_input_partitions: usize,
    ) -> Self {
        Self {
            receiver,
            _drop_helper: drop_helper,
            remaining_partitions: num_input_partitions,
        }
    }
}

impl Stream for PerPartitionPayloadStream {
    type Item = Result<AggPayloadBatch>;

    fn poll_next(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        loop {
            match self.receiver.recv().poll_unpin(cx) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(Some(Some(result))) => return Poll::Ready(Some(result)),
                Poll::Ready(Some(None)) => {
                    self.remaining_partitions -= 1;
                    if self.remaining_partitions == 0 {
                        return Poll::Ready(None);
                    }
                }
                Poll::Ready(None) => return Poll::Ready(None),
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct AggregateExchangeExec {
    input: Arc<dyn ExecutionPlan>,
    state: Arc<Mutex<AggregateExchangeState>>,
    metrics: ExecutionPlanMetricsSet,
    cache: Arc<PlanProperties>,
    group_exprs: Vec<Arc<dyn PhysicalExpr>>,
}

impl AggregateExchangeExec {
    pub fn try_new(
        input: Arc<dyn ExecutionPlan>,
        group_exprs: Vec<Arc<dyn PhysicalExpr>>,
        num_partitions: usize,
    ) -> Result<Self> {
        let partitioning = Partitioning::Hash(group_exprs.clone(), num_partitions);
        let cache = Self::compute_properties(&input, partitioning);
        Ok(Self {
            input,
            state: Default::default(),
            metrics: ExecutionPlanMetricsSet::new(),
            cache: Arc::new(cache),
            group_exprs,
        })
    }

    pub(crate) fn execute_payload(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableAggPayloadStream> {
        let num_input_partitions = self.input.output_partitioning().partition_count();
        let (receiver, abort_helper) = {
            let mut state = self.state.lock();
            let state = state.consume_input_streams(
                &self.input,
                &self.metrics,
                self.partitioning().partition_count(),
                context,
            )?;
            let receiver = state.receivers.remove(&partition).ok_or_else(|| {
                DataFusionError::Internal(format!(
                    "aggregate exchange partition {partition} has already been executed"
                ))
            })?;
            (receiver, Arc::clone(&state.abort_helper))
        };

        Ok(Box::pin(PerPartitionPayloadStream::new(
            receiver,
            abort_helper,
            num_input_partitions,
        )))
    }

    pub(crate) fn partitioning(&self) -> &Partitioning {
        &self.cache.partitioning
    }

    fn with_new_children_and_same_properties(
        &self,
        mut children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Self {
        Self {
            input: children.swap_remove(0),
            state: Default::default(),
            metrics: ExecutionPlanMetricsSet::new(),
            cache: Arc::clone(&self.cache),
            group_exprs: self.group_exprs.clone(),
        }
    }

    fn compute_properties(
        input: &Arc<dyn ExecutionPlan>,
        partitioning: Partitioning,
    ) -> PlanProperties {
        let mut eq_properties = input.equivalence_properties().clone();
        eq_properties.clear_orderings();
        if input.output_partitioning().partition_count() > 1 {
            eq_properties.clear_per_partition_constants();
        }
        PlanProperties::new(
            eq_properties,
            partitioning,
            EmissionType::Incremental,
            input.boundedness(),
        )
        .with_scheduling_type(SchedulingType::Cooperative)
        .with_evaluation_type(EvaluationType::Eager)
    }
}

impl DisplayAs for AggregateExchangeExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut Formatter) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => write!(
                f,
                "AggregateExchangeExec: partitioning={}, input_partitions={}",
                self.partitioning(),
                self.input.output_partitioning().partition_count()
            ),
            DisplayFormatType::TreeRender => {
                writeln!(f, "partitioning_scheme={}", self.partitioning())
            }
        }
    }
}

impl ExecutionPlan for AggregateExchangeExec {
    fn name(&self) -> &'static str {
        "AggregateExchangeExec"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.cache
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn apply_expressions(
        &self,
        f: &mut dyn FnMut(&dyn PhysicalExpr) -> Result<TreeNodeRecursion>,
    ) -> Result<TreeNodeRecursion> {
        let mut tnr = TreeNodeRecursion::Continue;
        for expr in &self.group_exprs {
            tnr = tnr.visit_sibling(|| f(expr.as_ref()))?;
        }
        Ok(tnr)
    }

    fn with_new_children(
        self: Arc<Self>,
        mut children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        check_if_same_properties!(self, children);
        Ok(Arc::new(Self::try_new(
            children.swap_remove(0),
            self.group_exprs.clone(),
            self.partitioning().partition_count(),
        )?))
    }

    fn benefits_from_input_partitioning(&self) -> Vec<bool> {
        vec![true]
    }

    fn maintains_input_order(&self) -> Vec<bool> {
        vec![false]
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let schema = self.schema();
        let stream = self.execute_payload(partition, context)?;
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            schema,
            stream.map_ok(|payload| payload.batch),
        )))
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }

    fn required_input_distribution(&self) -> Vec<Distribution> {
        vec![Distribution::UnspecifiedDistribution]
    }

    fn with_fetch(&self, _limit: Option<usize>) -> Option<Arc<dyn ExecutionPlan>> {
        None
    }

    fn repartitioned(
        &self,
        _target_partitions: usize,
        _config: &ConfigOptions,
    ) -> Result<Option<Arc<dyn ExecutionPlan>>> {
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aggregates::{AggregateMode, PhysicalGroupBy};
    use crate::execution_plan::collect;
    use crate::test::TestMemoryExec;
    use arrow::array::{Int64Array, StringArray, UInt32Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use datafusion_common::test_util::batches_to_sort_string;
    use datafusion_execution::TaskContext;
    use datafusion_functions_aggregate::sum::sum_udaf;
    use datafusion_physical_expr::aggregate::{
        AggregateExprBuilder, AggregateFunctionExpr,
    };
    use datafusion_physical_expr::expressions::col;

    #[tokio::test]
    async fn test_aggregate_exchange_exec_roundtrip() -> Result<()> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("k", DataType::Utf8, false),
            Field::new("p", DataType::UInt32, false),
            Field::new("v", DataType::Int64, false),
        ]));

        let batch1 = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(StringArray::from(vec!["alpha", "beta", "alpha"])),
                Arc::new(UInt32Array::from(vec![1, 2, 1])),
                Arc::new(Int64Array::from(vec![10, 20, 30])),
            ],
        )?;
        let batch2 = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(StringArray::from(vec!["alpha", "beta", "gamma"])),
                Arc::new(UInt32Array::from(vec![1, 2, 3])),
                Arc::new(Int64Array::from(vec![5, 7, 11])),
            ],
        )?;

        let groups = PhysicalGroupBy::new_single(vec![
            (col("k", &schema)?, "k".to_string()),
            (col("p", &schema)?, "p".to_string()),
        ]);
        let aggregates: Vec<Arc<AggregateFunctionExpr>> = vec![Arc::new(
            AggregateExprBuilder::new(sum_udaf(), vec![col("v", &schema)?])
                .schema(Arc::clone(&schema))
                .alias("sum_v")
                .build()?,
        )];

        let input = TestMemoryExec::try_new_exec(
            &[vec![batch1], vec![batch2]],
            Arc::clone(&schema),
            None,
        )?;
        let partial = Arc::new(AggregateExec::try_new(
            AggregateMode::Partial,
            groups.clone(),
            aggregates.clone(),
            vec![None],
            input,
            Arc::clone(&schema),
        )?);

        let exchange = Arc::new(AggregateExchangeExec::try_new(
            Arc::clone(&partial) as _,
            partial.output_group_expr(),
            2,
        )?);

        let final_agg = Arc::new(AggregateExec::try_new(
            AggregateMode::FinalPartitioned,
            groups.as_final(),
            aggregates,
            vec![None],
            exchange as _,
            Arc::clone(&schema),
        )?);

        let result = collect(final_agg, Arc::new(TaskContext::default())).await?;
        insta::assert_snapshot!(batches_to_sort_string(&result), @r"
            +-------+---+-------+
            | k     | p | sum_v |
            +-------+---+-------+
            | alpha | 1 | 45    |
            | beta  | 2 | 27    |
            | gamma | 3 | 11    |
            +-------+---+-------+
        ");

        Ok(())
    }

    #[tokio::test]
    async fn test_aggregate_exchange_exec_execute_drops_hash_sidecar() -> Result<()> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("k", DataType::Utf8, false),
            Field::new("v", DataType::Int64, false),
        ]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(StringArray::from(vec!["a", "b", "a"])),
                Arc::new(Int64Array::from(vec![1, 2, 3])),
            ],
        )?;

        let groups =
            PhysicalGroupBy::new_single(vec![(col("k", &schema)?, "k".to_string())]);
        let aggregates: Vec<Arc<AggregateFunctionExpr>> = vec![Arc::new(
            AggregateExprBuilder::new(sum_udaf(), vec![col("v", &schema)?])
                .schema(Arc::clone(&schema))
                .alias("sum_v")
                .build()?,
        )];

        let input =
            TestMemoryExec::try_new_exec(&[vec![batch]], Arc::clone(&schema), None)?;
        let partial = Arc::new(AggregateExec::try_new(
            AggregateMode::Partial,
            groups,
            aggregates,
            vec![None],
            input,
            Arc::clone(&schema),
        )?);
        let expected_schema = partial.schema();

        let exchange = Arc::new(AggregateExchangeExec::try_new(
            Arc::clone(&partial) as _,
            partial.output_group_expr(),
            2,
        )?);
        let result = collect(exchange, Arc::new(TaskContext::default())).await?;

        assert!(!result.is_empty());
        for batch in &result {
            assert_eq!(batch.schema(), expected_schema);
            assert!(
                batch.schema()
                    .fields()
                    .iter()
                    .all(|field| field.name() != crate::aggregates::GROUP_HASH_FIELD_NAME)
            );
        }

        Ok(())
    }
}
