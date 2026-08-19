use std::{
    cell::Cell,
    mem::MaybeUninit,
    slice,
    sync::atomic::{AtomicU64, Ordering},
};

use anyhow::ensure;
use singe_cuda::{
    event::{Event, EventFlags, EventRecordFlags},
    memory::{DeviceMemory, HostAllocationFlags, MemoryCopyKind},
};

use crate::{
    context::{Context, CudaStream},
    tensor::TensorView,
};

/// Utility for performing speed-of-light batched copies
/// between CPU and GPU memory, using a bump allocator
/// of page-locked memory.
#[derive(Debug)]
pub struct BulkCopyEngine {
    cx: Context,
    streams: Vec<CudaStream>,
    completion_events: Vec<Event>,
    did_record_batch: bool,
    did_observe_completion: Cell<bool>,
    memory_chunks: Vec<Chunk>,
    engine_id: u64,
    next_epoch: u64,
}

const INITIAL_CHUNK_SIZE: usize = 2 * 1024 * 1024;
const CHUNK_GROWTH_FACTOR: usize = 2;
const NUM_STREAMS: usize = 2;

impl BulkCopyEngine {
    pub fn new(cx: &Context) -> anyhow::Result<Self> {
        cx.cuda().bind()?;
        let streams = (0..NUM_STREAMS)
            .map(|_| CudaStream::new(cx))
            .collect::<Result<Vec<_>, _>>()?;

        let completion_events = (0..NUM_STREAMS)
            .map(|_| {
                cx.cuda()
                    .create_event_with_flags(EventFlags::DISABLE_TIMING)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let memory_chunks = vec![Chunk::alloc(cx, INITIAL_CHUNK_SIZE)?];

        static NEXT_ENGINE_ID: AtomicU64 = AtomicU64::new(0);

        Ok(Self {
            cx: cx.clone(),
            streams,
            completion_events,
            memory_chunks,
            did_record_batch: true,
            did_observe_completion: Cell::new(true),
            engine_id: NEXT_ENGINE_ID.fetch_add(1, Ordering::Relaxed),
            next_epoch: 0,
        })
    }

    fn wait(&self) -> anyhow::Result<()> {
        assert!(self.did_record_batch, "must call BulkCopyBatch::commit()");

        if !self.did_observe_completion.get() {
            self.cx.cuda().bind()?;
            for event in &self.completion_events {
                event.synchronize()?;
            }
            self.did_observe_completion.set(true);
        }

        Ok(())
    }

    /// Retrieves a data slice that is the result of a previous
    /// device=>host copy operation. Panics if the provided
    /// token was not from the most recent copy operation.
    /// This blocks on copy operation completion if needed.
    pub fn retrieve(&self, token: CopyToken) -> anyhow::Result<&[u8]> {
        assert_eq!(token.engine_id, self.engine_id);
        assert_eq!(token.epoch + 1, self.next_epoch);

        self.wait()?;

        let chunk = &self.memory_chunks[token.chunk_idx];
        let ptr = unsafe { chunk.ptr.add(token.offset_in_chunk) };
        let slice = unsafe { slice::from_raw_parts(ptr, token.num_bytes) };

        Ok(slice)
    }

    /// Starts a new copy operation.
    ///
    /// This waits for the previous operation to complete,
    /// so that its page-locked host memory can be reused.
    /// All previous CopyTokens are invalidated.
    pub fn start_batch(&mut self) -> anyhow::Result<BulkCopyBatch<'_>> {
        self.wait()?;

        // Keep only the last (largest-sized) chunk to reduce fragmentation.
        let c = self.memory_chunks.len();
        self.memory_chunks.drain(..c.checked_sub(1).unwrap());

        for chunk in &mut self.memory_chunks {
            chunk.used = 0;
        }

        let epoch = self.next_epoch;
        self.next_epoch += 1;
        self.did_observe_completion.set(false);
        self.did_record_batch = false;

        Ok(BulkCopyBatch {
            engine: self,
            epoch,
            num_ops: 0,
        })
    }

    fn chunk_with_space_for(&mut self, num_bytes: usize) -> anyhow::Result<(usize, &mut Chunk)> {
        let chunk = self.memory_chunks.last_mut().unwrap();
        if chunk.capacity - chunk.used >= num_bytes {
            return Ok((
                self.memory_chunks.len() - 1,
                self.memory_chunks.last_mut().unwrap(),
            ));
        }

        let new_chunk_size = chunk
            .capacity
            .checked_mul(CHUNK_GROWTH_FACTOR)
            .unwrap()
            .max(num_bytes);
        self.memory_chunks
            .push(Chunk::alloc(&self.cx, new_chunk_size)?);

        Ok((
            self.memory_chunks.len() - 1,
            self.memory_chunks.last_mut().unwrap(),
        ))
    }
}

#[derive(Debug)]
pub struct BulkCopyBatch<'a> {
    engine: &'a mut BulkCopyEngine,
    num_ops: usize,
    epoch: u64,
}

impl<'a> BulkCopyBatch<'a> {
    /// Copy some data to a device tensor, using in-place
    /// writing to page-locked memory. Write data to be copied
    /// into the provided slice.
    pub fn copy_to_device(
        &mut self,
        tensor: &TensorView,
        callback: impl FnOnce(&mut [u8]),
    ) -> anyhow::Result<()> {
        ensure!(
            tensor.is_contiguous(),
            "transfers for non-contiguous tensors are unimplemented"
        );
        let num_bytes = tensor.allocation_size();

        let (_, chunk) = self.engine.chunk_with_space_for(num_bytes)?;

        let slice = unsafe { slice::from_raw_parts_mut(chunk.ptr.add(chunk.used), num_bytes) };
        callback(slice);
        chunk.used += num_bytes;

        unsafe {
            DeviceMemory::<u8>::copy_async(
                tensor.data_ptr().as_raw().cast::<u8>(),
                slice.as_ptr(),
                num_bytes,
                MemoryCopyKind::HostToDevice,
                self.engine.streams[self.num_ops % NUM_STREAMS].get(),
            )?;
        }

        self.num_ops += 1;

        Ok(())
    }

    /// Copies device tensor to host buffer. Returns token
    /// which can later be used to access the data after
    /// completion of the copy.
    pub fn copy_to_host(&mut self, tensor: &TensorView) -> anyhow::Result<CopyToken> {
        ensure!(
            tensor.is_contiguous(),
            "transfers for non-contiguous tensors are unimplemented"
        );
        let num_bytes = tensor.allocation_size();

        let stream = self.engine.streams[self.num_ops % NUM_STREAMS].clone();

        let (chunk_idx, chunk) = self.engine.chunk_with_space_for(num_bytes)?;

        let slice = unsafe { slice::from_raw_parts_mut(chunk.ptr.add(chunk.used), num_bytes) };
        chunk.used += num_bytes;

        unsafe {
            DeviceMemory::<u8>::copy_async(
                slice.as_mut_ptr(),
                tensor.data_ptr().as_raw().cast::<u8>(),
                num_bytes,
                MemoryCopyKind::DeviceToHost,
                stream.get(),
            )?;
        }

        self.num_ops += 1;

        Ok(CopyToken {
            chunk_idx,
            offset_in_chunk: chunk.used - num_bytes,
            num_bytes,
            engine_id: self.engine.engine_id,
            epoch: self.epoch,
        })
    }

    pub fn commit(self) -> anyhow::Result<()> {
        for (stream, event) in self
            .engine
            .streams
            .iter()
            .zip(&self.engine.completion_events)
        {
            event.record(stream.get(), EventRecordFlags::DEFAULT)?;
        }

        self.engine.did_record_batch = true;

        Ok(())
    }
}

impl<'a> Drop for BulkCopyBatch<'a> {
    fn drop(&mut self) {
        assert!(
            self.engine.did_record_batch,
            "must call BulkCopyBatch::commit()"
        );
    }
}

/// Token that allows retrieving device=>host-copied
/// data once the transfer completes.
#[derive(Debug)]
pub struct CopyToken {
    chunk_idx: usize,
    offset_in_chunk: usize,
    num_bytes: usize,
    engine_id: u64,
    epoch: u64,
}

#[derive(Debug)]
struct Chunk {
    cx: Context,
    ptr: *mut u8,
    capacity: usize,
    used: usize,
}

impl Chunk {
    fn alloc(cx: &Context, capacity: usize) -> anyhow::Result<Self> {
        let ptr =
            unsafe { DeviceMemory::<u8>::alloc_pinned(capacity, HostAllocationFlags::DEFAULT)? };
        unsafe {
            slice::from_raw_parts_mut(ptr.cast::<MaybeUninit<u8>>(), capacity)
                .fill(MaybeUninit::new(0));
        }
        Ok(Self {
            cx: cx.clone(),
            ptr: ptr.cast(),
            capacity,
            used: 0,
        })
    }
}

impl Drop for Chunk {
    fn drop(&mut self) {
        unsafe {
            self.cx.cuda().bind().ok();
            DeviceMemory::<u8>::free_host(self.ptr.cast()).ok();
        }
    }
}
