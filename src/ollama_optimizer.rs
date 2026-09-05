use futures_util::stream::{self, StreamExt};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use sysinfo::System;
use tokio::sync::RwLock;

use crate::{
    ChatRequest, ChatResponse, EmbedRequest, EmbedResponse, Message, OllamaOptions, RagChunk,
    OLLAMA_URL,
};

// =================================================================================================
// 1. HIGH-THROUGHPUT CONNECTION POOLING CLIENT
// =================================================================================================

pub fn create_optimized_ollama_client() -> Client {
    Client::builder()
        .tcp_nodelay(true)
        .tcp_keepalive(Duration::from_secs(60))
        .pool_idle_timeout(Duration::from_secs(180))
        .pool_max_idle_per_host(64)
        .connect_timeout(Duration::from_secs(5))
        .timeout(Duration::from_secs(300))
        .build()
        .unwrap_or_else(|_| Client::new())
}

// =================================================================================================
// 2. DEEP HARDWARE & GPU PROFILER (AITUNER 2.0)
// =================================================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareProfile {
    pub physical_cores: usize,
    pub logical_cores: usize,
    pub total_memory_mb: u64,
    pub free_memory_mb: u64,
    pub is_apple_silicon: bool,
    pub recommended_mode: String,
    pub optimal_ctx: usize,
    pub optimal_threads: usize,
    pub optimal_gpu_layers: usize,
    pub f16_kv: bool,
    pub use_mmap: bool,
}

pub struct OllamaHardwareProfiler;

impl OllamaHardwareProfiler {
    pub fn profile() -> HardwareProfile {
        let mut sys = System::new_all();
        sys.refresh_memory();
        sys.refresh_cpu_usage();

        let logical_cores = sys.cpus().len().max(1);
        let physical_cores = sysinfo::System::physical_core_count().unwrap_or(std::cmp::max(1, logical_cores / 2));
        let total_mem = sys.total_memory() / (1024 * 1024);
        let free_mem = sys.free_memory() / (1024 * 1024);

        let is_apple = std::env::consts::OS == "macos" && std::env::consts::ARCH == "aarch64";

        let (mode, ctx, threads, gpu_layers, f16) = if is_apple {
            // Apple Silicon Unified Memory: saturate Metal GPU
            ("APPLE_SILICON_TURBO".to_string(), 16384, physical_cores, 999, true)
        } else if total_mem < 8192 || physical_cores <= 2 {
            // Low RAM potato machine: strictly minimize context to avoid OOM
            ("POTATO_ECO".to_string(), 2048, std::cmp::max(1, physical_cores), 1, true)
        } else if total_mem < 16384 {
            // Balanced Workstation
            ("BALANCED".to_string(), 4096, physical_cores, 33, true)
        } else {
            // High-End Workstation / Rig
            ("TURBO_MAX".to_string(), 8192, physical_cores, 999, true)
        };

        HardwareProfile {
            physical_cores,
            logical_cores,
            total_memory_mb: total_mem,
            free_memory_mb: free_mem,
            is_apple_silicon: is_apple,
            recommended_mode: mode,
            optimal_ctx: ctx,
            optimal_threads: threads,
            optimal_gpu_layers: gpu_layers,
            f16_kv: f16,
            use_mmap: true,
        }
    }

    pub fn build_optimized_options(base_temp: f32) -> OllamaOptions {
        let prof = Self::profile();
        OllamaOptions {
            temperature: base_temp,
            num_ctx: Some(prof.optimal_ctx),
            num_thread: Some(prof.optimal_threads),
            num_gpu: Some(prof.optimal_gpu_layers),
            top_k: Some(40),
            top_p: Some(0.9),
            repeat_penalty: Some(1.1),
            f16_kv: Some(prof.f16_kv),
            use_mmap: Some(prof.use_mmap),
            use_mlock: None,
            num_predict: Some(prof.optimal_ctx as i32),
            stop: None,
        }
    }
}

// =================================================================================================
// 3. BATCHED PARALLEL EMBEDDING ENGINE WITH IN-MEMORY / HASH CACHE
// =================================================================================================

#[derive(Clone)]
pub struct BatchedEmbeddingEngine {
    client: Client,
    cache: Arc<RwLock<HashMap<String, Vec<f32>>>>,
    cache_hits: Arc<AtomicUsize>,
    cache_misses: Arc<AtomicUsize>,
}

impl BatchedEmbeddingEngine {
    pub fn new(client: Client) -> Self {
        Self {
            client,
            cache: Arc::new(RwLock::new(HashMap::new())),
            cache_hits: Arc::new(AtomicUsize::new(0)),
            cache_misses: Arc::new(AtomicUsize::new(0)),
        }
    }

    pub fn compute_text_hash(text: &str) -> String {
        // Quick 64-bit FNV-1a hash formatted as hex for zero external dependencies
        let mut hash: u64 = 0xcbf29ce484222325;
        for byte in text.as_bytes() {
            hash ^= *byte as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
        format!("{:016x}", hash)
    }

    pub async fn embed_single(&self, text: &str, model: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let hash = Self::compute_text_hash(text);

        // Check cache first
        {
            let cache_read = self.cache.read().await;
            if let Some(vec) = cache_read.get(&hash) {
                self.cache_hits.fetch_add(1, Ordering::Relaxed);
                return Ok(vec.clone());
            }
        }

        self.cache_misses.fetch_add(1, Ordering::Relaxed);

        let req = EmbedRequest {
            model: model.to_string(),
            prompt: text.to_string(),
            keep_alive: Some(-1),
            options: None,
        };

        let res = self.client.post(format!("{}/api/embeddings", OLLAMA_URL))
            .json(&req)
            .send()
            .await?;

        if res.status().is_success() {
            let parsed: EmbedResponse = res.json().await?;
            let mut cache_write = self.cache.write().await;
            cache_write.insert(hash, parsed.embedding.clone());
            Ok(parsed.embedding)
        } else {
            Err("Failed to compute embedding from Ollama".into())
        }
    }

    pub async fn embed_batch(
        &self,
        chunks: Vec<(String, String)>, // (file_path, chunk_text)
        model: &str,
        concurrency: usize,
    ) -> Vec<RagChunk> {
        let concurrency_limit = concurrency.clamp(1, 32);

        let results = stream::iter(chunks)
            .map(|(file, text)| {
                let engine = self.clone();
                let mdl = model.to_string();
                async move {
                    match engine.embed_single(&text, &mdl).await {
                        Ok(vector) => Some(RagChunk { file, text, vector }),
                        Err(_) => None,
                    }
                }
            })
            .buffer_unordered(concurrency_limit)
            .collect::<Vec<Option<RagChunk>>>()
            .await;

        results.into_iter().flatten().collect()
    }

    pub fn get_cache_stats(&self) -> (usize, usize) {
        (
            self.cache_hits.load(Ordering::Relaxed),
            self.cache_misses.load(Ordering::Relaxed),
        )
    }
}

// =================================================================================================
// 4. ZERO-COPY RESILIENT STREAMING TOKEN ACCUMULATOR
// =================================================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamTelemetry {
    pub total_tokens: usize,
    pub time_to_first_token_ms: u64,
    pub total_duration_ms: u64,
    pub tokens_per_second: f64,
    pub reasoning_tokens: usize,
    pub prompt_eval_count: u64,
    pub prompt_eval_duration: u64,
}

pub struct StreamingTokenBuffer {
    pub raw_buffer: String,
    pub accumulated_text: String,
    pub reasoning_text: String,
    pub in_reasoning_block: bool,
    pub token_count: usize,
    pub reasoning_token_count: usize,
    pub start_time: Instant,
    pub first_token_time: Option<Instant>,
    pub prompt_eval_count: u64,
    pub prompt_eval_duration: u64,
}

impl StreamingTokenBuffer {
    pub fn new() -> Self {
        Self {
            raw_buffer: String::new(),
            accumulated_text: String::new(),
            reasoning_text: String::new(),
            in_reasoning_block: false,
            token_count: 0,
            reasoning_token_count: 0,
            start_time: Instant::now(),
            first_token_time: None,
            prompt_eval_count: 0,
            prompt_eval_duration: 0,
        }
    }

    pub fn push_chunk(&mut self, chunk_bytes: &[u8]) -> Vec<String> {
        let chunk_str = String::from_utf8_lossy(chunk_bytes);
        self.raw_buffer.push_str(&chunk_str);

        let mut emitted_tokens = Vec::new();

        // Process line-by-line while retaining partial trailing line in raw_buffer
        while let Some(nl_pos) = self.raw_buffer.find('\n') {
            let line = self.raw_buffer[..nl_pos].trim().to_string();
            self.raw_buffer.drain(..=nl_pos);

            if line.is_empty() {
                continue;
            }

            if let Ok(parsed) = serde_json::from_str::<ChatResponse>(&line) {
                if let Some(count) = parsed.prompt_eval_count {
                    self.prompt_eval_count = count;
                }
                if let Some(duration) = parsed.prompt_eval_duration {
                    self.prompt_eval_duration = duration;
                }

                if let Some(msg) = parsed.message {
                    let content = msg.content;
                    if !content.is_empty() {
                        if self.first_token_time.is_none() {
                            self.first_token_time = Some(Instant::now());
                        }

                        self.token_count += 1;

                        if content.contains("<think>") {
                            self.in_reasoning_block = true;
                        }

                        if self.in_reasoning_block {
                            self.reasoning_text.push_str(&content);
                            self.reasoning_token_count += 1;
                        } else {
                            self.accumulated_text.push_str(&content);
                            emitted_tokens.push(content.clone());
                        }

                        if content.contains("</think>") {
                            self.in_reasoning_block = false;
                        }
                    }
                }
            }
        }

        emitted_tokens
    }

    pub fn finalize(&self) -> StreamTelemetry {
        let total_ms = self.start_time.elapsed().as_millis() as u64;
        let ttft_ms = self.first_token_time
            .map(|t| t.duration_since(self.start_time).as_millis() as u64)
            .unwrap_or(total_ms);

        let gen_duration_secs = (total_ms.saturating_sub(ttft_ms) as f64) / 1000.0;
        let tps = if gen_duration_secs > 0.001 {
            self.token_count as f64 / gen_duration_secs
        } else {
            0.0
        };

        StreamTelemetry {
            total_tokens: self.token_count,
            time_to_first_token_ms: ttft_ms,
            total_duration_ms: total_ms,
            tokens_per_second: tps,
            reasoning_tokens: self.reasoning_token_count,
            prompt_eval_count: self.prompt_eval_count,
            prompt_eval_duration: self.prompt_eval_duration,
        }
    }
}

// =================================================================================================
// 5. PROMPT KV-CACHE OPTIMIZER & PREFIX PRESERVER
// =================================================================================================

pub struct PromptKvOptimizer;

impl PromptKvOptimizer {
    pub fn optimize_message_history(
        system_prompt: Option<&str>,
        history: &[Message],
        max_context_tokens: usize,
    ) -> Vec<Message> {
        let mut optimized = Vec::new();

        // 1. Fixed invariant prefix: System prompt MUST stay at index 0 without modifications
        if let Some(sys) = system_prompt {
            optimized.push(Message {
                role: "system".to_string(),
                content: sys.to_string(),
                tool_calls: None,
                images: None,
            });
        }

        // 2. Budget remaining tokens across conversation history
        // If history exceeds context window, keep recent turns intact to preserve KV cache
        let estimated_tokens_per_char = 0.25f32; // ~4 chars per token
        let mut budget = max_context_tokens.saturating_sub(512); // Reserve 512 for generation

        let mut kept_turns = Vec::new();
        for msg in history.iter().rev() {
            let msg_tokens = (msg.content.len() as f32 * estimated_tokens_per_char) as usize;
            if msg_tokens < budget {
                budget -= msg_tokens;
                kept_turns.push(msg.clone());
            } else {
                break;
            }
        }

        kept_turns.reverse();
        optimized.extend(kept_turns);
        optimized
    }
}

// =================================================================================================
// 6. MODEL PRE-WARMING & LATENCY BENCHMARK ENGINE
// =================================================================================================

pub struct OllamaBenchmarkEngine;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaBenchmarkReport {
    pub model: String,
    pub prompt_eval_tps: f64,
    pub generation_tps: f64,
    pub time_to_first_token_ms: u64,
    pub total_latency_ms: u64,
    pub status: String,
}

impl OllamaBenchmarkEngine {
    pub async fn prewarm_model(client: &Client, model: &str) -> Result<bool, Box<dyn std::error::Error>> {
        let req = ChatRequest {
            model: model.to_string(),
            messages: vec![Message {
                role: "user".to_string(),
                content: "1".to_string(),
                tool_calls: None,
                images: None,
            }],
            stream: false,
            tools: None,
            format: None,
            options: Some(OllamaOptions {
                temperature: 0.0,
                num_ctx: Some(256),
                num_thread: Some(2),
                num_gpu: Some(999),
                top_k: None,
                top_p: None,
                repeat_penalty: None,
                f16_kv: Some(true),
                use_mmap: Some(true),
                use_mlock: None,
                num_predict: Some(1),
                stop: None,
            }),
            keep_alive: Some(-1),
        };

        let res = client.post(format!("{}/api/chat", OLLAMA_URL))
            .json(&req)
            .send()
            .await?;

        Ok(res.status().is_success())
    }

    pub async fn run_benchmark(client: &Client, model: &str) -> Result<OllamaBenchmarkReport, Box<dyn std::error::Error>> {
        let test_prompt = "Explain in 2 sentences why Rust memory safety without a garbage collector is fast.";
        let options = OllamaHardwareProfiler::build_optimized_options(0.1);

        let req = ChatRequest {
            model: model.to_string(),
            messages: vec![Message {
                role: "user".to_string(),
                content: test_prompt.to_string(),
                tool_calls: None,
                images: None,
            }],
            stream: true,
            tools: None,
            format: None,
            options: Some(options),
            keep_alive: Some(-1),
        };

        let _start = Instant::now();
        let res = client.post(format!("{}/api/chat", OLLAMA_URL))
            .json(&req)
            .send()
            .await?;

        if !res.status().is_success() {
            return Ok(OllamaBenchmarkReport {
                model: model.to_string(),
                prompt_eval_tps: 0.0,
                generation_tps: 0.0,
                time_to_first_token_ms: 0,
                total_latency_ms: 0,
                status: format!("Model '{}' not available on local Ollama daemon", model),
            });
        }

        let mut stream = res.bytes_stream();
        let mut buffer = StreamingTokenBuffer::new();

        while let Some(chunk_result) = stream.next().await {
            let chunk = chunk_result?;
            let _ = buffer.push_chunk(&chunk);
        }

        let telemetry = buffer.finalize();

        let prompt_eval_tps = if telemetry.prompt_eval_duration > 0 {
            (telemetry.prompt_eval_count as f64) / (telemetry.prompt_eval_duration as f64 / 1_000_000_000.0)
        } else {
            0.0
        };

        Ok(OllamaBenchmarkReport {
            model: model.to_string(),
            prompt_eval_tps,
            generation_tps: telemetry.tokens_per_second,
            time_to_first_token_ms: telemetry.time_to_first_token_ms,
            total_latency_ms: telemetry.total_duration_ms,
            status: "success".to_string(),
        })
    }
}
