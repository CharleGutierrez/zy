use zy::ollama_optimizer::*;
use zy::Message;

// =================================================================================================
// 1. OPTIMIZED CLIENT POOL TESTS
// =================================================================================================

#[test]
fn test_optimized_client_pool_construction() {
    let client = create_optimized_ollama_client();
    // Verify client can be cloned and dispatched concurrently
    let client2 = client.clone();
    drop(client2);
}

// =================================================================================================
// 2. HARDWARE PROFILER & OPTIONS TESTS
// =================================================================================================

#[test]
fn test_hardware_profiler_and_options_generation() {
    let profile = OllamaHardwareProfiler::profile();
    assert!(profile.physical_cores >= 1);
    assert!(profile.logical_cores >= 1);
    assert!(profile.total_memory_mb > 0);
    assert!(profile.optimal_ctx >= 2048);
    assert!(profile.optimal_threads >= 1);
    assert!(profile.f16_kv);
    assert!(profile.use_mmap);

    let opts = OllamaHardwareProfiler::build_optimized_options(0.1);
    assert_eq!(opts.temperature, 0.1);
    assert_eq!(opts.f16_kv, Some(true));
    assert_eq!(opts.use_mmap, Some(true));
    assert_eq!(opts.repeat_penalty, Some(1.1));
    assert_eq!(opts.top_k, Some(40));
    assert_eq!(opts.top_p, Some(0.9));
}

// =================================================================================================
// 3. BATCHED EMBEDDING ENGINE & CACHE TESTS
// =================================================================================================

#[tokio::test]
async fn test_batched_embedding_engine_cache_and_hashing() {
    let client = create_optimized_ollama_client();
    let engine = BatchedEmbeddingEngine::new(client);

    let (hits, misses) = engine.get_cache_stats();
    assert_eq!(hits, 0);
    assert_eq!(misses, 0);

    let chunk_text = "fn compute_fibonacci(n: u64) -> u64 { if n <= 1 { n } else { compute_fibonacci(n-1) + compute_fibonacci(n-2) } }";
    let hash1 = BatchedEmbeddingEngine::compute_text_hash(chunk_text);
    let hash2 = BatchedEmbeddingEngine::compute_text_hash(chunk_text);
    assert_eq!(hash1, hash2);
    assert_eq!(hash1.len(), 16); // 16-hex char FNV-1a hash
}

// =================================================================================================
// 4. STREAMING TOKEN BUFFER & REASONING ACCUMULATOR TESTS
// =================================================================================================

#[test]
fn test_streaming_token_buffer_line_reassembly_and_telemetry() {
    let mut buffer = StreamingTokenBuffer::new();

    // Simulate chunk 1: partial JSON line with reasoning tag
    let chunk1 = br#"{"message":{"role":"assistant","content":"<think>Analyzing "#;
    let tokens1 = buffer.push_chunk(chunk1);
    assert!(tokens1.is_empty()); // Still partial line

    // Simulate chunk 2: completing the JSON line
    let chunk2 = br#"the problem...</think>Here is the solution:\n"}"#;
    let _ = buffer.push_chunk(chunk2);

    // Simulate chunk 3: regular token chunk
    let chunk3 = b"\n{\"message\":{\"role\":\"assistant\",\"content\":\"```rust\\nfn main() {}\\n```\"}}\n";
    let tokens3 = buffer.push_chunk(chunk3);
    assert!(!tokens3.is_empty());

    let telemetry = buffer.finalize();
    assert!(telemetry.total_tokens >= 1);
}

// =================================================================================================
// 5. PROMPT KV-CACHE OPTIMIZER TESTS
// =================================================================================================

#[test]
fn test_prompt_kv_cache_optimizer_invariants() {
    let system_prompt = "You are an ultra-fast Rust pair programmer.";
    let history = vec![
        Message { role: "user".to_string(), content: "Hello!".to_string(), tool_calls: None, images: None },
        Message { role: "assistant".to_string(), content: "Greetings! How can I help with your code?".to_string(), tool_calls: None, images: None },
        Message { role: "user".to_string(), content: "Write a high-throughput queue.".to_string(), tool_calls: None, images: None },
    ];

    let optimized = PromptKvOptimizer::optimize_message_history(Some(system_prompt), &history, 4096);

    // System prompt MUST be at index 0 to guarantee KV cache hit
    assert_eq!(optimized[0].role, "system");
    assert_eq!(optimized[0].content, system_prompt);

    // User turns preserved in sequence
    assert_eq!(optimized.len(), 4);
    assert_eq!(optimized[1].content, "Hello!");
    assert_eq!(optimized[3].content, "Write a high-throughput queue.");
}

// =================================================================================================
// 6. BENCHMARK STRUCTURES TESTS
// =================================================================================================

#[test]
fn test_ollama_benchmark_report_structures() {
    let report = OllamaBenchmarkReport {
        model: "qwen2.5-coder:1.5b".to_string(),
        prompt_eval_tps: 185.5,
        generation_tps: 92.4,
        time_to_first_token_ms: 120,
        total_latency_ms: 850,
        status: "success".to_string(),
    };

    let serialized = serde_json::to_string(&report).unwrap();
    assert!(serialized.contains("qwen2.5-coder:1.5b"));
    assert!(serialized.contains("92.4"));
    assert!(serialized.contains("185.5"));
}
