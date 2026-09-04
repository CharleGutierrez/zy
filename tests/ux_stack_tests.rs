use reqwest::Client;
use std::path::Path;
use zy::ux_stack::*;

// =================================================================================================
// SYSTEM 1 TESTS: TERMINAL UI & GRAPHICS PROTOCOL ENGINE
// =================================================================================================

#[test]
fn test_system1_terminal_capabilities_and_graphics_escape() {
    let caps = TerminalCapabilities::detect();
    assert!(caps.columns > 0);
    assert!(caps.rows > 0);

    // 4x2 RGBA image (8 pixels = 32 bytes)
    let raw_rgba: Vec<u8> = vec![
        255, 0, 0, 255,   0, 255, 0, 255,   0, 0, 255, 255,   255, 255, 255, 255,
        0, 0, 0, 255,     128, 128, 128, 255, 255, 255, 0, 255, 0, 255, 255, 255,
    ];

    // 1. Test Kitty protocol
    let mut kitty_caps = caps.clone();
    kitty_caps.protocol = TerminalGraphicsProtocol::Kitty;
    let kitty_esc = kitty_caps.render_image_escape(&raw_rgba, 4, 2);
    assert!(kitty_esc.starts_with("\x1b_Gf=32"));
    assert!(kitty_esc.ends_with("\x1b\\"));

    // 2. Test iTerm2 protocol
    let mut iterm_caps = caps.clone();
    iterm_caps.protocol = TerminalGraphicsProtocol::ITerm2;
    let iterm_esc = iterm_caps.render_image_escape(&raw_rgba, 4, 2);
    assert!(iterm_esc.starts_with("\x1b]1337;File=inline=1;width=4px;height=2px:"));
    assert!(iterm_esc.ends_with("\x07"));

    // 3. Test TrueColor half-block fallback
    let mut block_caps = caps.clone();
    block_caps.protocol = TerminalGraphicsProtocol::BlockTrueColor;
    let block_esc = block_caps.render_image_escape(&raw_rgba, 4, 2);
    assert!(block_esc.contains("▀"));
    assert!(block_esc.contains("\x1b[38;2;"));

    // 4. Test ASCII fallback
    let mut ascii_caps = caps.clone();
    ascii_caps.protocol = TerminalGraphicsProtocol::AsciiFallback;
    let ascii_esc = ascii_caps.render_image_escape(&raw_rgba, 4, 2);
    assert!(!ascii_esc.is_empty());
}

#[test]
fn test_system1_streaming_syntax_highlighter() {
    let mut highlighter = StreamingSyntaxHighlighter::new();
    assert!(!highlighter.in_code_block);

    // Stream regular text
    let out1 = highlighter.process_token("Here is the requested Rust function:\n");
    assert!(out1.contains("Here is the requested"));
    assert!(!highlighter.in_code_block);

    // Stream opening code block
    let _out2 = highlighter.process_token("```rust\n");
    assert!(highlighter.in_code_block);
    assert_eq!(highlighter.current_lang.as_deref(), Some("rust"));

    // Stream code token with keywords
    let out3 = highlighter.process_token("pub async fn execute() -> bool {\n");
    assert!(out3.contains("pub"));
    assert!(out3.contains("async"));
    assert!(out3.contains("fn"));

    // Stream closing code block
    let _out4 = highlighter.process_token("}\n```\n");
    assert!(!highlighter.in_code_block);
}

#[test]
fn test_system1_tui_split_multiplexer_navigation() {
    let mut mux = TuiSplitMultiplexer::new();
    assert_eq!(mux.active_index, 1);
    assert!(mux.center_pane.is_active);
    assert!(!mux.left_pane.is_active);

    // Cycle pane to right
    mux.cycle_pane();
    assert_eq!(mux.active_index, 2);
    assert!(mux.right_pane.is_active);

    // Cycle pane to left
    mux.cycle_pane();
    assert_eq!(mux.active_index, 0);
    assert!(mux.left_pane.is_active);

    // Format for terminal
    let formatted = mux.format_tui_layout_for_terminal();
    assert!(formatted.contains("CODEBASE & RAG TREE"));
    assert!(formatted.contains("AGENT REPL / LIVE DIFF"));
    assert!(formatted.contains("SWARM TELEMETRY & RADAR"));
}

// =================================================================================================
// SYSTEM 2 TESTS: DESKTOP GUI & HUD SPOTLIGHT OVERLAY PROTOCOL
// =================================================================================================

#[tokio::test]
async fn test_system2_desktop_hud_rpc_handshake_and_search() {
    let bridge = DesktopHudBridge::new(8105);

    // 1. Handshake request
    let handshake_req = DesktopHudMessage {
        jsonrpc: "2.0".to_string(),
        id: Some(101),
        method: "hud/handshake".to_string(),
        params: serde_json::json!({}),
    };
    let handshake_resp = DesktopHudBridge::handle_hud_rpc_message(&handshake_req, &bridge).await;
    assert_eq!(handshake_resp.jsonrpc, "2.0");
    assert_eq!(handshake_resp.id, Some(101));
    let res = handshake_resp.result.unwrap();
    assert_eq!(res["app"], "zy");
    assert_eq!(res["status"], "connected");
    assert_eq!(res["hud_protocol_version"], "2.0");

    // 2. Spotlight search query
    let search_req = DesktopHudMessage {
        jsonrpc: "2.0".to_string(),
        id: Some(102),
        method: "hud/spotlight_search".to_string(),
        params: serde_json::json!({ "query": "radar" }),
    };
    let search_resp = DesktopHudBridge::handle_hud_rpc_message(&search_req, &bridge).await;
    let results_arr = search_resp.result.unwrap();
    assert!(results_arr.as_array().unwrap().len() > 0);
    let first = &results_arr.as_array().unwrap()[0];
    assert!(first["title"].as_str().unwrap().contains("Radar") || first["title"].as_str().unwrap().contains("radar"));

    // 3. Telemetry request
    let telemetry_req = DesktopHudMessage {
        jsonrpc: "2.0".to_string(),
        id: Some(103),
        method: "hud/get_telemetry".to_string(),
        params: serde_json::json!({}),
    };
    let telemetry_resp = DesktopHudBridge::handle_hud_rpc_message(&telemetry_req, &bridge).await;
    let telem = telemetry_resp.result.unwrap();
    assert!(telem["total_memory_mb"].as_u64().unwrap() > 0);
    assert!(telem["mode"].as_str().unwrap().contains("MODE"));

    // 4. Tool approval request
    let approve_req = DesktopHudMessage {
        jsonrpc: "2.0".to_string(),
        id: Some(104),
        method: "hud/approve_tool".to_string(),
        params: serde_json::json!({ "call_id": "call_12345", "approved": true }),
    };
    let approve_resp = DesktopHudBridge::handle_hud_rpc_message(&approve_req, &bridge).await;
    let app_res = approve_resp.result.unwrap();
    assert_eq!(app_res["call_id"], "call_12345");
    assert_eq!(app_res["approved"], true);
    assert_eq!(app_res["status"], "executed");
}

// =================================================================================================
// SYSTEM 3 TESTS: RICH VISUALIZATIONS & INTERACTIVE COMPONENT ENGINE
// =================================================================================================

#[test]
fn test_system3_swarm_dag_layout_svg_and_mermaid() {
    let subtasks = vec![
        "Lexical & AST parsing".to_string(),
        "Generate RAG chunk embeddings".to_string(),
        "Synthesize unit tests".to_string(),
    ];
    let dag = DagLayout::build_swarm_workflow_dag("Refactor codebase architecture", &subtasks);

    assert_eq!(dag.nodes.len(), 5); // Architect + 3 Workers + Verifier
    assert_eq!(dag.edges.len(), 4);

    // Verify SVG export
    let svg = dag.to_svg();
    assert!(svg.starts_with("<svg xmlns=\"http://www.w3.org/2000/svg\""));
    assert!(svg.contains("marker id=\"arrow\""));
    assert!(svg.contains("filter id=\"glow\""));
    assert!(svg.contains("Architect:"));
    assert!(svg.contains("QA Verifier"));
    assert!(svg.ends_with("</svg>"));

    // Verify Mermaid export
    let mermaid = dag.to_mermaid();
    assert!(mermaid.starts_with("graph TD\n"));
    assert!(mermaid.contains("node_architect[\"Architect:"));
    assert!(mermaid.contains("node_verifier[\"QA Verifier"));
    assert!(mermaid.contains("-->"));
}

#[test]
fn test_system3_codebase_health_radar_metrics_and_svg() {
    let metrics = CodebaseRadarMetrics::calculate(Path::new("."));

    assert!(metrics.maintainability >= 0.0 && metrics.maintainability <= 100.0);
    assert!(metrics.complexity >= 0.0 && metrics.complexity <= 100.0);
    assert!(metrics.test_coverage >= 0.0 && metrics.test_coverage <= 100.0);
    assert!(metrics.security >= 0.0 && metrics.security <= 100.0);
    assert!(metrics.performance >= 0.0 && metrics.performance <= 100.0);
    assert!(metrics.documentation >= 0.0 && metrics.documentation <= 100.0);
    assert!(metrics.overall_score >= 0.0 && metrics.overall_score <= 100.0);

    let svg = metrics.to_svg();
    assert!(svg.starts_with("<svg xmlns=\"http://www.w3.org/2000/svg\""));
    assert!(svg.contains("Maintainability"));
    assert!(svg.contains("Security"));
    assert!(svg.contains("Overall Score:"));
    assert!(svg.contains("<polygon points="));
    assert!(svg.ends_with("</svg>"));
}

#[test]
fn test_system3_fast_fourier_voice_spectrum_engine() {
    // Generate 16kHz sine wave audio (440Hz A tone)
    let sample_rate = 16000u32;
    let freq = 440.0f32;
    let audio: Vec<f32> = (0..sample_rate)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            (2.0 * std::f32::consts::PI * freq * t).sin() * 0.5
        })
        .collect();

    let spectrum = FastFourierSpectrum::compute(&audio, sample_rate);
    assert_eq!(spectrum.sample_rate, 16000);
    assert_eq!(spectrum.bin_count, 32);
    assert!(spectrum.rms_volume > 0.05);
    assert!(spectrum.is_speech_active);

    // Peak frequency should be detected in the audible band
    assert!(spectrum.peak_frequency_hz > 0.0);

    let ascii_bars = spectrum.to_ascii_bars();
    assert_eq!(ascii_bars.chars().count(), 32);
}

// =================================================================================================
// SYSTEM 4 TESTS: EMBEDDED LOCAL WEB DASHBOARD (ZERO-INSTALL SERVER)
// =================================================================================================

#[tokio::test]
async fn test_system4_embedded_web_dashboard_live_endpoints() {
    let port = 18899u16;
    let handle = EmbeddedWebDashboard::start(port).await.expect("Failed to start embedded web server");

    // Give server 50ms to bind
    tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;

    let client = Client::new();

    // 1. Test GET / (HTML SPA DOM)
    let resp_root = client.get(format!("http://127.0.0.1:{}/", port)).send().await.expect("GET / failed");
    assert_eq!(resp_root.status(), 200);
    let html_body = resp_root.text().await.unwrap();
    assert!(html_body.contains("<!DOCTYPE html>"));
    assert!(html_body.contains("⚡ zy"));
    assert!(html_body.contains("Autonomous Agent REPL"));
    assert!(html_body.contains("System Telemetry"));

    // 2. Test GET /api/status (JSON Health)
    let resp_status = client.get(format!("http://127.0.0.1:{}/api/status", port)).send().await.expect("GET /api/status failed");
    assert_eq!(resp_status.status(), 200);
    let status_json: serde_json::Value = resp_status.json().await.unwrap();
    assert_eq!(status_json["status"], "healthy");
    assert_eq!(status_json["app"], "zy");
    assert_eq!(status_json["local"], true);

    // 3. Test GET /api/radar/svg (Live SVG Radar)
    let resp_radar = client.get(format!("http://127.0.0.1:{}/api/radar/svg", port)).send().await.expect("GET /api/radar/svg failed");
    assert_eq!(resp_radar.status(), 200);
    let radar_svg = resp_radar.text().await.unwrap();
    assert!(radar_svg.contains("<svg"));
    assert!(radar_svg.contains("Maintainability"));

    // 4. Test GET /api/dag/svg (Live DAG SVG)
    let resp_dag = client.get(format!("http://127.0.0.1:{}/api/dag/svg", port)).send().await.expect("GET /api/dag/svg failed");
    assert_eq!(resp_dag.status(), 200);
    let dag_svg = resp_dag.text().await.unwrap();
    assert!(dag_svg.contains("<svg"));
    assert!(dag_svg.contains("Architect:"));

    // 5. Test GET /api/dag/json (Live DAG JSON)
    let resp_dag_json = client.get(format!("http://127.0.0.1:{}/api/dag/json", port)).send().await.expect("GET /api/dag/json failed");
    assert_eq!(resp_dag_json.status(), 200);
    let dag_data: serde_json::Value = resp_dag_json.json().await.unwrap();
    assert!(dag_data["nodes"].as_array().unwrap().len() >= 3);
    assert!(dag_data["edges"].as_array().unwrap().len() >= 2);

    // Cleanup background task
    handle.abort();
}

// =================================================================================================
// SYSTEM 5 TESTS: UNIVERSAL IDE & EDITOR SIDECAR / LSP STACK
// =================================================================================================

#[test]
fn test_system5_universal_editor_lsp_protocol() {
    // 1. Test initialize request
    let init_req = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "rootUri": "file:///workspace/zy",
            "capabilities": {}
        }
    });
    let init_resp = UniversalEditorSidecarStack::handle_lsp_request(&init_req);
    assert_eq!(init_resp["jsonrpc"], "2.0");
    assert_eq!(init_resp["id"], 1);
    let caps = &init_resp["result"]["capabilities"];
    assert_eq!(caps["hoverProvider"], true);
    assert_eq!(caps["codeActionProvider"], true);
    assert_eq!(caps["diagnosticProvider"], true);

    // 2. Test textDocument/hover request
    let hover_req = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 2,
        "method": "textDocument/hover",
        "params": {
            "textDocument": { "uri": "file:///src/main.rs" },
            "position": { "line": 42, "character": 10 }
        }
    });
    let hover_resp = UniversalEditorSidecarStack::handle_lsp_request(&hover_req);
    let hover_content = hover_resp["result"]["contents"].as_str().unwrap();
    assert!(hover_content.contains("zy Copilot Hover"));
    assert!(hover_content.contains("file:///src/main.rs"));

    // 3. Test textDocument/codeAction request
    let code_action_req = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 3,
        "method": "textDocument/codeAction",
        "params": {
            "textDocument": { "uri": "file:///src/main.rs" },
            "range": {
                "start": { "line": 10, "character": 0 },
                "end": { "line": 20, "character": 0 }
            }
        }
    });
    let action_resp = UniversalEditorSidecarStack::handle_lsp_request(&code_action_req);
    let actions = action_resp["result"].as_array().unwrap();
    assert_eq!(actions.len(), 3);
    assert!(actions[0]["title"].as_str().unwrap().contains("Refactor"));
    assert!(actions[1]["title"].as_str().unwrap().contains("Security"));
    assert!(actions[2]["title"].as_str().unwrap().contains("Test Suite"));

    // 4. Test editor configuration generators
    let nvim_cfg = UniversalEditorSidecarStack::generate_neovim_config();
    assert!(nvim_cfg.contains("lspconfig.zy_lsp"));
    assert!(nvim_cfg.contains("8098"));

    let vscode_cfg = UniversalEditorSidecarStack::generate_vscode_config();
    assert!(vscode_cfg.contains("zy.sidecar.port"));
    assert!(vscode_cfg.contains("8098"));
}

// =================================================================================================
// BRUTAL STRESS & EDGE-CASE TESTS FOR ALL 5 UX/UI SYSTEMS
// =================================================================================================

#[test]
fn test_brutal_syntax_highlighter_multiple_languages_and_empty_blocks() {
    let mut highlighter = StreamingSyntaxHighlighter::new();

    // Python block
    let _ = highlighter.process_token("```python\n");
    assert!(highlighter.in_code_block);
    assert_eq!(highlighter.current_lang.as_deref(), Some("python"));
    let py_code = highlighter.process_token("def train_model(epochs=10):\n    for i in range(epochs):\n        print(f'Epoch {i}')\n");
    assert!(py_code.contains("def"));
    assert!(py_code.contains("for"));
    let _ = highlighter.process_token("```\n");
    assert!(!highlighter.in_code_block);

    // JavaScript / TypeScript block
    let _ = highlighter.process_token("```typescript\n");
    assert!(highlighter.in_code_block);
    assert_eq!(highlighter.current_lang.as_deref(), Some("typescript"));
    let js_code = highlighter.process_token("export async function fetchData(): Promise<boolean> {\n    const res = await fetch('/api');\n    return true;\n}\n");
    assert!(js_code.contains("export"));
    assert!(js_code.contains("async"));
    assert!(js_code.contains("const"));
    let _ = highlighter.process_token("```\n");
    assert!(!highlighter.in_code_block);
}

#[test]
fn test_brutal_terminal_graphics_dimensions_and_edge_cases() {
    let caps = TerminalCapabilities::detect();

    // 1x1 pixel image
    let pixel: Vec<u8> = vec![255, 128, 0, 255];
    let mut block_caps = caps.clone();
    block_caps.protocol = TerminalGraphicsProtocol::BlockTrueColor;
    let out = block_caps.render_image_escape(&pixel, 1, 1);
    assert!(out.contains("▀"));

    // 0x0 empty image
    let empty_out = block_caps.render_image_escape(&[], 0, 0);
    assert!(empty_out.is_empty());

    // Odd dimensions (7x3 pixels = 21 pixels * 4 = 84 bytes)
    let odd_rgba: Vec<u8> = vec![100; 84];
    let odd_out = block_caps.render_image_escape(&odd_rgba, 7, 3);
    assert!(odd_out.contains("▀"));
}

#[tokio::test]
async fn test_brutal_desktop_hud_error_handling_and_rejections() {
    let bridge = DesktopHudBridge::new(8105);

    // 1. Unknown RPC method
    let bad_req = DesktopHudMessage {
        jsonrpc: "2.0".to_string(),
        id: Some(999),
        method: "hud/invalidMethodXYZ".to_string(),
        params: serde_json::json!({}),
    };
    let bad_resp = DesktopHudBridge::handle_hud_rpc_message(&bad_req, &bridge).await;
    assert!(bad_resp.error.is_some());
    let err = bad_resp.error.unwrap();
    assert_eq!(err["code"], -32601);

    // 2. Reject tool execution
    let reject_req = DesktopHudMessage {
        jsonrpc: "2.0".to_string(),
        id: Some(1000),
        method: "hud/approve_tool".to_string(),
        params: serde_json::json!({ "call_id": "dangerous_call", "approved": false }),
    };
    let reject_resp = DesktopHudBridge::handle_hud_rpc_message(&reject_req, &bridge).await;
    let res = reject_resp.result.unwrap();
    assert_eq!(res["approved"], false);
    assert_eq!(res["status"], "rejected");
}

#[test]
fn test_brutal_swarm_dag_massive_subtasks_scalability() {
    // Test with 0 subtasks
    let empty_dag = DagLayout::build_swarm_workflow_dag("Single Step Execution", &[]);
    assert_eq!(empty_dag.nodes.len(), 2);
    assert_eq!(empty_dag.edges.len(), 1);
    let empty_svg = empty_dag.to_svg();
    assert!(empty_svg.contains("<svg"));

    // Test with 15 subtasks
    let tasks: Vec<String> = (1..=15).map(|i| format!("Subtask item #{}", i)).collect();
    let massive_dag = DagLayout::build_swarm_workflow_dag("Large Scale Refactor Workflow", &tasks);
    assert_eq!(massive_dag.nodes.len(), 17); // Architect + 15 Workers + Verifier
    assert_eq!(massive_dag.edges.len(), 16);
    let massive_svg = massive_dag.to_svg();
    assert!(massive_svg.contains("Worker #15:"));
    assert!(massive_svg.contains("QA Verifier"));
}

#[test]
fn test_brutal_voice_spectrum_silence_and_noise() {
    let sample_rate = 16000u32;

    // 1. Absolute silence
    let silence: Vec<f32> = vec![0.0; 16000];
    let spec_silence = FastFourierSpectrum::compute(&silence, sample_rate);
    assert_eq!(spec_silence.rms_volume, 0.0);
    assert!(!spec_silence.is_speech_active);

    // 2. High amplitude white noise
    let noise: Vec<f32> = (0..16000).map(|i| if i % 2 == 0 { 0.8 } else { -0.8 }).collect();
    let spec_noise = FastFourierSpectrum::compute(&noise, sample_rate);
    assert!(spec_noise.rms_volume > 0.5);
    assert!(spec_noise.is_speech_active);
    assert_eq!(spec_noise.bin_count, 32);
}

#[tokio::test]
async fn test_brutal_embedded_web_concurrent_requests_stress() {
    let port = 18950u16;
    let handle = EmbeddedWebDashboard::start(port).await.expect("Failed to start web server");
    tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;

    let client = Client::new();

    // Fire 30 concurrent requests
    let mut join_set = tokio::task::JoinSet::new();
    for i in 0..30 {
        let c = client.clone();
        join_set.spawn(async move {
            let path = match i % 5 {
                0 => "/",
                1 => "/api/status",
                2 => "/api/radar/svg",
                3 => "/api/dag/svg",
                _ => "/api/dag/json",
            };
            let resp = c.get(format!("http://127.0.0.1:{}{}", port, path)).send().await.unwrap();
            assert_eq!(resp.status(), 200);
        });
    }

    while let Some(res) = join_set.join_next().await {
        res.expect("Concurrent HTTP request failed");
    }

    handle.abort();
}
