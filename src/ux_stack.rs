use base64::Engine;
use colored::Colorize;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::sync::RwLock;

// =================================================================================================
// SYSTEM 1: ADVANCED TERMINAL UI & GRAPHICS PROTOCOL ENGINE (TUI STACK)
// =================================================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TerminalGraphicsProtocol {
    Kitty,
    Sixel,
    ITerm2,
    BlockTrueColor,
    AsciiFallback,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TerminalCapabilities {
    pub protocol: TerminalGraphicsProtocol,
    pub true_color: bool,
    pub unicode_support: bool,
    pub term_program: String,
    pub columns: u16,
    pub rows: u16,
}

impl TerminalCapabilities {
    pub fn detect() -> Self {
        let term_program = std::env::var("TERM_PROGRAM").unwrap_or_default();
        let term = std::env::var("TERM").unwrap_or_default();
        let colorterm = std::env::var("COLORTERM").unwrap_or_default();
        let is_kitty = std::env::var("KITTY_WINDOW_ID").is_ok() || term.contains("kitty");
        let is_ghostty = std::env::var("GHOSTTY_RESOURCES_DIR").is_ok() || term_program.contains("ghostty");
        let is_iterm = term_program.contains("iTerm.app") || term_program.contains("WezTerm");

        let true_color = colorterm == "truecolor" || colorterm == "24bit" || is_kitty || is_ghostty || is_iterm;
        let unicode_support = true;

        let protocol = if is_kitty || is_ghostty {
            TerminalGraphicsProtocol::Kitty
        } else if is_iterm {
            TerminalGraphicsProtocol::ITerm2
        } else if term.contains("sixel") || term.contains("mlterm") {
            TerminalGraphicsProtocol::Sixel
        } else if true_color {
            TerminalGraphicsProtocol::BlockTrueColor
        } else {
            TerminalGraphicsProtocol::AsciiFallback
        };

        let (columns, rows) = crossterm::terminal::size().unwrap_or((80, 24));

        Self {
            protocol,
            true_color,
            unicode_support,
            term_program,
            columns,
            rows,
        }
    }

    pub fn render_image_escape(&self, raw_rgba: &[u8], width: u32, height: u32) -> String {
        match self.protocol {
            TerminalGraphicsProtocol::Kitty => {
                let encoded = base64::engine::general_purpose::STANDARD.encode(raw_rgba);
                format!("\x1b_Gf=32,s={},v={},a=T,m=0;{}\x1b\\", width, height, encoded)
            }
            TerminalGraphicsProtocol::ITerm2 => {
                let encoded = base64::engine::general_purpose::STANDARD.encode(raw_rgba);
                format!("\x1b]1337;File=inline=1;width={}px;height={}px:{}\x07", width, height, encoded)
            }
            TerminalGraphicsProtocol::BlockTrueColor | TerminalGraphicsProtocol::Sixel => {
                let mut out = String::new();
                for y in (0..height).step_by(2) {
                    for x in 0..width {
                        let idx_top = ((y * width + x) * 4) as usize;
                        let r1 = raw_rgba.get(idx_top).copied().unwrap_or(0);
                        let g1 = raw_rgba.get(idx_top + 1).copied().unwrap_or(0);
                        let b1 = raw_rgba.get(idx_top + 2).copied().unwrap_or(0);

                        let idx_bot = (((y + 1) * width + x) * 4) as usize;
                        let r2 = raw_rgba.get(idx_bot).copied().unwrap_or(0);
                        let g2 = raw_rgba.get(idx_bot + 1).copied().unwrap_or(0);
                        let b2 = raw_rgba.get(idx_bot + 2).copied().unwrap_or(0);

                        out.push_str(&format!("\x1b[38;2;{};{};{}m\x1b[48;2;{};{};{}m▀\x1b[0m", r1, g1, b1, r2, g2, b2));
                    }
                    out.push('\n');
                }
                out
            }
            TerminalGraphicsProtocol::AsciiFallback => {
                let mut out = String::new();
                for y in (0..height).step_by(2) {
                    for x in 0..width {
                        let idx = ((y * width + x) * 4) as usize;
                        let r = raw_rgba.get(idx).copied().unwrap_or(0) as u32;
                        let g = raw_rgba.get(idx + 1).copied().unwrap_or(0) as u32;
                        let b = raw_rgba.get(idx + 2).copied().unwrap_or(0) as u32;
                        let lum = (r * 299 + g * 587 + b * 114) / 1000;
                        let ch = if lum > 200 { '#' } else if lum > 120 { '*' } else if lum > 50 { '.' } else { ' ' };
                        out.push(ch);
                    }
                    out.push('\n');
                }
                out
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingSyntaxHighlighter {
    pub in_code_block: bool,
    pub current_lang: Option<String>,
    pub buffer: String,
}

impl StreamingSyntaxHighlighter {
    pub fn new() -> Self {
        Self {
            in_code_block: false,
            current_lang: None,
            buffer: String::new(),
        }
    }

    pub fn process_token(&mut self, token: &str) -> String {
        self.buffer.push_str(token);
        let mut highlighted = String::new();

        if let Some(pos) = token.find("```") {
            if !self.in_code_block {
                self.in_code_block = true;
                let remainder = &token[pos + 3..];
                let lang_line = remainder.lines().next().unwrap_or_default().trim();
                if !lang_line.is_empty() {
                    self.current_lang = Some(lang_line.to_string());
                } else {
                    self.current_lang = Some("rust".to_string());
                }
            } else {
                self.in_code_block = false;
                self.current_lang = None;
            }
        }

        if self.in_code_block {
            let lang = self.current_lang.as_deref().unwrap_or("rust");
            highlighted.push_str(&Self::highlight_code_snippet(token, lang));
        } else {
            highlighted.push_str(token);
        }

        highlighted
    }

    pub fn highlight_code_snippet(code: &str, lang: &str) -> String {
        let keywords = match lang.to_lowercase().as_str() {
            "rust" | "rs" => vec!["fn", "let", "mut", "pub", "struct", "enum", "impl", "use", "match", "async", "await", "return", "true", "false"],
            "python" | "py" => vec!["def", "class", "import", "from", "return", "if", "elif", "else", "for", "while", "try", "except", "with", "as", "True", "False"],
            "javascript" | "js" | "typescript" | "ts" => vec!["function", "const", "let", "var", "import", "export", "return", "async", "await", "if", "else", "class", "extends", "true", "false"],
            _ => vec!["fn", "def", "func", "function", "return", "if", "else", "class", "struct"],
        };

        let mut out = String::new();
        for word in code.split_inclusive(|c: char| !c.is_alphanumeric() && c != '_') {
            let trimmed = word.trim_matches(|c: char| !c.is_alphanumeric() && c != '_');
            if keywords.contains(&trimmed) {
                out.push_str(&word.replace(trimmed, &trimmed.cyan().bold().to_string()));
            } else if trimmed.starts_with('"') || trimmed.ends_with('"') {
                out.push_str(&word.green().to_string());
            } else if trimmed.chars().all(|c| c.is_numeric()) && !trimmed.is_empty() {
                out.push_str(&word.yellow().to_string());
            } else {
                out.push_str(word);
            }
        }
        out
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SplitMultiplexerPane {
    pub title: String,
    pub width_percent: u16,
    pub is_active: bool,
    pub content: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TuiSplitMultiplexer {
    pub left_pane: SplitMultiplexerPane,
    pub center_pane: SplitMultiplexerPane,
    pub right_pane: SplitMultiplexerPane,
    pub active_index: usize,
}

impl TuiSplitMultiplexer {
    pub fn new() -> Self {
        Self {
            left_pane: SplitMultiplexerPane {
                title: "📁 CODEBASE & RAG TREE".to_string(),
                width_percent: 25,
                is_active: false,
                content: Vec::new(),
            },
            center_pane: SplitMultiplexerPane {
                title: "⚡ AGENT REPL / LIVE DIFF".to_string(),
                width_percent: 50,
                is_active: true,
                content: Vec::new(),
            },
            right_pane: SplitMultiplexerPane {
                title: "📊 SWARM TELEMETRY & RADAR".to_string(),
                width_percent: 25,
                is_active: false,
                content: Vec::new(),
            },
            active_index: 1,
        }
    }

    pub fn cycle_pane(&mut self) {
        self.active_index = (self.active_index + 1) % 3;
        self.left_pane.is_active = self.active_index == 0;
        self.center_pane.is_active = self.active_index == 1;
        self.right_pane.is_active = self.active_index == 2;
    }

    pub fn format_tui_layout_for_terminal(&self) -> String {
        let mut out = String::new();
        let border = "═".repeat(78);
        out.push_str(&format!("╔{}╗\n", border.cyan()));
        out.push_str(&format!(
            "║ {:<24} │ {:<24} │ {:<24} ║\n",
            if self.left_pane.is_active { self.left_pane.title.green().bold() } else { self.left_pane.title.dimmed() },
            if self.center_pane.is_active { self.center_pane.title.green().bold() } else { self.center_pane.title.dimmed() },
            if self.right_pane.is_active { self.right_pane.title.green().bold() } else { self.right_pane.title.dimmed() },
        ));
        out.push_str(&format!("╠{}╣\n", border.cyan()));
        out.push_str(&format!("║ {} ║\n", "Multi-pane Split Multiplexer Active: [Tab] Switch Pane │ [Ctrl+C] Exit".yellow()));
        out.push_str(&format!("╚{}╝\n", border.cyan()));
        out
    }
}

// =================================================================================================
// SYSTEM 2: DESKTOP GUI & HUD SPOTLIGHT OVERLAY PROTOCOL (TAURI / SLINT / WEBVIEW)
// =================================================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DesktopHudMessage {
    pub jsonrpc: String,
    pub id: Option<u64>,
    pub method: String,
    pub params: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DesktopHudResponse {
    pub jsonrpc: String,
    pub id: Option<u64>,
    pub result: Option<serde_json::Value>,
    pub error: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpotlightSearchResult {
    pub title: String,
    pub category: String,
    pub description: String,
    pub action_command: String,
    pub score: f32,
}

pub struct DesktopHudBridge {
    pub port: u16,
    pub is_running: Arc<AtomicBool>,
    pub active_session_id: Arc<RwLock<String>>,
    pub pending_tool_approvals: Arc<RwLock<Vec<serde_json::Value>>>,
}

impl DesktopHudBridge {
    pub fn new(port: u16) -> Self {
        Self {
            port,
            is_running: Arc::new(AtomicBool::new(false)),
            active_session_id: Arc::new(RwLock::new("zy-hud-default".to_string())),
            pending_tool_approvals: Arc::new(RwLock::new(Vec::new())),
        }
    }

    pub fn query_spotlight(query: &str, workspace: &Path) -> Vec<SpotlightSearchResult> {
        let mut results = Vec::new();
        let q_lower = query.to_lowercase();

        // 1. Match core commands
        let commands = vec![
            ("Chat Agent", "mode", "Start autonomous coding chat session", "/chat"),
            ("Index Codebase", "rag", "Build vector embeddings index for fast RAG retrieval", "/rag index"),
            ("Code Health Radar", "analytics", "Analyze codebase maintainability, complexity, and security", "/radar"),
            ("Git DAG Graph", "git", "Inspect branch history and visual merge tree", "/git dag"),
            ("Duplex Voice", "voice", "Launch real-time full-duplex conversational voice mode", "/voice"),
            ("Web Dashboard", "dashboard", "Launch localhost web GUI at http://localhost:7890", "/web"),
        ];

        for (title, cat, desc, cmd) in commands {
            if title.to_lowercase().contains(&q_lower) || desc.to_lowercase().contains(&q_lower) || cmd.contains(&q_lower) {
                results.push(SpotlightSearchResult {
                    title: title.to_string(),
                    category: cat.to_string(),
                    description: desc.to_string(),
                    action_command: cmd.to_string(),
                    score: 0.9,
                });
            }
        }

        // 2. Match local files
        if let Ok(entries) = fs::read_dir(workspace) {
            for entry in entries.flatten() {
                let path = entry.path();
                let file_name = path.file_name().unwrap_or_default().to_string_lossy().to_string();
                if file_name.starts_with('.') || file_name == "target" {
                    continue;
                }
                if file_name.to_lowercase().contains(&q_lower) {
                    results.push(SpotlightSearchResult {
                        title: file_name.clone(),
                        category: if path.is_dir() { "directory".to_string() } else { "file".to_string() },
                        description: path.to_string_lossy().to_string(),
                        action_command: format!("/file {}", file_name),
                        score: 0.75,
                    });
                }
            }
        }

        results
    }

    pub async fn handle_hud_rpc_message(msg: &DesktopHudMessage, bridge: &DesktopHudBridge) -> DesktopHudResponse {
        match msg.method.as_str() {
            "hud/handshake" => DesktopHudResponse {
                jsonrpc: "2.0".to_string(),
                id: msg.id,
                result: Some(serde_json::json!({
                    "status": "connected",
                    "app": "zy",
                    "version": "0.1.0",
                    "bridge_port": bridge.port,
                    "hud_protocol_version": "2.0",
                    "capabilities": ["spotlight", "token_stream", "tool_approval", "state_sync", "radar_render"]
                })),
                error: None,
            },
            "hud/spotlight_search" => {
                let query = msg.params.get("query").and_then(|v| v.as_str()).unwrap_or("");
                let results = Self::query_spotlight(query, Path::new("."));
                DesktopHudResponse {
                    jsonrpc: "2.0".to_string(),
                    id: msg.id,
                    result: Some(serde_json::to_value(results).unwrap()),
                    error: None,
                }
            }
            "hud/get_telemetry" => {
                let mut sys = sysinfo::System::new_all();
                sys.refresh_all();
                let total_mem = sys.total_memory() / (1024 * 1024);
                let used_mem = sys.used_memory() / (1024 * 1024);
                let cpu_usage = sys.global_cpu_usage();

                DesktopHudResponse {
                    jsonrpc: "2.0".to_string(),
                    id: msg.id,
                    result: Some(serde_json::json!({
                        "total_memory_mb": total_mem,
                        "used_memory_mb": used_mem,
                        "cpu_usage_percent": cpu_usage,
                        "os": std::env::consts::OS,
                        "arch": std::env::consts::ARCH,
                        "mode": if total_mem < 12000 { "ECO_MODE" } else { "TURBO_MODE" },
                    })),
                    error: None,
                }
            }
            "hud/approve_tool" => {
                let call_id = msg.params.get("call_id").and_then(|v| v.as_str()).unwrap_or("");
                let approved = msg.params.get("approved").and_then(|v| v.as_bool()).unwrap_or(true);
                let mut queue = bridge.pending_tool_approvals.write().await;
                queue.retain(|v| v.get("call_id").and_then(|s| s.as_str()) != Some(call_id));

                DesktopHudResponse {
                    jsonrpc: "2.0".to_string(),
                    id: msg.id,
                    result: Some(serde_json::json!({
                        "call_id": call_id,
                        "approved": approved,
                        "status": if approved { "executed" } else { "rejected" }
                    })),
                    error: None,
                }
            }
            _ => DesktopHudResponse {
                jsonrpc: "2.0".to_string(),
                id: msg.id,
                result: None,
                error: Some(serde_json::json!({
                    "code": -32601,
                    "message": format!("Method '{}' not found in zy HUD protocol", msg.method)
                })),
            },
        }
    }
}

// =================================================================================================
// SYSTEM 3: RICH VISUALIZATIONS & INTERACTIVE COMPONENT ENGINE
// =================================================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DagNode {
    pub id: String,
    pub label: String,
    pub node_type: String, // "architect", "planner", "coder", "reviewer", "tool", "checkpoint"
    pub status: String,    // "pending", "running", "completed", "failed"
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DagEdge {
    pub source: String,
    pub target: String,
    pub label: Option<String>,
    pub control_point_x1: f32,
    pub control_point_y1: f32,
    pub control_point_x2: f32,
    pub control_point_y2: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DagLayout {
    pub nodes: Vec<DagNode>,
    pub edges: Vec<DagEdge>,
    pub canvas_width: f32,
    pub canvas_height: f32,
}

impl DagLayout {
    pub fn build_swarm_workflow_dag(goal: &str, subtasks: &[String]) -> Self {
        let mut nodes = Vec::new();
        let mut edges = Vec::new();

        // Root Node (Architect)
        nodes.push(DagNode {
            id: "node_architect".to_string(),
            label: format!("Architect: {}", if goal.len() > 24 { &goal[..24] } else { goal }),
            node_type: "architect".to_string(),
            status: "completed".to_string(),
            x: 250.0,
            y: 40.0,
            width: 200.0,
            height: 50.0,
        });

        // Intermediate Subtask Coder Nodes
        let mut prev_id = "node_architect".to_string();
        for (i, task) in subtasks.iter().enumerate() {
            let task_id = format!("node_task_{}", i + 1);
            let y_pos = 140.0 + (i as f32 * 90.0);
            let x_pos = if i % 2 == 0 { 180.0 } else { 320.0 };

            nodes.push(DagNode {
                id: task_id.clone(),
                label: format!("Worker #{}: {}", i + 1, if task.len() > 20 { &task[..20] } else { task }),
                node_type: "coder".to_string(),
                status: "completed".to_string(),
                x: x_pos,
                y: y_pos,
                width: 190.0,
                height: 48.0,
            });

            edges.push(DagEdge {
                source: prev_id.clone(),
                target: task_id.clone(),
                label: Some(format!("Step {}", i + 1)),
                control_point_x1: 250.0,
                control_point_y1: y_pos - 45.0,
                control_point_x2: x_pos + 95.0,
                control_point_y2: y_pos - 20.0,
            });

            prev_id = task_id;
        }

        // Terminal Reviewer / Verifier Node
        let final_y = 140.0 + (subtasks.len() as f32 * 90.0);
        nodes.push(DagNode {
            id: "node_verifier".to_string(),
            label: "QA Verifier & Test Suite".to_string(),
            node_type: "reviewer".to_string(),
            status: "completed".to_string(),
            x: 250.0,
            y: final_y,
            width: 200.0,
            height: 50.0,
        });

        edges.push(DagEdge {
            source: prev_id,
            target: "node_verifier".to_string(),
            label: Some("Final Verification".to_string()),
            control_point_x1: 250.0,
            control_point_y1: final_y - 45.0,
            control_point_x2: 250.0,
            control_point_y2: final_y - 20.0,
        });

        Self {
            nodes,
            edges,
            canvas_width: 600.0,
            canvas_height: final_y + 100.0,
        }
    }

    pub fn to_svg(&self) -> String {
        let mut svg = format!(
            "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {} {}\" width=\"100%\" height=\"100%\" style=\"background:#0f172a; font-family:sans-serif;\">\n",
            self.canvas_width, self.canvas_height
        );

        // Marker definition for arrows
        svg.push_str("<defs>\n");
        svg.push_str("  <marker id=\"arrow\" viewBox=\"0 0 10 10\" refX=\"10\" refY=\"5\" markerWidth=\"6\" markerHeight=\"6\" orient=\"auto-start-reverse\">\n");
        svg.push_str("    <path d=\"M 0 0 L 10 5 L 0 10 z\" fill=\"#38bdf8\" />\n");
        svg.push_str("  </marker>\n");
        svg.push_str("  <filter id=\"glow\" x=\"-20%\" y=\"-20%\" width=\"140%\" height=\"140%\">\n");
        svg.push_str("    <feGaussianBlur stdDeviation=\"4\" result=\"blur\" />\n");
        svg.push_str("    <feComposite in=\"SourceGraphic\" in2=\"blur\" operator=\"over\" />\n");
        svg.push_str("  </filter>\n");
        svg.push_str("</defs>\n");

        // Draw Edges with Bezier curves
        for edge in &self.edges {
            let src_node = self.nodes.iter().find(|n| n.id == edge.source);
            let tgt_node = self.nodes.iter().find(|n| n.id == edge.target);
            if let (Some(s), Some(t)) = (src_node, tgt_node) {
                let sx = s.x + s.width / 2.0;
                let sy = s.y + s.height;
                let tx = t.x + t.width / 2.0;
                let ty = t.y;
                svg.push_str(&format!(
                    "  <path d=\"M {} {} C {} {}, {} {}, {} {}\" stroke=\"#38bdf8\" stroke-width=\"2.5\" fill=\"none\" marker-end=\"url(#arrow)\" stroke-dasharray=\"6,3\" />\n",
                    sx, sy, edge.control_point_x1, edge.control_point_y1, edge.control_point_x2, edge.control_point_y2, tx, ty
                ));
            }
        }

        // Draw Nodes
        for node in &self.nodes {
            let (bg_color, border_color) = match node.node_type.as_str() {
                "architect" => ("#1e293b", "#818cf8"),
                "coder" => ("#1e293b", "#38bdf8"),
                "reviewer" => ("#1e293b", "#34d399"),
                _ => ("#1e293b", "#94a3b8"),
            };

            svg.push_str(&format!(
                "  <g transform=\"translate({}, {})\">\n",
                node.x, node.y
            ));
            svg.push_str(&format!(
                "    <rect width=\"{}\" height=\"{}\" rx=\"8\" fill=\"{}\" stroke=\"{}\" stroke-width=\"2\" filter=\"url(#glow)\" />\n",
                node.width, node.height, bg_color, border_color
            ));
            svg.push_str(&format!(
                "    <text x=\"{}\" y=\"28\" fill=\"#f8fafc\" font-size=\"12\" font-weight=\"bold\" text-anchor=\"middle\">{}</text>\n",
                node.width / 2.0, node.label
            ));
            svg.push_str("  </g>\n");
        }

        svg.push_str("</svg>");
        svg
    }

    pub fn to_mermaid(&self) -> String {
        let mut out = String::from("graph TD\n");
        for node in &self.nodes {
            out.push_str(&format!("  {}[\"{}\"]\n", node.id, node.label));
        }
        for edge in &self.edges {
            if let Some(ref l) = edge.label {
                out.push_str(&format!("  {} -->|\"{}\"| {}\n", edge.source, l, edge.target));
            } else {
                out.push_str(&format!("  {} --> {}\n", edge.source, edge.target));
            }
        }
        out
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodebaseRadarMetrics {
    pub maintainability: f32, // 0 - 100
    pub complexity: f32,      // 0 - 100 (lower cyclomatic = higher score)
    pub test_coverage: f32,   // 0 - 100
    pub security: f32,        // 0 - 100
    pub performance: f32,     // 0 - 100
    pub documentation: f32,   // 0 - 100
    pub overall_score: f32,
}

impl CodebaseRadarMetrics {
    pub fn calculate(workspace: &Path) -> Self {
        let mut rust_files = 0usize;
        let mut total_lines = 0usize;
        let mut doc_comments = 0usize;
        let mut test_functions = 0usize;
        let mut unwrap_calls = 0usize;
        let mut unsafe_blocks = 0usize;
        let mut allocations_and_locks = 0usize;
        let mut conditionals = 0usize;
        let mut todos = 0usize;

        for entry in walkdir::WalkDir::new(workspace).into_iter().filter_map(|e| e.ok()) {
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("rs") {
                rust_files += 1;
                if let Ok(content) = fs::read_to_string(path) {
                    for line in content.lines() {
                        total_lines += 1;
                        let trimmed = line.trim();
                        if trimmed.starts_with("///") || trimmed.starts_with("//!") {
                            doc_comments += 1;
                        }
                        if trimmed.starts_with("#[test]") {
                            test_functions += 1;
                        }
                        if trimmed.contains(".unwrap()") || trimmed.contains(".expect(") || trimmed.contains("panic!(") {
                            unwrap_calls += 1;
                        }
                        if trimmed.contains("unsafe {") || trimmed.contains("unsafe fn ") {
                            unsafe_blocks += 1;
                        }
                        if trimmed.contains(".clone()") || trimmed.contains("Box::new(") || trimmed.contains("Arc::new(") || trimmed.contains("Mutex::new(") {
                            allocations_and_locks += 1;
                        }
                        if trimmed.starts_with("if ") || trimmed.starts_with("match ") || trimmed.starts_with("while ") || trimmed.starts_with("for ") {
                            conditionals += 1;
                        }
                        if trimmed.contains("TODO") || trimmed.contains("FIXME") {
                            todos += 1;
                        }
                    }
                }
            }
        }

        let avg_file_len = if rust_files > 0 { total_lines as f32 / rust_files as f32 } else { 0.0 };
        
        let maintainability = if total_lines > 0 { (100.0 - (todos as f32 * 0.5) - (avg_file_len * 0.02)).clamp(40.0, 99.0) } else { 85.0 };
        let complexity = if total_lines > 0 { (100.0 - (conditionals as f32 / total_lines as f32 * 500.0)).clamp(30.0, 99.0) } else { 95.0 };
        let test_coverage = if rust_files > 0 { (test_functions as f32 * 8.0).clamp(10.0, 99.0) } else { 80.0 };
        let security = if total_lines > 0 { (100.0 - (unsafe_blocks as f32 * 5.0) - (unwrap_calls as f32 * 0.5)).clamp(20.0, 99.0) } else { 96.0 };
        let performance = if total_lines > 0 { (100.0 - (allocations_and_locks as f32 / total_lines as f32 * 200.0)).clamp(40.0, 99.0) } else { 95.0 };
        let documentation = if total_lines > 0 { ((doc_comments as f32 / total_lines as f32) * 600.0).clamp(20.0, 99.0) } else { 85.0 };

        let overall = (maintainability + complexity + test_coverage + security + performance + documentation) / 6.0;

        Self {
            maintainability,
            complexity,
            test_coverage,
            security,
            performance,
            documentation,
            overall_score: overall,
        }
    }

    pub fn to_svg(&self) -> String {
        let size = 400.0f32;
        let cx = size / 2.0;
        let cy = size / 2.0;
        let r = 140.0f32;

        let labels = ["Maintainability", "Complexity", "Coverage", "Security", "Performance", "Docs"];
        let values = [
            self.maintainability,
            self.complexity,
            self.test_coverage,
            self.security,
            self.performance,
            self.documentation,
        ];
        let n = values.len();

        let mut polygon_points = Vec::new();
        let mut grid_polygons = Vec::new();
        let mut axis_lines = Vec::new();
        let mut text_elements = Vec::new();

        // 4 concentric grid rings (25%, 50%, 75%, 100%)
        for ring in 1..=4 {
            let ring_r = r * (ring as f32 / 4.0);
            let mut ring_pts = Vec::new();
            for i in 0..n {
                let angle = (i as f32 * 2.0 * std::f32::consts::PI / n as f32) - (std::f32::consts::PI / 2.0);
                let px = cx + ring_r * angle.cos();
                let py = cy + ring_r * angle.sin();
                ring_pts.push(format!("{},{}", px, py));
            }
            grid_polygons.push(ring_pts.join(" "));
        }

        // Data Polygon points & Axes
        for i in 0..n {
            let angle = (i as f32 * 2.0 * std::f32::consts::PI / n as f32) - (std::f32::consts::PI / 2.0);
            let val_r = r * (values[i] / 100.0);
            let px = cx + val_r * angle.cos();
            let py = cy + val_r * angle.sin();
            polygon_points.push(format!("{},{}", px, py));

            let edge_x = cx + (r + 20.0) * angle.cos();
            let edge_y = cy + (r + 20.0) * angle.sin();
            axis_lines.push(format!("<line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" stroke=\"#334155\" stroke-width=\"1.5\" />", cx, cy, cx + r * angle.cos(), cy + r * angle.sin()));
            text_elements.push(format!(
                "<text x=\"{}\" y=\"{}\" fill=\"#94a3b8\" font-size=\"11\" font-weight=\"bold\" text-anchor=\"middle\" dominant-baseline=\"central\">{} ({:.0}%)</text>",
                edge_x, edge_y, labels[i], values[i]
            ));
        }

        let poly_str = polygon_points.join(" ");

        let mut svg = format!(
            "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {} {}\" width=\"100%\" height=\"100%\" style=\"background:#0f172a; font-family:sans-serif;\">\n",
            size, size
        );

        svg.push_str("  <style>\n");
        svg.push_str("    @keyframes pulseRadar { 0% { filter: drop-shadow(0 0 4px #38bdf8); transform: scale(1); } 50% { filter: drop-shadow(0 0 15px #818cf8); transform: scale(1.02); } 100% { filter: drop-shadow(0 0 4px #38bdf8); transform: scale(1); } }\n");
        svg.push_str("    .live-radar { animation: pulseRadar 3s ease-in-out infinite; transform-origin: center; }\n");
        svg.push_str("  </style>\n");

        svg.push_str("  <defs>\n");
        svg.push_str("    <linearGradient id=\"radarGrad\" x1=\"0%\" y1=\"0%\" x2=\"100%\" y2=\"100%\">\n");
        svg.push_str("      <stop offset=\"0%\" stop-color=\"#38bdf8\" stop-opacity=\"0.6\" />\n");
        svg.push_str("      <stop offset=\"100%\" stop-color=\"#818cf8\" stop-opacity=\"0.3\" />\n");
        svg.push_str("    </linearGradient>\n");
        svg.push_str("  </defs>\n");

        for ring_poly in grid_polygons {
            svg.push_str(&format!("  <polygon points=\"{}\" fill=\"none\" stroke=\"#1e293b\" stroke-width=\"1.5\" />\n", ring_poly));
        }

        for axis in axis_lines {
            svg.push_str(&format!("  {}\n", axis));
        }

        svg.push_str(&format!(
            "  <polygon class=\"live-radar\" points=\"{}\" fill=\"url(#radarGrad)\" stroke=\"#38bdf8\" stroke-width=\"2.5\" />\n",
            poly_str
        ));

        for txt in text_elements {
            svg.push_str(&format!("  {}\n", txt));
        }

        svg.push_str(&format!(
            "  <text x=\"{}\" y=\"{}\" fill=\"#38bdf8\" font-size=\"16\" font-weight=\"bold\" text-anchor=\"middle\">Overall Score: {:.1}/100</text>\n",
            cx, cy + r + 45.0, self.overall_score
        ));

        svg.push_str("</svg>");
        svg
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FastFourierSpectrum {
    pub sample_rate: u32,
    pub bin_count: usize,
    pub energy_spectrum: Vec<f32>,
    pub peak_frequency_hz: f32,
    pub rms_volume: f32,
    pub is_speech_active: bool,
}

impl FastFourierSpectrum {
    pub fn compute(audio_pcm: &[f32], sample_rate: u32) -> Self {
        let n = 32usize;
        let mut spectrum = vec![0.0f32; n];
        let mut sum_sq = 0.0f32;

        for &sample in audio_pcm {
            sum_sq += sample * sample;
        }
        let rms = (sum_sq / (audio_pcm.len().max(1) as f32)).sqrt();

        // 32-bin discrete approximation of frequency power distribution
        for i in 0..n {
            let freq_step = i as f32 * 0.15;
            let mut bin_energy = 0.0f32;
            for (t, &s) in audio_pcm.iter().enumerate() {
                bin_energy += (s * (t as f32 * freq_step).sin()).abs();
            }
            spectrum[i] = (bin_energy / (audio_pcm.len().max(1) as f32) * 10.0).clamp(0.0, 1.0);
        }

        let peak_idx = spectrum.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(idx, _)| idx).unwrap_or(0);
        let peak_freq = (peak_idx as f32 / n as f32) * (sample_rate as f32 / 2.0);
        let is_speech = rms > 0.015;

        Self {
            sample_rate,
            bin_count: n,
            energy_spectrum: spectrum,
            peak_frequency_hz: peak_freq,
            rms_volume: rms,
            is_speech_active: is_speech,
        }
    }

    pub fn to_ascii_bars(&self) -> String {
        let chars = [" ", " ", "▂", "▃", "▄", "▅", "▆", "▇", "█"];
        let mut out = String::new();
        for &val in &self.energy_spectrum {
            let idx = ((val * 8.0) as usize).clamp(0, 8);
            out.push_str(chars[idx]);
        }
        out
    }
}

// =================================================================================================
// SYSTEM 4: EMBEDDED LOCAL WEB DASHBOARD (ZERO-INSTALL HTTP & WEBSOCKET ENGINE)
// =================================================================================================

pub struct EmbeddedWebDashboard {
    pub port: u16,
    pub is_running: Arc<AtomicBool>,
}

impl EmbeddedWebDashboard {
    pub fn new(port: u16) -> Self {
        Self {
            port,
            is_running: Arc::new(AtomicBool::new(false)),
        }
    }

    fn create_checkpoint(name: &str) -> String {
        let cp_dir = std::path::Path::new("zy_checkpoints").join(name);
        if cp_dir.exists() {
            return format!("Checkpoint '{}' already exists.", name);
        }
        let _ = std::fs::create_dir_all(&cp_dir);
        let mut count = 0;
        for entry in walkdir::WalkDir::new(".").into_iter().filter_map(|e| e.ok()) {
            let path = entry.path();
            if path.components().any(|c| c.as_os_str() == ".git" || c.as_os_str() == "target" || c.as_os_str() == "node_modules" || c.as_os_str() == "zy_checkpoints") {
                continue;
            }
            if path.is_file() {
                let rel = path.strip_prefix(".").unwrap_or(path);
                let target = cp_dir.join(rel);
                if let Some(p) = target.parent() {
                    let _ = std::fs::create_dir_all(p);
                }
                if std::fs::copy(path, target).is_ok() {
                    count += 1;
                }
            }
        }
        format!("Checkpoint '{}' created with {} files.", name, count)
    }

    fn restore_checkpoint(name: &str) -> String {
        let cp_dir = std::path::Path::new("zy_checkpoints").join(name);
        if !cp_dir.exists() {
            return format!("Checkpoint '{}' does not exist.", name);
        }
        let mut count = 0;
        for entry in walkdir::WalkDir::new(&cp_dir).into_iter().filter_map(|e| e.ok()) {
            let path = entry.path();
            if path.is_file() {
                let rel = path.strip_prefix(&cp_dir).unwrap_or(path);
                let target = std::path::Path::new(".").join(rel);
                if let Some(p) = target.parent() {
                    let _ = std::fs::create_dir_all(p);
                }
                if std::fs::copy(path, target).is_ok() {
                    count += 1;
                }
            }
        }
        format!("Checkpoint '{}' restored with {} files.", name, count)
    }

    fn list_checkpoints() -> String {
        let cp_dir = std::path::Path::new("zy_checkpoints");
        if !cp_dir.exists() {
            return "No checkpoints found.".to_string();
        }
        let mut cps = vec![];
        if let Ok(entries) = std::fs::read_dir(cp_dir) {
            for entry in entries.filter_map(|e| e.ok()) {
                if entry.path().is_dir() {
                    cps.push(entry.file_name().to_string_lossy().into_owned());
                }
            }
        }
        if cps.is_empty() {
            "No checkpoints found.".to_string()
        } else {
            format!("Available checkpoints: {}", cps.join(", "))
        }
    }

    fn get_agent_tools() -> serde_json::Value {
        serde_json::json!([
            {
                "type": "function",
                "function": {
                    "name": "run_command",
                    "description": "Execute a bash command on the local system",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": { "type": "string", "description": "The command to run" }
                        },
                        "required": ["command"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read the contents of a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": { "type": "string", "description": "Path to the file" }
                        },
                        "required": ["path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "Write or overwrite a file with content",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": { "type": "string" },
                            "content": { "type": "string" }
                        },
                        "required": ["path", "content"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "create_checkpoint",
                    "description": "Take a snapshot of the current workspace before making risky changes",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": { "type": "string", "description": "A unique name for the checkpoint (e.g. 'before_refactor')" }
                        },
                        "required": ["name"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "restore_checkpoint",
                    "description": "Restore the workspace to a previously taken snapshot",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": { "type": "string", "description": "The name of the checkpoint to restore" }
                        },
                        "required": ["name"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "list_checkpoints",
                    "description": "List all available checkpoints",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            }
        ])
    }

    async fn handle_events_sse(stream: &mut tokio::net::TcpStream, tx: tokio::sync::broadcast::Sender<String>) {
        use tokio::io::AsyncWriteExt;
        let headers = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: keep-alive\r\nAccess-Control-Allow-Origin: *\r\n\r\n";
        if stream.write_all(headers.as_bytes()).await.is_err() { return; }

        let mut rx = tx.subscribe();
        while let Ok(msg) = rx.recv().await {
            if stream.write_all(format!("data: {}\n\n", msg).as_bytes()).await.is_err() {
                break;
            }
        }
    }

    async fn handle_dag_execute(req_str: String, tx: tokio::sync::broadcast::Sender<String>, cancel_token: std::sync::Arc<std::sync::atomic::AtomicBool>) {
        let body_str = req_str.split("\r\n\r\n").nth(1).unwrap_or("").trim_matches('\0');
        let mut target = "Refactor".to_string();
        let mut subtasks = vec![];
        if let Ok(json) = serde_json::from_str::<serde_json::Value>(body_str) {
            if let Some(goal) = json.get("goal").and_then(|v| v.as_str()) {
                if !goal.is_empty() { target = goal.to_string(); }
            }
            if let Some(tasks) = json.get("subtasks").and_then(|v| v.as_array()) {
                for t in tasks {
                    if let Some(t_str) = t.as_str() { subtasks.push(t_str.to_string()); }
                }
            }
        }
        
        let _ = tx.send(serde_json::json!({ "type": "status", "msg": format!("Orchestrating {} parallel agents...", subtasks.len()) }).to_string());

        let mut handles = vec![];
        for (i, task) in subtasks.into_iter().enumerate() {
            let tx_clone = tx.clone();
            let cancel_clone = cancel_token.clone();
            let handle = tokio::spawn(async move {
                if cancel_clone.load(std::sync::atomic::Ordering::Relaxed) { return format!("Agent {} cancelled.", i+1); }
                let _ = tx_clone.send(serde_json::json!({ "type": "thought", "msg": format!("Agent {} started: {}", i+1, task) }).to_string());
                
                let mut tuning_options = crate::OllamaOptions::default();
                tuning_options.num_thread = Some(std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).saturating_sub(2).max(1));
                tuning_options.num_ctx = Some(4096);
                tuning_options.num_predict = Some(1024);
                let req = crate::ChatRequest {
                    model: "qwen2.5-coder:1.5b".to_string(),
                    messages: vec![crate::Message { 
                        role: "user".to_string(), 
                        content: format!("You are an agent assigned a subtask.\nSubtask: {}\nExecute it and return your result.", task), 
                        tool_calls: None, images: None 
                    }],
                    stream: false, tools: None, format: None, options: Some(tuning_options), keep_alive: None,
                };
                let client = reqwest::Client::new();
                if let Ok(res) = client.post(format!("{}/api/chat", crate::OLLAMA_URL)).json(&req).send().await {
                    if let Ok(chat_res) = res.json::<crate::ChatResponse>().await {
                        if let Some(msg) = chat_res.message {
                            let _ = tx_clone.send(serde_json::json!({ "type": "tool_result", "name": format!("Agent {}", i+1), "result": msg.content.clone() }).to_string());
                            return format!("Agent {} ({}):\n{}", i+1, task, msg.content);
                        }
                    }
                }
                format!("Agent {} ({}) failed.", i+1, task)
            });
            handles.push(handle);
        }

        let mut all_results = vec![];
        for handle in handles {
            if let Ok(res) = handle.await {
                all_results.push(res);
            }
        }

        let final_report = format!("# Swarm Execution Complete\n\n## Goal: {}\n\n{}", target, all_results.join("\n\n---\n\n"));
        let _ = tx.send(serde_json::json!({ "type": "done", "msg": final_report }).to_string());
    }

    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm_a == 0.0 || norm_b == 0.0 { 0.0 } else { dot / (norm_a * norm_b) }
    }

    async fn handle_rag_index(req_str: String, tx: tokio::sync::broadcast::Sender<String>, _cancel_token: std::sync::Arc<std::sync::atomic::AtomicBool>) {
        let _ = tx.send(serde_json::json!({ "type": "status", "msg": "Starting Codebase Indexing for Local Auto-RAG...".to_string() }).to_string());
        
        let mut selected_model = "qwen2.5-coder:1.5b".to_string();
        let body_str = req_str.split("\r\n\r\n").nth(1).unwrap_or("").trim_matches('\0');
        if let Ok(json) = serde_json::from_str::<serde_json::Value>(body_str) {
            if let Some(m) = json.get("model").and_then(|v| v.as_str()) {
                selected_model = m.to_string();
            }
        }
        let mut chunks = vec![];
        let client = reqwest::Client::new();
        
        let extensions = vec!["rs", "md", "js", "html", "css", "toml", "json"];
        let mut count = 0;
        
        for entry in walkdir::WalkDir::new(".") {
            let entry = match entry { Ok(e) => e, Err(_) => continue };
            if entry.file_type().is_dir() { continue; }
            let path = entry.path();
            let path_str = path.to_string_lossy().to_string();
            
            if path_str.contains("/target/") || path_str.contains("/.git/") || path_str.contains("node_modules") { continue; }
            
            let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
            if !extensions.contains(&ext) { continue; }
            
            if let Ok(content) = std::fs::read_to_string(path) {
                let lines: Vec<&str> = content.lines().collect();
                for (i, chunk_lines) in lines.chunks(100).enumerate() {
                    let text = chunk_lines.join("\n");
                    if text.trim().is_empty() { continue; }
                    
                    let mut tuning_options = crate::OllamaOptions::default();
                    tuning_options.num_thread = Some(std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).saturating_sub(2).max(1));
                    let embed_req = crate::EmbedRequest {
                        model: selected_model.clone(),
                        prompt: text.clone(),
                        keep_alive: None,
                        options: Some(tuning_options),
                    };
                    
                    if let Ok(res) = client.post(format!("{}/api/embeddings", crate::OLLAMA_URL)).json(&embed_req).send().await {
                        if let Ok(embed_res) = res.json::<crate::EmbedResponse>().await {
                            chunks.push(crate::RagChunk {
                                file: format!("{} (Part {})", path_str, i+1),
                                text,
                                vector: embed_res.embedding,
                            });
                        }
                    }
                }
                count += 1;
                if count % 10 == 0 {
                    let _ = tx.send(serde_json::json!({ "type": "status", "msg": format!("Indexed {} files...", count) }).to_string());
                }
            }
        }
        
        if let Ok(json) = serde_json::to_string(&chunks) {
            let _ = std::fs::write(".zy_index.json", json);
            let _ = tx.send(serde_json::json!({ "type": "done", "msg": format!("Successfully generated local RAG vector index with {} chunks from {} files using model `{}`.", chunks.len(), count, selected_model) }).to_string());
        } else {
            let _ = tx.send(serde_json::json!({ "type": "done", "msg": "Failed to serialize RAG index.".to_string() }).to_string());
        }
    }

    async fn handle_chat_async(req_str: String, tx: tokio::sync::broadcast::Sender<String>, cancel_token: std::sync::Arc<std::sync::atomic::AtomicBool>) {
        let body_str = req_str.split("\r\n\r\n").nth(1).unwrap_or("").trim_matches('\0');
        let mut messages = vec![
            crate::Message { 
                role: "system".to_string(), 
                content: "You are an autonomous AI coding agent. You can execute bash commands and read/write files to solve user requests. Once you have solved the request, output your final response in Markdown without using any tools.".to_string(), 
                tool_calls: None, 
                images: None 
            }
        ];
        
        let mut has_user_prompt = false;
        let mut context_files: Vec<String> = Vec::new();
        let mut context_images: Vec<String> = Vec::new();
        let mut selected_model = "qwen2.5-coder:1.5b".to_string();
        
        if let Ok(json) = serde_json::from_str::<serde_json::Value>(body_str) {
            if let Some(m) = json.get("model").and_then(|v| v.as_str()) {
                selected_model = m.to_string();
            }
            if let Some(files) = json.get("context_files").and_then(|v| v.as_array()) {
                for f in files {
                    if let Some(f_str) = f.as_str() { context_files.push(f_str.to_string()); }
                }
            }
            if let Some(images) = json.get("context_images").and_then(|v| v.as_array()) {
                for i in images {
                    if let Some(i_str) = i.as_str() { context_images.push(i_str.to_string()); }
                }
            }

            if let Some(msgs) = json.get("messages").and_then(|v| v.as_array()) {
                for (idx, m) in msgs.iter().enumerate() {
                    if let (Some(role), Some(content)) = (m.get("role").and_then(|v| v.as_str()), m.get("content").and_then(|v| v.as_str())) {
                        let mut final_content = content.to_string();
                        let mut final_images = None;
                        
                        // Inject context into the very last user message
                        if idx == msgs.len() - 1 && role == "user" {
                            if !context_files.is_empty() {
                                final_content.push_str("\n\n### Attached Files Context ###\n");
                                for path in &context_files {
                                    if let Ok(file_content) = std::fs::read_to_string(path) {
                                        final_content.push_str(&format!("--- {} ---\n{}\n\n", path, file_content));
                                    }
                                }
                            }
                            if !context_images.is_empty() {
                                final_images = Some(context_images.clone());
                            }
                        }

                        messages.push(crate::Message {
                            role: role.to_string(),
                            content: final_content,
                            tool_calls: None,
                            images: final_images
                        });
                        has_user_prompt = true;
                    }
                }
            } else if let Some(p) = json.get("prompt").and_then(|v| v.as_str()) {
                messages.push(crate::Message { 
                    role: "user".to_string(), 
                    content: p.to_string(), 
                    tool_calls: None, 
                    images: None 
                });
                has_user_prompt = true;
            }
        }
        
        if !has_user_prompt { return; }

        // Local Auto-RAG Retrieval
        if let Ok(index_json) = std::fs::read_to_string(".zy_index.json") {
            if let Ok(chunks) = serde_json::from_str::<Vec<crate::RagChunk>>(&index_json) {
                // Find the latest user message
                let last_user_msg = messages.iter().rev().find(|m| m.role == "user").map(|m| m.content.clone()).unwrap_or_default();
                
                let client = reqwest::Client::new();
                let mut tuning_options = crate::OllamaOptions::default();
                tuning_options.num_thread = Some(std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).saturating_sub(2).max(1));
                let embed_req = crate::EmbedRequest {
                    model: selected_model.clone(),
                    prompt: last_user_msg,
                    keep_alive: None,
                    options: Some(tuning_options),
                };
                
                if let Ok(res) = client.post(format!("{}/api/embeddings", crate::OLLAMA_URL)).json(&embed_req).send().await {
                    if let Ok(embed_res) = res.json::<crate::EmbedResponse>().await {
                        let query_vec = embed_res.embedding;
                        let mut scored_chunks: Vec<(&crate::RagChunk, f32)> = chunks.iter().map(|c| {
                            (c, Self::cosine_similarity(&query_vec, &c.vector))
                        }).collect();
                        
                        scored_chunks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                        
                        let top_k = 3;
                        let mut rag_context = String::new();
                        for (chunk, score) in scored_chunks.into_iter().take(top_k) {
                            if score > 0.5 { // relevance threshold
                                rag_context.push_str(&format!("--- Auto-Retrieved Context from {} (Score: {:.2}) ---\n{}\n\n", chunk.file, score, chunk.text));
                            }
                        }
                        
                        if !rag_context.is_empty() {
                            let _ = tx.send(serde_json::json!({ "type": "status", "msg": format!("Auto-RAG: Injected context from {} files.", top_k) }).to_string());
                            messages[0].content.push_str("\n\nYou have automatically retrieved codebase context:\n");
                            messages[0].content.push_str(&rag_context);
                        }
                    }
                }
            }
        }

        tokio::spawn(async move {
            let client = reqwest::Client::new();
            let mut previous_tool_call = String::new();
            let mut loop_count = 0;

        loop {
            if cancel_token.load(std::sync::atomic::Ordering::Relaxed) {
                break;
            }
            if loop_count > 10 {
                let limit_msg = "*Agent paused after 10 autonomous steps to prevent infinite loop. Please provide further instructions.*".to_string();
                let sse_msg = serde_json::json!({ "type": "done", "msg": limit_msg }).to_string();
                let _ = tx.send(sse_msg);
                break;
            }
            loop_count += 1;

            let mut tuning_options = crate::OllamaOptions::default();
            tuning_options.num_thread = Some(std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).saturating_sub(2).max(1));
            tuning_options.num_ctx = Some(4096);
            tuning_options.num_predict = Some(1024);
            let req = crate::ChatRequest {
                model: selected_model.clone(),
                messages: messages.clone(),
                stream: true,
                tools: Some(Self::get_agent_tools()),
                format: None, options: Some(tuning_options), keep_alive: None,
            };

            let res = match client.post(format!("{}/api/chat", crate::OLLAMA_URL)).json(&req).send().await {
                Ok(r) => r,
                Err(e) => {
                    let err_json = serde_json::json!({ "type": "done", "msg": format!("Error connecting to Ollama: {}", e) }).to_string();
                    let _ = tx.send(err_json);
                    break;
                }
            };

            if !res.status().is_success() {
                let status = res.status();
                let err_text = res.text().await.unwrap_or_default();
                let err_msg = format!("Ollama API Error ({}): {}", status, err_text);
                let err_json = serde_json::json!({ "type": "done", "msg": err_msg }).to_string();
                let _ = tx.send(err_json);
                break;
            }

            use futures_util::StreamExt;
            let mut stream = res.bytes_stream();
            let mut full_content = String::new();
            let mut full_tool_calls = vec![];
            let mut line_buf = String::new();
            
            while let Some(chunk) = stream.next().await {
                match chunk {
                    Ok(bytes) => {
                        let chunk_str = String::from_utf8_lossy(&bytes);
                        line_buf.push_str(&chunk_str);
                        
                        while let Some(idx) = line_buf.find('\n') {
                            let line = line_buf[..idx].to_string();
                            line_buf = line_buf[idx+1..].to_string();
                            
                            if line.trim().is_empty() { continue; }
                            println!("RAW CHUNK: {}", line);
                            match serde_json::from_str::<crate::ChatResponse>(&line) {
                                Ok(chat_res) => {
                                    if let Some(err_msg) = chat_res.error {
                                        let err_json = serde_json::json!({ "type": "done", "msg": format!("Ollama Stream Error: {}", err_msg) }).to_string();
                                        let _ = tx.send(err_json);
                                        break;
                                    }
                                    if let Some(msg) = chat_res.message {
                                        if !msg.content.is_empty() {
                                            full_content.push_str(&msg.content);
                                            let _ = tx.send(serde_json::json!({ "type": "chunk", "msg": msg.content }).to_string());
                                        }
                                        if let Some(mut tc) = msg.tool_calls {
                                            full_tool_calls.append(&mut tc);
                                        }
                                    }
                                    if chat_res.done.unwrap_or(false) {
                                        break;
                                    }
                                },
                                Err(e) => {
                                    println!("JSON Parse Error: {}. Line: {}", e, line);
                                }
                            }
                        }
                    }
                    Err(e) => {
                        let err_msg = format!("Stream read error: {}", e);
                        println!("{}", err_msg);
                        let err_json = serde_json::json!({ "type": "done", "msg": err_msg }).to_string();
                        let _ = tx.send(err_json);
                        break;
                    }
                }
            }
            
            if !line_buf.trim().is_empty() {
                println!("LEFTOVER BUF: {}", line_buf);
            }
            
            if true {
                if true {
                    let msg = crate::Message {
                        role: "assistant".to_string(),
                        content: full_content.clone(),
                        tool_calls: if full_tool_calls.is_empty() { None } else { Some(full_tool_calls) },
                        images: None,
                    };

                    println!("\n=== LOOP {} ===", loop_count);
                    println!("LLM Output: {:#?}", msg);
                    
                    messages.push(msg.clone());
                    
                    let mut executed_tools = false;
                    let mut tool_calls = msg.tool_calls.clone().unwrap_or_default();
                    
                    if tool_calls.is_empty() && !msg.content.is_empty() {
                        let clean_content = msg.content
                            .trim_start_matches("```json")
                            .trim_start_matches("```")
                            .trim_end_matches("```")
                            .trim();
                        if let Ok(tc) = serde_json::from_str::<crate::ToolCallFunction>(clean_content) {
                            tool_calls.push(crate::ToolCall { function: tc });
                        } else if let Ok(tcs) = serde_json::from_str::<Vec<crate::ToolCallFunction>>(clean_content) {
                            for tc in tcs {
                                tool_calls.push(crate::ToolCall { function: tc });
                            }
                        }
                    }

                    if !msg.content.is_empty() {
                        let sse_msg = serde_json::json!({ "type": "thought", "msg": msg.content.clone() }).to_string();
                        let _ = tx.send(sse_msg);
                    }

                    if !tool_calls.is_empty() {
                        executed_tools = true;
                        for tc in tool_calls {
                            let fname = tc.function.name;
                            let args = tc.function.arguments;
                            
                            let sse_msg = serde_json::json!({ "type": "status", "msg": format!("Executing {}...", fname) }).to_string();
                            let _ = tx.send(sse_msg);
                            
                            let sse_call = serde_json::json!({ "type": "tool_call", "name": fname, "args": args.clone() }).to_string();
                            let _ = tx.send(sse_call);
                            
                            let mut output = match fname.as_str() {
                                "run_command" => {
                                    if let Some(cmd) = args.get("command").and_then(|c| c.as_str()) {
                                        // Check if Docker is available
                                        if let Ok(docker_check) = std::process::Command::new("docker").arg("--version").output() {
                                            if docker_check.status.success() {
                                                // Execute securely in Docker
                                                let cwd = std::env::current_dir().unwrap_or_default().display().to_string();
                                                let out = std::process::Command::new("docker")
                                                    .arg("run")
                                                    .arg("--rm")
                                                    .arg("-v")
                                                    .arg(format!("{}:/workspace", cwd))
                                                    .arg("-w")
                                                    .arg("/workspace")
                                                    .arg("ubuntu:latest") // or rust:latest
                                                    .arg("bash")
                                                    .arg("-c")
                                                    .arg(cmd)
                                                    .output();
                                                    
                                                match out {
                                                    Ok(o) => String::from_utf8_lossy(&o.stdout).to_string() + "\n" + &String::from_utf8_lossy(&o.stderr),
                                                    Err(e) => format!("Error executing docker container: {}", e),
                                                }
                                            } else {
                                                "Error: Docker is not running or available. Execution sandboxing failed. Please start Docker.".to_string()
                                            }
                                        } else {
                                            "Error: Docker is not installed or available. Execution sandboxing failed. Please install Docker.".to_string()
                                        }
                                    } else { "Error: Missing command argument".to_string() }
                                },
                                "read_file" => {
                                    if let Some(path) = args.get("path").and_then(|p| p.as_str()) {
                                        std::fs::read_to_string(path).unwrap_or_else(|e| e.to_string())
                                    } else { "Error: Missing path".to_string() }
                                },
                                "write_file" => {
                                    if let (Some(path), Some(content)) = (args.get("path").and_then(|p| p.as_str()), args.get("content").and_then(|c| c.as_str())) {
                                        std::fs::write(path, content).map(|_| format!("Successfully wrote to {}", path)).unwrap_or_else(|e| e.to_string())
                                    } else { "Error: Missing path or content".to_string() }
                                },
                                "create_checkpoint" => {
                                    if let Some(name) = args.get("name").and_then(|n| n.as_str()) {
                                        Self::create_checkpoint(name)
                                    } else { "Error: Missing name parameter".to_string() }
                                },
                                "restore_checkpoint" => {
                                    if let Some(name) = args.get("name").and_then(|n| n.as_str()) {
                                        Self::restore_checkpoint(name)
                                    } else { "Error: Missing name parameter".to_string() }
                                },
                                "list_checkpoints" => {
                                    Self::list_checkpoints()
                                },
                                _ => "Error: Unknown tool".to_string()
                            };
                            
                            let current_call_str = format!("{}:{:?}", fname, args);
                            if current_call_str == previous_tool_call {
                                output = format!("Error: You just ran {} with the exact same arguments. DO NOT repeat the same tool call. Analyze the output and move on to the next step, or give your final response.", fname);
                            } else {
                                previous_tool_call = current_call_str.clone();
                            }
                            
                            let truncated_output = if output.len() > 3000 {
                                output[..3000].to_string() + "\n...[truncated]"
                            } else {
                                output
                            };
                            
                            let sse_result = serde_json::json!({ "type": "tool_result", "name": fname, "result": truncated_output.clone() }).to_string();
                            let _ = tx.send(sse_result);
                            
                            let role = if msg.tool_calls.is_some() { "tool" } else { "user" };
                            let formatted_content = if role == "user" {
                                format!("Tool execution result for {}:\n{}\n\n(System: If you have completed the user's request, output your final response in Markdown. Do NOT output a tool call if you are done.)", fname, truncated_output)
                            } else {
                                truncated_output
                            };
                            
                            println!("Tool Output: {}", formatted_content);
                            
                            messages.push(crate::Message {
                                role: role.to_string(),
                                content: formatted_content,
                                tool_calls: None, images: None
                            });
                        }
                    }

                    if !executed_tools {
                        let sse_msg = serde_json::json!({ "type": "done", "msg": msg.content }).to_string();
                        let _ = tx.send(sse_msg);
                        break;
                    }
                } else { break; }
            } else { break; }
        }
        });
    }
    pub fn generate_dashboard_html() -> String {
        let mut default_model = "qwen2.5-coder:1.5b".to_string();
        if let Ok(home) = std::env::var("HOME") {
            if let Ok(m) = std::fs::read_to_string(std::path::PathBuf::from(home).join(".zy_model")) {
                let trimmed = m.trim();
                if !trimmed.is_empty() { default_model = trimmed.to_string(); }
            }
        }

        r#"<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>⚡ zy: Autonomous Local AI Agent Dashboard</title>
  <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/tokyo-night-dark.min.css">
  <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
  <style>
    :root {
      --bg: #0b0f19;
      --card-bg: #151d30;
      --border: #24304f;
      --accent: #38bdf8;
      --accent-glow: rgba(56, 189, 248, 0.25);
      --text: #f8fafc;
      --text-muted: #94a3b8;
      --success: #34d399;
      --warning: #fbbf24;
      --danger: #f87171;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace; }
    body { background: var(--bg); color: var(--text); min-height: 100vh; display: flex; flex-direction: column; }
    header { background: var(--card-bg); border-bottom: 1px solid var(--border); padding: 1rem 2rem; display: flex; justify-content: space-between; align-items: center; }
    .logo { font-size: 1.5rem; font-weight: 800; color: var(--accent); display: flex; align-items: center; gap: 0.5rem; }
    .badge { background: var(--border); color: var(--accent); padding: 0.25rem 0.6rem; border-radius: 9999px; font-size: 0.75rem; font-weight: 600; }
    .model-select { background: var(--card-bg); color: var(--accent); border: 1px solid var(--border); border-radius: 9999px; padding: 0.25rem 0.6rem; font-size: 0.75rem; font-weight: 600; outline: none; cursor: pointer; }
    .model-select option { background: var(--card-bg); color: var(--text); }
    .container { display: grid; grid-template-columns: 320px 1fr 340px; gap: 1.5rem; padding: 1.5rem; flex: 1; }
    .card { background: var(--card-bg); border: 1px solid var(--border); border-radius: 12px; padding: 1.25rem; display: flex; flex-direction: column; gap: 1rem; box-shadow: 0 4px 20px rgba(0,0,0,0.3); min-height: 0; }
    .card h3 { font-size: 0.95rem; color: var(--accent); border-bottom: 1px solid var(--border); padding-bottom: 0.5rem; text-transform: uppercase; letter-spacing: 0.05em; flex-shrink: 0; }
    .chat-box { flex: 1; overflow-y: auto; display: flex; flex-direction: column; gap: 0.75rem; min-height: 0; background: #0b0f19; padding: 1rem; border-radius: 8px; border: 1px solid var(--border); }
    .msg { padding: 0.75rem 1rem; border-radius: 8px; font-size: 0.88rem; line-height: 1.5; max-width: 85%; }
    .msg.user { background: #1e293b; color: #fff; align-self: flex-end; border-left: 3px solid var(--accent); }
    .msg.agent { background: #1e1b4b; color: #e0e7ff; align-self: flex-start; border-left: 3px solid #818cf8; }
    .input-row { display: flex; gap: 0.5rem; position: relative; align-items: flex-end; flex-shrink: 0; }
    textarea { flex: 1; background: #0b0f19; border: 1px solid var(--border); color: #fff; padding: 0.8rem 1rem; border-radius: 8px; outline: none; resize: vertical; min-height: 48px; max-height: 400px; overflow-y: auto; line-height: 1.4; transition: border-color 0.2s; }
    textarea:focus { border-color: var(--accent); box-shadow: 0 0 10px var(--accent-glow); }
    .send-btn { background: var(--accent); color: #000; border: none; padding: 0 1.25rem; height: 48px; border-radius: 8px; cursor: pointer; transition: all 0.2s; display: flex; justify-content: center; align-items: center; }
    .send-btn:hover { opacity: 0.9; transform: translateY(-1px); }
    .send-btn svg { width: 20px; height: 20px; fill: currentColor; }
    .autocomplete { position: absolute; bottom: 100%; left: 0; background: var(--card-bg); border: 1px solid var(--border); border-radius: 8px; margin-bottom: 0.5rem; width: 300px; max-height: 200px; overflow-y: auto; box-shadow: 0 4px 12px rgba(0,0,0,0.5); z-index: 50; display: none; }
    .ac-item { padding: 0.75rem 1rem; cursor: pointer; color: var(--text); font-size: 0.9rem; border-bottom: 1px solid var(--border); word-break: break-all; }
    .ac-item:last-child { border-bottom: none; }
    .ac-item:hover, .ac-item.active { background: #1e293b; color: var(--accent); }
    .context-bar { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 0.5rem; }
    .chip { background: #1e293b; border: 1px solid var(--accent); border-radius: 4px; padding: 0.25rem 0.5rem; font-size: 0.8rem; display: flex; align-items: center; gap: 0.5rem; color: #fff; }
    .chip-remove { cursor: pointer; color: var(--danger); font-weight: bold; }
    .input-wrapper { display: flex; flex-direction: column; flex: 1; position: relative; }
    .attach-btn { background: none; border: none; color: var(--text-muted); cursor: pointer; font-size: 1.5rem; display: flex; align-items: center; padding: 0 0.5rem; transition: color 0.2s; height: 48px; }
    .attach-btn:hover { color: var(--accent); }
    
    .typing-dots { display: inline-flex; align-items: center; gap: 4px; height: 1.5rem; vertical-align: middle; }
    .typing-dots span { width: 6px; height: 6px; background-color: var(--accent); border-radius: 50%; animation: typing 1.4s infinite ease-in-out both; }
    .typing-dots span:nth-child(1) { animation-delay: -0.32s; }
    .typing-dots span:nth-child(2) { animation-delay: -0.16s; }
    @keyframes typing { 0%, 80%, 100% { transform: scale(0); } 40% { transform: scale(1); } }
    
    .msg pre { background: #000; border-radius: 6px; padding: 1rem; margin: 0.5rem 0; overflow-x: auto; }
    .msg code { font-family: monospace; }
    .msg p { margin-bottom: 0.5rem; }
    .msg p:last-child { margin-bottom: 0; }
    .metric-row { display: flex; justify-content: space-between; font-size: 0.85rem; color: var(--text-muted); }
    .metric-val { color: var(--text); font-weight: 600; }
    .visual-frame { width: 100%; border-radius: 8px; border: 1px solid var(--border); background: #080c14; display: flex; justify-content: center; align-items: center; min-height: 220px; transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1); transform-origin: top right; position: relative; z-index: 1; cursor: zoom-in; }
    .visual-frame:hover { transform: scale(1.8) translateX(-15%); z-index: 100; box-shadow: -10px 15px 40px rgba(0,0,0,0.9); border-color: var(--accent); }
    .visual-frame svg { width: 100%; height: auto; display: block; }
  </style>
</head>
<body>
  <header>
    <div class="logo">⚡ zy <span class="badge">v0.1.0 • 100% LOCAL</span></div>
    <div style="display: flex; gap: 1rem; align-items: center;">
      <button onclick="indexCodebase()" style="background: none; border: 1px solid var(--warning); color: var(--warning); border-radius: 4px; padding: 0.2rem 0.5rem; cursor: pointer; font-size: 0.8rem;">🔄 Index Codebase</button>
      <button onclick="toggleEventTrail()" style="background: none; border: 1px solid var(--accent); color: var(--accent); border-radius: 4px; padding: 0.2rem 0.5rem; cursor: pointer; font-size: 0.8rem;">👁️ Event Trail</button>
      <select id="model-select" class="model-select" onchange="updateGlobalModel()">
        <option value="{default_model}">{default_model}</option>
      </select>
      <span class="badge" id="tuner-badge">DYNAMIC AITUNER: ACTIVE</span>
      <span class="badge" style="color: var(--success);" id="status-badge">● ONLINE</span>
    </div>
  </header>

  <div class="container">
    <!-- Left Column: Navigation & Telemetry -->
    <div class="card">
      <h3>💻 System Telemetry</h3>
      <div class="metric-row"><span>Engine:</span><span class="metric-val">Rust (Bare-Metal)</span></div>
      <div class="metric-row"><span>Mode:</span><span class="metric-val">AiTuner Turbo</span></div>
      <div class="metric-row"><span>Vella RAG:</span><span class="metric-val" style="color: var(--success);">Zero-Latency Active</span></div>
      <div class="metric-row"><span>Editor Bridge:</span><span class="metric-val">Port 8098 (Ready)</span></div>
      <div class="metric-row"><span>Duplex Audio:</span><span class="metric-val">16kHz VAD Active</span></div>

      <h3 style="margin-top: 1rem;">📑 Quick Commands</h3>
      <button onclick="sendQuick('/radar')">📊 Code Health Radar</button>
      <button onclick="sendQuick('/dag')">🔀 Swarm DAG Flow</button>
      <button onclick="sendQuick('/models')">🧠 Ollama Model List</button>
      <button onclick="sendQuick('/clear')">🧹 Clear Session</button>
      <button id="btn-voice-mode" onclick="toggleVoiceMode()" style="border-color: var(--accent); color: var(--accent);">🎙️ Enable Voice Mode</button>
    </div>

    <!-- Center Column: Agent Chat & REPL -->
    <div class="card">
      <h3>🤖 Autonomous Agent REPL <span style="font-size: 0.6em; color: var(--accent);">(v1.1)</span></h3>
      <div class="chat-box" id="chat-box">
      </div>
      <div class="input-row">
        <button class="attach-btn" onclick="document.getElementById('file-upload').click()" title="Attach Image">📎</button>
        <input type="file" id="file-upload" accept="image/*" style="display:none;" onchange="handleImageUpload(event)">
        <div class="input-wrapper">
          <div id="context-bar" class="context-bar"></div>
          <div id="autocomplete-box" class="autocomplete"></div>
          <textarea id="user-input" rows="1" placeholder="Type prompt, @file, or /command... (Shift+Enter for new line)"></textarea>
        </div>
        <button class="send-btn" onclick="sendPrompt()" title="Send">
          <svg viewBox="0 0 24 24"><path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"></path></svg>
        </button>
      </div>
    </div>

    <!-- Right Column: Visualizations & Radars -->
    <div class="card">
      <h3>📊 Live Visualizations</h3>
      <div class="visual-frame" id="visual-container">
        <p style="color: var(--text-muted); font-size: 0.8rem;">Click 'Code Health Radar' or 'Swarm DAG' to inspect.</p>
      </div>
      <h3 style="margin-top: 1rem;">🛡️ Safety Permissions</h3>
      <div class="metric-row"><span>Agent Force Mode:</span><span class="metric-val" style="color: var(--warning);">Prompt (Safe)</span></div>
      <div class="metric-row"><span>Sandbox Worktrees:</span><span class="metric-val" style="color: var(--success);">Enabled</span></div>
      
      <div id="event-trail-container" style="display: block;">
          <h3 style="margin-top: 1rem; display: flex; justify-content: space-between; align-items: center;">
            👁️ Event Trail
            <button onclick="document.getElementById('event-trail').innerHTML=''" style="background: none; border: 1px solid var(--border); color: var(--text-muted); border-radius: 4px; padding: 2px 5px; cursor: pointer; font-size: 0.7rem;">Clear</button>
          </h3>
          <div id="event-trail" style="background: #000; border-radius: 8px; border: 1px solid var(--border); padding: 1rem; height: 350px; overflow-y: auto; font-family: monospace; font-size: 0.8rem; margin-top: 0.5rem; white-space: pre-wrap; word-break: break-all;">
            <div style="color: var(--text-muted)">Waiting for agent events...</div>
          </div>
      </div>
    </div>
  </div>

  <script>
    window.onerror = function(msg, url, lineNo, columnNo, error) {
      const chat = document.getElementById('chat-box');
      if (chat) {
        chat.insertAdjacentHTML('beforeend', `<div class="msg agent" style="border-left-color:red; color:red;"><b>GLOBAL ERROR:</b><br/>${msg}<br/>Line: ${lineNo}</div>`);
      }
      return false;
    };
    window.addEventListener("unhandledrejection", function(event) {
      const chat = document.getElementById('chat-box');
      if (chat) {
        chat.insertAdjacentHTML('beforeend', `<div class="msg agent" style="border-left-color:red; color:red;"><b>PROMISE ERROR:</b><br/>${event.reason}</div>`);
      }
    });

    const inp = document.getElementById('user-input');
    const acBox = document.getElementById('autocomplete-box');
    const commands = [
      { cmd: '/radar', desc: 'Code Health Radar' },
      { cmd: '/dag', desc: 'Swarm DAG Flow' },
      { cmd: '/models', desc: 'Ollama Model List' },
      { cmd: '/clear', desc: 'Clear Session' },
      { cmd: '/stop', desc: 'Stop Agent Execution' }
    ];
    let currentVisMode = null;
    let contextFiles = [];
    let contextImages = [];
    let allFilesCache = [];
    let isFetchingFiles = false;

    function renderContextBar() {
      const bar = document.getElementById('context-bar');
      bar.innerHTML = '';
      contextFiles.forEach((f, i) => {
        bar.innerHTML += `<div class="chip">📄 ${f} <span class="chip-remove" onclick="removeFile(${i})">×</span></div>`;
      });
      contextImages.forEach((img, i) => {
        bar.innerHTML += `<div class="chip">🖼️ Image ${i+1} <span class="chip-remove" onclick="removeImage(${i})">×</span></div>`;
      });
    }

    function removeFile(idx) { contextFiles.splice(idx, 1); renderContextBar(); }
    function removeImage(idx) { contextImages.splice(idx, 1); renderContextBar(); }

    function handleImageUpload(e) {
      const file = e.target.files[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = function(ev) {
        const base64 = ev.target.result.split(',')[1];
        contextImages.push(base64);
        renderContextBar();
      };
      reader.readAsDataURL(file);
      e.target.value = '';
    }

    inp.addEventListener('input', async function() {
      this.style.height = '48px';
      this.style.height = (this.scrollHeight) + 'px';
      
      const val = this.value;
      const cursor = this.selectionStart;
      const textBeforeCursor = val.substring(0, cursor);
      const lastWordMatch = textBeforeCursor.match(/[\w/@.-]+$/);
      const lastWord = lastWordMatch ? lastWordMatch[0] : "";

      if (lastWord.startsWith('/')) {
        const matches = commands.filter(c => c.cmd.startsWith(lastWord));
        if (matches.length > 0) {
          acBox.innerHTML = matches.map(m => `<div class="ac-item" onclick="selectCmd('${m.cmd}')"><b>${m.cmd}</b> - ${m.desc}</div>`).join('');
          acBox.style.display = 'block';
        } else { acBox.style.display = 'none'; }
      } else if (lastWord.startsWith('@')) {
        if (allFilesCache.length === 0 && !isFetchingFiles) {
          isFetchingFiles = true;
          try {
            const res = await fetch('/api/files');
            const data = await res.json();
            allFilesCache = data.files || [];
          } catch (e) { console.error(e); }
        }
        const search = lastWord.substring(1).toLowerCase();
        let matches = allFilesCache;
        if (search) {
          matches = allFilesCache.filter(f => f.toLowerCase().includes(search));
        }
        matches = matches.slice(0, 10); // limit 10
        if (matches.length > 0) {
          acBox.innerHTML = matches.map(m => `<div class="ac-item" onclick="selectFile('${m}')">📄 ${m}</div>`).join('');
          acBox.style.display = 'block';
        } else { acBox.style.display = 'none'; }
      } else { 
        acBox.style.display = 'none'; 
      }
    });

    function selectCmd(cmd) {
      const val = inp.value;
      const lastWordMatch = val.match(/[\w/@.-]+$/);
      if (lastWordMatch) {
         inp.value = val.substring(0, val.length - lastWordMatch[0].length) + cmd + ' ';
      }
      inp.focus();
      acBox.style.display = 'none';
    }

    function selectFile(file) {
      if (!contextFiles.includes(file)) {
        contextFiles.push(file);
        renderContextBar();
      }
      const val = inp.value;
      const lastWordMatch = val.match(/[\w/@.-]+$/);
      if (lastWordMatch) {
         inp.value = val.substring(0, val.length - lastWordMatch[0].length);
      }
      inp.focus();
      acBox.style.display = 'none';
    }

    inp.addEventListener('keydown', function(e) {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        acBox.style.display = 'none';
        sendPrompt();
      }
    });

    if (typeof marked !== 'undefined') {
      marked.setOptions({
        highlight: function(code, lang) {
          if (typeof hljs !== 'undefined') {
            if (lang && hljs.getLanguage(lang)) {
              return hljs.highlight(code, { language: lang }).value;
            }
            return hljs.highlightAuto(code).value;
          }
          return code;
        }
      });
    }

    let chatHistory = [];
    try {
      const stored = localStorage.getItem('zy_chat');
      if (stored) {
        chatHistory = JSON.parse(stored);
      }
    } catch (e) {
      console.error('Failed to load chat history', e);
    }

    const typingHtml = `<div class="typing-dots"><span></span><span></span><span></span></div>`;

    function renderHistory() {
      const chat = document.getElementById('chat-box');
      if (chatHistory.length === 0) {
        chat.innerHTML = `<div class="msg agent">⚡ <b>zy agent ready.</b> 100% local, air-gapped, zero-latency pair programmer online. How can I assist you today?</div>`;
        return;
      }
      chat.innerHTML = '';
      chatHistory.forEach(msg => {
        if (msg.role === 'user') {
          chat.insertAdjacentHTML('beforeend', `<div class="msg user">${msg.content.replace(/\\n/g, '<br/>')}</div>`);
        } else {
          let formattedHtml = msg.content;
          try { formattedHtml = typeof marked !== 'undefined' ? marked.parse(msg.content) : msg.content; } catch(e) {}
          chat.insertAdjacentHTML('beforeend', `<div class="msg agent">⚡ <b>zy:</b><br/>${formattedHtml}</div>`);
        }
      });
      window.scrollTo(0, document.body.scrollHeight);
    }
    
    async function loadModels() {
      try {
        const res = await fetch('/api/models');
        const data = await res.json();
        const select = document.getElementById('model-select');
        const currentSelected = select.value;
        if (data.models && data.models.length > 0) {
          select.innerHTML = '';
          data.models.forEach(m => {
            const opt = document.createElement('option');
            opt.value = m.name;
            opt.textContent = m.name;
            if (m.name === currentSelected || m.name.includes(currentSelected)) {
              opt.selected = true;
            }
            select.appendChild(opt);
          });
        }
      } catch (e) {
        console.error("Failed to load models for dropdown", e);
      }
    }
    
    async function updateGlobalModel() {
        const model = document.getElementById('model-select').value;
        try {
            await fetch('/api/config/model', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ model })
            });
        } catch (e) { console.error(e); }
    }
    
    loadModels();
    
    // Render on load
    renderHistory();

    const evtSource = new EventSource('/api/events');
    evtSource.onmessage = function(e) {
      try {
        const data = JSON.parse(e.data);
        const chat = document.getElementById('chat-box');
        const trail = document.getElementById('event-trail');
        
        if (data.type === 'thought' || data.type === 'status') {
            console.log('SSE thought/status:', data.msg);
            let pMsg = document.getElementById('processing-msg');
            if (!pMsg) {
                chat.insertAdjacentHTML('beforeend', `<div class="msg agent" id="processing-msg"></div>`);
                pMsg = document.getElementById('processing-msg');
                chat.scrollTop = chat.scrollHeight;
            }
            
            if (data.type === 'thought') {
                pMsg.innerHTML = `<span style="color:var(--accent);">[Thinking]</span> ... ${typingHtml}`;
            } else {
                pMsg.innerHTML = `<span style="color:var(--accent);">[Agent Action]</span> ${data.msg} ... ${typingHtml}`;
            }

            if (trail && trail.parentElement.style.display !== 'none') {
                const el = document.createElement('div');
                el.style.color = 'var(--text-muted)';
                el.innerText = data.type === 'thought' ? `🤔 Thought: ${data.msg}` : `⚙️ Status: ${data.msg}`;
                trail.appendChild(el);
                trail.scrollTop = trail.scrollHeight;
            }
        } else if (data.type === 'tool_call') {
            if (trail && trail.parentElement.style.display !== 'none') {
                const el = document.createElement('div');
                el.style.color = 'var(--warning)';
                el.innerText = `🛠️ Tool Call: ${data.name}\n${JSON.stringify(data.args, null, 2)}`;
                trail.appendChild(el);
                trail.scrollTop = trail.scrollHeight;
            }
        } else if (data.type === 'tool_result') {
            if (trail && trail.parentElement.style.display !== 'none') {
                const el = document.createElement('div');
                el.style.color = '#66d9ef';
                el.innerText = data.result;
                trail.appendChild(el);
                trail.scrollTop = trail.scrollHeight;
            }
        } else if (data.type === 'chunk') {
            console.log('SSE chunk received');
            let pMsg = document.getElementById('processing-msg');
            if (pMsg) {
                pMsg.remove();
            }
            let currentMsg = document.getElementById('agent-msg');
            if (!currentMsg) {
                chat.insertAdjacentHTML('beforeend', `<div class="msg agent" id="agent-msg">⚡ <b>zy:</b><br/></div>`);
                currentMsg = document.getElementById('agent-msg');
            }
            const safeText = data.msg.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/\n/g, '<br/>');
            currentMsg.innerHTML += safeText;
            chat.scrollTop = chat.scrollHeight;
        } else if (data.type === 'done') {
           const pMsg = document.getElementById('processing-msg');
           if (pMsg) { pMsg.remove(); }
           const aMsg = document.getElementById('agent-msg');
           if (aMsg) { aMsg.removeAttribute('id'); }
           
           if (data.msg && data.msg.trim() !== '') {
               chatHistory.push({ role: 'assistant', content: data.msg });
               localStorage.setItem('zy_chat', JSON.stringify(chatHistory));
               
               let formattedHtml = data.msg;
               try { formattedHtml = typeof marked !== 'undefined' ? marked.parse(data.msg) : data.msg; } catch(e) {}
               chat.insertAdjacentHTML('beforeend', `<div class="msg agent">⚡ <b>zy:</b><br/>${formattedHtml}</div>`);
               chat.scrollTop = chat.scrollHeight;
               
               if (isVoiceModeEnabled) {
                   // Strip markdown for TTS
                   const cleanText = data.msg
                       .replace(/\\*\\*/g, '')
                       .replace(/`/g, '')
                       .replace(/#/g, '')
                       .replace(/\\[.*?\\]\\(.*?\\)/g, '')
                       .replace(/<[^>]*>/g, '');
                   speakText(cleanText);
               }
           }
        }
      } catch (err) {
        console.error("SSE parse error", err);
      }
    };

    async function indexCodebase() {
      const chat = document.getElementById('chat-box');
      chat.insertAdjacentHTML('beforeend', `<div class="msg agent" id="processing-msg">Initializing Codebase Index for Local Auto-RAG... ${typingHtml}</div>`);
      chat.scrollTop = chat.scrollHeight;
      
      try {
        await fetch('/api/rag/index', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ model: document.getElementById('model-select').value })
        });
      } catch (e) {
        console.error('RAG Index error', e);
      }
    }

    async function sendPrompt() {
      const text = inp.value.trim();
      if (!text) return;
      
      console.log('Sending prompt:', text);

      chatHistory.push({ role: 'user', content: text });
      try {
        localStorage.setItem('zy_chat', JSON.stringify(chatHistory));
      } catch (e) {
        console.warn('LocalStorage error:', e);
      }

      const chat = document.getElementById('chat-box');
      chat.insertAdjacentHTML('beforeend', `<div class="msg user">${text.replace(/\\n/g, '<br/>')}</div>`);
      inp.value = '';
      inp.style.height = '48px';
      chat.scrollTop = chat.scrollHeight;

      if (text.startsWith('/radar')) {
        currentVisMode = 'radar';
        chat.insertAdjacentHTML('beforeend', `<div class="msg agent" id="processing-msg">Scanning codebase health... ${typingHtml}</div>`);
        const res = await fetch('/api/radar/svg');
        const svg = await res.text();
        const pMsg = document.getElementById('processing-msg');
        if (pMsg) { pMsg.remove(); }
        document.getElementById('visual-container').innerHTML = svg;
        const msgHtml = "Rendered Codebase Health & Architecture Radar Chart. Polling activated.";
        chatHistory.push({ role: 'assistant', content: msgHtml });
        localStorage.setItem('zy_chat', JSON.stringify(chatHistory));
        chat.insertAdjacentHTML('beforeend', `<div class="msg agent">⚡ <b>zy:</b><br/>${msgHtml}</div>`);
      } else if (text.startsWith('/dag')) {
        currentVisMode = 'dag';
        chat.insertAdjacentHTML('beforeend', `<div class="msg agent" id="processing-msg">Planning swarm tasks via LLM... ${typingHtml}</div>`);
        let prompt = text.replace('/dag', '').trim();
        if (!prompt) prompt = "Refactor and optimize zy codebase";
        
        // 1. Get the SVG + subtasks array
        const res = await fetch('/api/dag/svg', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ goal: prompt })
        });
        const data = await res.json();
        const pMsg = document.getElementById('processing-msg');
        if (pMsg) { pMsg.remove(); }
        document.getElementById('visual-container').innerHTML = data.svg;
        
        // 2. Trigger the true execution (which streams via SSE)
        chat.insertAdjacentHTML('beforeend', `<div class="msg agent" id="processing-msg">Executing DAG with multiple parallel agents... ${typingHtml}</div>`);
        chat.scrollTop = chat.scrollHeight;
        
        try {
            await fetch('/api/dag/execute', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ goal: prompt, subtasks: data.subtasks })
            });
        } catch (e) {
            console.error('DAG execute error', e);
        }
      } else if (text.startsWith('/models')) {
        chat.insertAdjacentHTML('beforeend', `<div class="msg agent" id="processing-msg">Fetching local models... ${typingHtml}</div>`);
        const res = await fetch('/api/models');
        const data = await res.json();
        const pMsg = document.getElementById('processing-msg');
        if (pMsg) { pMsg.remove(); }
        let html = "<b>Available Local Models:</b><br/><ul>";
        if (data.models) {
          for (const m of data.models) {
            html += `<li>${m.name}</li>`;
          }
        } else { html += "<li>No models found or Ollama is offline.</li>"; }
        html += "</ul>";
        chatHistory.push({ role: 'assistant', content: html });
        localStorage.setItem('zy_chat', JSON.stringify(chatHistory));
        chat.insertAdjacentHTML('beforeend', `<div class="msg agent">⚡ <b>zy:</b><br/>${html}</div>`);
      } else if (text.startsWith('/clear')) {
        chatHistory = [];
        localStorage.removeItem('zy_chat');
        chat.innerHTML = '<div class="msg agent">⚡ <b>zy agent ready.</b> Session cleared.</div>';
      } else if (text.startsWith('/stop')) {
        chat.insertAdjacentHTML('beforeend', `<div class="msg agent" id="processing-msg">Stopping agent... ${typingHtml}</div>`);
        const res = await fetch('/api/stop');
        const pMsg = document.getElementById('processing-msg');
        if (pMsg) { pMsg.remove(); }
      } else {
        console.log('Displaying processing msg');
        chat.insertAdjacentHTML('beforeend', `<div class="msg agent" id="processing-msg">Processing request with local LLM engine... ${typingHtml}</div>`);
        chat.scrollTop = chat.scrollHeight;
        
        try {
          const payload = { 
            model: document.getElementById('model-select').value,
            messages: chatHistory,
            context_files: contextFiles,
            context_images: contextImages
          };
          
          contextFiles = [];
          contextImages = [];
          renderContextBar();

          const res = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
          });
          if (!res.ok) {
            throw new Error(`HTTP error! status: ${res.status}`);
          }
        } catch (e) {
          console.error('Chat error', e);
          const pMsg = document.getElementById('processing-msg');
          if (pMsg) { pMsg.remove(); }
          chat.insertAdjacentHTML('beforeend', `<div class="msg agent" style="border-left-color:var(--danger)">⚡ <b>zy error:</b><br/>${e.message}</div>`);
        }
      }
      chat.scrollTop = chat.scrollHeight;
    }

    // Live telemetry background polling
    setInterval(async () => {
      if (currentVisMode === 'radar') {
        try {
          const res = await fetch('/api/radar/svg');
          if (res.ok) {
            const svg = await res.text();
            document.getElementById('visual-container').innerHTML = svg;
          }
        } catch (e) { console.error('Radar polling failed', e); }
      }
    }, 5000);

    function sendQuick(cmd) {
      document.getElementById('user-input').value = cmd;
      sendPrompt();
    }
    
    function toggleEventTrail() {
        const container = document.getElementById('event-trail-container');
        if (container.style.display === 'none') {
            container.style.display = 'block';
        } else {
            container.style.display = 'none';
        }
    }
    
    // Voice Mode Implementation
    let isVoiceModeEnabled = false;
    let recognition = null;
    let isRecognizing = false;

    if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        recognition = new SpeechRecognition();
        recognition.continuous = false; // We will restart on end for better VAD chunking
        recognition.interimResults = false;
        recognition.lang = 'en-US';

        recognition.onstart = function() {
            isRecognizing = true;
        };

        recognition.onresult = function(event) {
            const transcript = event.results[0][0].transcript;
            if (transcript.trim() !== '') {
                document.getElementById('user-input').value = transcript;
                sendPrompt(); // Auto submit!
            }
        };

        recognition.onerror = function(event) {
            console.error('Speech recognition error', event.error);
        };

        recognition.onend = function() {
            isRecognizing = false;
            // Restart if voice mode is still enabled and we are not currently speaking
            if (isVoiceModeEnabled && !window.speechSynthesis.speaking) {
                try { recognition.start(); } catch(e) {}
            }
        };
    }

    function toggleVoiceMode() {
        const btn = document.getElementById('btn-voice-mode');
        isVoiceModeEnabled = !isVoiceModeEnabled;
        
        if (isVoiceModeEnabled) {
            btn.innerHTML = '🛑 Disable Voice Mode';
            btn.style.backgroundColor = 'var(--danger)';
            btn.style.color = '#fff';
            if (recognition && !isRecognizing) {
                try { recognition.start(); } catch(e) {}
            }
        } else {
            btn.innerHTML = '🎙️ Enable Voice Mode';
            btn.style.backgroundColor = 'transparent';
            btn.style.color = 'var(--accent)';
            window.speechSynthesis.cancel(); // Stop talking
            if (recognition && isRecognizing) {
                recognition.stop();
            }
        }
    }
    
    function speakText(text) {
        if (!('speechSynthesis' in window)) return;
        
        // Stop listening while speaking so it doesn't hear itself
        if (recognition && isRecognizing) {
            recognition.stop();
        }
        
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.rate = 1.05;
        
        utterance.onend = function() {
            // Resume listening after speaking finishes
            if (isVoiceModeEnabled && recognition && !isRecognizing) {
                try { recognition.start(); } catch(e) {}
            }
        };
        
        window.speechSynthesis.speak(utterance);
    }
  </script>
</body>
</html>"#.to_string().replace("{default_model}", &default_model)
    }

    pub async fn start(port: u16) -> Result<tokio::task::JoinHandle<()>, Box<dyn std::error::Error>> {
        let listener = TcpListener::bind(format!("127.0.0.1:{}", port)).await?;
        println!("{}", format!("⚡ zy Embedded Web Dashboard listening at http://127.0.0.1:{}", port).green().bold());

        let (tx, _rx) = tokio::sync::broadcast::channel::<String>(100);
        let cancel_token = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));

        let handle = tokio::spawn(async move {
            loop {
                if let Ok((mut stream, _)) = listener.accept().await {
                    let tx = tx.clone();
                    let cancel_token = cancel_token.clone();
                    tokio::spawn(async move {
                        let mut buf = [0u8; 4096];
                        if let Ok(n) = stream.read(&mut buf).await {
                            if n == 0 { return; }
                            let req_str = String::from_utf8_lossy(&buf[..n]);
                            let first_line = req_str.lines().next().unwrap_or_default();
                            let parts: Vec<&str> = first_line.split_whitespace().collect();
                            let path = if parts.len() > 1 { parts[1] } else { "/" };

                            if path == "/api/events" {
                                Self::handle_events_sse(&mut stream, tx).await;
                                return;
                            }

                            if path == "/api/stop" {
                                cancel_token.store(true, std::sync::atomic::Ordering::Relaxed);
                                let sse_msg = serde_json::json!({ "type": "done", "msg": "❌ Agent execution forcefully aborted by user." }).to_string();
                                let _ = tx.send(sse_msg);
                                use tokio::io::AsyncWriteExt;
                                let headers = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: close\r\nAccess-Control-Allow-Origin: *\r\n\r\n{}";
                                let _ = stream.write_all(headers.as_bytes()).await;
                                return;
                            }

                            if path == "/api/chat" {
                                cancel_token.store(false, std::sync::atomic::Ordering::Relaxed);
                                Self::handle_chat_async(req_str.to_string(), tx, cancel_token.clone()).await;
                                use tokio::io::AsyncWriteExt;
                                let headers = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: close\r\nAccess-Control-Allow-Origin: *\r\n\r\n{}";
                                let _ = stream.write_all(headers.as_bytes()).await;
                                return;
                            }
                            
                            if path == "/api/dag/execute" {
                                cancel_token.store(false, std::sync::atomic::Ordering::Relaxed);
                                Self::handle_dag_execute(req_str.to_string(), tx, cancel_token.clone()).await;
                                use tokio::io::AsyncWriteExt;
                                let headers = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: close\r\nAccess-Control-Allow-Origin: *\r\n\r\n{}";
                                let _ = stream.write_all(headers.as_bytes()).await;
                                return;
                            }
                            
                            if path == "/api/rag/index" {
                                cancel_token.store(false, std::sync::atomic::Ordering::Relaxed);
                                Self::handle_rag_index(req_str.to_string(), tx, cancel_token.clone()).await;
                                use tokio::io::AsyncWriteExt;
                                let headers = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: close\r\nAccess-Control-Allow-Origin: *\r\n\r\n{}";
                                let _ = stream.write_all(headers.as_bytes()).await;
                                return;
                            }
                            
                            if path == "/api/config/model" {
                                let body_str = req_str.split("\r\n\r\n").nth(1).unwrap_or("").trim_matches('\0');
                                if let Ok(json) = serde_json::from_str::<serde_json::Value>(body_str) {
                                    if let Some(model) = json.get("model").and_then(|v| v.as_str()) {
                                        if let Ok(home) = std::env::var("HOME") {
                                            let _ = std::fs::write(std::path::PathBuf::from(home).join(".zy_model"), model);
                                        }
                                    }
                                }
                                use tokio::io::AsyncWriteExt;
                                let headers = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: close\r\nAccess-Control-Allow-Origin: *\r\n\r\n{\"status\":\"ok\"}";
                                let _ = stream.write_all(headers.as_bytes()).await;
                                return;
                            }


                            let (status, content_type, body) = match path {
                                "/" | "/index.html" => ("200 OK".to_string(), "text/html; charset=utf-8".to_string(), Self::generate_dashboard_html()),
                                "/api/status" => {
                                    let json = serde_json::json!({
                                        "status": "healthy",
                                        "version": "0.1.0",
                                        "app": "zy",
                                        "local": true,
                                        "tuner": "active"
                                    }).to_string();
                                    ("200 OK".to_string(), "application/json".to_string(), json)
                                }
                                "/api/files" => {
                                    let mut files = Vec::new();
                                    for entry in walkdir::WalkDir::new(".").into_iter().filter_map(|e| e.ok()) {
                                        if entry.file_type().is_file() {
                                            let path_str = entry.path().display().to_string();
                                            if !path_str.contains("/.git/") && !path_str.contains("/target/") {
                                                files.push(path_str.trim_start_matches("./").to_string());
                                            }
                                        }
                                    }
                                    let json = serde_json::json!({ "files": files }).to_string();
                                    ("200 OK".to_string(), "application/json".to_string(), json)
                                }
                                "/api/radar/svg" => {
                                    let metrics = CodebaseRadarMetrics::calculate(Path::new("."));
                                    ("200 OK".to_string(), "image/svg+xml".to_string(), metrics.to_svg())
                                }
                                "/api/dag/svg" => {
                                    let body_str = req_str.split("\r\n\r\n").nth(1).unwrap_or("").trim_matches('\0');
                                    let mut target = "Refactor and optimize zy codebase".to_string();
                                    if let Ok(json) = serde_json::from_str::<serde_json::Value>(body_str) {
                                        if let Some(goal) = json.get("goal").and_then(|v| v.as_str()) {
                                            if !goal.is_empty() { target = goal.to_string(); }
                                        }
                                    }
                                    let mut subtasks = vec!["Scan symbols and AST".to_string(), "Run LSP diagnostics".to_string(), "Generate automated tests".to_string()];
                                    let req = crate::ChatRequest {
                                        model: "qwen2.5-coder:1.5b".to_string(),
                                        messages: vec![crate::Message { role: "user".to_string(), content: format!("Break down this goal into 3-5 technical subtasks. Goal: {}\n\nReturn ONLY a JSON array of strings, nothing else. Example: [\"task 1\", \"task 2\"]", target), tool_calls: None, images: None }],
                                        stream: false, tools: None, format: None, options: None, keep_alive: None,
                                    };
                                    let client = reqwest::Client::new();
                                    if let Ok(res) = client.post(format!("{}/api/chat", crate::OLLAMA_URL)).json(&req).send().await {
                                        if let Ok(chat_res) = res.json::<crate::ChatResponse>().await {
                                            if let Some(msg) = chat_res.message {
                                                if let Ok(parsed) = serde_json::from_str::<Vec<String>>(&msg.content) {
                                                    if !parsed.is_empty() { subtasks = parsed; }
                                                }
                                            }
                                        }
                                    }
                                    let dag = DagLayout::build_swarm_workflow_dag(&target, &subtasks);
                                    let json_resp = serde_json::json!({
                                        "svg": dag.to_svg(),
                                        "subtasks": subtasks
                                    });
                                    ("200 OK".to_string(), "application/json".to_string(), json_resp.to_string())
                                }
                                "/api/dag/json" => {
                                    let dag = DagLayout::build_swarm_workflow_dag("Refactor and optimize zy codebase", &[
                                        "Scan symbols and AST".to_string(),
                                        "Run LSP diagnostics".to_string(),
                                        "Generate automated tests".to_string(),
                                    ]);
                                    ("200 OK".to_string(), "application/json".to_string(), serde_json::to_string(&dag).unwrap())
                                }
                                "/api/models" => {
                                    let client = reqwest::Client::new();
                                    let mut reply = r#"{"models":[]}"#.to_string();
                                    if let Ok(res) = client.get(format!("{}/api/tags", crate::OLLAMA_URL)).send().await {
                                        if let Ok(text) = res.text().await {
                                            reply = text;
                                        }
                                    }
                                    ("200 OK".to_string(), "application/json".to_string(), reply)
                                }
                                _ => ("404 NOT FOUND".to_string(), "text/plain".to_string(), "Not Found".to_string()),
                            };

                            let resp = format!(
                                "HTTP/1.1 {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nCache-Control: no-store, no-cache, must-revalidate, max-age=0\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n{}",
                                status,
                                content_type,
                                body.len(),
                                body
                            );
                            let _ = stream.write_all(resp.as_bytes()).await;
                        }
                    });
                }
            }
        });

        Ok(handle)
    }
}

// =================================================================================================
// SYSTEM 5: UNIVERSAL IDE & EDITOR SIDECAR / LSP STACK
// =================================================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LspPosition {
    pub line: u32,
    pub character: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LspRange {
    pub start: LspPosition,
    pub end: LspPosition,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LspDiagnostic {
    pub range: LspRange,
    pub severity: u32, // 1: Error, 2: Warning, 3: Information, 4: Hint
    pub code: Option<String>,
    pub source: String,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LspCodeAction {
    pub title: String,
    pub kind: String,
    pub is_preferred: bool,
    pub command: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LspHoverResult {
    pub contents: String,
    pub range: Option<LspRange>,
}

pub struct UniversalEditorSidecarStack;

impl UniversalEditorSidecarStack {
    pub fn handle_lsp_request(req_json: &serde_json::Value) -> serde_json::Value {
        let method = req_json.get("method").and_then(|v| v.as_str()).unwrap_or("");
        let id = req_json.get("id");

        match method {
            "initialize" => serde_json::json!({
                "jsonrpc": "2.0",
                "id": id,
                "result": {
                    "capabilities": {
                        "textDocumentSync": 1,
                        "hoverProvider": true,
                        "codeActionProvider": true,
                        "diagnosticProvider": true,
                        "completionProvider": {
                            "triggerCharacters": [".", ":", "/", "@"]
                        },
                        "executeCommandProvider": {
                            "commands": ["zy.explain", "zy.refactor", "zy.generateTests", "zy.securityAudit"]
                        }
                    },
                    "serverInfo": {
                        "name": "zy-lsp-sidecar",
                        "version": "0.1.0"
                    }
                }
            }),
            "textDocument/hover" => {
                let file_path = req_json.pointer("/params/textDocument/uri").and_then(|v| v.as_str()).unwrap_or("unknown");
                let line = req_json.pointer("/params/position/line").and_then(|v| v.as_u64()).unwrap_or(0);

                serde_json::json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "result": {
                        "contents": format!("### ⚡ zy Copilot Hover\n* **File:** `{}`\n* **Line:** `{}`\n* **Status:** 100% indexed in Vella RAG\n* **AiTuner:** Turbo Mode (Active)", file_path, line),
                    }
                })
            }
            "textDocument/codeAction" => {
                serde_json::json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "result": [
                        {
                            "title": "⚡ zy: Optimize & Refactor Function",
                            "kind": "refactor.rewrite",
                            "isPreferred": true,
                            "command": { "command": "zy.refactor", "title": "Refactor" }
                        },
                        {
                            "title": "🛡️ zy: Perform Security & Vulnerability Audit",
                            "kind": "quickfix",
                            "isPreferred": false,
                            "command": { "command": "zy.securityAudit", "title": "Audit" }
                        },
                        {
                            "title": "🧪 zy: Auto-Generate Unit Test Suite",
                            "kind": "source.generateTests",
                            "isPreferred": false,
                            "command": { "command": "zy.generateTests", "title": "Generate Tests" }
                        }
                    ]
                })
            }
            _ => serde_json::json!({
                "jsonrpc": "2.0",
                "id": id,
                "result": null
            }),
        }
    }

    pub fn generate_neovim_config() -> String {
        r#"-- zy Native Neovim Integration
local lspconfig = require('lspconfig')
local configs = require('lspconfig.configs')

if not configs.zy_lsp then
  configs.zy_lsp = {
    default_config = {
      cmd = { 'zy', 'sidecar', '--port', '8098' },
      filetypes = { 'rust', 'python', 'javascript', 'typescript', 'c', 'cpp', 'go' },
      root_dir = lspconfig.util.root_pattern('.git', 'Cargo.toml', 'package.json'),
      settings = {},
    },
  }
end

lspconfig.zy_lsp.setup{
  on_attach = function(client, bufnr)
    print("⚡ zy AI Sidecar connected to Neovim buffer")
  end
}
"#
        .to_string()
    }

    pub fn generate_vscode_config() -> String {
        r#"{
  "zy.sidecar.port": 8098,
  "zy.sidecar.autostart": true,
  "zy.agent.forceMode": false,
  "zy.rag.autoIndex": true,
  "zy.tuner.mode": "auto"
}"#
        .to_string()
    }
}
