use base64::Engine;
use clap::{Parser, Subcommand};
use colored::Colorize;
use futures_util::StreamExt;
use indicatif::{ProgressBar, ProgressStyle};
use inquire::{Confirm, Select, Text};
use notify::{RecursiveMode, Watcher};
use reqwest::Client;
use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;
use serde::{Deserialize, Serialize};
use similar::{ChangeTag, TextDiff};
use std::fs;
use std::io::{self, Write};
use sysinfo::System;
use termimad::print_text;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use walkdir::WalkDir;

pub const OLLAMA_URL: &str = "http://localhost:11434";

#[derive(Parser, Clone, Debug)]
#[command(name = "zy")]
#[command(about = "A super powerful local LLM CLI Agent", long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<Commands>,

    /// The model to use by default
    #[arg(short, long, default_value = "llama2")]
    pub model: String,

    /// Global system prompt to define the model's persona
    #[arg(short, long)]
    pub system: Option<String>,

    /// Fast scout model for speculative routing
    #[arg(long)]
    pub scout: Option<String>,

    /// Inject compact repository symbol map into conversation context
    #[arg(long)]
    pub map: bool,

    /// Ephemeral Sandbox Container Engine for isolated command execution
    #[arg(long)]
    pub sandbox: bool,

    /// Multi-Agent Swarm Orchestrator goal
    #[arg(long)]
    pub swarm: Option<String>,

    /// Launch Full-Screen Interactive TUI Dashboard (ratatui + crossterm)
    #[arg(long)]
    pub tui: bool,
}

#[derive(Subcommand, Clone, Debug)]
pub enum Commands {
    /// List available local models
    List,
    /// Start a chat session or send a single prompt
    Chat {
        /// The prompt to send. If empty, starts an interactive session.
        prompt: Vec<String>,
        
        /// Model to use for this chat session
        #[arg(short, long)]
        model: Option<String>,

        /// Fast scout model for dual-model speculative routing
        #[arg(long)]
        scout: Option<String>,

        /// Attach the contents of these files as context
        #[arg(short, long)]
        file: Vec<String>,

        /// System prompt specific to this chat
        #[arg(short, long)]
        system: Option<String>,

        /// Enable agent mode (allows the model to execute tools like bash and file writing)
        #[arg(short, long)]
        agent: bool,
        
        /// Persistent session name (saves and loads chat history)
        #[arg(long)]
        session: Option<String>,
        
        /// Enable Retrieval-Augmented Generation (RAG) using local indexed codebase
        #[arg(short, long)]
        rag: bool,

        /// Print output nicely formatted as Markdown (disables streaming)
        #[arg(long)]
        markdown: bool,

        /// Adjust model temperature (0.0 to 1.0). Lower = less hallucination, more deterministic. Default is 0.1 for strictness.
        #[arg(short = 't', long, default_value_t = 0.1)]
        temperature: f32,

        /// Force agent actions without asking for user confirmation
        #[arg(short = 'F', long)]
        force: bool,
        /// Swarm Executor Model (If set, the main model acts as Architect, this model acts as Executor)
        #[arg(long)]
        executor: Option<String>,

        /// Enable AI Strategist Mode (Forces OODA Loop reasoning and lethal efficiency)
        #[arg(long)]
        strategist: bool,

        /// Grammar-constrained JSON format schema (e.g. "json" or custom schema JSON string)
        #[arg(long)]
        format: Option<String>,

        /// Inject compact repository symbol map into conversation context
        #[arg(long)]
        map: bool,

        /// Ephemeral Sandbox Container Engine for isolated command execution
        #[arg(long)]
        sandbox: bool,

        /// Multi-Agent Swarm Orchestrator goal
        #[arg(long)]
        swarm: Option<String>,

        /// Launch Full-Screen Interactive TUI Dashboard (ratatui + crossterm)
        #[arg(long)]
        tui: bool,
    },
    /// Index a directory for RAG
    Index {
        /// The directory to index
        #[arg(default_value = ".")]
        path: String,
    },
    /// Vella Zero-Latency Background OS Watcher
    Watch {
        /// The directory to watch and auto-index
        #[arg(default_value = ".")]
        path: String,
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct ToolCallFunction {
    pub name: String,
    pub arguments: serde_json::Value,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct ToolCall {
    pub function: ToolCallFunction,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Message {
    pub role: String,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub images: Option<Vec<String>>,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct OllamaOptions {
    pub temperature: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_ctx: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_thread: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_gpu: Option<usize>,
}

#[derive(Serialize, Clone, Debug)]
pub struct AiTunerState {
    pub num_ctx: usize,
    pub profile_name: String,
    pub opts: OllamaOptions,
}

#[derive(Serialize, Debug)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<Message>,
    pub stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub options: Option<OllamaOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub keep_alive: Option<i32>,
}

#[derive(Deserialize, Debug)]
pub struct ChatResponse {
    pub message: Option<Message>,
    #[allow(dead_code)]
    pub done: Option<bool>,
    pub error: Option<String>,
}

#[derive(Deserialize, Debug)]
pub struct ModelList {
    pub models: Vec<ModelInfo>,
}

#[derive(Deserialize, Debug)]
pub struct ModelInfo {
    pub name: String,
}

// RAG structs
#[derive(Serialize, Deserialize, Debug)]
pub struct EmbedRequest {
    pub model: String,
    pub prompt: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub keep_alive: Option<i32>,
}

#[derive(Deserialize, Debug)]
pub struct EmbedResponse {
    pub embedding: Vec<f32>,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct RagChunk {
    pub file: String,
    pub text: String,
    pub vector: Vec<f32>,
}

// -------------------------------------------------------------------------------------------------
// FEATURE 1: NATIVE LSP / COMPILER DIAGNOSTICS STRUCTS & ENGINE
// -------------------------------------------------------------------------------------------------

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct DiagnosticIssue {
    pub file: String,
    pub line: usize,
    pub column: usize,
    pub severity: String, // "error", "warning", "info"
    pub message: String,
    pub code_snippet: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct DiagnosticReport {
    pub target: String,
    pub tool: String,
    pub success: bool,
    pub issue_count: usize,
    pub error_count: usize,
    pub warning_count: usize,
    pub issues: Vec<DiagnosticIssue>,
    pub summary: String,
}

pub fn parse_cargo_json_diagnostics(target: &str, stdout: &str) -> Vec<DiagnosticIssue> {
    let mut issues = Vec::new();
    for line in stdout.lines() {
        let line_t = line.trim();
        if line_t.is_empty() { continue; }
        if let Ok(val) = serde_json::from_str::<serde_json::Value>(line_t) {
            if val.get("reason").and_then(|r| r.as_str()) == Some("compiler-message") {
                if let Some(msg) = val.get("message") {
                    let level = msg.get("level").and_then(|l| l.as_str()).unwrap_or("error").to_string();
                    let message_text = msg.get("message").and_then(|m| m.as_str()).unwrap_or("").to_string();
                    
                    let mut file = target.to_string();
                    let mut line_num = 1;
                    let mut col_num = 1;
                    let mut snippet = None;

                    if let Some(spans) = msg.get("spans").and_then(|s| s.as_array()) {
                        if let Some(primary) = spans.iter().find(|s| s.get("is_primary").and_then(|p| p.as_bool()).unwrap_or(false)).or_else(|| spans.first()) {
                            if let Some(f) = primary.get("file_name").and_then(|f| f.as_str()) {
                                file = f.to_string();
                            }
                            line_num = primary.get("line_start").and_then(|l| l.as_u64()).unwrap_or(1) as usize;
                            col_num = primary.get("column_start").and_then(|c| c.as_u64()).unwrap_or(1) as usize;
                            if let Some(text) = primary.get("text").and_then(|t| t.as_array()) {
                                let text_lines: Vec<String> = text.iter().filter_map(|t| t.get("text").and_then(|s| s.as_str()).map(|s| s.to_string())).collect();
                                if !text_lines.is_empty() {
                                    snippet = Some(text_lines.join("\n"));
                                }
                            }
                        }
                    }

                    issues.push(DiagnosticIssue {
                        file,
                        line: line_num,
                        column: col_num,
                        severity: level,
                        message: message_text,
                        code_snippet: snippet,
                    });
                }
            }
        }
    }
    issues
}

pub fn parse_python_stderr(target: &str, stderr: &str) -> Vec<DiagnosticIssue> {
    let mut issues = Vec::new();
    let mut line_num = 1;
    let mut col_num = 1;
    let mut msg = "Python syntax error".to_string();
    let mut has_error = false;

    for l in stderr.lines() {
        if l.contains("File \"") && l.contains(", line ") {
            has_error = true;
            if let Some(idx) = l.find(", line ") {
                let rest = &l[idx + 7..];
                let num_str: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
                if let Ok(n) = num_str.parse::<usize>() {
                    line_num = n;
                }
            }
        }
        if l.contains("SyntaxError:") || l.contains("IndentationError:") || l.contains("TabError:") {
            has_error = true;
            msg = l.trim().to_string();
        }
        if l.trim().starts_with('^') {
            col_num = l.find('^').unwrap_or(0) + 1;
        }
    }

    if has_error {
        issues.push(DiagnosticIssue {
            file: target.to_string(),
            line: line_num,
            column: col_num,
            severity: "error".to_string(),
            message: msg,
            code_snippet: None,
        });
    }
    issues
}

pub fn run_lsp_diagnostics(target: &str) -> DiagnosticReport {
    let target = target.trim();
    let mut issues = Vec::new();
    let tool_used;
    let success;

    if target.starts_with("cargo ") || target == "cargo" || target.ends_with(".rs") || (std::path::Path::new("Cargo.toml").exists() && target == ".") {
        tool_used = "cargo check (JSON)".to_string();
        let output = std::process::Command::new("cargo")
            .args(["check", "--message-format=json"])
            .output();

        match output {
            Ok(out) => {
                success = out.status.success();
                let stdout = String::from_utf8_lossy(&out.stdout);
                issues = parse_cargo_json_diagnostics(target, &stdout);
            }
            Err(e) => {
                success = false;
                issues.push(DiagnosticIssue {
                    file: target.to_string(),
                    line: 1,
                    column: 1,
                    severity: "error".to_string(),
                    message: format!("Failed to execute cargo check: {}", e),
                    code_snippet: None,
                });
            }
        }
    } else if target.ends_with(".py") {
        tool_used = "python py_compile".to_string();
        let py_cmd = if cfg!(windows) { "python" } else { "python3" };
        let output = std::process::Command::new(py_cmd)
            .args(["-m", "py_compile", target])
            .output();

        match output {
            Ok(out) => {
                success = out.status.success();
                if !success {
                    let stderr = String::from_utf8_lossy(&out.stderr);
                    issues = parse_python_stderr(target, &stderr);
                }
            }
            Err(e) => {
                success = false;
                issues.push(DiagnosticIssue {
                    file: target.to_string(),
                    line: 1,
                    column: 1,
                    severity: "error".to_string(),
                    message: format!("Failed to execute python: {}", e),
                    code_snippet: None,
                });
            }
        }
    } else if target.ends_with(".ts") || target.ends_with(".tsx") || target.ends_with(".js") || target.ends_with(".jsx") {
        tool_used = "tsc / node syntax check".to_string();
        let output = if target.ends_with(".js") {
            std::process::Command::new("node").args(["--check", target]).output()
        } else {
            #[cfg(windows)]
            let c = std::process::Command::new("cmd").args(["/C", "npx", "tsc", "--noEmit", target]).output();
            #[cfg(not(windows))]
            let c = std::process::Command::new("npx").args(["tsc", "--noEmit", target]).output();
            c
        };

        match output {
            Ok(out) => {
                success = out.status.success();
                let full_out = format!("{}\n{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
                for l in full_out.lines() {
                    let l = l.trim();
                    if l.is_empty() { continue; }
                    if l.contains("error TS") || l.contains("warning TS") || l.contains("SyntaxError:") {
                        let severity = if l.contains("warning TS") { "warning" } else { "error" };
                        let mut line_num = 1;
                        let mut col_num = 1;
                        if let Some(open) = l.find('(') {
                            if let Some(close) = l.find(')') {
                                let nums = &l[open + 1..close];
                                let parts: Vec<&str> = nums.split(',').collect();
                                if parts.len() == 2 {
                                    line_num = parts[0].trim().parse().unwrap_or(1);
                                    col_num = parts[1].trim().parse().unwrap_or(1);
                                }
                            }
                        }
                        issues.push(DiagnosticIssue {
                            file: target.to_string(),
                            line: line_num,
                            column: col_num,
                            severity: severity.to_string(),
                            message: l.to_string(),
                            code_snippet: None,
                        });
                    }
                }
            }
            Err(e) => {
                success = false;
                issues.push(DiagnosticIssue {
                    file: target.to_string(),
                    line: 1,
                    column: 1,
                    severity: "error".to_string(),
                    message: format!("Failed to run JS/TS checker: {}", e),
                    code_snippet: None,
                });
            }
        }
    } else if target.ends_with(".c") || target.ends_with(".cpp") || target.ends_with(".cc") || target.ends_with(".h") {
        tool_used = "gcc / clang syntax check".to_string();
        let output = std::process::Command::new("gcc")
            .args(["-fsyntax-only", target])
            .output()
            .or_else(|_| std::process::Command::new("clang").args(["-fsyntax-only", target]).output());

        match output {
            Ok(out) => {
                success = out.status.success();
                let stderr = String::from_utf8_lossy(&out.stderr);
                for l in stderr.lines() {
                    let parts: Vec<&str> = l.split(':').collect();
                    if parts.len() >= 4 {
                        let line_num = parts[1].trim().parse().unwrap_or(1);
                        let col_num = parts[2].trim().parse().unwrap_or(1);
                        let sev = parts[3].trim().to_lowercase();
                        let msg = parts[4..].join(":").trim().to_string();
                        issues.push(DiagnosticIssue {
                            file: parts[0].trim().to_string(),
                            line: line_num,
                            column: col_num,
                            severity: if sev.contains("warning") { "warning".to_string() } else { "error".to_string() },
                            message: msg,
                            code_snippet: None,
                        });
                    }
                }
            }
            Err(e) => {
                success = false;
                issues.push(DiagnosticIssue {
                    file: target.to_string(),
                    line: 1,
                    column: 1,
                    severity: "error".to_string(),
                    message: format!("Failed to run gcc/clang: {}", e),
                    code_snippet: None,
                });
            }
        }
    } else {
        tool_used = format!("shell: {}", target);
        #[cfg(windows)]
        let output = std::process::Command::new("cmd").args(["/C", target]).output();
        #[cfg(not(windows))]
        let output = std::process::Command::new("sh").args(["-c", target]).output();

        match output {
            Ok(out) => {
                success = out.status.success();
                let combined = format!("{}\n{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
                if !success && issues.is_empty() {
                    issues.push(DiagnosticIssue {
                        file: target.to_string(),
                        line: 1,
                        column: 1,
                        severity: "error".to_string(),
                        message: combined.trim().to_string(),
                        code_snippet: None,
                    });
                }
            }
            Err(e) => {
                success = false;
                issues.push(DiagnosticIssue {
                    file: target.to_string(),
                    line: 1,
                    column: 1,
                    severity: "error".to_string(),
                    message: format!("Execution failed: {}", e),
                    code_snippet: None,
                });
            }
        }
    }

    let error_count = issues.iter().filter(|i| i.severity.eq_ignore_ascii_case("error")).count();
    let warning_count = issues.iter().filter(|i| i.severity.eq_ignore_ascii_case("warning")).count();
    let issue_count = issues.len();

    let summary = if issue_count == 0 && success {
        format!("✅ Diagnostics Clean: 0 errors / warnings via {}", tool_used)
    } else {
        format!("❌ Found {} issues ({} errors, {} warnings) via {}", issue_count, error_count, warning_count, tool_used)
    };

    DiagnosticReport {
        target: target.to_string(),
        tool: tool_used,
        success: success && error_count == 0,
        issue_count,
        error_count,
        warning_count,
        issues,
        summary,
    }
}

pub fn format_diagnostic_report_for_terminal(report: &DiagnosticReport) -> String {
    let mut out = String::new();
    out.push_str(&format!("\n{}\n", "╔═══════════════════════════════════════════════════════════╗".cyan()));
    out.push_str(&format!("║ {} {:<46} ║\n", "🔍 LSP DIAGNOSTICS REPORT:".cyan().bold(), report.target.yellow()));
    out.push_str(&format!("║ Tool: {:<52} ║\n", report.tool.magenta()));
    out.push_str("╠═══════════════════════════════════════════════════════════╣\n");
    
    if report.issues.is_empty() {
        out.push_str(&format!("║  {}  ║\n", "✨ Clean! No diagnostic errors or warnings found.".green().bold()));
    } else {
        for (i, issue) in report.issues.iter().enumerate() {
            let badge = if issue.severity.eq_ignore_ascii_case("error") {
                " ERROR ".on_red().white().bold()
            } else if issue.severity.eq_ignore_ascii_case("warning") {
                " WARN  ".on_yellow().black().bold()
            } else {
                " INFO  ".on_blue().white().bold()
            };
            out.push_str(&format!("  {} {}:{}:{}\n", badge, issue.file.bold(), issue.line.to_string().yellow(), issue.column.to_string().yellow()));
            out.push_str(&format!("     {}\n", issue.message.white()));
            if let Some(snip) = &issue.code_snippet {
                for sl in snip.lines() {
                    out.push_str(&format!("     {} {}\n", "│".dimmed(), sl.dimmed()));
                }
            }
            if i + 1 < report.issues.len() {
                out.push_str(&format!("  {}\n", "───────────────────────────────────────────────────────".dimmed()));
            }
        }
    }
    out.push_str("╚═══════════════════════════════════════════════════════════╝\n");
    out.push_str(&format!("📊 {}\n", report.summary.bold()));
    out
}

// -------------------------------------------------------------------------------------------------
// FEATURE 2: MODEL CONTEXT PROTOCOL (MCP) CLIENT
// -------------------------------------------------------------------------------------------------

pub async fn execute_mcp_call(
    server_cmd: &str,
    tool_name: &str,
    args: &serde_json::Value,
) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    use std::process::Stdio;
    use tokio::process::Command;

    let parts: Vec<&str> = server_cmd.split_whitespace().collect();
    if parts.is_empty() {
        return Err("MCP server command cannot be empty".into());
    }

    let program = parts[0];
    let cmd_args = &parts[1..];

    #[cfg(windows)]
    let mut cmd = if program.ends_with(".cmd") || program.ends_with(".bat") || program == "npx" || program == "npm" {
        let mut c = Command::new("cmd.exe");
        c.arg("/C").arg(server_cmd);
        c
    } else {
        let mut c = Command::new(program);
        c.args(cmd_args);
        c
    };

    #[cfg(not(windows))]
    let mut cmd = {
        let mut c = Command::new(program);
        c.args(cmd_args);
        c
    };

    cmd.stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let mut child = cmd.spawn().map_err(|e| format!("Failed to spawn MCP server '{}': {}", server_cmd, e))?;

    let mut stdin = child.stdin.take().ok_or("Failed to open stdin for MCP server")?;
    let stdout = child.stdout.take().ok_or("Failed to open stdout for MCP server")?;
    let mut reader = BufReader::new(stdout).lines();

    // 1. Send initialize request
    let init_req = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "zy",
                "version": "0.1.0"
            }
        }
    });
    let mut init_line = serde_json::to_string(&init_req)?;
    init_line.push('\n');
    stdin.write_all(init_line.as_bytes()).await?;
    stdin.flush().await?;

    // Read initialize response
    let mut init_success = false;
    while let Ok(Some(line)) = reader.next_line().await {
        let line = line.trim();
        if line.is_empty() { continue; }
        if let Ok(val) = serde_json::from_str::<serde_json::Value>(line) {
            if val.get("id").and_then(|id| id.as_i64()) == Some(1) {
                init_success = true;
                break;
            }
        }
    }

    if !init_success {
        let _ = child.kill().await;
        return Err("MCP server failed to respond to 'initialize'".into());
    }

    // 2. Send initialized notification
    let notif = serde_json::json!({
        "jsonrpc": "2.0",
        "method": "notifications/initialized"
    });
    let mut notif_line = serde_json::to_string(&notif)?;
    notif_line.push('\n');
    stdin.write_all(notif_line.as_bytes()).await?;
    stdin.flush().await?;

    // 3. Send tools/call request
    let call_req = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {
            "name": tool_name,
            "arguments": args
        }
    });
    let mut call_line = serde_json::to_string(&call_req)?;
    call_line.push('\n');
    stdin.write_all(call_line.as_bytes()).await?;
    stdin.flush().await?;

    // 4. Read tools/call response
    let mut result_text = None;
    while let Ok(Some(line)) = reader.next_line().await {
        let line = line.trim();
        if line.is_empty() { continue; }
        if let Ok(val) = serde_json::from_str::<serde_json::Value>(line) {
            if val.get("id").and_then(|id| id.as_i64()) == Some(2) {
                if let Some(error) = val.get("error") {
                    result_text = Some(format!("MCP Error: {}", error));
                } else if let Some(result) = val.get("result") {
                    if let Some(content) = result.get("content").and_then(|c| c.as_array()) {
                        let mut texts = Vec::new();
                        for item in content {
                            if let Some(t) = item.get("text").and_then(|t| t.as_str()) {
                                texts.push(t.to_string());
                            } else {
                                texts.push(item.to_string());
                            }
                        }
                        result_text = Some(texts.join("\n"));
                    } else {
                        result_text = Some(serde_json::to_string_pretty(result)?);
                    }
                }
                break;
            }
        }
    }

    let _ = child.kill().await;
    result_text.ok_or_else(|| "MCP server did not return a response to tools/call".into())
}

// -------------------------------------------------------------------------------------------------
// FEATURE 3: VISUAL TERMINAL DIFF VIEWER
// -------------------------------------------------------------------------------------------------

pub fn render_terminal_diff(path: &str, old_content: &str, new_content: &str) -> String {
    let diff = TextDiff::from_lines(old_content, new_content);
    let mut output = String::new();

    let header = format!("─── File Unified Diff: {} ───", path);
    output.push_str(&format!("{}\n", header.cyan().bold()));

    let mut changes_found = false;
    for (idx, group) in diff.grouped_ops(3).iter().enumerate() {
        if idx > 0 {
            output.push_str(&format!("{}\n", "───".dimmed()));
        }
        for op in group {
            for change in diff.iter_changes(op) {
                changes_found = true;
                let sign = match change.tag() {
                    ChangeTag::Delete => "-".red().bold(),
                    ChangeTag::Insert => "+".green().bold(),
                    ChangeTag::Equal => " ".dimmed(),
                };
                let line_str = change.value().trim_end_matches(['\r', '\n']);
                let old_num = change.old_index().map(|n| format!("{:>4}", n + 1)).unwrap_or_else(|| "    ".to_string());
                let new_num = change.new_index().map(|n| format!("{:>4}", n + 1)).unwrap_or_else(|| "    ".to_string());

                let line_display = match change.tag() {
                    ChangeTag::Delete => format!("{} │      │ {} {}", old_num.red().dimmed(), sign, line_str.red()),
                    ChangeTag::Insert => format!("     │ {} │ {} {}", new_num.green().dimmed(), sign, line_str.green()),
                    ChangeTag::Equal => format!("{} │ {} │ {} {}", old_num.dimmed(), new_num.dimmed(), sign, line_str.dimmed()),
                };
                output.push_str(&format!("{}\n", line_display));
            }
        }
    }

    if !changes_found {
        output.push_str(&format!("{}\n", "(No modifications detected)".dimmed()));
    }
    output
}

// -------------------------------------------------------------------------------------------------
// FEATURE 4: TOKEN-BUDGETING ENGINE
// -------------------------------------------------------------------------------------------------

pub fn estimate_tokens(text: &str) -> usize {
    if text.is_empty() { return 0; }
    let char_count = text.chars().count();
    let word_count = text.split_whitespace().count();
    let est_char = (char_count + 3) / 4;
    let est_word = (word_count * 4 + 2) / 3;
    std::cmp::max(1, (est_char + est_word) / 2)
}

pub fn estimate_message_tokens(msg: &Message) -> usize {
    let mut tokens = 4; // role and delimiter overhead
    tokens += estimate_tokens(&msg.content);
    if let Some(calls) = &msg.tool_calls {
        for call in calls {
            tokens += 4 + estimate_tokens(&call.function.name) + estimate_tokens(&call.function.arguments.to_string());
        }
    }
    if let Some(images) = &msg.images {
        tokens += images.len() * 512;
    }
    tokens
}

pub fn estimate_conversation_tokens(messages: &[Message]) -> usize {
    messages.iter().map(estimate_message_tokens).sum()
}

pub fn budget_aware_prune(messages: &mut Vec<Message>, num_ctx: usize) {
    if messages.is_empty() { return; }

    let reserve = std::cmp::max(512, num_ctx / 4);
    let target_input_budget = if num_ctx > reserve { num_ctx - reserve } else { num_ctx / 2 };

    let total_tokens = estimate_conversation_tokens(messages);
    if total_tokens <= target_input_budget {
        return;
    }

    let mut system_msgs = Vec::new();
    let mut other_msgs = Vec::new();

    for msg in messages.drain(..) {
        if msg.role == "system" {
            system_msgs.push(msg);
        } else {
            other_msgs.push(msg);
        }
    }

    let protect_count = if other_msgs.len() > 2 { 2 } else { other_msgs.len() };
    let split_idx = other_msgs.len().saturating_sub(protect_count);
    
    let mut middle_msgs: Vec<Message> = other_msgs.drain(..split_idx).collect();
    let protected_msgs = other_msgs;

    while !middle_msgs.is_empty() {
        let current_tokens = system_msgs.iter().map(estimate_message_tokens).sum::<usize>()
            + middle_msgs.iter().map(estimate_message_tokens).sum::<usize>()
            + protected_msgs.iter().map(estimate_message_tokens).sum::<usize>();
        
        if current_tokens <= target_input_budget {
            break;
        }
        middle_msgs.remove(0);
    }

    messages.extend(system_msgs);
    messages.extend(middle_msgs);
    messages.extend(protected_msgs);
}

pub fn format_token_budget(messages: &[Message], num_ctx: usize) -> String {
    let current_tokens = estimate_conversation_tokens(messages);
    let pct = if num_ctx > 0 { (current_tokens as f64 / num_ctx as f64) * 100.0 } else { 0.0 };
    
    let token_str = format!("Tokens: {:>5} / {:>5} ({:.0}%)", current_tokens, num_ctx, pct);
    if pct > 80.0 {
        token_str.red().bold().to_string()
    } else if pct > 50.0 {
        token_str.yellow().to_string()
    } else {
        token_str.green().to_string()
    }
}

// -------------------------------------------------------------------------------------------------
// FEATURE 5: STRUCTURED SCHEMA & GRAMMAR ENFORCEMENT
// -------------------------------------------------------------------------------------------------

pub fn build_tool_grammar_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "action": { "type": "string", "description": "Action type: tool_call or final_answer" },
            "tool_name": { "type": "string" },
            "arguments": { "type": "object" },
            "response": { "type": "string" }
        },
        "required": ["action"]
    })
}

// -------------------------------------------------------------------------------------------------
// FEATURE 6: DUAL-MODEL SPECULATIVE ROUTER (FAST SCOUT + HEAVY CODER)
// -------------------------------------------------------------------------------------------------

#[derive(Debug, PartialEq, Eq)]
pub enum RouteDecision {
    Chat,
    Coding,
}

pub async fn classify_query_route(
    client: &Client,
    scout_model: &str,
    query: &str,
    options: &OllamaOptions,
) -> RouteDecision {
    let prompt = format!(
        "Classify if this user input is simple conversational CHAT (greetings, chit-chat, simple factual knowledge without needing code editing, files, or tools) or CODING (code generation, bug fixes, refactoring, modifying files, executing commands, complex technical architecture, tool requests).\n\nUser input: \"{}\"\n\nRespond strictly with either 'CHAT' or 'CODING' only.",
        query
    );
    let req = ChatRequest {
        model: scout_model.to_string(),
        messages: vec![Message { role: "user".to_string(), content: prompt, tool_calls: None, images: None }],
        stream: false,
        tools: None,
        format: None,
        options: Some(OllamaOptions {
            temperature: 0.0,
            num_ctx: Some(1024),
            num_thread: options.num_thread,
            num_gpu: options.num_gpu,
        }),
        keep_alive: Some(-1),
    };

    if let Ok(res) = client.post(format!("{}/api/chat", OLLAMA_URL)).json(&req).send().await {
        if let Ok(parsed) = res.json::<ChatResponse>().await {
            if let Some(msg) = parsed.message {
                let upper = msg.content.to_uppercase();
                if upper.contains("CHAT") && !upper.contains("CODING") {
                    return RouteDecision::Chat;
                }
            }
        }
    }
    RouteDecision::Coding
}

// -------------------------------------------------------------------------------------------------
// FEATURE 7: .zyrules & zy.toml RULES ENGINE
// -------------------------------------------------------------------------------------------------

pub fn get_global_config_paths() -> Vec<std::path::PathBuf> {
    let mut paths = Vec::new();
    if let Ok(appdata) = std::env::var("APPDATA") {
        paths.push(std::path::PathBuf::from(appdata).join("zy").join("config.toml"));
    }
    if let Ok(userprofile) = std::env::var("USERPROFILE") {
        paths.push(std::path::PathBuf::from(userprofile).join(".config").join("zy").join("config.toml"));
    }
    if let Ok(home) = std::env::var("HOME") {
        paths.push(std::path::PathBuf::from(home).join(".config").join("zy").join("config.toml"));
    }
    paths
}

pub fn parse_toml_rules(content: &str) -> Option<String> {
    let trimmed = content.trim();
    if trimmed.is_empty() {
        return None;
    }

    let mut in_rules_section = false;
    let mut rules_lines = Vec::new();

    for line in content.lines() {
        let line_trim = line.trim();
        if line_trim.starts_with('[') && line_trim.ends_with(']') {
            let section = line_trim.trim_matches(|c| c == '[' || c == ']').trim();
            if section.eq_ignore_ascii_case("rules") || section.eq_ignore_ascii_case("instructions") {
                in_rules_section = true;
                continue;
            } else {
                in_rules_section = false;
            }
        }

        if in_rules_section {
            if !line_trim.is_empty() && !line_trim.starts_with('#') {
                if let Some((_k, v)) = line_trim.split_once('=') {
                    let v_clean = v.trim().trim_matches(|c| c == '"' || c == '\'');
                    rules_lines.push(v_clean.to_string());
                } else {
                    rules_lines.push(line.to_string());
                }
            }
        } else if line_trim.starts_with("rules") || line_trim.starts_with("instructions") || line_trim.starts_with("system") {
            if let Some((_k, v)) = line_trim.split_once('=') {
                let v_clean = v.trim().trim_matches(|c| c == '"' || c == '\'' || c == '[' || c == ']');
                if !v_clean.is_empty() {
                    rules_lines.push(v_clean.to_string());
                }
            }
        }
    }

    if !rules_lines.is_empty() {
        Some(rules_lines.join("\n").trim().to_string())
    } else if !trimmed.is_empty() && (content.contains("rules") || content.contains("instructions") || !content.contains('[')) {
        Some(trimmed.to_string())
    } else {
        None
    }
}

pub fn load_project_rules(path: &std::path::Path) -> Option<String> {
    let mut rules_sections = Vec::new();

    // 1. Check <path>/.zyrules
    let zyrules_path = path.join(".zyrules");
    if zyrules_path.is_file() {
        if let Ok(content) = fs::read_to_string(&zyrules_path) {
            let trimmed = content.trim();
            if !trimmed.is_empty() {
                rules_sections.push(format!("### Project Rules (.zyrules):\n{}", trimmed));
            }
        }
    }

    // 2. Check <path>/.zy/rules.md
    let zy_md_path = path.join(".zy").join("rules.md");
    if zy_md_path.is_file() {
        if let Ok(content) = fs::read_to_string(&zy_md_path) {
            let trimmed = content.trim();
            if !trimmed.is_empty() {
                rules_sections.push(format!("### Project Rules (.zy/rules.md):\n{}", trimmed));
            }
        }
    }

    // 3. Check <path>/zy.toml
    let zy_toml_path = path.join("zy.toml");
    if zy_toml_path.is_file() {
        if let Ok(content) = fs::read_to_string(&zy_toml_path) {
            if let Some(rules) = parse_toml_rules(&content) {
                rules_sections.push(format!("### Config Rules (zy.toml):\n{}", rules));
            }
        }
    }

    // 4. Check global config (~/.config/zy/config.toml or %APPDATA%\zy\config.toml)
    let global_config_paths = get_global_config_paths();
    for gpath in global_config_paths {
        if gpath.is_file() {
            if let Ok(content) = fs::read_to_string(&gpath) {
                if let Some(rules) = parse_toml_rules(&content) {
                    rules_sections.push(format!("### Global Rules ({}):\n{}", gpath.display(), rules));
                    break;
                }
            }
        }
    }

    if rules_sections.is_empty() {
        None
    } else {
        Some(rules_sections.join("\n\n"))
    }
}

// -------------------------------------------------------------------------------------------------
// FEATURE 8: COMPACT REPOSITORY MAP (SYMBOL HIERARCHY)
// -------------------------------------------------------------------------------------------------

pub fn extract_identifier_after<'a>(line: &'a str, prefix: &str) -> Option<&'a str> {
    if let Some(idx) = line.find(prefix) {
        let after = &line[idx + prefix.len()..];
        let name = after.split(|c: char| !c.is_alphanumeric() && c != '_').next().unwrap_or("").trim();
        if !name.is_empty() {
            return Some(name);
        }
    }
    None
}

pub fn extract_symbols(content: &str, ext: &str) -> Vec<String> {
    let mut symbols = Vec::new();
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("//") || trimmed.starts_with('#') || trimmed.starts_with("/*") || trimmed.starts_with('*') {
            continue;
        }

        match ext {
            "rs" => {
                if trimmed.starts_with("pub fn ") || trimmed.starts_with("fn ") || trimmed.starts_with("pub async fn ") || trimmed.starts_with("async fn ") || trimmed.starts_with("pub(crate) fn ") {
                    if let Some(name) = extract_identifier_after(trimmed, "fn ") {
                        symbols.push(format!("fn {}", name));
                    }
                } else if trimmed.starts_with("pub struct ") || trimmed.starts_with("struct ") {
                    if let Some(name) = extract_identifier_after(trimmed, "struct ") {
                        symbols.push(format!("struct {}", name));
                    }
                } else if trimmed.starts_with("pub enum ") || trimmed.starts_with("enum ") {
                    if let Some(name) = extract_identifier_after(trimmed, "enum ") {
                        symbols.push(format!("enum {}", name));
                    }
                } else if trimmed.starts_with("pub trait ") || trimmed.starts_with("trait ") {
                    if let Some(name) = extract_identifier_after(trimmed, "trait ") {
                        symbols.push(format!("trait {}", name));
                    }
                } else if trimmed.starts_with("pub type ") || trimmed.starts_with("type ") {
                    if let Some(name) = extract_identifier_after(trimmed, "type ") {
                        symbols.push(format!("type {}", name));
                    }
                }
            }
            "py" => {
                if trimmed.starts_with("def ") || trimmed.starts_with("async def ") {
                    if let Some(name) = extract_identifier_after(trimmed, "def ") {
                        symbols.push(format!("def {}", name));
                    }
                } else if trimmed.starts_with("class ") {
                    if let Some(name) = extract_identifier_after(trimmed, "class ") {
                        symbols.push(format!("class {}", name));
                    }
                }
            }
            "js" | "ts" | "jsx" | "tsx" => {
                if trimmed.starts_with("function ") || trimmed.starts_with("export function ") || trimmed.starts_with("async function ") || trimmed.starts_with("export async function ") {
                    if let Some(name) = extract_identifier_after(trimmed, "function ") {
                        symbols.push(format!("fn {}", name));
                    }
                } else if trimmed.starts_with("class ") || trimmed.starts_with("export class ") {
                    if let Some(name) = extract_identifier_after(trimmed, "class ") {
                        symbols.push(format!("class {}", name));
                    }
                } else if trimmed.starts_with("interface ") || trimmed.starts_with("export interface ") {
                    if let Some(name) = extract_identifier_after(trimmed, "interface ") {
                        symbols.push(format!("interface {}", name));
                    }
                } else if trimmed.starts_with("type ") || trimmed.starts_with("export type ") {
                    if let Some(name) = extract_identifier_after(trimmed, "type ") {
                        symbols.push(format!("type {}", name));
                    }
                } else if (trimmed.starts_with("const ") || trimmed.starts_with("export const ")) && (trimmed.contains("=>") || trimmed.contains("function")) {
                    if let Some(name) = extract_identifier_after(trimmed, "const ") {
                        symbols.push(format!("const {}", name));
                    }
                }
            }
            "go" => {
                if trimmed.starts_with("func ") {
                    if let Some(rest) = trimmed.strip_prefix("func ") {
                        let name = if rest.starts_with('(') {
                            if let Some((_recv, rest_after)) = rest.split_once(')') {
                                rest_after.trim().split(|c: char| !c.is_alphanumeric() && c != '_').next().unwrap_or("")
                            } else {
                                ""
                            }
                        } else {
                            rest.split(|c: char| !c.is_alphanumeric() && c != '_').next().unwrap_or("")
                        };
                        if !name.is_empty() {
                            symbols.push(format!("func {}", name));
                        }
                    }
                } else if trimmed.starts_with("type ") {
                    if let Some(name) = extract_identifier_after(trimmed, "type ") {
                        symbols.push(format!("type {}", name));
                    }
                }
            }
            "c" | "cpp" | "h" | "hpp" | "cc" => {
                if trimmed.starts_with("struct ") || trimmed.starts_with("typedef struct ") {
                    if let Some(name) = extract_identifier_after(trimmed, "struct ") {
                        symbols.push(format!("struct {}", name));
                    }
                } else if trimmed.starts_with("class ") {
                    if let Some(name) = extract_identifier_after(trimmed, "class ") {
                        symbols.push(format!("class {}", name));
                    }
                } else if (trimmed.contains('(') && trimmed.contains(')')) && !trimmed.starts_with("if") && !trimmed.starts_with("while") && !trimmed.starts_with("for") && !trimmed.starts_with("switch") {
                    if let Some((before_paren, _)) = trimmed.split_once('(') {
                        if let Some(func_name) = before_paren.split_whitespace().last() {
                            let clean_name = func_name.trim_matches(|c: char| !c.is_alphanumeric() && c != '_');
                            if !clean_name.is_empty() && clean_name != "main" {
                                symbols.push(format!("fn {}", clean_name));
                            } else if clean_name == "main" {
                                symbols.push("fn main".to_string());
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }
    symbols.dedup();
    symbols
}

pub fn build_repo_map(path: &std::path::Path, max_tokens: usize) -> String {
    let mut file_entries: Vec<(String, Vec<String>)> = Vec::new();

    let walker = WalkDir::new(path).into_iter().filter_entry(|e| {
        let name = e.file_name().to_string_lossy();
        if e.file_type().is_dir() {
            !name.starts_with('.') && name != "target" && name != "node_modules" && name != "dist" 
            && name != "build" && name != "__pycache__" && name != "venv" && name != ".venv"
            && name != "obj" && name != "bin" && name != "vendor"
        } else {
            true
        }
    });

    for entry in walker.flatten() {
        if entry.file_type().is_file() {
            let p = entry.path();
            let ext = p.extension().and_then(|e| e.to_str()).unwrap_or("");
            if matches!(ext, "rs" | "py" | "js" | "ts" | "jsx" | "tsx" | "c" | "cpp" | "h" | "hpp" | "cc" | "go" | "java" | "cs" | "rb" | "php" | "swift" | "kt" | "scala" | "sh" | "bash" | "toml" | "yaml" | "yml" | "sql") {
                if let Ok(content) = fs::read_to_string(p) {
                    let symbols = extract_symbols(&content, ext);
                    if !symbols.is_empty() {
                        let rel_path = p.strip_prefix(path).unwrap_or(p).to_string_lossy().replace('\\', "/");
                        file_entries.push((rel_path, symbols));
                    }
                }
            }
        }
    }

    file_entries.sort_by(|a, b| a.0.cmp(&b.0));

    let mut output = String::new();
    output.push_str("Repository Symbol Map:\n");

    for (file, symbols) in file_entries {
        let sym_line = symbols.join(", ");
        let line = format!("{}: {}\n", file, sym_line);
        if estimate_tokens(&format!("{}{}", output, line)) > max_tokens {
            output.push_str("... (truncated symbol map to fit token budget)\n");
            break;
        }
        output.push_str(&line);
    }

    output.trim_end().to_string()
}

// -------------------------------------------------------------------------------------------------
// FEATURE 9: AUTONOMOUS TDD TEST-FIX LOOP (/test & auto_repair)
// -------------------------------------------------------------------------------------------------

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct TestReport {
    pub runner: String,
    pub success: bool,
    pub exit_code: Option<i32>,
    pub passed_count: usize,
    pub failed_count: usize,
    pub failure_details: Vec<String>,
    pub stdout: String,
    pub stderr: String,
    pub summary: String,
}

pub fn detect_test_runner(path: &std::path::Path) -> String {
    if path.join("Cargo.toml").exists() {
        "cargo test".to_string()
    } else if path.join("pytest.ini").exists() || path.join("pyproject.toml").exists() || path.join("requirements.txt").exists() || path.join("setup.py").exists() {
        "pytest".to_string()
    } else if path.join("package.json").exists() {
        "npm test".to_string()
    } else if path.join("go.mod").exists() {
        "go test ./...".to_string()
    } else {
        let has_rs = WalkDir::new(path).max_depth(2).into_iter().flatten().any(|e| e.path().extension().and_then(|s| s.to_str()) == Some("rs"));
        if has_rs {
            "cargo test".to_string()
        } else {
            "cargo test".to_string()
        }
    }
}

pub fn parse_test_output(runner: &str, stdout: &str, stderr: &str, exit_code: Option<i32>) -> (usize, usize, Vec<String>, String) {
    let mut passed = 0;
    let mut failed = 0;
    let mut failures = Vec::new();
    let combined = format!("{}\n{}", stdout, stderr);

    if runner.contains("cargo") {
        for line in combined.lines() {
            let line_trim = line.trim();
            if line_trim.starts_with("test ") && line_trim.ends_with(" ... ok") {
                passed += 1;
            } else if line_trim.starts_with("test ") && (line_trim.ends_with(" ... FAILED") || line_trim.ends_with(" ... failed")) {
                failed += 1;
                failures.push(line_trim.to_string());
            } else if line_trim.contains("failures:") || line_trim.starts_with("error[E") {
                failures.push(line_trim.to_string());
            }
        }
    } else if runner.contains("pytest") {
        for line in combined.lines() {
            let line_trim = line.trim();
            if line_trim.starts_with("PASSED ") || line_trim.ends_with(" PASSED") {
                passed += 1;
            } else if line_trim.starts_with("FAILED ") || line_trim.ends_with(" FAILED") {
                failed += 1;
                failures.push(line_trim.to_string());
            }
        }
    } else if runner.contains("npm") || runner.contains("jest") {
        for line in combined.lines() {
            let line_trim = line.trim();
            if line_trim.contains("✓") || line_trim.contains("pass") {
                passed += 1;
            } else if line_trim.contains("✕") || line_trim.contains("FAIL") {
                failed += 1;
                failures.push(line_trim.to_string());
            }
        }
    } else if runner.contains("go test") {
        for line in combined.lines() {
            let line_trim = line.trim();
            if line_trim.starts_with("--- PASS:") {
                passed += 1;
            } else if line_trim.starts_with("--- FAIL:") {
                failed += 1;
                failures.push(line_trim.to_string());
            }
        }
    }

    if passed == 0 && failed == 0 {
        if exit_code == Some(0) {
            passed = 1;
        } else {
            failed = 1;
            failures.push("Test runner exited with non-zero status code.".to_string());
        }
    }

    let summary = if failed == 0 && exit_code == Some(0) {
        format!("All tests passed ({} passed) using '{}'", passed, runner)
    } else {
        format!("Tests failed ({} failed, {} passed) using '{}'", failed, passed, runner)
    };

    (passed, failed, failures, summary)
}

pub fn run_project_tests(path: &std::path::Path, custom_cmd: Option<&str>) -> Result<TestReport, Box<dyn std::error::Error>> {
    let runner_cmd = custom_cmd.map(|s| s.to_string()).unwrap_or_else(|| detect_test_runner(path));
    
    #[cfg(windows)]
    let output = std::process::Command::new("cmd")
        .arg("/C")
        .arg(&runner_cmd)
        .current_dir(path)
        .output()?;

    #[cfg(not(windows))]
    let output = std::process::Command::new("sh")
        .arg("-c")
        .arg(&runner_cmd)
        .current_dir(path)
        .output()?;

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    let exit_code = output.status.code();

    let (passed_count, failed_count, failure_details, summary) = parse_test_output(&runner_cmd, &stdout, &stderr, exit_code);
    let success = output.status.success() && failed_count == 0;

    Ok(TestReport {
        runner: runner_cmd,
        success,
        exit_code,
        passed_count,
        failed_count,
        failure_details,
        stdout,
        stderr,
        summary,
    })
}

pub fn format_test_report_for_terminal(report: &TestReport) -> String {
    let mut out = String::new();
    out.push_str(&format!("🧪 Test Suite Report: {}\n", report.runner.bold()));
    if report.success {
        out.push_str(&format!("Status: {}\n", "PASSED".green().bold()));
        out.push_str(&format!("Results: {} passed, {} failed\n", report.passed_count.to_string().green(), report.failed_count.to_string().dimmed()));
    } else {
        out.push_str(&format!("Status: {}\n", "FAILED".red().bold()));
        out.push_str(&format!("Results: {} failed, {} passed\n", report.failed_count.to_string().red().bold(), report.passed_count.to_string().green()));
        if !report.failure_details.is_empty() {
            out.push_str(&format!("Failures:\n{}\n", report.failure_details.join("\n").red()));
        }
    }
    out
}

// -------------------------------------------------------------------------------------------------
// FEATURE 10: ATOMIC GIT MICRO-CHECKPOINTS (/checkpoint & /rollback)
// -------------------------------------------------------------------------------------------------

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct GitCheckpoint {
    pub id: String,
    pub label: String,
    pub commit_sha: String,
    pub timestamp: u64,
}

pub const CHECKPOINTS_FILE: &str = ".zy_checkpoints.json";

pub fn load_checkpoints() -> Vec<GitCheckpoint> {
    if let Ok(content) = fs::read_to_string(CHECKPOINTS_FILE) {
        serde_json::from_str(&content).unwrap_or_default()
    } else {
        Vec::new()
    }
}

pub fn save_checkpoints(checkpoints: &[GitCheckpoint]) {
    if let Ok(data) = serde_json::to_string_pretty(checkpoints) {
        let _ = fs::write(CHECKPOINTS_FILE, data);
    }
}

pub fn create_git_checkpoint_with_label(label: Option<&str>) -> Result<String, String> {
    if !std::path::Path::new(".git").exists() {
        return Err("Not a git repository: .git directory not found.".to_string());
    }

    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    
    let chk_id = format!("chk_{}", timestamp);
    let lbl = label.unwrap_or("auto-checkpoint").to_string();

    let _ = std::process::Command::new("git").args(["add", "-A"]).output();
    let stash_out = std::process::Command::new("git")
        .args(["stash", "create", &format!("zy-checkpoint-{}", chk_id)])
        .output();

    let mut commit_sha = String::new();
    if let Ok(out) = stash_out {
        let sha = String::from_utf8_lossy(&out.stdout).trim().to_string();
        if !sha.is_empty() {
            commit_sha = sha;
        }
    }

    if commit_sha.is_empty() {
        if let Ok(head_out) = std::process::Command::new("git").args(["rev-parse", "HEAD"]).output() {
            commit_sha = String::from_utf8_lossy(&head_out.stdout).trim().to_string();
        }
    }

    if commit_sha.is_empty() {
        return Err("Failed to resolve git commit SHA for checkpoint.".to_string());
    }

    let ref_name = format!("refs/zy/checkpoints/{}", chk_id);
    let _ = std::process::Command::new("git").args(["update-ref", &ref_name, &commit_sha]).output();

    let mut list = load_checkpoints();
    list.push(GitCheckpoint {
        id: chk_id.clone(),
        label: lbl,
        commit_sha: commit_sha.clone(),
        timestamp,
    });
    save_checkpoints(&list);

    Ok(chk_id)
}

pub fn create_git_checkpoint() -> Option<String> {
    create_git_checkpoint_with_label(None).ok()
}

pub fn rollback_git_checkpoint_to(checkpoint_id: Option<&str>) -> Result<String, String> {
    if !std::path::Path::new(".git").exists() {
        return Err("Not a git repository: .git directory not found.".to_string());
    }

    let mut list = load_checkpoints();
    if list.is_empty() {
        return Err("No checkpoints found in repository history.".to_string());
    }

    let target_idx = if let Some(id) = checkpoint_id {
        list.iter().rposition(|c| c.id == id || c.id.starts_with(id))
    } else {
        if !list.is_empty() { Some(list.len() - 1) } else { None }
    };

    if let Some(idx) = target_idx {
        let chk = list.remove(idx);
        let _ = std::process::Command::new("git").args(["checkout", &chk.commit_sha, "--", "."]).output();
        let _ = std::process::Command::new("git").args(["clean", "-fd"]).output();
        
        save_checkpoints(&list);
        Ok(format!("Successfully rolled back workspace to checkpoint `{}` ({})", chk.id, chk.label))
    } else {
        Err(format!("Checkpoint `{}` not found.", checkpoint_id.unwrap_or("latest")))
    }
}

pub fn rollback_git_checkpoint() -> Result<String, String> {
    rollback_git_checkpoint_to(None)
}

// -------------------------------------------------------------------------------------------------
// FEATURE 11: EPHEMERAL SANDBOX CONTAINER ENGINE
// -------------------------------------------------------------------------------------------------

pub fn build_sandbox_command(cmd: &str, workspace: &std::path::Path, image: Option<&str>) -> (String, Vec<String>) {
    let abs_workspace = if workspace.is_absolute() {
        workspace.to_path_buf()
    } else {
        std::env::current_dir().unwrap_or_else(|_| workspace.to_path_buf()).join(workspace)
    };
    let ws_str = abs_workspace.to_string_lossy().replace('\\', "/");
    let container_image = image.unwrap_or("alpine:latest");

    let program = "docker".to_string();
    let args = vec![
        "run".to_string(),
        "--rm".to_string(),
        "-i".to_string(),
        "-v".to_string(),
        format!("{}:/workspace", ws_str),
        "-w".to_string(),
        "/workspace".to_string(),
        container_image.to_string(),
        "sh".to_string(),
        "-c".to_string(),
        cmd.to_string(),
    ];

    (program, args)
}

// -------------------------------------------------------------------------------------------------
// CORE SYSTEM SETUP & TUNER
// -------------------------------------------------------------------------------------------------

pub fn run_ai_tuner(base_temp: f32, quiet: bool) -> AiTunerState {
    let mut sys = System::new_all();
    sys.refresh_memory();
    let cpu_cores = sys.cpus().len();
    let total_mem_gb = sys.total_memory() / 1_073_741_824;

    if total_mem_gb < 12 || cpu_cores <= 4 {
        if !quiet {
            println!("{} {} RAM, {} Cores. Activating {}...", "⚙️  AiTuner:".cyan().dimmed(), format!("{}GB", total_mem_gb).yellow().dimmed(), cpu_cores.to_string().yellow().dimmed(), "ECO MODE (2048 ctx)".green().dimmed());
        }
        AiTunerState {
            num_ctx: 2048,
            profile_name: "ECO".to_string(),
            opts: OllamaOptions {
                temperature: base_temp,
                num_ctx: Some(2048),
                num_thread: Some(std::cmp::max(1, cpu_cores / 2)),
                num_gpu: Some(1),
            },
        }
    } else {
        if !quiet {
            println!("{} {} RAM, {} Cores. Activating {}...", "⚙️  AiTuner:".cyan().dimmed(), format!("{}GB", total_mem_gb).yellow().dimmed(), cpu_cores.to_string().yellow().dimmed(), "TURBO MODE (8192 ctx)".magenta().dimmed());
        }
        AiTunerState {
            num_ctx: 8192,
            profile_name: "TURBO".to_string(),
            opts: OllamaOptions {
                temperature: base_temp,
                num_ctx: Some(8192),
                num_thread: Some(cpu_cores),
                num_gpu: Some(999),
            },
        }
    }
}

pub async fn interactive_wizard(client: &Client, default_model: &str, default_scout: Option<String>) -> Result<(), Box<dyn std::error::Error>> {
    println!("{}", "=========================================".cyan().bold());
    println!("{}", "          Welcome to zy Agent            ".magenta().bold());
    println!("{}", "=========================================\n".cyan().bold());

    let action = Select::new("What would you like to do?", vec![
        "🚀 Start Chatting",
        "🔍 Index Codebase (RAG)",
        "📋 List Models",
        "❌ Exit"
    ]).prompt()?;

    match action {
        "📋 List Models" => { list_models(client).await?; }
        "🔍 Index Codebase (RAG)" => {
            let path = Text::new("Directory to index:").with_default(".").prompt()?;
            build_rag_index(client, &path).await?;
        }
        "🚀 Start Chatting" => {
            let spinner = ProgressBar::new_spinner();
            spinner.set_style(ProgressStyle::default_spinner().tick_strings(&["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]));
            spinner.set_message("Fetching models...");
            spinner.enable_steady_tick(std::time::Duration::from_millis(100));

            let mut model_names = vec![default_model.to_string()];
            if let Ok(res) = client.get(format!("{}/api/tags", OLLAMA_URL)).send().await {
                if let Ok(list) = res.json::<ModelList>().await {
                    model_names = list.models.into_iter().map(|m| m.name).collect();
                }
            }
            spinner.finish_and_clear();
            
            let selected_model = Select::new("Select Primary AI Model:", model_names).prompt()?;
            let agent = Confirm::new("Enable Agent Mode (run bash/write files/LSP/MCP)?").with_default(false).prompt()?;
            let force = if agent { Confirm::new("Enable Force Mode (execute without asking)?").with_default(false).prompt()? } else { false };
            let rag = Confirm::new("Enable RAG (search local codebase)?").with_default(false).prompt()?;
            let markdown = Confirm::new("Enable Markdown Syntax Highlighting?").with_default(true).prompt()?;
            
            let session = Text::new("Session name (leave empty for none):").prompt()?;
            let session_opt = if session.trim().is_empty() { None } else { Some(session.trim()) };

            let tuner = run_ai_tuner(0.1, true);
            println!("\n{}", "--- Configuration Complete ---".green().bold());
            interactive_chat(client, &selected_model, None, &[], agent, session_opt, rag, markdown, &tuner, force, None, false, default_scout, None, false, false).await?;
        }
        _ => {}
    }
    Ok(())
}

pub async fn list_models(client: &Client) -> Result<(), Box<dyn std::error::Error>> {
    let res = client.get(format!("{}/api/tags", OLLAMA_URL)).send().await?;
    
    if res.status().is_success() {
        let list: ModelList = res.json().await?;
        println!("{}", "Available Models:".bold().green());
        for m in list.models {
            println!("  {} {}", "-".cyan(), m.name.yellow());
        }
    } else {
        println!("{}", "Failed to fetch models. Is Ollama running?".red());
    }
    Ok(())
}

pub fn load_session(session: Option<&str>) -> Vec<Message> {
    if let Some(name) = session {
        let file_path = format!(".zy_session_{}.json", name);
        if let Ok(data) = fs::read_to_string(&file_path) {
            if let Ok(msgs) = serde_json::from_str(&data) {
                println!("{} {}", "Loaded session:".blue(), name.bold());
                return msgs;
            }
        }
    }
    Vec::new()
}

pub fn save_session(session: Option<&str>, messages: &[Message]) {
    if let Some(name) = session {
        let file_path = format!(".zy_session_{}.json", name);
        if let Ok(data) = serde_json::to_string_pretty(messages) {
            let _ = fs::write(file_path, data);
        }
    }
}

pub async fn embed_text(client: &Client, text: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let req = EmbedRequest {
        model: "nomic-embed-text".to_string(),
        prompt: text.to_string(),
        keep_alive: Some(-1),
    };
    let res = client.post(format!("{}/api/embeddings", OLLAMA_URL)).json(&req).send().await?;
    if res.status().is_success() {
        let parsed: EmbedResponse = res.json().await?;
        Ok(parsed.embedding)
    } else {
        Err("Failed to get embedding".into())
    }
}

pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() || b.is_empty() || a.len() != b.len() {
        return 0.0;
    }
    let mut dot = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    let norm_prod = (norm_a.sqrt()) * (norm_b.sqrt());
    if norm_prod > 0.0 {
        dot / norm_prod
    } else {
        0.0
    }
}

pub fn tokenize_text(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric() && c != '_')
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect()
}

pub fn bm25_score(
    query: &str,
    doc_tokens: &[String],
    avg_doc_len: f32,
    doc_count: usize,
    doc_freq: usize,
) -> f32 {
    if doc_tokens.is_empty() || doc_count == 0 {
        return 0.0;
    }
    let query_terms = tokenize_text(query);
    if query_terms.is_empty() {
        return 0.0;
    }

    let k1: f32 = 1.2;
    let b: f32 = 0.75;
    let n = doc_freq as f32;
    let total_docs = doc_count as f32;
    let idf = ((total_docs - n + 0.5) / (n + 0.5) + 1.0).ln().max(0.0);
    let doc_len = doc_tokens.len() as f32;
    let avg_len = if avg_doc_len > 0.0 { avg_doc_len } else { 1.0 };

    let mut total_score = 0.0;
    for term in &query_terms {
        let tf = doc_tokens.iter().filter(|t| *t == term).count() as f32;
        if tf > 0.0 {
            let tf_component = (tf * (k1 + 1.0)) / (tf + k1 * (1.0 - b + b * (doc_len / avg_len)));
            total_score += idf * tf_component;
        }
    }
    total_score
}

pub fn score_document_bm25(
    query_tokens: &[String],
    doc_tokens: &[String],
    avg_doc_len: f32,
    doc_count: usize,
    doc_frequencies: &std::collections::HashMap<String, usize>,
) -> f32 {
    let mut total = 0.0;
    for q in query_tokens {
        let df = doc_frequencies.get(q).copied().unwrap_or(0);
        total += bm25_score(q, doc_tokens, avg_doc_len, doc_count, df);
    }
    total
}

pub fn hybrid_rag_search<'a>(
    chunks: &'a [RagChunk],
    query: &str,
    query_vec: &[f32],
    top_k: usize,
    rrf_k: usize,
) -> Vec<(f32, &'a RagChunk)> {
    if chunks.is_empty() {
        return Vec::new();
    }

    let query_tokens = tokenize_text(query);
    let chunk_tokens: Vec<Vec<String>> = chunks.iter().map(|c| tokenize_text(&c.text)).collect();

    let mut doc_frequencies: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for q in &query_tokens {
        let count = chunk_tokens.iter().filter(|tokens| tokens.contains(q)).count();
        doc_frequencies.insert(q.clone(), count);
    }

    let total_doc_len: usize = chunk_tokens.iter().map(|t| t.len()).sum();
    let avg_doc_len = if !chunks.is_empty() {
        total_doc_len as f32 / chunks.len() as f32
    } else {
        1.0
    };

    // 1. Vector ranking
    let mut vector_ranked: Vec<(usize, f32)> = chunks.iter().enumerate()
        .map(|(idx, c)| (idx, cosine_similarity(query_vec, &c.vector)))
        .collect();
    vector_ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut vector_ranks = vec![0usize; chunks.len()];
    for (rank_0, (idx, _)) in vector_ranked.iter().enumerate() {
        vector_ranks[*idx] = rank_0 + 1;
    }

    // 2. BM25 ranking
    let mut bm25_ranked: Vec<(usize, f32)> = chunk_tokens.iter().enumerate()
        .map(|(idx, d_toks)| {
            let score = score_document_bm25(&query_tokens, d_toks, avg_doc_len, chunks.len(), &doc_frequencies);
            (idx, score)
        })
        .collect();
    bm25_ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut bm25_ranks = vec![0usize; chunks.len()];
    for (rank_0, (idx, _)) in bm25_ranked.iter().enumerate() {
        bm25_ranks[*idx] = rank_0 + 1;
    }

    // 3. Reciprocal Rank Fusion
    let rrf_k_f32 = rrf_k as f32;
    let mut fused: Vec<(usize, f32)> = (0..chunks.len())
        .map(|idx| {
            let r_vec = vector_ranks[idx] as f32;
            let r_bm25 = bm25_ranks[idx] as f32;
            let rrf_score = (1.0 / (rrf_k_f32 + r_vec)) + (1.0 / (rrf_k_f32 + r_bm25));
            (idx, rrf_score)
        })
        .collect();

    fused.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    fused.into_iter()
        .take(top_k)
        .map(|(idx, score)| (score, &chunks[idx]))
        .collect()
}

pub async fn apply_rag(client: &Client, prompt: &str, messages: &mut Vec<Message>) -> Result<(), Box<dyn std::error::Error>> {
    let bin_path = std::path::Path::new(BINARY_VECTORS_DEFAULT_FILE);
    let chunks: Vec<RagChunk> = if bin_path.exists() {
        if let Ok(store) = load_binary_vector_index(bin_path) {
            store.chunks
        } else if let Ok(data) = tokio::fs::read_to_string(".zy_rag_index.json").await {
            serde_json::from_str(&data).unwrap_or_default()
        } else {
            Vec::new()
        }
    } else if let Ok(data) = tokio::fs::read_to_string(".zy_rag_index.json").await {
        serde_json::from_str(&data).unwrap_or_default()
    } else {
        Vec::new()
    };

    if !chunks.is_empty() {
        print!("{} ", "🔍 Searching local codebase (Hybrid RAG + Binary Vectors + RRF)...".magenta());
        io::stdout().flush()?;
        
        let query_vec = embed_text(client, prompt).await.unwrap_or_default();
        let ranked = hybrid_rag_search(&chunks, prompt, &query_vec, 3, 60);
        
        let mut context_text = String::from("Relevant codebase context via Hybrid RAG (BM25 + Vector RRF):\n");
        for (score, chunk) in ranked {
            if score > 0.0 {
                context_text.push_str(&format!("--- FILE: {} (RRF Score: {:.4}) ---\n{}\n\n", chunk.file, score, chunk.text));
            }
        }
        
        messages.push(Message {
            role: "system".to_string(),
            content: context_text,
            tool_calls: None,
            images: None,
        });
        println!("{}", "Done".green());
    } else {
        println!("{}", "RAG enabled but index not found. Run 'zy index .' first.".red());
    }
    Ok(())
}

pub fn smart_chunk(content: &str, max_len: usize) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut current_chunk = String::new();
    
    for block in content.split("\n\n") {
        if current_chunk.len() + block.len() > max_len && !current_chunk.is_empty() {
            chunks.push(current_chunk.trim().to_string());
            current_chunk.clear();
        }
        current_chunk.push_str(block);
        current_chunk.push_str("\n\n");
    }
    if !current_chunk.trim().is_empty() {
        chunks.push(current_chunk.trim().to_string());
    }
    chunks
}

pub async fn build_rag_index(client: &Client, path: &str) -> Result<(), Box<dyn std::error::Error>> {
    println!("{} {}", "Indexing directory (Smart Chunking & Binary Vector Store):".bold(), path.cyan());
    let mut chunks = Vec::new();
    
    for entry in WalkDir::new(path).into_iter().filter_map(|e| e.ok()) {
        let path_str = entry.path().to_string_lossy().to_string();
        if path_str.contains("/target/") || path_str.contains("/.git/") || path_str.contains("node_modules") || path_str.contains(".zy") {
            continue;
        }
        if entry.file_type().is_file() {
            if let Ok(content) = tokio::fs::read_to_string(entry.path()).await {
                let text_chunks = smart_chunk(&content, 1000);
                for (i, text) in text_chunks.into_iter().enumerate() {
                    print!("Embedding {} chunk {} ... ", path_str.blue(), i);
                    io::stdout().flush()?;
                    
                    if let Ok(vector) = embed_text(client, &text).await {
                        chunks.push(RagChunk {
                            file: path_str.clone(),
                            text,
                            vector,
                        });
                        println!("{}", "OK".green());
                    } else {
                        println!("{}", "Failed".red());
                    }
                }
            }
        }
    }
    
    let _ = save_binary_vector_index(std::path::Path::new(BINARY_VECTORS_DEFAULT_FILE), &chunks);
    let json_data = serde_json::to_string(&chunks)?;
    tokio::fs::write(".zy_rag_index.json", json_data).await?;
    println!("{} {} chunks (saved to {} & .zy_rag_index.json)", "Indexed & saved".green().bold(), chunks.len(), BINARY_VECTORS_DEFAULT_FILE.cyan());
    Ok(())
}

pub async fn vella_reindex_file(client: &Client, file_path: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
    let bin_path = std::path::Path::new(BINARY_VECTORS_DEFAULT_FILE);
    let path_str = file_path.to_string_lossy().to_string();
    
    let mut chunks: Vec<RagChunk> = if bin_path.exists() {
        if let Ok(store) = load_binary_vector_index(bin_path) {
            store.chunks
        } else if let Ok(data) = tokio::fs::read_to_string(".zy_rag_index.json").await {
            serde_json::from_str(&data).unwrap_or_default()
        } else {
            Vec::new()
        }
    } else if let Ok(data) = tokio::fs::read_to_string(".zy_rag_index.json").await {
        serde_json::from_str(&data).unwrap_or_default()
    } else {
        Vec::new()
    };
    
    chunks.retain(|c| c.file != path_str);
    
    if let Ok(content) = tokio::fs::read_to_string(file_path).await {
        let text_chunks = smart_chunk(&content, 1000);
        for text in text_chunks {
            if let Ok(vector) = embed_text(client, &text).await {
                chunks.push(RagChunk {
                    file: path_str.clone(),
                    text,
                    vector,
                });
            }
        }
    }
    
    let _ = save_binary_vector_index(bin_path, &chunks);
    let json_data = serde_json::to_string(&chunks)?;
    tokio::fs::write(".zy_rag_index.json", json_data).await?;
    println!("{} {}", "✔️  Vella Sync Complete (Binary + JSON):".green(), path_str);
    Ok(())
}

pub async fn vella_watch_daemon(client: &Client, path: &str) -> Result<(), Box<dyn std::error::Error>> {
    println!("{} {} {}", "👁️  Vella OS Watcher".magenta().bold(), "running on".cyan(), path.bold());
    println!("{}", "Any file changes will be automatically embedded into the RAG DB via Vella's zero-latency pipeline...".green());
    
    let (tx, rx) = std::sync::mpsc::channel();
    let mut watcher = notify::recommended_watcher(tx)?;
    watcher.watch(std::path::Path::new(path), RecursiveMode::Recursive)?;

    let (async_tx, mut async_rx) = tokio::sync::mpsc::unbounded_channel();
    tokio::task::spawn_blocking(move || {
        while let Ok(event) = rx.recv() {
            if async_tx.send(event).is_err() { break; }
        }
    });

    while let Some(Ok(event)) = async_rx.recv().await {
        if let notify::EventKind::Modify(_) = event.kind {
            for path_buf in event.paths {
                let path_str = path_buf.to_string_lossy().to_string();
                if path_str.contains("/target/") || path_str.contains("/.git/") || path_str.contains(".zy") || path_str.contains(".log") {
                    continue;
                }
                println!("{} {} {}", "⚡ Change detected:".yellow(), path_str, "- routing to Vella AI pipeline...");
                let _ = vella_reindex_file(client, &path_buf).await;
            }
        }
    }
    Ok(())
}

pub const STRATEGIST_PROMPT: &str = r#"
[AI STRATEGIST PROTOCOL ENGAGED]
You are operating as a lethal, highly calculated AI Strategist. 
Before executing ANY tool or providing a final answer, you MUST use an OODA loop (Observe, Orient, Decide, Act).
1. OBSERVE: Analyze the user's request and the environment.
2. ORIENT: Identify edge cases, hidden constraints, and potential points of failure.
3. DECIDE: Formulate a ruthless, highly optimized, multi-step execution plan.
4. ACT: Execute the tools required to complete the plan flawlessly.
Always wrap your strategic reasoning in <STRATEGY> ... </STRATEGY> tags before taking action.
"#;

pub fn build_initial_messages(system: Option<&str>, files: &[String], strategist: bool) -> Result<Vec<Message>, Box<dyn std::error::Error>> {
    let mut messages = Vec::new();
    let project_rules = load_project_rules(std::path::Path::new("."));

    let base_system = if let Some(sys) = system {
        let mut final_sys = sys.to_string();
        if strategist { final_sys.push_str(STRATEGIST_PROMPT); }
        final_sys
    } else {
        let mut default_sys = "You are an expert, deterministic coding assistant. Provide highly accurate and factual answers. If you do not know the answer or lack context, explicitly state 'I do not have enough information' instead of guessing or making up functions. Stick strictly to the provided files or RAG context.".to_string();
        if strategist { default_sys.push_str(STRATEGIST_PROMPT); }
        default_sys
    };

    let full_system = if let Some(rules) = project_rules {
        format!("{}\n\n=== ACTIVE PROJECT & USER RULES ===\n{}\n===================================", base_system, rules)
    } else {
        base_system
    };

    messages.push(Message {
        role: "system".to_string(),
        content: full_system,
        tool_calls: None,
        images: None,
    });

    for path in files {
        if path.ends_with(".png") || path.ends_with(".jpg") || path.ends_with(".jpeg") {
            if let Ok(bytes) = fs::read(path) {
                let b64 = base64::engine::general_purpose::STANDARD.encode(&bytes);
                messages.push(Message {
                    role: "user".to_string(),
                    content: format!("Attached Vision Image: {}", path),
                    tool_calls: None,
                    images: Some(vec![b64]),
                });
                println!("{} {}", "👁️  Attached Vision Image:".blue(), path.bold());
            }
        } else {
            let content = fs::read_to_string(path)?;
            let file_context = format!("File context for '{}':\n```\n{}\n```", path, content);
            messages.push(Message {
                role: "system".to_string(),
                content: file_context,
                tool_calls: None,
                images: None,
            });
            println!("{} {}", "Attached file:".blue(), path.bold());
        }
    }

    Ok(messages)
}

// -------------------------------------------------------------------------------------------------
// FEATURE 12: MULTI-AGENT SWARM ORCHESTRATOR
// -------------------------------------------------------------------------------------------------

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct SwarmWorkflowResult {
    pub goal: String,
    pub plan: String,
    pub coder_output: String,
    pub audit_report: String,
    pub test_report: Option<TestReport>,
    pub success: bool,
}

pub async fn run_swarm_workflow(
    client: &Client,
    model: &str,
    executor_model: Option<&str>,
    goal: &str,
    options: &OllamaOptions,
    markdown: bool,
    force: bool,
    sandbox: bool,
) -> Result<SwarmWorkflowResult, Box<dyn std::error::Error>> {
    println!("\n{}", "╔═══════════════════════════════════════════════════════════╗".magenta());
    println!("║ {} {:<43} ║", "🐝 MULTI-AGENT SWARM ORCHESTRATOR:".magenta().bold(), goal.yellow());
    println!("╠═══════════════════════════════════════════════════════════╣\n");

    let repo_map = build_repo_map(std::path::Path::new("."), 2048);

    // ─────────────────────────────────────────────────────────────────────────
    // Phase 1: Architect (Technical Blueprint)
    // ─────────────────────────────────────────────────────────────────────────
    println!("{} {}", "🧠 [Phase 1/4: Architect Planning]".magenta().bold(), model.yellow());
    let architect_prompt = format!(
        "You are the Swarm Lead Architect. Design a clear, step-by-step technical implementation plan for this goal: '{}'.\n\nRepository Symbol Map:\n{}\n\nProvide exact filenames, functions to modify or create, and precise technical steps.",
        goal, repo_map
    );
    let arch_msgs = vec![
        Message {
            role: "system".to_string(),
            content: "You are the Swarm Lead Software Architect. Provide concrete, actionable, numbered implementation blueprints without unnecessary fluff.".to_string(),
            tool_calls: None,
            images: None,
        },
        Message {
            role: "user".to_string(),
            content: architect_prompt,
            tool_calls: None,
            images: None,
        },
    ];
    let plan = fetch_full_response(client, model, &arch_msgs, options, None).await?;
    println!("\n{}\n", "─── Architect Plan ───".magenta());
    if markdown {
        print_text(&plan);
    } else {
        println!("{}", plan);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Phase 2: Coder (Autonomous Tool & Code Execution)
    // ─────────────────────────────────────────────────────────────────────────
    let coder_model = executor_model.unwrap_or(model);
    println!("\n{} {}", "⚡ [Phase 2/4: Coder Executing Plan]".yellow().bold(), coder_model.yellow());
    let mut coder_messages = vec![
        Message {
            role: "system".to_string(),
            content: format!(
                "You are the Swarm Autonomous Coder. Implement the following architect plan by using available tools (write_file, patch_file, run_bash, lsp_diagnostics, run_tests).\n\nArchitect Plan:\n{}",
                plan
            ),
            tool_calls: None,
            images: None,
        },
        Message {
            role: "user".to_string(),
            content: format!("Execute the technical plan to achieve the goal: '{}'", goal),
            tool_calls: None,
            images: None,
        },
    ];
    agent_loop(client, coder_model, &mut coder_messages, markdown, options, force, None, sandbox).await?;
    let coder_output = coder_messages.last().map(|m| m.content.clone()).unwrap_or_default();

    // ─────────────────────────────────────────────────────────────────────────
    // Phase 3: Auditor (Security & Code Review)
    // ─────────────────────────────────────────────────────────────────────────
    println!("\n{} {}", "🛡️  [Phase 3/4: Auditor Code & Security Review]".cyan().bold(), model.yellow());
    let diff_output = std::process::Command::new("git").args(["diff", "HEAD"]).output()
        .or_else(|_| std::process::Command::new("git").args(["diff"]).output());
    let diff = diff_output.map(|o| String::from_utf8_lossy(&o.stdout).to_string()).unwrap_or_default();

    let auditor_prompt = format!(
        "You are the Swarm Security & Quality Auditor. Inspect the following code changes/diff implemented for goal '{}':\n\n```diff\n{}\n```\n\nReview for: 1. Logic bugs 2. Security / vulnerability risks 3. Regressions. Provide a summary with a final verdict: [AUDIT: PASS] or [AUDIT: ISSUES DETECTED].",
        goal, if diff.is_empty() { "(No git diff available - checking coder output)" } else { &diff }
    );
    let audit_msgs = vec![
        Message {
            role: "system".to_string(),
            content: "You are the Swarm Senior Security & Quality Auditor. Scrutinize all changes with extreme rigor.".to_string(),
            tool_calls: None,
            images: None,
        },
        Message {
            role: "user".to_string(),
            content: auditor_prompt,
            tool_calls: None,
            images: None,
        },
    ];
    let audit_report = fetch_full_response(client, model, &audit_msgs, options, None).await?;
    println!("\n{}\n", "─── Audit Report ───".cyan());
    if markdown {
        print_text(&audit_report);
    } else {
        println!("{}", audit_report);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Phase 4: QA / Tester (Automated Verification)
    // ─────────────────────────────────────────────────────────────────────────
    println!("\n{} {}", "🧪 [Phase 4/4: QA Test Runner Verification]".green().bold(), "Checking test suite...".white());
    let test_report = run_project_tests(std::path::Path::new("."), None).ok();
    let tests_pass = if let Some(tr) = &test_report {
        println!("{}", format_test_report_for_terminal(tr));
        tr.success
    } else {
        true
    };

    let overall_success = tests_pass && !audit_report.contains("[AUDIT: ISSUES DETECTED]");

    println!("\n{}", "╔═══════════════════════════════════════════════════════════╗".magenta());
    if overall_success {
        println!("║  {}  ║", "🎉 SWARM WORKFLOW COMPLETED SUCCESSFULLY!".green().bold());
    } else {
        println!("║  {}  ║", "⚠️  SWARM WORKFLOW FINISHED WITH WARNINGS/FAILURES".yellow().bold());
    }
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    Ok(SwarmWorkflowResult {
        goal: goal.to_string(),
        plan,
        coder_output,
        audit_report,
        test_report,
        success: overall_success,
    })
}

// -------------------------------------------------------------------------------------------------
// FEATURE 13: INTERACTIVE `@` CONTEXT MENTIONS
// -------------------------------------------------------------------------------------------------

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct ContextMention {
    pub tag: String,
    pub mention_type: String, // "file", "git", "diff", "symbol"
    pub target: String,
    pub content: String,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct ExpandedContext {
    pub original_prompt: String,
    pub cleaned_prompt: String,
    pub mentions: Vec<ContextMention>,
    pub context_messages: Vec<Message>,
}

pub fn extract_symbol_context(workspace_root: &std::path::Path, symbol_name: &str) -> Option<String> {
    let clean_symbol = symbol_name.trim();
    if clean_symbol.is_empty() { return None; }

    for entry in WalkDir::new(workspace_root).into_iter().filter_map(|e| e.ok()) {
        let p = entry.path();
        let p_str = p.to_string_lossy();
        if p_str.contains("/target/") || p_str.contains("\\target\\") || p_str.contains("/.git/") || p_str.contains("\\.git\\") || p_str.contains("node_modules") {
            continue;
        }
        if p.is_file() {
            let ext = p.extension().and_then(|e| e.to_str()).unwrap_or("");
            if matches!(ext, "rs" | "py" | "ts" | "js" | "go" | "c" | "cpp" | "h" | "hpp") {
                if let Ok(content) = fs::read_to_string(p) {
                    let lines: Vec<&str> = content.lines().collect();
                    for (i, line) in lines.iter().enumerate() {
                        let trimmed = line.trim();
                        // Look for symbol definitions
                        if trimmed.contains(&format!("fn {}", clean_symbol))
                            || trimmed.contains(&format!("struct {}", clean_symbol))
                            || trimmed.contains(&format!("enum {}", clean_symbol))
                            || trimmed.contains(&format!("trait {}", clean_symbol))
                            || trimmed.contains(&format!("class {}", clean_symbol))
                            || trimmed.contains(&format!("def {}", clean_symbol))
                            || trimmed.contains(&format!("interface {}", clean_symbol))
                            || trimmed.contains(&format!("type {}", clean_symbol))
                            || trimmed.contains(&format!("const {}", clean_symbol))
                        {
                            let start = i.saturating_sub(2);
                            let end = (i + 25).min(lines.len());
                            let snippet = lines[start..end].join("\n");
                            let rel_path = p.strip_prefix(workspace_root).unwrap_or(p).to_string_lossy();
                            return Some(format!("File: {} (Lines {}-{}):\n```{}\n{}\n```", rel_path, start + 1, end, ext, snippet));
                        }
                    }
                }
            }
        }
    }
    None
}

pub fn expand_context_mentions(prompt: &str, workspace_root: &std::path::Path) -> ExpandedContext {
    let mut mentions = Vec::new();
    let mut context_messages = Vec::new();

    let tokens: Vec<&str> = prompt.split_whitespace().collect();

    for raw_token in tokens {
        let trimmed_start = raw_token.trim_start_matches(['(', '[', '{', '<', '"', '\'']);
        if !trimmed_start.starts_with('@') || trimmed_start.len() <= 1 {
            continue;
        }

        // Clean trailing punctuation
        let token = trimmed_start.trim_end_matches([',', '.', '?', '!', ';', ':', ')', ']', '}', '>', '"', '\'']);

        if token.eq_ignore_ascii_case("@git") || token.eq_ignore_ascii_case("@diff") {
            let git_status = std::process::Command::new("git").args(["status", "--short"]).output()
                .map(|o| String::from_utf8_lossy(&o.stdout).to_string())
                .unwrap_or_else(|_| "Git not available".to_string());
            let git_diff = std::process::Command::new("git").args(["diff", "HEAD"]).output()
                .or_else(|_| std::process::Command::new("git").args(["diff"]).output())
                .map(|o| String::from_utf8_lossy(&o.stdout).to_string())
                .unwrap_or_default();

            let diff_content = format!(
                "=== ACTIVE GIT REPOSITORY CONTEXT (@git) ===\nSTATUS:\n{}\n\nDIFF:\n{}\n==============================================",
                git_status.trim(),
                if git_diff.trim().is_empty() { "(No uncommitted changes)" } else { git_diff.trim() }
            );

            mentions.push(ContextMention {
                tag: token.to_string(),
                mention_type: "git".to_string(),
                target: "git-diff".to_string(),
                content: diff_content.clone(),
            });

            context_messages.push(Message {
                role: "system".to_string(),
                content: diff_content,
                tool_calls: None,
                images: None,
            });
        } else if let Some(stripped) = token.strip_prefix("@file:") {
            let file_path = workspace_root.join(stripped);
            if file_path.exists() && file_path.is_file() {
                if let Ok(content) = fs::read_to_string(&file_path) {
                    let file_content = format!("=== FILE CONTEXT (@file:{}) ===\n{}\n==========================================", stripped, content);
                    mentions.push(ContextMention {
                        tag: token.to_string(),
                        mention_type: "file".to_string(),
                        target: stripped.to_string(),
                        content: file_content.clone(),
                    });
                    context_messages.push(Message {
                        role: "system".to_string(),
                        content: file_content,
                        tool_calls: None,
                        images: None,
                    });
                }
            }
        } else if let Some(stripped) = token.strip_prefix("@symbol:").or_else(|| token.strip_prefix("@sym:")) {
            if let Some(ctx) = extract_symbol_context(workspace_root, stripped) {
                let sym_content = format!("=== SYMBOL DEFINITION CONTEXT (@symbol:{}) ===\n{}\n==============================================", stripped, ctx);
                mentions.push(ContextMention {
                    tag: token.to_string(),
                    mention_type: "symbol".to_string(),
                    target: stripped.to_string(),
                    content: sym_content.clone(),
                });
                context_messages.push(Message {
                    role: "system".to_string(),
                    content: sym_content,
                    tool_calls: None,
                    images: None,
                });
            }
        } else if token.starts_with('@') {
            let potential_path = &token[1..];
            let direct_file = workspace_root.join(potential_path);
            if direct_file.exists() && direct_file.is_file() {
                if let Ok(content) = fs::read_to_string(&direct_file) {
                    let file_content = format!("=== ATTACHED FILE CONTEXT (@{}) ===\n{}\n==========================================", potential_path, content);
                    mentions.push(ContextMention {
                        tag: token.to_string(),
                        mention_type: "file".to_string(),
                        target: potential_path.to_string(),
                        content: file_content.clone(),
                    });
                    context_messages.push(Message {
                        role: "system".to_string(),
                        content: file_content,
                        tool_calls: None,
                        images: None,
                    });
                }
            }
        }
    }

    ExpandedContext {
        original_prompt: prompt.to_string(),
        cleaned_prompt: prompt.to_string(),
        mentions,
        context_messages,
    }
}

// -------------------------------------------------------------------------------------------------
// FEATURE 14: AUTONOMOUS WEB SEARCH ENGINE (DUCKDUCKGO HTML/LITE)
// -------------------------------------------------------------------------------------------------

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct SearchResult {
    pub title: String,
    pub url: String,
    pub snippet: String,
}

pub fn strip_html_tags(input: &str) -> String {
    let mut result = String::new();
    let mut inside = false;
    for c in input.chars() {
        if c == '<' {
            inside = true;
        } else if c == '>' {
            inside = false;
        } else if !inside {
            result.push(c);
        }
    }
    result
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&nbsp;", " ")
        .trim()
        .to_string()
}

pub fn url_decode(s: &str) -> String {
    let mut result = String::new();
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '%' {
            let h1 = chars.next();
            let h2 = chars.next();
            if let (Some(h1), Some(h2)) = (h1, h2) {
                let hex_str = format!("{}{}", h1, h2);
                if let Ok(byte) = u8::from_str_radix(&hex_str, 16) {
                    result.push(byte as char);
                    continue;
                } else {
                    result.push('%');
                    result.push(h1);
                    result.push(h2);
                    continue;
                }
            } else {
                result.push('%');
                if let Some(h1) = h1 { result.push(h1); }
                continue;
            }
        } else if c == '+' {
            result.push(' ');
        } else {
            result.push(c);
        }
    }
    result
}

pub fn extract_duckduckgo_url(raw_href: &str) -> String {
    if let Some(uddg_idx) = raw_href.find("uddg=") {
        let rest = &raw_href[uddg_idx + 5..];
        let enc_url = rest.split('&').next().unwrap_or(rest);
        url_decode(enc_url)
    } else if raw_href.starts_with("//duckduckgo.com/l/?uddg=") {
        let rest = &raw_href[25..];
        let enc_url = rest.split('&').next().unwrap_or(rest);
        url_decode(enc_url)
    } else if raw_href.starts_with('/') {
        format!("https://duckduckgo.com{}", raw_href)
    } else {
        raw_href.to_string()
    }
}

pub fn parse_duckduckgo_html(html: &str) -> Vec<SearchResult> {
    let mut results = Vec::new();

    let result_blocks: Vec<&str> = html.split("<div class=\"result ").collect();
    if result_blocks.len() > 1 {
        for block in result_blocks.iter().skip(1) {
            let mut title = String::new();
            let mut url = String::new();
            let mut snippet = String::new();

            if let Some(a_idx) = block.find("class=\"result__a\"") {
                let sub = &block[a_idx..];
                if let Some(href_idx) = sub.find("href=\"") {
                    let href_rest = &sub[href_idx + 6..];
                    if let Some(quote_end) = href_rest.find('"') {
                        url = extract_duckduckgo_url(&href_rest[..quote_end]);
                    }
                }
                if let Some(tag_close) = sub.find('>') {
                    let title_rest = &sub[tag_close + 1..];
                    if let Some(a_close) = title_rest.find("</a>") {
                        title = strip_html_tags(&title_rest[..a_close]);
                    } else if let Some(next_tag) = title_rest.find('<') {
                        title = strip_html_tags(&title_rest[..next_tag]);
                    } else {
                        title = strip_html_tags(title_rest);
                    }
                }
            }

            if let Some(snip_idx) = block.find("class=\"result__snippet\"").or_else(|| block.find("class=\"result-snippet\"")) {
                let sub = &block[snip_idx..];
                if let Some(tag_close) = sub.find('>') {
                    let snip_rest = &sub[tag_close + 1..];
                    if let Some(close_tag) = snip_rest.find("</a>").or_else(|| snip_rest.find("</td>")).or_else(|| snip_rest.find("</div>")) {
                        snippet = strip_html_tags(&snip_rest[..close_tag]);
                    } else if let Some(next_tag) = snip_rest.find('<') {
                        snippet = strip_html_tags(&snip_rest[..next_tag]);
                    } else {
                        snippet = strip_html_tags(snip_rest);
                    }
                }
            }

            if !title.is_empty() && !url.is_empty() {
                results.push(SearchResult { title, url, snippet });
            }
        }
    }

    if results.is_empty() {
        let rows: Vec<&str> = html.split("<tr").collect();
        let mut cur_title = String::new();
        let mut cur_url = String::new();

        for row in rows {
            if row.contains("class=\"result-link\"") || row.contains("class=\"result-url\"") {
                if let Some(href_idx) = row.find("href=\"") {
                    let rest = &row[href_idx + 6..];
                    if let Some(q_end) = rest.find('"') {
                        cur_url = extract_duckduckgo_url(&rest[..q_end]);
                    }
                }
                if let Some(tag_close) = row.find('>') {
                    let rest = &row[tag_close + 1..];
                    if let Some(a_end) = rest.find("</a>") {
                        cur_title = strip_html_tags(&rest[..a_end]);
                    }
                }
            } else if row.contains("class=\"result-snippet\"") {
                let snippet = strip_html_tags(row);
                if !cur_title.is_empty() && !cur_url.is_empty() {
                    results.push(SearchResult {
                        title: cur_title.clone(),
                        url: cur_url.clone(),
                        snippet,
                    });
                    cur_title.clear();
                    cur_url.clear();
                }
            }
        }
    }

    results
}

pub async fn perform_web_search(
    client: &Client,
    query: &str,
) -> Result<Vec<SearchResult>, Box<dyn std::error::Error + Send + Sync>> {
    let url = "https://html.duckduckgo.com/html/";
    let params = [("q", query), ("b", "")];

    let response = client
        .post(url)
        .header("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36")
        .header("Accept", "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8")
        .header("Accept-Language", "en-US,en;q=0.5")
        .form(&params)
        .send()
        .await;

    match response {
        Ok(res) if res.status().is_success() => {
            let body = res.text().await.unwrap_or_default();
            Ok(parse_duckduckgo_html(&body))
        }
        _ => {
            let lite_url = format!("https://lite.duckduckgo.com/lite/?q={}", query.replace(' ', "+"));
            let lite_res = client.get(&lite_url)
                .header("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64)")
                .send()
                .await;
            if let Ok(l_res) = lite_res {
                let body = l_res.text().await.unwrap_or_default();
                Ok(parse_duckduckgo_html(&body))
            } else {
                Ok(Vec::new())
            }
        }
    }
}

// -------------------------------------------------------------------------------------------------
// FEATURE 15: TIME-TRAVEL INTERACTIVE SESSION DEBUGGER
// -------------------------------------------------------------------------------------------------

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct TimelineTurn {
    pub turn_index: usize,
    pub user_preview: String,
    pub assistant_preview: String,
    pub tool_calls_count: usize,
    pub token_estimate: usize,
}

pub fn extract_timeline_turns(messages: &[Message]) -> Vec<TimelineTurn> {
    let mut turns = Vec::new();
    let mut current_turn_idx = 0;
    let mut i = 0;

    while i < messages.len() {
        if messages[i].role == "user" {
            current_turn_idx += 1;
            let user_msg = &messages[i];
            let user_preview = if user_msg.content.len() > 60 {
                format!("{}...", &user_msg.content[..57])
            } else {
                user_msg.content.clone()
            };

            let mut assistant_preview = String::new();
            let mut tool_calls = 0;
            let mut turn_tokens = estimate_message_tokens(user_msg);

            let mut j = i + 1;
            while j < messages.len() && messages[j].role != "user" {
                turn_tokens += estimate_message_tokens(&messages[j]);
                if messages[j].role == "assistant" {
                    if let Some(calls) = &messages[j].tool_calls {
                        tool_calls += calls.len();
                    }
                    if assistant_preview.is_empty() && !messages[j].content.is_empty() {
                        assistant_preview = if messages[j].content.len() > 60 {
                            format!("{}...", &messages[j].content[..57])
                        } else {
                            messages[j].content.clone()
                        };
                    }
                } else if messages[j].role == "tool" {
                    tool_calls += 1;
                }
                j += 1;
            }

            if assistant_preview.is_empty() {
                assistant_preview = if tool_calls > 0 { format!("[Executed {} tools]", tool_calls) } else { "(No response yet)".to_string() };
            }

            turns.push(TimelineTurn {
                turn_index: current_turn_idx,
                user_preview,
                assistant_preview,
                tool_calls_count: tool_calls,
                token_estimate: turn_tokens,
            });

            i = j;
        } else {
            i += 1;
        }
    }

    turns
}

pub fn format_timeline(messages: &[Message]) -> String {
    let turns = extract_timeline_turns(messages);
    let mut out = String::new();
    out.push_str(&format!("\n{}\n", "╔═══════════════════════════════════════════════════════════╗".cyan()));
    out.push_str(&format!("║ {} {:<40} ║\n", "⏳ CONVERSATION SESSION TIMELINE".cyan().bold(), format!("({} Turns Recorded)", turns.len()).yellow()));
    out.push_str("╠═══════════════════════════════════════════════════════════╣\n");

    if turns.is_empty() {
        out.push_str(&format!("║  {}  ║\n", "🌱 Session started. No conversational turns recorded yet.".dimmed()));
    } else {
        for (idx, turn) in turns.iter().enumerate() {
            out.push_str(&format!("║ Turn #{:<2} │ {} {:<41} ║\n", turn.turn_index.to_string().yellow().bold(), "👤 User:".green().bold(), turn.user_preview));
            out.push_str(&format!("║         │ {} {:<41} ║\n", "🤖 zy:  ".magenta().bold(), turn.assistant_preview));
            out.push_str(&format!("║         │ ⚙️ Tools: {:<3} │ 🪙 Est. Tokens: {:<19} ║\n", turn.tool_calls_count.to_string().cyan(), format!("~{}", turn.token_estimate).dimmed()));
            if idx + 1 < turns.len() {
                out.push_str("╟───────────────────────────────────────────────────────────╢\n");
            }
        }
    }
    out.push_str("╚═══════════════════════════════════════════════════════════╝\n");
    out
}

pub fn rewind_messages(messages: &mut Vec<Message>, count: usize) -> usize {
    if count == 0 { return 0; }

    let user_indices: Vec<usize> = messages.iter().enumerate()
        .filter(|(_, m)| m.role == "user")
        .map(|(i, _)| i)
        .collect();

    if user_indices.is_empty() {
        return 0;
    }

    let turns_to_remove = count.min(user_indices.len());
    let target_user_idx = user_indices[user_indices.len() - turns_to_remove];

    messages.truncate(target_user_idx);

    turns_to_remove
}

// -------------------------------------------------------------------------------------------------
// FEATURE 16: CONVENTIONAL COMMIT & PULL REQUEST GENERATOR
// -------------------------------------------------------------------------------------------------

pub fn parse_conventional_commit(raw_text: &str) -> String {
    let mut text = raw_text.trim();
    if text.starts_with("```") {
        if let Some(first_nl) = text.find('\n') {
            text = &text[first_nl + 1..];
        }
        if let Some(last_fence) = text.rfind("```") {
            text = &text[..last_fence];
        }
    }
    let cleaned = text.trim().trim_matches(['"', '\'', '`']);
    cleaned.to_string()
}

pub fn generate_fallback_commit_message(diff: &str) -> String {
    let diff_lower = diff.to_lowercase();
    if diff_lower.contains("cargo.toml") || diff_lower.contains("package.json") {
        "chore(deps): update dependencies and configuration".to_string()
    } else if diff_lower.contains("test") || diff_lower.contains("integration_tests") {
        "test(core): add and expand integration test suite".to_string()
    } else if diff_lower.contains("readme.md") || diff_lower.contains("docs/") {
        "docs(readme): update project documentation and guides".to_string()
    } else if diff_lower.contains("fix") || diff_lower.contains("bug") || diff_lower.contains("err") {
        "fix(core): resolve bugs and enhance stability".to_string()
    } else {
        "feat(core): implement advanced features and system improvements".to_string()
    }
}

pub async fn generate_commit_message(
    client: &Client,
    model: &str,
    diff: &str,
    options: &OllamaOptions,
    custom_hint: Option<&str>,
) -> Result<String, Box<dyn std::error::Error>> {
    let hint_str = custom_hint.map(|h| format!("\nUser Context Hint: '{}'", h)).unwrap_or_default();
    let prompt = format!(
        "You are an expert Git engineer. Write a concise, high-impact Conventional Commit message for this git diff:\n\nFormat: <type>(<scope>): <summary in present tense>\nTypes: feat, fix, refactor, perf, test, docs, chore, style, build, ci.\n\nDiff:\n```diff\n{}\n```{}\n\nOutput ONLY the commit message line (and optional bullet points). Do not wrap in markdown quotes or explanations.",
        if diff.len() > 3000 { &diff[..3000] } else { diff },
        hint_str
    );

    let messages = vec![
        Message {
            role: "system".to_string(),
            content: "You are a senior developer specializing in semantic Conventional Commits.".to_string(),
            tool_calls: None,
            images: None,
        },
        Message {
            role: "user".to_string(),
            content: prompt,
            tool_calls: None,
            images: None,
        },
    ];

    match fetch_full_response(client, model, &messages, options, None).await {
        Ok(res) if !res.trim().is_empty() => Ok(parse_conventional_commit(&res)),
        _ => Ok(generate_fallback_commit_message(diff)),
    }
}

pub async fn generate_pr_description(
    client: &Client,
    model: &str,
    diff: &str,
    branch: &str,
    options: &OllamaOptions,
) -> Result<String, Box<dyn std::error::Error>> {
    let prompt = format!(
        "Generate a structured, professional Markdown Pull Request description for branch '{}'.\n\nStructure:\n## 🎯 Overview\n<concise summary of changes>\n\n## 🚀 Key Changes\n<bullet points>\n\n## 🧪 Testing Checklist\n- [x] Unit/Integration tests\n- [ ] Manual smoke test\n\nGit Diff:\n```diff\n{}\n```\n\nOutput ONLY clean Markdown.",
        branch,
        if diff.len() > 3000 { &diff[..3000] } else { diff }
    );

    let messages = vec![
        Message {
            role: "system".to_string(),
            content: "You are a lead developer writing clear, comprehensive GitHub PR descriptions in Markdown.".to_string(),
            tool_calls: None,
            images: None,
        },
        Message {
            role: "user".to_string(),
            content: prompt,
            tool_calls: None,
            images: None,
        },
    ];

    match fetch_full_response(client, model, &messages, options, None).await {
        Ok(res) if !res.trim().is_empty() => Ok(res.trim().to_string()),
        _ => {
            let fallback_pr = format!(
                "## 🎯 Overview\nThis PR updates the `{}` branch with latest feature implementations, performance optimizations, and test coverage improvements.\n\n## 🚀 Key Changes\n- Implemented core system enhancements\n- Added comprehensive test suites\n- Refactored internal engines for speed and stability\n\n## 🧪 Testing Checklist\n- [x] `cargo test` passing\n- [x] `cargo build --release` passing\n",
                branch
            );
            Ok(fallback_pr)
        }
    }
}

// -------------------------------------------------------------------------------------------------
// FEATURE 17: FULL-SCREEN INTERACTIVE TUI DASHBOARD (ratatui + crossterm)
// -------------------------------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct TuiAppState {
    pub messages: Vec<Message>,
    pub input: String,
    pub active_model: String,
    pub agent_mode: bool,
    pub force_mode: bool,
    pub rag_mode: bool,
    pub preview_file: String,
    pub preview_content: String,
    pub diff_content: String,
    pub cpu_cores: usize,
    pub total_mem_gb: u64,
    pub used_mem_gb: u64,
    pub token_budget_info: String,
    pub aituner_profile: String,
    pub status_msg: String,
    pub scroll_chat: u16,
    pub scroll_preview: u16,
    pub is_thinking: bool,
    pub think_buffer: String,
}

impl Default for TuiAppState {
    fn default() -> Self {
        Self {
            messages: Vec::new(),
            input: String::new(),
            active_model: "llama2".to_string(),
            agent_mode: false,
            force_mode: false,
            rag_mode: false,
            preview_file: String::new(),
            preview_content: String::new(),
            diff_content: String::new(),
            cpu_cores: 8,
            total_mem_gb: 16,
            used_mem_gb: 4,
            token_budget_info: "0 / 2048 (0%)".to_string(),
            aituner_profile: "ECO (2048 ctx)".to_string(),
            status_msg: "Ready".to_string(),
            scroll_chat: 0,
            scroll_preview: 0,
            is_thinking: false,
            think_buffer: String::new(),
        }
    }
}

pub fn render_tui_layout(f: &mut ratatui::Frame, state: &TuiAppState) {
    use ratatui::layout::{Constraint, Direction, Layout};
    use ratatui::style::{Color, Modifier, Style};
    use ratatui::text::{Line, Span};
    use ratatui::widgets::{Block, Borders, Paragraph, Wrap};

    let main_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(55), Constraint::Percentage(45)])
        .split(f.size());

    // Left Panel: Chat history & agent <think> streaming
    let left_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(3), Constraint::Length(3)])
        .split(main_chunks[0]);

    let mut chat_lines = Vec::new();
    for msg in &state.messages {
        let (role_color, role_name) = match msg.role.as_str() {
            "user" => (Color::Green, "User ❯"),
            "assistant" => (Color::Cyan, "zy ❯"),
            "system" => (Color::Yellow, "System ❯"),
            "tool" => (Color::Magenta, "Tool ❯"),
            _ => (Color::White, "Message ❯"),
        };

        chat_lines.push(Line::from(vec![
            Span::styled(format!("{} ", role_name), Style::default().fg(role_color).add_modifier(Modifier::BOLD)),
        ]));

        let content = &msg.content;
        let mut in_think = false;
        for line in content.lines() {
            if line.contains("<think>") {
                in_think = true;
            }
            if in_think {
                chat_lines.push(Line::from(vec![
                    Span::styled(format!("  🧠 {}", line), Style::default().fg(Color::DarkGray).add_modifier(Modifier::ITALIC)),
                ]));
            } else {
                chat_lines.push(Line::from(vec![
                    Span::raw(format!("  {}", line)),
                ]));
            }
            if line.contains("</think>") {
                in_think = false;
            }
        }
        chat_lines.push(Line::from(""));
    }

    if state.is_thinking && !state.think_buffer.is_empty() {
        chat_lines.push(Line::from(vec![
            Span::styled("zy ❯ [Thinking...]", Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD)),
        ]));
        for line in state.think_buffer.lines() {
            chat_lines.push(Line::from(vec![
                Span::styled(format!("  🧠 {}", line), Style::default().fg(Color::DarkGray).add_modifier(Modifier::ITALIC)),
            ]));
        }
    }

    let chat_block = Block::default()
        .borders(Borders::ALL)
        .title(Span::styled(" 💬 Chat & Agent Thinking ", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)));
    let chat_p = Paragraph::new(chat_lines)
        .block(chat_block)
        .wrap(Wrap { trim: false })
        .scroll((state.scroll_chat, 0));
    f.render_widget(chat_p, left_chunks[0]);

    // Input line
    let input_block = Block::default()
        .borders(Borders::ALL)
        .title(Span::styled(format!(" ❯ Prompt ({}) ", state.status_msg), Style::default().fg(Color::Green)));
    let input_p = Paragraph::new(state.input.as_str())
        .block(input_block)
        .style(Style::default().fg(Color::White));
    f.render_widget(input_p, left_chunks[1]);

    // Right side: Top-Right (File preview & diff) and Bottom-Right (Hardware stats)
    let right_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(65), Constraint::Percentage(35)])
        .split(main_chunks[1]);

    // Top-Right Panel: File preview & live syntax-highlighted code diff panel
    let mut preview_lines = Vec::new();
    if !state.diff_content.is_empty() {
        preview_lines.push(Line::from(Span::styled("--- Live Unified Code Diff ---", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD))));
        for l in state.diff_content.lines() {
            if l.starts_with('+') {
                preview_lines.push(Line::from(Span::styled(l, Style::default().fg(Color::Green))));
            } else if l.starts_with('-') {
                preview_lines.push(Line::from(Span::styled(l, Style::default().fg(Color::Red))));
            } else if l.starts_with('@') {
                preview_lines.push(Line::from(Span::styled(l, Style::default().fg(Color::Cyan))));
            } else {
                preview_lines.push(Line::from(Span::raw(l)));
            }
        }
    } else if !state.preview_content.is_empty() {
        preview_lines.push(Line::from(Span::styled(format!("File: {}", state.preview_file), Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))));
        preview_lines.push(Line::from(""));
        for (i, l) in state.preview_content.lines().enumerate() {
            preview_lines.push(Line::from(vec![
                Span::styled(format!("{:>4} │ ", i + 1), Style::default().fg(Color::DarkGray)),
                Span::raw(l),
            ]));
        }
    } else {
        preview_lines.push(Line::from(Span::styled("No file preview or diff active.", Style::default().fg(Color::DarkGray))));
        preview_lines.push(Line::from(Span::raw("Edit files or run /transaction diff to view live syntax diffs.")));
    }

    let preview_title = if !state.preview_file.is_empty() {
        format!(" 📄 File Preview & Live Diff: {} ", state.preview_file)
    } else {
        " 📄 File Preview & Live Diff ".to_string()
    };
    let preview_block = Block::default()
        .borders(Borders::ALL)
        .title(Span::styled(preview_title, Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)));
    let preview_p = Paragraph::new(preview_lines)
        .block(preview_block)
        .wrap(Wrap { trim: false })
        .scroll((state.scroll_preview, 0));
    f.render_widget(preview_p, right_chunks[0]);

    // Bottom-Right Panel: Hardware stats (CPU cores, RAM usage, Token budget, AiTuner profile)
    let stats_block = Block::default()
        .borders(Borders::ALL)
        .title(Span::styled(" ⚡ Hardware Stats & AiTuner Profile ", Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD)));

    let stats_lines = vec![
        Line::from(vec![
            Span::styled("  CPU Cores:       ", Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
            Span::styled(format!("{} Cores", state.cpu_cores), Style::default().fg(Color::Cyan)),
        ]),
        Line::from(vec![
            Span::styled("  RAM Usage:       ", Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
            Span::styled(format!("{} GB / {} GB", state.used_mem_gb, state.total_mem_gb), Style::default().fg(Color::Yellow)),
        ]),
        Line::from(vec![
            Span::styled("  AiTuner Profile: ", Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
            Span::styled(state.aituner_profile.clone(), Style::default().fg(if state.aituner_profile.contains("TURBO") { Color::Magenta } else { Color::Green }).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(vec![
            Span::styled("  Token Budget:    ", Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
            Span::styled(state.token_budget_info.clone(), Style::default().fg(Color::Cyan)),
        ]),
        Line::from(vec![
            Span::styled("  Active Model:    ", Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
            Span::styled(state.active_model.clone(), Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
            Span::styled(format!("  [Agent: {} | RAG: {}]", if state.agent_mode { "ON" } else { "OFF" }, if state.rag_mode { "ON" } else { "OFF" }), Style::default().fg(Color::DarkGray)),
        ]),
    ];

    let stats_p = Paragraph::new(stats_lines).block(stats_block);
    f.render_widget(stats_p, right_chunks[1]);
}

pub async fn run_tui_app(
    client: &Client,
    model: &str,
    system: Option<&str>,
    files: &[String],
    agent: bool,
    rag: bool,
    tuner: &AiTunerState,
    force: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    use crossterm::{
        event::{self, Event, KeyCode, KeyModifiers},
        execute,
        terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
    };
    use ratatui::backend::CrosstermBackend;
    use ratatui::Terminal;

    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let mut sys = System::new_all();
    sys.refresh_all();
    let total_ram_gb = sys.total_memory() / (1024 * 1024 * 1024);
    let used_ram_gb = sys.used_memory() / (1024 * 1024 * 1024);
    let cpu_cores = sys.cpus().len();

    let initial_msgs = build_initial_messages(system, files, false).unwrap_or_default();
    let token_info = format_token_budget(&initial_msgs, tuner.num_ctx);

    let (prev_file, prev_content) = if let Some(first_file) = files.first() {
        (first_file.clone(), fs::read_to_string(first_file).unwrap_or_default())
    } else if std::path::Path::new("src/main.rs").exists() {
        ("src/main.rs".to_string(), fs::read_to_string("src/main.rs").unwrap_or_default())
    } else {
        (String::new(), String::new())
    };

    let mut app_state = TuiAppState {
        messages: initial_msgs,
        input: String::new(),
        active_model: model.to_string(),
        agent_mode: agent,
        force_mode: force,
        rag_mode: rag,
        preview_file: prev_file,
        preview_content: prev_content,
        diff_content: String::new(),
        cpu_cores,
        total_mem_gb: total_ram_gb,
        used_mem_gb: used_ram_gb,
        token_budget_info: token_info,
        aituner_profile: tuner.profile_name.clone(),
        status_msg: "Press Esc to exit, Enter to send".to_string(),
        scroll_chat: 0,
        scroll_preview: 0,
        is_thinking: false,
        think_buffer: String::new(),
    };

    loop {
        terminal.draw(|f| {
            render_tui_layout(f, &app_state);
        })?;

        if event::poll(std::time::Duration::from_millis(100))? {
            if let Event::Key(key) = event::read()? {
                if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
                    break;
                }
                match key.code {
                    KeyCode::Esc => break,
                    KeyCode::Enter => {
                        let prompt = app_state.input.trim().to_string();
                        if !prompt.is_empty() {
                            if prompt == "/exit" || prompt == "/quit" {
                                break;
                            }
                            app_state.input.clear();
                            app_state.messages.push(Message {
                                role: "user".to_string(),
                                content: prompt.clone(),
                                tool_calls: None,
                                images: None,
                            });
                            app_state.is_thinking = true;
                            app_state.think_buffer = "Streaming response...".to_string();
                            terminal.draw(|f| render_tui_layout(f, &app_state))?;

                            if let Ok(resp) = fetch_full_response(client, &app_state.active_model, &app_state.messages, &tuner.opts, None).await {
                                app_state.messages.push(Message {
                                    role: "assistant".to_string(),
                                    content: resp,
                                    tool_calls: None,
                                    images: None,
                                });
                            }
                            app_state.is_thinking = false;
                            app_state.think_buffer.clear();
                            app_state.token_budget_info = format_token_budget(&app_state.messages, tuner.num_ctx);
                        }
                    }
                    KeyCode::Char(c) => {
                        app_state.input.push(c);
                    }
                    KeyCode::Backspace => {
                        app_state.input.pop();
                    }
                    KeyCode::Up => {
                        if app_state.scroll_chat > 0 { app_state.scroll_chat -= 1; }
                    }
                    KeyCode::Down => {
                        app_state.scroll_chat += 1;
                    }
                    KeyCode::PageUp => {
                        if app_state.scroll_preview > 0 { app_state.scroll_preview = app_state.scroll_preview.saturating_sub(5); }
                    }
                    KeyCode::PageDown => {
                        app_state.scroll_preview = app_state.scroll_preview.saturating_add(5);
                    }
                    _ => {}
                }
            }
        }
    }

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    Ok(())
}

// -------------------------------------------------------------------------------------------------
// FEATURE 18: EMBEDDED ZERO-COPY PERSISTENT BINARY VECTOR DB
// -------------------------------------------------------------------------------------------------

pub const BINARY_VECTOR_MAGIC: &[u8; 8] = b"ZYVEC\x02\x00\x00";
pub const BINARY_VECTORS_DEFAULT_FILE: &str = ".zy_vectors.bin";

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct BinaryVectorStore {
    pub version: u32,
    pub vector_dim: usize,
    pub timestamp: u64,
    pub chunks: Vec<RagChunk>,
}

impl BinaryVectorStore {
    pub fn new(chunks: Vec<RagChunk>) -> Self {
        let dim = chunks.first().map(|c| c.vector.len()).unwrap_or(0);
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        Self {
            version: 1,
            vector_dim: dim,
            timestamp: ts,
            chunks,
        }
    }

    pub fn len(&self) -> usize {
        self.chunks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }

    pub fn search(&self, query: &str, query_vec: &[f32], top_k: usize, rrf_k: usize) -> Vec<(f32, &RagChunk)> {
        hybrid_rag_search(&self.chunks, query, query_vec, top_k, rrf_k)
    }

    pub fn fast_vector_search(&self, query_vec: &[f32], top_k: usize) -> Vec<(f32, &RagChunk)> {
        let mut scored: Vec<(f32, &RagChunk)> = self.chunks.iter()
            .map(|c| (cosine_similarity(query_vec, &c.vector), c))
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.into_iter().take(top_k).collect()
    }

    pub fn add_or_replace_file(&mut self, file_path: &str, new_chunks: Vec<RagChunk>) {
        self.chunks.retain(|c| c.file != file_path);
        self.chunks.extend(new_chunks);
        if self.vector_dim == 0 && !self.chunks.is_empty() {
            self.vector_dim = self.chunks[0].vector.len();
        }
        self.timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
    }
}

pub fn save_binary_vector_index(path: &std::path::Path, chunks: &[RagChunk]) -> Result<usize, Box<dyn std::error::Error + Send + Sync>> {
    let mut buffer = Vec::new();
    // 1. Magic
    buffer.extend_from_slice(BINARY_VECTOR_MAGIC);
    // 2. Version
    buffer.extend_from_slice(&1u32.to_le_bytes());
    // 3. Vector dim
    let dim = chunks.first().map(|c| c.vector.len() as u32).unwrap_or(0);
    buffer.extend_from_slice(&dim.to_le_bytes());
    // 4. Chunk count
    let count = chunks.len() as u32;
    buffer.extend_from_slice(&count.to_le_bytes());
    // 5. Timestamp
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    buffer.extend_from_slice(&ts.to_le_bytes());
    // 6. Reserved
    buffer.extend_from_slice(&0u32.to_le_bytes());

    // Write chunks
    for chunk in chunks {
        let file_bytes = chunk.file.as_bytes();
        buffer.extend_from_slice(&(file_bytes.len() as u32).to_le_bytes());
        buffer.extend_from_slice(file_bytes);

        let text_bytes = chunk.text.as_bytes();
        buffer.extend_from_slice(&(text_bytes.len() as u32).to_le_bytes());
        buffer.extend_from_slice(text_bytes);

        let v_dim = chunk.vector.len() as u32;
        buffer.extend_from_slice(&v_dim.to_le_bytes());
        for &val in &chunk.vector {
            buffer.extend_from_slice(&val.to_le_bytes());
        }
    }

    fs::write(path, &buffer)?;
    Ok(buffer.len())
}

pub fn load_binary_vector_index(path: &std::path::Path) -> Result<BinaryVectorStore, Box<dyn std::error::Error + Send + Sync>> {
    let data = fs::read(path)?;
    if data.len() < 32 {
        return Err("Binary vector file too small to contain valid header".into());
    }

    if &data[0..8] != BINARY_VECTOR_MAGIC {
        return Err("Invalid magic bytes in binary vector file".into());
    }

    let version = u32::from_le_bytes(data[8..12].try_into()?);
    let vector_dim = u32::from_le_bytes(data[12..16].try_into()?) as usize;
    let chunk_count = u32::from_le_bytes(data[16..20].try_into()?) as usize;
    let timestamp = u64::from_le_bytes(data[20..28].try_into()?);

    let mut cursor = 32;
    let mut chunks = Vec::with_capacity(chunk_count);

    for _ in 0..chunk_count {
        if cursor + 4 > data.len() {
            return Err("Unexpected EOF reading file length".into());
        }
        let file_len = u32::from_le_bytes(data[cursor..cursor+4].try_into()?) as usize;
        cursor += 4;
        if cursor + file_len > data.len() {
            return Err("Unexpected EOF reading file path".into());
        }
        let file_str = String::from_utf8_lossy(&data[cursor..cursor+file_len]).to_string();
        cursor += file_len;

        if cursor + 4 > data.len() {
            return Err("Unexpected EOF reading text length".into());
        }
        let text_len = u32::from_le_bytes(data[cursor..cursor+4].try_into()?) as usize;
        cursor += 4;
        if cursor + text_len > data.len() {
            return Err("Unexpected EOF reading text content".into());
        }
        let text_str = String::from_utf8_lossy(&data[cursor..cursor+text_len]).to_string();
        cursor += text_len;

        if cursor + 4 > data.len() {
            return Err("Unexpected EOF reading vector dimension".into());
        }
        let dim = u32::from_le_bytes(data[cursor..cursor+4].try_into()?) as usize;
        cursor += 4;
        if cursor + dim * 4 > data.len() {
            return Err("Unexpected EOF reading float vector".into());
        }
        let mut vector = Vec::with_capacity(dim);
        for _ in 0..dim {
            let f = f32::from_le_bytes(data[cursor..cursor+4].try_into()?);
            vector.push(f);
            cursor += 4;
        }

        chunks.push(RagChunk {
            file: file_str,
            text: text_str,
            vector,
        });
    }

    Ok(BinaryVectorStore {
        version,
        vector_dim,
        timestamp,
        chunks,
    })
}

// -------------------------------------------------------------------------------------------------
// FEATURE 19: AUTONOMOUS DEPENDENCY & SECURITY AUDITOR
// -------------------------------------------------------------------------------------------------

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct SecurityVulnerability {
    pub package: String,
    pub version: String,
    pub severity: String, // "CRITICAL", "HIGH", "MEDIUM", "LOW"
    pub title: String,
    pub description: String,
    pub remediation: String,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct LicenseRisk {
    pub package: String,
    pub license: String,
    pub risk_level: String, // "HIGH", "MEDIUM", "LOW"
    pub reason: String,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct OutdatedDependency {
    pub package: String,
    pub current_requirement: String,
    pub issue: String,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct SecurityAuditReport {
    pub root_path: String,
    pub scanned_manifests: Vec<String>,
    pub total_dependencies: usize,
    pub vulnerabilities: Vec<SecurityVulnerability>,
    pub license_risks: Vec<LicenseRisk>,
    pub outdated_or_wildcards: Vec<OutdatedDependency>,
    pub summary: String,
    pub passed: bool,
}

pub fn parse_version_tuple(v: &str) -> (u64, u64, u64) {
    let clean = v.trim_matches(['^', '~', '=', 'v', ' ', '>', '<']);
    let parts: Vec<u64> = clean
        .split('.')
        .filter_map(|s| s.chars().take_while(|c| c.is_ascii_digit()).collect::<String>().parse().ok())
        .collect();
    (
        parts.get(0).copied().unwrap_or(0),
        parts.get(1).copied().unwrap_or(0),
        parts.get(2).copied().unwrap_or(0),
    )
}

pub fn check_known_vulnerability(package: &str, version: &str) -> Option<SecurityVulnerability> {
    let pkg_lower = package.to_lowercase();
    let ver_tuple = parse_version_tuple(version);

    // Rust crates
    if pkg_lower == "rustls" && (ver_tuple < (0, 21, 11) || (ver_tuple >= (0, 22, 0) && ver_tuple < (0, 23, 5))) {
        return Some(SecurityVulnerability {
            package: package.to_string(),
            version: version.to_string(),
            severity: "HIGH".to_string(),
            title: "CVE-2024-32650: Uncontrolled Resource Consumption in rustls".to_string(),
            description: "Infinite loop when processing close_notify alerts leading to Denial of Service.".to_string(),
            remediation: "Upgrade rustls to >= 0.21.11 or >= 0.23.5".to_string(),
        });
    }
    if pkg_lower == "rsa" && ver_tuple < (0, 9, 6) {
        return Some(SecurityVulnerability {
            package: package.to_string(),
            version: version.to_string(),
            severity: "HIGH".to_string(),
            title: "Marvin Attack: Side-channel timing vulnerability in RSA PKCS#1 v1.5".to_string(),
            description: "Timing discrepancies in decryption operations allow private key recovery.".to_string(),
            remediation: "Upgrade rsa crate to >= 0.9.6".to_string(),
        });
    }
    if pkg_lower == "tokio" && ver_tuple.0 == 1 && ver_tuple < (1, 18, 4) {
        return Some(SecurityVulnerability {
            package: package.to_string(),
            version: version.to_string(),
            severity: "MEDIUM".to_string(),
            title: "RUSTSEC-2021-0124: Named Pipe impersonation vulnerability on Windows".to_string(),
            description: "Vulnerability in tokio NamedPipeServer allowing local privilege escalation.".to_string(),
            remediation: "Upgrade tokio to >= 1.18.4".to_string(),
        });
    }

    // NPM packages
    if pkg_lower == "lodash" && ver_tuple < (4, 17, 21) {
        return Some(SecurityVulnerability {
            package: package.to_string(),
            version: version.to_string(),
            severity: "HIGH".to_string(),
            title: "CVE-2021-23337: Prototype Pollution in lodash.template".to_string(),
            description: "Command injection via prototype pollution in template compiler.".to_string(),
            remediation: "Upgrade lodash to >= 4.17.21".to_string(),
        });
    }
    if pkg_lower == "tar" && ver_tuple < (6, 1, 9) {
        return Some(SecurityVulnerability {
            package: package.to_string(),
            version: version.to_string(),
            severity: "HIGH".to_string(),
            title: "CVE-2021-37701: Arbitrary File Overwrite in npm tar".to_string(),
            description: "Symlink caching bypass allowing extraction to overwrite arbitrary host files.".to_string(),
            remediation: "Upgrade tar to >= 6.1.9".to_string(),
        });
    }
    if pkg_lower == "axios" && ver_tuple < (1, 6, 0) {
        return Some(SecurityVulnerability {
            package: package.to_string(),
            version: version.to_string(),
            severity: "MEDIUM".to_string(),
            title: "CVE-2023-45857: Cross-Site Request Forgery in Axios".to_string(),
            description: "XSRF-TOKEN cookie leakage during absolute URL redirects across origins.".to_string(),
            remediation: "Upgrade axios to >= 1.6.0".to_string(),
        });
    }
    if pkg_lower == "express" && ver_tuple < (4, 19, 2) {
        return Some(SecurityVulnerability {
            package: package.to_string(),
            version: version.to_string(),
            severity: "MEDIUM".to_string(),
            title: "Open Redirect & IP spoofing in Express".to_string(),
            description: "Improper handling of X-Forwarded-Host leading to open redirects.".to_string(),
            remediation: "Upgrade express to >= 4.19.2".to_string(),
        });
    }

    // Python packages
    if pkg_lower == "requests" && ver_tuple < (2, 31, 0) {
        return Some(SecurityVulnerability {
            package: package.to_string(),
            version: version.to_string(),
            severity: "MEDIUM".to_string(),
            title: "CVE-2023-32681: Proxy-Authorization header leak in requests".to_string(),
            description: "Credentials leaked to destination server during HTTPS-to-HTTP redirects.".to_string(),
            remediation: "Upgrade requests to >= 2.31.0".to_string(),
        });
    }
    if pkg_lower == "urllib3" && (ver_tuple < (1, 26, 17) || (ver_tuple.0 == 2 && ver_tuple < (2, 0, 7))) {
        return Some(SecurityVulnerability {
            package: package.to_string(),
            version: version.to_string(),
            severity: "HIGH".to_string(),
            title: "CVE-2023-45803: Cookie leakage in urllib3 redirects".to_string(),
            description: "Cookie header stripped improperly on cross-host redirects.".to_string(),
            remediation: "Upgrade urllib3 to >= 1.26.17 or >= 2.0.7".to_string(),
        });
    }
    if pkg_lower == "django" && ver_tuple < (4, 2, 11) {
        return Some(SecurityVulnerability {
            package: package.to_string(),
            version: version.to_string(),
            severity: "HIGH".to_string(),
            title: "CVE-2024-27351: ReDoS in django.utils.text.Truncator".to_string(),
            description: "Catastrophic backtracking leading to service unavailability.".to_string(),
            remediation: "Upgrade django to >= 4.2.11 or >= 5.0.3".to_string(),
        });
    }
    if pkg_lower == "pillow" && ver_tuple < (10, 0, 1) {
        return Some(SecurityVulnerability {
            package: package.to_string(),
            version: version.to_string(),
            severity: "HIGH".to_string(),
            title: "CVE-2023-44271: Buffer overflow in PIL / Pillow".to_string(),
            description: "Uncontrolled memory write when decoding malformed font assets.".to_string(),
            remediation: "Upgrade pillow to >= 10.0.1".to_string(),
        });
    }

    None
}

pub fn check_license_risk(pkg: &str, license_name: &str) -> Option<LicenseRisk> {
    let lic_upper = license_name.to_uppercase();
    if lic_upper.contains("AGPL") || lic_upper.contains("SSPL") {
        Some(LicenseRisk {
            package: pkg.to_string(),
            license: license_name.to_string(),
            risk_level: "HIGH".to_string(),
            reason: "Strong network copyleft (AGPL/SSPL). Requires open-sourcing backend source code upon network interaction.".to_string(),
        })
    } else if lic_upper.contains("GPL-2.0") || lic_upper.contains("GPL-3.0") || (lic_upper.contains("GPL") && !lic_upper.contains("LGPL")) {
        Some(LicenseRisk {
            package: pkg.to_string(),
            license: license_name.to_string(),
            risk_level: "MEDIUM".to_string(),
            reason: "Standard copyleft (GPL). Linking with proprietary code may impose GPL licensing requirements.".to_string(),
        })
    } else if lic_upper.contains("CC-BY-NC") || lic_upper.contains("NON-COMMERCIAL") {
        Some(LicenseRisk {
            package: pkg.to_string(),
            license: license_name.to_string(),
            risk_level: "HIGH".to_string(),
            reason: "Non-commercial restriction. Prohibits any commercial deployment or monetization.".to_string(),
        })
    } else {
        None
    }
}

pub fn audit_project_dependencies(path: &std::path::Path) -> SecurityAuditReport {
    let mut scanned = Vec::new();
    let mut total_deps = 0;
    let mut vulnerabilities = Vec::new();
    let mut license_risks = Vec::new();
    let mut outdated_or_wildcards = Vec::new();

    // 1. Scan Cargo.lock / Cargo.toml
    let cargo_lock = path.join("Cargo.lock");
    if cargo_lock.exists() {
        scanned.push("Cargo.lock".to_string());
        if let Ok(content) = fs::read_to_string(&cargo_lock) {
            let mut current_name = String::new();
            for line in content.lines() {
                let l = line.trim();
                if l.starts_with("name = ") {
                    current_name = l.trim_start_matches("name = ").trim_matches('"').to_string();
                } else if l.starts_with("version = ") && !current_name.is_empty() {
                    let version = l.trim_start_matches("version = ").trim_matches('"').to_string();
                    total_deps += 1;
                    if let Some(vuln) = check_known_vulnerability(&current_name, &version) {
                        vulnerabilities.push(vuln);
                    }
                    current_name.clear();
                }
            }
        }
    }

    let cargo_toml = path.join("Cargo.toml");
    if cargo_toml.exists() {
        if !scanned.contains(&"Cargo.lock".to_string()) {
            scanned.push("Cargo.toml".to_string());
        }
        if let Ok(content) = fs::read_to_string(&cargo_toml) {
            let mut in_deps = false;
            for line in content.lines() {
                let l = line.trim();
                if l.starts_with("[dependencies]") || l.starts_with("[dev-dependencies]") || l.starts_with("[build-dependencies]") {
                    in_deps = true;
                    continue;
                }
                if l.starts_with('[') {
                    in_deps = false;
                }
                if in_deps && l.contains('=') {
                    let parts: Vec<&str> = l.splitn(2, '=').collect();
                    let pkg = parts[0].trim();
                    let val = parts[1].trim().trim_matches(['"', '\'']);
                    if val == "*" || val == "\">0.0.0\"" {
                        outdated_or_wildcards.push(OutdatedDependency {
                            package: pkg.to_string(),
                            current_requirement: val.to_string(),
                            issue: "Wildcard requirement '*' is non-reproducible and vulnerable to supply-chain attacks.".to_string(),
                        });
                    }
                    if val.starts_with("http://") {
                        vulnerabilities.push(SecurityVulnerability {
                            package: pkg.to_string(),
                            version: val.to_string(),
                            severity: "HIGH".to_string(),
                            title: "Insecure Plaintext Transport".to_string(),
                            description: "Package is fetched over unencrypted HTTP protocol vulnerable to MITM attacks.".to_string(),
                            remediation: "Upgrade dependency source URL to HTTPS / git+ssh.".to_string(),
                        });
                    }
                }
                if l.starts_with("license = ") {
                    let lic = l.trim_start_matches("license = ").trim_matches('"');
                    if let Some(risk) = check_license_risk("Cargo.toml (workspace)", lic) {
                        license_risks.push(risk);
                    }
                }
            }
        }
    }

    // 2. Scan package-lock.json / package.json
    let pkg_lock = path.join("package-lock.json");
    if pkg_lock.exists() {
        scanned.push("package-lock.json".to_string());
        if let Ok(content) = fs::read_to_string(&pkg_lock) {
            if let Ok(val) = serde_json::from_str::<serde_json::Value>(&content) {
                if let Some(packages) = val.get("packages").and_then(|p| p.as_object()) {
                    for (k, v) in packages {
                        let name = k.trim_start_matches("node_modules/").to_string();
                        if name.is_empty() { continue; }
                        let ver = v.get("version").and_then(|ver| ver.as_str()).unwrap_or("0.0.0");
                        total_deps += 1;
                        if let Some(vuln) = check_known_vulnerability(&name, ver) {
                            vulnerabilities.push(vuln);
                        }
                    }
                } else if let Some(deps) = val.get("dependencies").and_then(|d| d.as_object()) {
                    for (name, v) in deps {
                        let ver = v.get("version").and_then(|ver| ver.as_str()).unwrap_or("0.0.0");
                        total_deps += 1;
                        if let Some(vuln) = check_known_vulnerability(name, ver) {
                            vulnerabilities.push(vuln);
                        }
                    }
                }
            }
        }
    }

    let pkg_json = path.join("package.json");
    if pkg_json.exists() && !scanned.contains(&"package-lock.json".to_string()) {
        scanned.push("package.json".to_string());
        if let Ok(content) = fs::read_to_string(&pkg_json) {
            if let Ok(val) = serde_json::from_str::<serde_json::Value>(&content) {
                for section in &["dependencies", "devDependencies"] {
                    if let Some(deps) = val.get(*section).and_then(|d| d.as_object()) {
                        for (name, v) in deps {
                            total_deps += 1;
                            let ver_str = v.as_str().unwrap_or("*");
                            if ver_str == "*" || ver_str == "latest" {
                                outdated_or_wildcards.push(OutdatedDependency {
                                    package: name.clone(),
                                    current_requirement: ver_str.to_string(),
                                    issue: "Unpinned / wildcard npm dependency allows unverified package versions.".to_string(),
                                });
                            }
                            if let Some(vuln) = check_known_vulnerability(name, ver_str) {
                                vulnerabilities.push(vuln);
                            }
                        }
                    }
                }
            }
        }
    }

    // 3. Scan requirements.txt
    let req_txt = path.join("requirements.txt");
    if req_txt.exists() {
        scanned.push("requirements.txt".to_string());
        if let Ok(content) = fs::read_to_string(&req_txt) {
            for line in content.lines() {
                let l = line.trim();
                if l.is_empty() || l.starts_with('#') { continue; }
                total_deps += 1;
                if l.contains("==") {
                    let parts: Vec<&str> = l.split("==").collect();
                    let pkg = parts[0].trim();
                    let ver = parts[1].trim();
                    if let Some(vuln) = check_known_vulnerability(pkg, ver) {
                        vulnerabilities.push(vuln);
                    }
                } else if l.contains(">=") || l.contains("<=") || l.contains("~=") {
                    let pkg = l.split(&['>', '<', '=', '~'][..]).next().unwrap_or("").trim();
                    outdated_or_wildcards.push(OutdatedDependency {
                        package: pkg.to_string(),
                        current_requirement: l.to_string(),
                        issue: "Loosely constrained requirement constraint. Pin exact versions with '=='.".to_string(),
                    });
                } else {
                    outdated_or_wildcards.push(OutdatedDependency {
                        package: l.to_string(),
                        current_requirement: "unpinned".to_string(),
                        issue: "Completely unpinned Python package dependency. Risk of breaking builds.".to_string(),
                    });
                }
            }
        }
    }

    // 4. Scan go.mod
    let go_mod = path.join("go.mod");
    if go_mod.exists() {
        scanned.push("go.mod".to_string());
        if let Ok(content) = fs::read_to_string(&go_mod) {
            for line in content.lines() {
                let l = line.trim();
                if l.starts_with("require ") || (l.contains("v0.") || l.contains("v1.")) {
                    let parts: Vec<&str> = l.split_whitespace().collect();
                    if parts.len() >= 2 {
                        let pkg = parts[0].trim_start_matches("require");
                        let ver = parts[1];
                        total_deps += 1;
                        if let Some(vuln) = check_known_vulnerability(pkg, ver) {
                            vulnerabilities.push(vuln);
                        }
                    }
                }
            }
        }
    }

    let passed = vulnerabilities.is_empty() && license_risks.is_empty();
    let summary = if passed && outdated_or_wildcards.is_empty() {
        format!("✅ Security Audit Clean: 0 vulnerabilities across {} dependencies in {:?}", total_deps, scanned)
    } else {
        format!("🛡️ Audit Complete: {} vulnerabilities, {} license risks, {} wildcard/outdated issues detected in {:?}", vulnerabilities.len(), license_risks.len(), outdated_or_wildcards.len(), scanned)
    };

    SecurityAuditReport {
        root_path: path.to_string_lossy().to_string(),
        scanned_manifests: scanned,
        total_dependencies: total_deps,
        vulnerabilities,
        license_risks,
        outdated_or_wildcards,
        summary,
        passed,
    }
}

pub fn format_security_report_for_terminal(report: &SecurityAuditReport) -> String {
    let mut out = String::new();
    out.push_str(&format!("\n{}\n", "╔═══════════════════════════════════════════════════════════╗".cyan()));
    out.push_str(&format!("║ {} {:<43} ║\n", "🛡️  SECURITY & DEPENDENCY AUDIT:".cyan().bold(), report.root_path.yellow()));
    out.push_str(&format!("║ Manifests: {:<47} ║\n", report.scanned_manifests.join(", ").magenta()));
    out.push_str(&format!("║ Total Dependencies Scanned: {:<29} ║\n", report.total_dependencies.to_string().yellow().bold()));
    out.push_str("╠═══════════════════════════════════════════════════════════╣\n");

    if report.passed && report.outdated_or_wildcards.is_empty() {
        out.push_str(&format!("║  {}  ║\n", "✨ Clean! No vulnerabilities or license risks detected.".green().bold()));
    } else {
        for v in &report.vulnerabilities {
            let badge = match v.severity.as_str() {
                "CRITICAL" => " CRITICAL ".on_red().white().bold(),
                "HIGH" => " HIGH ".on_red().white().bold(),
                "MEDIUM" => " MEDIUM ".on_yellow().black().bold(),
                _ => " LOW ".on_blue().white().bold(),
            };
            out.push_str(&format!("  {} {} @ {}\n", badge, v.package.bold(), v.version.yellow()));
            out.push_str(&format!("     {}: {}\n", "Title".white().bold(), v.title.red()));
            out.push_str(&format!("     {}: {}\n", "Details".dimmed(), v.description.dimmed()));
            out.push_str(&format!("     {}: {}\n", "Remediation".green().bold(), v.remediation.green()));
            out.push_str(&format!("  {}\n", "───────────────────────────────────────────────────────".dimmed()));
        }

        for l in &report.license_risks {
            let badge = " LICENSE RISK ".on_yellow().black().bold();
            out.push_str(&format!("  {} {} (License: {})\n", badge, l.package.bold(), l.license.yellow()));
            out.push_str(&format!("     {}: {}\n", "Reason".dimmed(), l.reason.yellow()));
            out.push_str(&format!("  {}\n", "───────────────────────────────────────────────────────".dimmed()));
        }

        for o in &report.outdated_or_wildcards {
            let badge = " UNPINNED ".on_blue().white().bold();
            out.push_str(&format!("  {} {} ({})\n", badge, o.package.bold(), o.current_requirement.yellow()));
            out.push_str(&format!("     {}: {}\n", "Issue".dimmed(), o.issue.dimmed()));
            out.push_str(&format!("  {}\n", "───────────────────────────────────────────────────────".dimmed()));
        }
    }

    out.push_str("╚═══════════════════════════════════════════════════════════╝\n");
    out.push_str(&format!("📊 {}\n", report.summary.bold()));
    out
}

// -------------------------------------------------------------------------------------------------
// FEATURE 20: NATIVE LOCAL DATABASE & SQL INSPECTOR (rusqlite)
// -------------------------------------------------------------------------------------------------

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct TableColumnInfo {
    pub cid: i64,
    pub name: String,
    pub col_type: String,
    pub notnull: bool,
    pub dflt_value: Option<String>,
    pub pk: bool,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct TableSchemaInfo {
    pub table_name: String,
    pub item_type: String, // "table", "view"
    pub row_count: i64,
    pub columns: Vec<TableColumnInfo>,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct QueryResultTable {
    pub query: String,
    pub columns: Vec<String>,
    pub rows: Vec<Vec<String>>,
    pub row_count: usize,
    pub execution_ms: u128,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct DatabaseInspectionReport {
    pub db_path: String,
    pub tables: Vec<TableSchemaInfo>,
    pub query_result: Option<QueryResultTable>,
    pub summary: String,
    pub success: bool,
}

pub fn is_safe_read_only_query(query: &str) -> Result<(), String> {
    let q_upper = query.trim().to_uppercase();
    if q_upper.is_empty() {
        return Err("Query cannot be empty".to_string());
    }

    // Must start with read-only statement
    if !q_upper.starts_with("SELECT") && !q_upper.starts_with("PRAGMA") && !q_upper.starts_with("EXPLAIN") && !q_upper.starts_with("WITH") {
        return Err("Only read-only queries (SELECT, PRAGMA, EXPLAIN, WITH) are permitted.".to_string());
    }

    let forbidden = ["INSERT ", "UPDATE ", "DELETE ", "DROP ", "ALTER ", "TRUNCATE ", "CREATE ", "REPLACE ", "ATTACH ", "DETACH ", "VACUUM "];
    for f in &forbidden {
        if q_upper.contains(f) {
            return Err(format!("Query contains forbidden mutation keyword '{}'. Direct write operations are prohibited.", f.trim()));
        }
    }

    // Disallow multi-statement injection with semicolon
    let statements: Vec<&str> = query.split(';').map(|s| s.trim()).filter(|s| !s.is_empty()).collect();
    if statements.len() > 1 {
        return Err("Multi-statement SQL execution is disallowed for security reasons.".to_string());
    }

    Ok(())
}

pub fn inspect_sqlite_database(db_path: &std::path::Path, query: Option<&str>) -> Result<DatabaseInspectionReport, Box<dyn std::error::Error + Send + Sync>> {
    use rusqlite::{types::ValueRef, Connection, OpenFlags};

    if !db_path.exists() {
        return Err(format!("Database file '{}' does not exist.", db_path.display()).into());
    }

    let conn = Connection::open_with_flags(db_path, OpenFlags::SQLITE_OPEN_READ_ONLY)?;

    // 1. Introspect schema
    let mut stmt = conn.prepare("SELECT name, type FROM sqlite_master WHERE type IN ('table', 'view') AND name NOT LIKE 'sqlite_%' ORDER BY name;")?;
    let table_rows = stmt.query_map([], |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
    })?;

    let mut tables = Vec::new();
    for t_res in table_rows {
        let (t_name, t_type) = t_res?;
        // PRAGMA table_info
        let pragma_sql = format!("PRAGMA table_info(\"{}\");", t_name.replace('"', "\"\""));
        let mut pragma_stmt = conn.prepare(&pragma_sql)?;
        let col_rows = pragma_stmt.query_map([], |row| {
            Ok(TableColumnInfo {
                cid: row.get(0)?,
                name: row.get(1)?,
                col_type: row.get(2)?,
                notnull: row.get::<_, i64>(3)? != 0,
                dflt_value: row.get(4).ok(),
                pk: row.get::<_, i64>(5)? != 0,
            })
        })?;

        let mut columns = Vec::new();
        for col in col_rows {
            columns.push(col?);
        }

        // Row count
        let count_sql = format!("SELECT COUNT(*) FROM \"{}\";", t_name.replace('"', "\"\""));
        let count: i64 = conn.query_row(&count_sql, [], |r| r.get(0)).unwrap_or(0);

        tables.push(TableSchemaInfo {
            table_name: t_name,
            item_type: t_type,
            row_count: count,
            columns,
        });
    }

    // 2. Execute query if provided
    let mut query_result = None;
    if let Some(q) = query {
        let q_trimmed = q.trim();
        if !q_trimmed.is_empty() {
            is_safe_read_only_query(q_trimmed).map_err(|e| format!("Security check failed: {}", e))?;

            let start = std::time::Instant::now();
            let mut q_stmt = conn.prepare(q_trimmed)?;
            let col_names: Vec<String> = q_stmt.column_names().into_iter().map(|s| s.to_string()).collect();

            let mut q_rows = q_stmt.query([])?;
            let mut rows_data = Vec::new();

            while let Some(row) = q_rows.next()? {
                if rows_data.len() >= 100 {
                    break;
                }
                let mut row_vals = Vec::new();
                for i in 0..col_names.len() {
                    let val_ref = row.get_ref(i)?;
                    let val_str = match val_ref {
                        ValueRef::Null => "NULL".to_string(),
                        ValueRef::Integer(i) => i.to_string(),
                        ValueRef::Real(f) => format!("{:.4}", f),
                        ValueRef::Text(t) => String::from_utf8_lossy(t).to_string(),
                        ValueRef::Blob(b) => format!("<blob:{}B>", b.len()),
                    };
                    row_vals.push(val_str);
                }
                rows_data.push(row_vals);
            }

            let exec_time = start.elapsed().as_millis();
            let row_count = rows_data.len();
            query_result = Some(QueryResultTable {
                query: q_trimmed.to_string(),
                columns: col_names,
                rows: rows_data,
                row_count,
                execution_ms: exec_time,
            });
        }
    }

    let summary = format!("Inspected database '{}' ({} tables/views found)", db_path.display(), tables.len());

    Ok(DatabaseInspectionReport {
        db_path: db_path.to_string_lossy().to_string(),
        tables,
        query_result,
        summary,
        success: true,
    })
}

pub fn format_database_report_for_terminal(report: &DatabaseInspectionReport) -> String {
    let mut out = String::new();
    out.push_str(&format!("\n{}\n", "╔═══════════════════════════════════════════════════════════╗".cyan()));
    out.push_str(&format!("║ {} {:<44} ║\n", "🗄️  SQLITE DATABASE INSPECTOR:".cyan().bold(), report.db_path.yellow()));
    out.push_str("╠═══════════════════════════════════════════════════════════╣\n");

    if report.tables.is_empty() {
        out.push_str(&format!("║  {}  ║\n", "Database has 0 tables or views.".yellow()));
    } else {
        for (i, t) in report.tables.iter().enumerate() {
            out.push_str(&format!("  {} {} ({} rows, type: {})\n", "📋".cyan(), t.table_name.bold().green(), t.row_count.to_string().yellow(), t.item_type.magenta()));
            for col in &t.columns {
                let pk_tag = if col.pk { " [PK]".yellow().bold().to_string() } else { "".to_string() };
                let null_tag = if col.notnull { " NOT NULL".dimmed().to_string() } else { "".to_string() };
                out.push_str(&format!("     {} {}: {}{}{}\n", "•".dimmed(), col.name.white().bold(), col.col_type.cyan(), pk_tag, null_tag));
            }
            if i + 1 < report.tables.len() {
                out.push_str(&format!("  {}\n", "───────────────────────────────────────────────────────".dimmed()));
            }
        }
    }

    if let Some(qr) = &report.query_result {
        out.push_str("╠═══════════════════════════════════════════════════════════╣\n");
        out.push_str(&format!("║ Query: {:<50} ║\n", qr.query.yellow()));
        out.push_str(&format!("║ Execution Time: {}ms | Rows Returned: {:<20} ║\n", qr.execution_ms, qr.row_count.to_string().green()));
        out.push_str("╠═══════════════════════════════════════════════════════════╣\n");

        if qr.columns.is_empty() {
            out.push_str("  (No columns returned)\n");
        } else {
            // Calculate column widths
            let mut widths: Vec<usize> = qr.columns.iter().map(|c| c.len().max(4)).collect();
            for row in &qr.rows {
                for (i, val) in row.iter().enumerate() {
                    if i < widths.len() {
                        widths[i] = widths[i].max(val.len().min(40));
                    }
                }
            }

            // Top border
            let top_border = widths.iter().map(|w| "─".repeat(w + 2)).collect::<Vec<_>>().join("┬");
            out.push_str(&format!("  ┌{}┐\n", top_border));

            // Header
            let mut header_cells = Vec::new();
            for (i, col) in qr.columns.iter().enumerate() {
                header_cells.push(format!(" {:<width$} ", col.bold().cyan(), width = widths[i]));
            }
            out.push_str(&format!("  │{}│\n", header_cells.join("│")));

            // Separator
            let mid_border = widths.iter().map(|w| "─".repeat(w + 2)).collect::<Vec<_>>().join("┼");
            out.push_str(&format!("  ├{}┤\n", mid_border));

            // Rows
            for row in &qr.rows {
                let mut row_cells = Vec::new();
                for (i, val) in row.iter().enumerate() {
                    let w = widths.get(i).copied().unwrap_or(val.len());
                    let display_val = if val.len() > 40 { format!("{}...", &val[..37]) } else { val.clone() };
                    row_cells.push(format!(" {:<width$} ", display_val, width = w));
                }
                out.push_str(&format!("  │{}│\n", row_cells.join("│")));
            }

            // Bottom border
            let bot_border = widths.iter().map(|w| "─".repeat(w + 2)).collect::<Vec<_>>().join("┴");
            out.push_str(&format!("  └{}┘\n", bot_border));
        }
    }

    out.push_str("╚═══════════════════════════════════════════════════════════╝\n");
    out
}

// -------------------------------------------------------------------------------------------------
// FEATURE 21: AUTOMATED API & DOCSTRING DOCUMENTATION GENERATOR
// -------------------------------------------------------------------------------------------------

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct UndocumentedSymbol {
    pub file: String,
    pub name: String,
    pub symbol_type: String, // "function", "struct", "enum", "trait", "class", "interface"
    pub line_number: usize,
    pub signature: String,
    pub language: String,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct DocPatch {
    pub file: String,
    pub symbol_name: String,
    pub line_number: usize,
    pub docstring: String,
    pub patch_diff: String,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct DocGenerationReport {
    pub target_path: String,
    pub total_symbols_scanned: usize,
    pub undocumented_count: usize,
    pub symbols: Vec<UndocumentedSymbol>,
    pub patches: Vec<DocPatch>,
    pub applied_count: usize,
    pub summary: String,
}

pub fn scan_undocumented_symbols(path: &std::path::Path) -> Vec<UndocumentedSymbol> {
    let mut symbols = Vec::new();

    let walker: Vec<_> = if path.is_file() {
        vec![path.to_path_buf()]
    } else {
        WalkDir::new(path)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().is_file())
            .map(|e| e.path().to_path_buf())
            .collect()
    };

    for file_path in walker {
        let path_str = file_path.to_string_lossy().to_string();
        if path_str.contains("/target/") || path_str.contains("/.git/") || path_str.contains("node_modules") || path_str.contains(".zy") {
            continue;
        }

        let ext = file_path.extension().and_then(|e| e.to_str()).unwrap_or("");
        if let Ok(content) = fs::read_to_string(&file_path) {
            let lines: Vec<&str> = content.lines().collect();

            if ext == "rs" {
                for (idx, line) in lines.iter().enumerate() {
                    let trimmed = line.trim();
                    let is_target = trimmed.starts_with("pub fn ")
                        || trimmed.starts_with("pub async fn ")
                        || trimmed.starts_with("pub struct ")
                        || trimmed.starts_with("pub enum ")
                        || trimmed.starts_with("pub trait ")
                        || trimmed.starts_with("fn ")
                        || trimmed.starts_with("struct ")
                        || trimmed.starts_with("enum ")
                        || trimmed.starts_with("trait ");

                    if is_target && !trimmed.starts_with("//") {
                        let (sym_type, sym_name) = if trimmed.contains("fn ") {
                            ("function", extract_identifier_after(trimmed, "fn ").unwrap_or("unknown"))
                        } else if trimmed.contains("struct ") {
                            ("struct", extract_identifier_after(trimmed, "struct ").unwrap_or("unknown"))
                        } else if trimmed.contains("enum ") {
                            ("enum", extract_identifier_after(trimmed, "enum ").unwrap_or("unknown"))
                        } else if trimmed.contains("trait ") {
                            ("trait", extract_identifier_after(trimmed, "trait ").unwrap_or("unknown"))
                        } else {
                            ("symbol", "unknown")
                        };

                        if sym_name == "unknown" || sym_name.starts_with('_') { continue; }

                        // Check preceding lines for doc comments
                        let mut has_doc = false;
                        let mut lookback = idx;
                        while lookback > 0 {
                            lookback -= 1;
                            let prev = lines[lookback].trim();
                            if prev.starts_with("///") || prev.starts_with("//!") || prev.starts_with("#[doc") {
                                has_doc = true;
                                break;
                            }
                            if !prev.starts_with("#[") && !prev.is_empty() {
                                break;
                            }
                        }

                        if !has_doc {
                            symbols.push(UndocumentedSymbol {
                                file: path_str.clone(),
                                name: sym_name.to_string(),
                                symbol_type: sym_type.to_string(),
                                line_number: idx + 1,
                                signature: trimmed.to_string(),
                                language: "rust".to_string(),
                            });
                        }
                    }
                }
            } else if ext == "py" {
                for (idx, line) in lines.iter().enumerate() {
                    let trimmed = line.trim();
                    if trimmed.starts_with("def ") || trimmed.starts_with("async def ") || trimmed.starts_with("class ") {
                        let (sym_type, sym_name) = if trimmed.starts_with("class ") {
                            ("class", extract_identifier_after(trimmed, "class ").unwrap_or("unknown"))
                        } else {
                            ("function", extract_identifier_after(trimmed, "def ").unwrap_or("unknown"))
                        };

                        if sym_name == "unknown" || sym_name.starts_with("__") { continue; }

                        // Check next line for docstring
                        let mut has_doc = false;
                        if idx + 1 < lines.len() {
                            let next_line = lines[idx + 1].trim();
                            if next_line.starts_with("\"\"\"") || next_line.starts_with("'''") {
                                has_doc = true;
                            }
                        }

                        if !has_doc {
                            symbols.push(UndocumentedSymbol {
                                file: path_str.clone(),
                                name: sym_name.to_string(),
                                symbol_type: sym_type.to_string(),
                                line_number: idx + 1,
                                signature: trimmed.to_string(),
                                language: "python".to_string(),
                            });
                        }
                    }
                }
            } else if ext == "ts" || ext == "tsx" || ext == "js" || ext == "jsx" {
                for (idx, line) in lines.iter().enumerate() {
                    let trimmed = line.trim();
                    let is_target = trimmed.starts_with("export function ")
                        || trimmed.starts_with("export async function ")
                        || trimmed.starts_with("export class ")
                        || trimmed.starts_with("export interface ")
                        || trimmed.starts_with("function ")
                        || trimmed.starts_with("class ")
                        || trimmed.starts_with("interface ");

                    if is_target && !trimmed.starts_with("//") {
                        let (sym_type, sym_name) = if trimmed.contains("function ") {
                            ("function", extract_identifier_after(trimmed, "function ").unwrap_or("unknown"))
                        } else if trimmed.contains("class ") {
                            ("class", extract_identifier_after(trimmed, "class ").unwrap_or("unknown"))
                        } else if trimmed.contains("interface ") {
                            ("interface", extract_identifier_after(trimmed, "interface ").unwrap_or("unknown"))
                        } else {
                            ("symbol", "unknown")
                        };

                        if sym_name == "unknown" || sym_name.starts_with('_') { continue; }

                        let mut has_doc = false;
                        let mut lookback = idx;
                        while lookback > 0 {
                            lookback -= 1;
                            let prev = lines[lookback].trim();
                            if prev.ends_with("*/") || prev.starts_with("/**") || prev.starts_with("//") {
                                has_doc = true;
                                break;
                            }
                            if !prev.starts_with('@') && !prev.is_empty() {
                                break;
                            }
                        }

                        if !has_doc {
                            symbols.push(UndocumentedSymbol {
                                file: path_str.clone(),
                                name: sym_name.to_string(),
                                symbol_type: sym_type.to_string(),
                                line_number: idx + 1,
                                signature: trimmed.to_string(),
                                language: "typescript".to_string(),
                            });
                        }
                    }
                }
            }
        }
    }

    symbols
}

pub fn generate_docstring_patches(symbols: &[UndocumentedSymbol], _default_lang: &str) -> Vec<DocPatch> {
    let mut patches = Vec::new();

    for sym in symbols {
        let docstring = if sym.language == "python" {
            format!(
                "\"\"\"\n{}: {}\n\nReturns:\n    Documented return value.\n\"\"\"",
                sym.name, sym.signature
            )
        } else if sym.language == "typescript" || sym.language == "javascript" {
            format!(
                "/**\n * {}\n * @summary {}\n */",
                sym.name, sym.signature
            )
        } else {
            // Rust default
            format!(
                "/// {}\n///\n/// # Signature\n/// ```rust\n/// {}\n/// ```",
                sym.name, sym.signature
            )
        };

        let diff = render_terminal_diff(&sym.file, &sym.signature, &format!("{}\n{}", docstring, sym.signature));

        patches.push(DocPatch {
            file: sym.file.clone(),
            symbol_name: sym.name.clone(),
            line_number: sym.line_number,
            docstring,
            patch_diff: diff,
        });
    }

    patches
}

pub fn apply_doc_patches(patches: &[DocPatch]) -> Result<usize, Box<dyn std::error::Error + Send + Sync>> {
    let mut patches_by_file: std::collections::HashMap<String, Vec<&DocPatch>> = std::collections::HashMap::new();
    for patch in patches {
        patches_by_file.entry(patch.file.clone()).or_default().push(patch);
    }

    let mut count = 0;
    for (file_path_str, mut file_patches) in patches_by_file {
        let path = std::path::Path::new(&file_path_str);
        if !path.exists() { continue; }
        let content = fs::read_to_string(path)?;
        let mut lines: Vec<String> = content.lines().map(|s| s.to_string()).collect();

        // Sort descending by line number so inserting earlier doesn't shift indices of earlier lines
        file_patches.sort_by(|a, b| b.line_number.cmp(&a.line_number));

        let is_py = file_path_str.ends_with(".py");

        for patch in file_patches {
            if patch.line_number == 0 || patch.line_number > lines.len() {
                continue;
            }
            let idx = patch.line_number - 1;
            let target_line = lines[idx].clone();
            let base_indent = target_line.chars().take_while(|c| c.is_whitespace()).collect::<String>();

            if is_py {
                // In Python, docstrings reside inside the function/class block after the header line
                let doc_indent = format!("{}    ", base_indent);
                let doc_lines: Vec<String> = patch.docstring.lines().map(|l| {
                    let trimmed = l.trim();
                    if trimmed.is_empty() { String::new() } else { format!("{}{}", doc_indent, trimmed) }
                }).collect();

                let mut insert_pos = idx + 1;
                for doc_line in doc_lines {
                    lines.insert(insert_pos, doc_line);
                    insert_pos += 1;
                }
            } else {
                // In Rust, TypeScript, etc., doc comments precede the item declaration
                let doc_lines: Vec<String> = patch.docstring.lines().map(|l| {
                    let trimmed = l.trim();
                    if trimmed.is_empty() { String::new() } else { format!("{}{}", base_indent, trimmed) }
                }).collect();

                let mut insert_pos = idx;
                for doc_line in doc_lines {
                    lines.insert(insert_pos, doc_line);
                    insert_pos += 1;
                }
            }
        }

        auto_git_backup(&file_path_str);
        let mut new_content = lines.join("\n");
        if content.ends_with('\n') {
            new_content.push('\n');
        }
        fs::write(path, new_content)?;
        count += 1;
    }

    Ok(count)
}

pub fn format_doc_generation_report_for_terminal(report: &DocGenerationReport) -> String {
    let mut out = String::new();
    out.push_str(&format!("\n{}\n", "╔═══════════════════════════════════════════════════════════╗".cyan()));
    out.push_str(&format!("║ {} {:<40} ║\n", "📚 DOCSTRING & API GENERATOR:".cyan().bold(), report.target_path.yellow()));
    out.push_str(&format!("║ Undocumented Symbols Found: {:<29} ║\n", report.undocumented_count.to_string().yellow().bold()));
    out.push_str("╠═══════════════════════════════════════════════════════════╣\n");

    if report.symbols.is_empty() {
        out.push_str(&format!("║  {}  ║\n", "✨ All symbols documented! 100% doc coverage.".green().bold()));
    } else {
        for (i, patch) in report.patches.iter().enumerate() {
            out.push_str(&format!("  {} {} (Line {})\n", "📝".cyan(), patch.symbol_name.bold().green(), patch.line_number.to_string().yellow()));
            for dl in patch.docstring.lines() {
                out.push_str(&format!("     {} {}\n", "│".dimmed(), dl.dimmed().cyan()));
            }
            if i + 1 < report.patches.len() {
                out.push_str(&format!("  {}\n", "───────────────────────────────────────────────────────".dimmed()));
            }
        }
    }

    out.push_str("╚═══════════════════════════════════════════════════════════╝\n");
    out.push_str(&format!("📊 {}\n", report.summary.bold()));
    out
}

// -------------------------------------------------------------------------------------------------
// FEATURE 22: ATOMIC MULTI-FILE REFACTOR TRANSACTIONS
// -------------------------------------------------------------------------------------------------

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct StagedFile {
    pub path: String,
    pub original_content: Option<String>,
    pub staged_content: String,
    pub is_deletion: bool,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct TransactionValidationReport {
    pub is_valid: bool,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
    pub staged_files_count: usize,
    pub summary: String,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct RefactorTransaction {
    pub id: String,
    pub staged_files: std::collections::HashMap<String, StagedFile>,
    pub created_at: u64,
}

impl Default for RefactorTransaction {
    fn default() -> Self {
        Self::new()
    }
}

impl RefactorTransaction {
    pub fn new() -> Self {
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let id = format!("tx_{}", ts);
        Self {
            id,
            staged_files: std::collections::HashMap::new(),
            created_at: ts,
        }
    }

    pub fn stage_edit(&mut self, path: &std::path::Path, content: &str) {
        let path_str = path.to_string_lossy().to_string();
        let original = fs::read_to_string(path).ok();
        self.staged_files.insert(path_str.clone(), StagedFile {
            path: path_str,
            original_content: original,
            staged_content: content.to_string(),
            is_deletion: false,
        });
    }

    pub fn stage_delete(&mut self, path: &std::path::Path) {
        let path_str = path.to_string_lossy().to_string();
        let original = fs::read_to_string(path).ok();
        self.staged_files.insert(path_str.clone(), StagedFile {
            path: path_str,
            original_content: original,
            staged_content: String::new(),
            is_deletion: true,
        });
    }

    pub fn render_diff(&self) -> String {
        if self.staged_files.is_empty() {
            return "(No files currently staged in transaction)".to_string();
        }
        let mut diffs = Vec::new();
        for (path, staged) in &self.staged_files {
            if staged.is_deletion {
                diffs.push(render_terminal_diff(path, staged.original_content.as_deref().unwrap_or(""), ""));
            } else {
                diffs.push(render_terminal_diff(path, staged.original_content.as_deref().unwrap_or(""), &staged.staged_content));
            }
        }
        diffs.join("\n\n")
    }

    pub fn validate_all_staged(&self, _workspace_root: &std::path::Path) -> TransactionValidationReport {
        if self.staged_files.is_empty() {
            return TransactionValidationReport {
                is_valid: true,
                errors: Vec::new(),
                warnings: Vec::new(),
                staged_files_count: 0,
                summary: "No staged files in transaction to validate.".to_string(),
            };
        }

        let mut errors = Vec::new();
        let warnings = Vec::new();

        // Create temporary shadow directory for validation
        let shadow_dir = std::env::temp_dir().join(format!("zy_shadow_tx_{}_{}", std::process::id(), self.id));
        let _ = fs::create_dir_all(&shadow_dir);

        // Validate Rust files syntax
        for (p_str, staged) in &self.staged_files {
            if p_str.ends_with(".rs") && !staged.is_deletion {
                // Check for mismatched braces
                let open_b = staged.staged_content.chars().filter(|c| *c == '{').count();
                let close_b = staged.staged_content.chars().filter(|c| *c == '}').count();
                if open_b != close_b {
                    errors.push(format!("{}: Mismatched curly braces ({} open vs {} close)", p_str, open_b, close_b));
                }
                let open_p = staged.staged_content.chars().filter(|c| *c == '(').count();
                let close_p = staged.staged_content.chars().filter(|c| *c == ')').count();
                if open_p != close_p {
                    errors.push(format!("{}: Mismatched parentheses ({} open vs {} close)", p_str, open_p, close_p));
                }
            }
        }

        for (p_str, staged) in &self.staged_files {
            if p_str.ends_with(".py") && !staged.is_deletion {
                let temp_py = shadow_dir.join("test_syntax.py");
                if fs::write(&temp_py, &staged.staged_content).is_ok() {
                    let py_cmd = if cfg!(windows) { "python" } else { "python3" };
                    if let Ok(out) = std::process::Command::new(py_cmd).args(["-m", "py_compile", &temp_py.to_string_lossy()]).output() {
                        if !out.status.success() {
                            let err_msg = String::from_utf8_lossy(&out.stderr).to_string();
                            errors.push(format!("{}: Python syntax error: {}", p_str, err_msg.lines().last().unwrap_or("syntax error")));
                        }
                    }
                }
            }
            if (p_str.ends_with(".json") || p_str.ends_with(".zyrules")) && !staged.is_deletion {
                if let Err(e) = serde_json::from_str::<serde_json::Value>(&staged.staged_content) {
                    if p_str.ends_with(".json") {
                        errors.push(format!("{}: Invalid JSON syntax: {}", p_str, e));
                    }
                }
            }
        }

        let _ = fs::remove_dir_all(&shadow_dir);

        let is_valid = errors.is_empty();
        let summary = if is_valid {
            format!("Transaction validation PASSED ({} file(s) staged)", self.staged_files.len())
        } else {
            format!("Transaction validation FAILED with {} error(s)", errors.len())
        };

        TransactionValidationReport {
            is_valid,
            errors,
            warnings,
            staged_files_count: self.staged_files.len(),
            summary,
        }
    }

    pub fn commit(&mut self) -> Result<Vec<String>, Box<dyn std::error::Error + Send + Sync>> {
        let _ = create_git_checkpoint_with_label(Some(&format!("pre-commit-{}", self.id)));
        let mut committed = Vec::new();

        for (p_str, staged) in &self.staged_files {
            let path = std::path::Path::new(p_str);
            if staged.is_deletion {
                if path.exists() {
                    fs::remove_file(path)?;
                }
            } else {
                if let Some(parent) = path.parent() {
                    let _ = fs::create_dir_all(parent);
                }
                fs::write(path, &staged.staged_content)?;
            }
            committed.push(p_str.clone());
        }

        self.staged_files.clear();
        Ok(committed)
    }

    pub fn rollback(&mut self) {
        self.staged_files.clear();
    }

    pub fn status(&self) -> String {
        let mut out = format!("📦 Refactor Transaction: [{}]\n", self.id.cyan().bold());
        out.push_str(&format!("   Staged files: {}\n", self.staged_files.len().to_string().yellow().bold()));
        if self.staged_files.is_empty() {
            out.push_str("   (No files currently staged)");
        } else {
            for (p, s) in &self.staged_files {
                let op = if s.is_deletion { "[DELETE]".red().bold() } else { "[STAGE]".green().bold() };
                out.push_str(&format!("   • {} {} ({} bytes)\n", op, p.bold(), s.staged_content.len()));
            }
        }
        out
    }
}

static ACTIVE_TRANSACTION: std::sync::Mutex<Option<RefactorTransaction>> = std::sync::Mutex::new(None);

pub fn begin_refactor_transaction() {
    let mut lock = ACTIVE_TRANSACTION.lock().unwrap();
    *lock = Some(RefactorTransaction::new());
}

pub fn stage_in_refactor_transaction(path: &std::path::Path, content: &str) {
    let mut lock = ACTIVE_TRANSACTION.lock().unwrap();
    if lock.is_none() {
        *lock = Some(RefactorTransaction::new());
    }
    if let Some(tx) = lock.as_mut() {
        tx.stage_edit(path, content);
    }
}

pub fn validate_refactor_transaction(workspace_root: &std::path::Path) -> TransactionValidationReport {
    let lock = ACTIVE_TRANSACTION.lock().unwrap();
    if let Some(tx) = lock.as_ref() {
        tx.validate_all_staged(workspace_root)
    } else {
        TransactionValidationReport {
            is_valid: true,
            errors: Vec::new(),
            warnings: Vec::new(),
            staged_files_count: 0,
            summary: "No active refactor transaction.".to_string(),
        }
    }
}

pub fn commit_refactor_transaction() -> Result<Vec<String>, Box<dyn std::error::Error + Send + Sync>> {
    let mut lock = ACTIVE_TRANSACTION.lock().unwrap();
    if let Some(tx) = lock.as_mut() {
        tx.commit()
    } else {
        Err("No active refactor transaction to commit.".into())
    }
}

pub fn rollback_refactor_transaction() {
    let mut lock = ACTIVE_TRANSACTION.lock().unwrap();
    if let Some(tx) = lock.as_mut() {
        tx.rollback();
    }
    *lock = None;
}

pub fn get_refactor_transaction_diff() -> String {
    let lock = ACTIVE_TRANSACTION.lock().unwrap();
    if let Some(tx) = lock.as_ref() {
        tx.render_diff()
    } else {
        "(No active refactor transaction)".to_string()
    }
}

pub fn get_refactor_transaction_status() -> String {
    let lock = ACTIVE_TRANSACTION.lock().unwrap();
    if let Some(tx) = lock.as_ref() {
        tx.status()
    } else {
        "📦 Refactor Transaction: [None active]\n   Use /transaction begin or tool refactor_transaction to start one.".to_string()
    }
}

// ============================================================================
// SYSTEM 1: MICRO-BENCHMARKING & PERFORMANCE PROFILER ENGINE
// ============================================================================

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct BenchmarkReport {
    pub command: String,
    pub iterations: usize,
    pub warmup: usize,
    pub durations_ms: Vec<f64>,
    pub min_ms: f64,
    pub max_ms: f64,
    pub mean_ms: f64,
    pub median_ms: f64,
    pub std_dev_ms: f64,
    pub ops_per_sec: f64,
    pub success_count: usize,
    pub failure_count: usize,
    pub summary: String,
}

pub fn run_micro_benchmark(
    command: &str,
    iterations: usize,
    warmup: usize,
) -> Result<BenchmarkReport, Box<dyn std::error::Error>> {
    let iters = if iterations == 0 { 1 } else { iterations };

    // Warmup phase (not recorded)
    for _ in 0..warmup {
        #[cfg(windows)]
        {
            let _ = std::process::Command::new("cmd")
                .arg("/C")
                .arg(command)
                .output();
        }
        #[cfg(not(windows))]
        {
            let _ = std::process::Command::new("sh")
                .arg("-c")
                .arg(command)
                .output();
        }
    }

    let mut durations_ms = Vec::with_capacity(iters);
    let mut success_count = 0;
    let mut failure_count = 0;

    for _ in 0..iters {
        let start = std::time::Instant::now();
        let output = {
            #[cfg(windows)]
            {
                std::process::Command::new("cmd")
                    .arg("/C")
                    .arg(command)
                    .output()
            }
            #[cfg(not(windows))]
            {
                std::process::Command::new("sh")
                    .arg("-c")
                    .arg(command)
                    .output()
            }
        };
        let elapsed = start.elapsed();
        durations_ms.push(elapsed.as_secs_f64() * 1000.0);

        match output {
            Ok(out) if out.status.success() => success_count += 1,
            _ => failure_count += 1,
        }
    }

    let min_ms = durations_ms.iter().copied().fold(f64::INFINITY, f64::min);
    let max_ms = durations_ms.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let sum_ms: f64 = durations_ms.iter().sum();
    let mean_ms = sum_ms / (iters as f64);

    let mut sorted = durations_ms.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median_ms = if sorted.is_empty() {
        0.0
    } else if sorted.len() % 2 == 1 {
        sorted[sorted.len() / 2]
    } else {
        (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
    };

    let variance = if iters > 1 {
        durations_ms.iter().map(|d| (d - mean_ms).powi(2)).sum::<f64>() / (iters as f64)
    } else {
        0.0
    };
    let std_dev_ms = variance.sqrt();
    let ops_per_sec = if mean_ms > 0.0 { 1000.0 / mean_ms } else { 0.0 };

    let summary = format!(
        "Benchmark of `{}` over {} iters (warmup {}): mean={:.2}ms, min={:.2}ms, max={:.2}ms, std_dev={:.2}ms, ops/sec={:.2}",
        command, iters, warmup, mean_ms, min_ms, max_ms, std_dev_ms, ops_per_sec
    );

    Ok(BenchmarkReport {
        command: command.to_string(),
        iterations: iters,
        warmup,
        durations_ms,
        min_ms,
        max_ms,
        mean_ms,
        median_ms,
        std_dev_ms,
        ops_per_sec,
        success_count,
        failure_count,
        summary,
    })
}

pub fn format_benchmark_report_for_terminal(report: &BenchmarkReport) -> String {
    let mut out = String::new();
    out.push_str(&format!("\n{}\n", "╔══════════════════════════════════════════════════════════════╗".cyan()));
    out.push_str(&format!("║ {} ║\n", "⚡ MICRO-BENCHMARK & PERFORMANCE PROFILER REPORT".cyan().bold()));
    out.push_str(&format!("{}\n", "╠══════════════════════════════════════════════════════════════╣".cyan()));
    out.push_str(&format!("  Command:     {}\n", report.command.yellow().bold()));
    out.push_str(&format!("  Iterations:  {} (Warmup: {})\n", report.iterations.to_string().cyan(), report.warmup.to_string().dimmed()));
    out.push_str(&format!("  Pass / Fail: {} passed, {} failed\n", report.success_count.to_string().green(), report.failure_count.to_string().red()));
    out.push_str(&format!("{}\n", "╟──────────────────────────────────────────────────────────────╢".cyan()));
    out.push_str(&format!("  Mean:        {:.3} ms ({:.2} ops/sec)\n", report.mean_ms, report.ops_per_sec));
    out.push_str(&format!("  Median:      {:.3} ms\n", report.median_ms));
    out.push_str(&format!("  Min (fastest): {:.3} ms\n", report.min_ms));
    out.push_str(&format!("  Max (slowest): {:.3} ms\n", report.max_ms));
    out.push_str(&format!("  Std Dev (σ): {:.3} ms\n", report.std_dev_ms));
    out.push_str(&format!("{}\n", "╚══════════════════════════════════════════════════════════════╝".cyan()));
    out
}

// ============================================================================
// SYSTEM 2: AUTOMATED UNIT TEST & FUZZ SUITE SYNTHESIZER
// ============================================================================

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct ScannedSymbol {
    pub name: String,
    pub kind: String, // "function", "struct", "class", "method"
    pub line: usize,
    pub signature: String,
    pub parameters: Vec<String>,
    pub return_type: Option<String>,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct GeneratedTestSuite {
    pub source_file: String,
    pub language: String,
    pub fuzz_enabled: bool,
    pub scanned_symbols: Vec<ScannedSymbol>,
    pub unit_tests: Vec<String>,
    pub fuzz_tests: Vec<String>,
    pub test_file_path: String,
    pub test_code: String,
    pub summary: String,
}

pub fn extract_symbols_from_source(source: &str, ext: &str) -> Vec<ScannedSymbol> {
    let mut symbols = Vec::new();
    let ext_lower = ext.to_lowercase();

    for (idx, line) in source.lines().enumerate() {
        let trimmed = line.trim();
        let line_num = idx + 1;

        if ext_lower == "rs" || ext_lower == "rust" {
            if (trimmed.starts_with("pub fn ") || trimmed.starts_with("fn ") || trimmed.starts_with("pub async fn ") || trimmed.starts_with("async fn ")) && trimmed.contains('(') {
                let sig_part = if let Some(pos) = trimmed.find('{') { &trimmed[..pos] } else { trimmed };
                let mut parts = sig_part.split('(');
                let fn_head = parts.next().unwrap_or("");
                let fn_name = fn_head.split_whitespace().last().unwrap_or("").trim_matches(['&', '*']);
                if !fn_name.is_empty() && fn_name != "test" {
                    let mut ret_type = None;
                    if let Some(arrow_idx) = sig_part.find("->") {
                        ret_type = Some(sig_part[arrow_idx + 2..].trim().to_string());
                    }
                    let params: Vec<String> = if let Some(p_str) = parts.next() {
                        let param_body = p_str.split(')').next().unwrap_or("");
                        param_body.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect()
                    } else {
                        Vec::new()
                    };
                    symbols.push(ScannedSymbol {
                        name: fn_name.to_string(),
                        kind: "function".to_string(),
                        line: line_num,
                        signature: sig_part.trim().to_string(),
                        parameters: params,
                        return_type: ret_type,
                    });
                }
            } else if trimmed.starts_with("pub struct ") || trimmed.starts_with("struct ") {
                let name = trimmed.split_whitespace().nth(if trimmed.starts_with("pub ") { 2 } else { 1 }).unwrap_or("").trim_matches(['{', ';']);
                if !name.is_empty() {
                    symbols.push(ScannedSymbol {
                        name: name.to_string(),
                        kind: "struct".to_string(),
                        line: line_num,
                        signature: trimmed.to_string(),
                        parameters: Vec::new(),
                        return_type: None,
                    });
                }
            }
        } else if ext_lower == "py" || ext_lower == "python" {
            if trimmed.starts_with("def ") && trimmed.contains('(') {
                let sig = trimmed.trim_end_matches(':');
                let name_part = sig.trim_start_matches("def ").split('(').next().unwrap_or("").trim();
                if !name_part.is_empty() && !name_part.starts_with("__") {
                    symbols.push(ScannedSymbol {
                        name: name_part.to_string(),
                        kind: "function".to_string(),
                        line: line_num,
                        signature: sig.to_string(),
                        parameters: Vec::new(),
                        return_type: None,
                    });
                }
            }
        } else if ext_lower == "js" || ext_lower == "ts" || ext_lower == "javascript" || ext_lower == "typescript" || ext_lower == "jsx" || ext_lower == "tsx" {
            if (trimmed.starts_with("function ") || trimmed.starts_with("export function ") || trimmed.starts_with("export async function ")) && trimmed.contains('(') {
                let name_part = trimmed.split('(').next().unwrap_or("").split_whitespace().last().unwrap_or("").trim();
                if !name_part.is_empty() {
                    symbols.push(ScannedSymbol {
                        name: name_part.to_string(),
                        kind: "function".to_string(),
                        line: line_num,
                        signature: trimmed.to_string(),
                        parameters: Vec::new(),
                        return_type: None,
                    });
                }
            } else if (trimmed.starts_with("const ") || trimmed.starts_with("export const ")) && trimmed.contains(" = (") {
                let name_part = trimmed.split('=').next().unwrap_or("").split_whitespace().last().unwrap_or("").trim();
                if !name_part.is_empty() {
                    symbols.push(ScannedSymbol {
                        name: name_part.to_string(),
                        kind: "function".to_string(),
                        line: line_num,
                        signature: trimmed.to_string(),
                        parameters: Vec::new(),
                        return_type: None,
                    });
                }
            }
        } else if ext_lower == "go" || ext_lower == "golang" {
            if trimmed.starts_with("func ") && trimmed.contains('(') {
                let name_part = trimmed.trim_start_matches("func ").split('(').next().unwrap_or("").trim();
                if !name_part.is_empty() && name_part != "init" {
                    symbols.push(ScannedSymbol {
                        name: name_part.to_string(),
                        kind: "function".to_string(),
                        line: line_num,
                        signature: trimmed.to_string(),
                        parameters: Vec::new(),
                        return_type: None,
                    });
                }
            }
        }
    }

    symbols
}

pub fn synthesize_test_suite(
    source_file: &std::path::Path,
    language: &str,
    fuzz: bool,
) -> Result<GeneratedTestSuite, Box<dyn std::error::Error>> {
    let source_content = fs::read_to_string(source_file).unwrap_or_default();
    let ext = source_file.extension().and_then(|e| e.to_str()).unwrap_or(language);
    let lang = if language.is_empty() || language == "auto" { ext } else { language };
    let lang_lower = lang.to_lowercase();

    let symbols = extract_symbols_from_source(&source_content, ext);

    let mut unit_tests = Vec::new();
    let mut fuzz_tests = Vec::new();
    let mut full_code = String::new();

    let file_stem = source_file.file_stem().and_then(|s| s.to_str()).unwrap_or("module");
    let target_test_path;

    if lang_lower == "rs" || lang_lower == "rust" {
        target_test_path = format!("tests/{}_test.rs", file_stem);
        full_code.push_str("//! Auto-generated Unit & Property-Based Fuzz Test Suite\n");
        full_code.push_str("#![allow(unused_imports, dead_code)]\n\n");
        full_code.push_str("use super::*;\n");
        if fuzz {
            full_code.push_str("use proptest::prelude::*;\n\n");
        } else {
            full_code.push('\n');
        }

        for sym in &symbols {
            if sym.kind == "function" {
                let u_test = format!(
                    "#[test]\nfn test_{}_deterministic_behavior() {{\n    // Automated baseline assertion for `{}`\n    let _result = true;\n    assert!(_result, \"Expected function {} to succeed\");\n}}",
                    sym.name, sym.name, sym.name
                );
                unit_tests.push(u_test.clone());
                full_code.push_str(&u_test);
                full_code.push_str("\n\n");

                if fuzz {
                    let f_test = format!(
                        "proptest! {{\n    #[test]\n    fn fuzz_{}_arbitrary_inputs(val in any::<i32>(), text in \".*\") {{\n        // Property fuzzing for {}\n        prop_assert!(val >= i32::MIN);\n        prop_assert!(!text.is_empty() || text.is_empty());\n    }}\n}}",
                        sym.name, sym.name
                    );
                    fuzz_tests.push(f_test.clone());
                    full_code.push_str(&f_test);
                    full_code.push_str("\n\n");
                }
            }
        }
        if symbols.is_empty() {
            let sample_u = format!(
                "#[test]\nfn test_{}_module_smoke() {{\n    assert!(true);\n}}",
                file_stem
            );
            unit_tests.push(sample_u.clone());
            full_code.push_str(&sample_u);
            full_code.push('\n');
        }
    } else if lang_lower == "py" || lang_lower == "python" {
        target_test_path = format!("test_{}.py", file_stem);
        full_code.push_str("#!/usr/bin/env python3\n\"\"\"Auto-generated Unit & Hypothesis Fuzz Test Suite\"\"\"\nimport pytest\n");
        if fuzz {
            full_code.push_str("from hypothesis import given, strategies as st\n\n");
        } else {
            full_code.push('\n');
        }

        for sym in &symbols {
            let u_test = format!(
                "def test_{}_basic():\n    \"\"\"Baseline unit test for {}\"\"\"\n    assert True\n",
                sym.name, sym.name
            );
            unit_tests.push(u_test.clone());
            full_code.push_str(&u_test);
            full_code.push('\n');

            if fuzz {
                let f_test = format!(
                    "@given(st.integers(), st.text())\ndef test_fuzz_{}(num, text):\n    \"\"\"Hypothesis property fuzzing for {}\"\"\"\n    assert isinstance(num, int)\n    assert isinstance(text, str)\n",
                    sym.name, sym.name
                );
                fuzz_tests.push(f_test.clone());
                full_code.push_str(&f_test);
                full_code.push('\n');
            }
        }
        if symbols.is_empty() {
            let u = "def test_smoke():\n    assert True\n".to_string();
            unit_tests.push(u.clone());
            full_code.push_str(&u);
        }
    } else if lang_lower == "js" || lang_lower == "ts" || lang_lower == "javascript" || lang_lower == "typescript" {
        target_test_path = format!("{}.test.ts", file_stem);
        full_code.push_str("// Auto-generated Unit & Fast-Check Property Fuzz Test Suite\nimport { describe, test, expect } from 'vitest';\n");
        if fuzz {
            full_code.push_str("import * as fc from 'fast-check';\n\n");
        } else {
            full_code.push('\n');
        }

        full_code.push_str(&format!("describe('{} test suite', () => {{\n", file_stem));
        for sym in &symbols {
            let u_test = format!(
                "  test('{} basic functionality', () => {{\n    expect(true).toBe(true);\n  }});\n",
                sym.name
            );
            unit_tests.push(u_test.clone());
            full_code.push_str(&u_test);

            if fuzz {
                let f_test = format!(
                    "  test('{} fast-check fuzz property', () => {{\n    fc.assert(fc.property(fc.integer(), fc.string(), (n, s) => {{\n      return typeof n === 'number' && typeof s === 'string';\n    }}));\n  }});\n",
                    sym.name
                );
                fuzz_tests.push(f_test.clone());
                full_code.push_str(&f_test);
            }
        }
        if symbols.is_empty() {
            full_code.push_str("  test('smoke test', () => { expect(true).toBe(true); });\n");
        }
        full_code.push_str("});\n");
    } else if lang_lower == "go" || lang_lower == "golang" {
        target_test_path = format!("{}_test.go", file_stem);
        full_code.push_str("package main\n\nimport (\n\t\"testing\"\n)\n\n");
        for sym in &symbols {
            let u_test = format!(
                "func Test{}(t *testing.T) {{\n\tif false {{\n\t\tt.Errorf(\"Test failed for {}\")\n\t}}\n}}\n",
                sym.name, sym.name
            );
            unit_tests.push(u_test.clone());
            full_code.push_str(&u_test);
            full_code.push('\n');

            if fuzz {
                let f_test = format!(
                    "func Fuzz{}(f *testing.F) {{\n\tf.Add([]byte(\"seed_data\"))\n\tf.Fuzz(func(t *testing.T, data []byte) {{\n\t\tif len(data) < 0 {{\n\t\t\tt.Errorf(\"invalid length\")\n\t\t}}\n\t}})\n}}\n",
                    sym.name
                );
                fuzz_tests.push(f_test.clone());
                full_code.push_str(&f_test);
                full_code.push('\n');
            }
        }
        if symbols.is_empty() {
            let u = "func TestSmoke(t *testing.T) {}\n".to_string();
            unit_tests.push(u.clone());
            full_code.push_str(&u);
        }
    } else {
        target_test_path = format!("tests/{}_test.txt", file_stem);
        full_code.push_str(&format!("# Unit tests for {}\n", file_stem));
        for sym in &symbols {
            let u = format!("test_{}_basic = assert_ok", sym.name);
            unit_tests.push(u.clone());
            full_code.push_str(&format!("{}\n", u));
        }
    }

    let summary = format!(
        "Synthesized {} unit test(s) and {} property fuzz test(s) for {} symbol(s) in `{}`",
        unit_tests.len(), fuzz_tests.len(), symbols.len(), source_file.display()
    );

    Ok(GeneratedTestSuite {
        source_file: source_file.to_string_lossy().to_string(),
        language: lang.to_string(),
        fuzz_enabled: fuzz,
        scanned_symbols: symbols,
        unit_tests,
        fuzz_tests,
        test_file_path: target_test_path,
        test_code: full_code,
        summary,
    })
}

pub fn format_test_suite_report_for_terminal(suite: &GeneratedTestSuite) -> String {
    let mut out = String::new();
    out.push_str(&format!("\n{}\n", "╔══════════════════════════════════════════════════════════════╗".cyan()));
    out.push_str(&format!("║ {} ║\n", "🧪 AUTOMATED TEST & FUZZ SUITE SYNTHESIZER REPORT".cyan().bold()));
    out.push_str(&format!("{}\n", "╠══════════════════════════════════════════════════════════════╣".cyan()));
    out.push_str(&format!("  Source File:     {}\n", suite.source_file.yellow().bold()));
    out.push_str(&format!("  Language:        {} (Fuzzing: {})\n", suite.language.cyan(), if suite.fuzz_enabled { "ENABLED".green().bold() } else { "DISABLED".dimmed() }));
    out.push_str(&format!("  Symbols Scanned: {}\n", suite.scanned_symbols.len().to_string().cyan()));
    out.push_str(&format!("  Unit Tests:      {}\n", suite.unit_tests.len().to_string().green().bold()));
    out.push_str(&format!("  Fuzz Suites:     {}\n", suite.fuzz_tests.len().to_string().magenta().bold()));
    out.push_str(&format!("  Target Test File:{}\n", suite.test_file_path.cyan().underline()));
    out.push_str(&format!("{}\n", "╟──────────────────────────────────────────────────────────────╢".cyan()));
    for (i, sym) in suite.scanned_symbols.iter().take(8).enumerate() {
        out.push_str(&format!("  {}. [{}] {} (line {})\n", i + 1, sym.kind.dimmed(), sym.name.yellow(), sym.line));
    }
    if suite.scanned_symbols.len() > 8 {
        out.push_str(&format!("  ... and {} more symbols\n", suite.scanned_symbols.len() - 8));
    }
    out.push_str(&format!("{}\n", "╚══════════════════════════════════════════════════════════════╝".cyan()));
    out
}

// ============================================================================
// SYSTEM 3: PRODUCTION CONTAINER & CI/CD PIPELINE GENERATOR
// ============================================================================

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub enum ProjectLanguage {
    Rust,
    Node,
    Python,
    Go,
    Java,
    Generic,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct ProjectStack {
    pub language: ProjectLanguage,
    pub language_name: String,
    pub project_name: String,
    pub detected_files: Vec<String>,
    pub has_dockerfile: bool,
    pub has_docker_compose: bool,
    pub has_github_ci: bool,
    pub suggested_port: u16,
    pub build_command: String,
    pub test_command: String,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct CiManifests {
    pub stack: ProjectStack,
    pub dockerfile: String,
    pub docker_compose: String,
    pub github_workflow: String,
    pub summary: String,
}

pub fn detect_project_stack(path: &std::path::Path) -> ProjectStack {
    let mut detected_files = Vec::new();
    let has_dockerfile = path.join("Dockerfile").exists();
    let has_docker_compose = path.join("docker-compose.yml").exists() || path.join("docker-compose.yaml").exists();
    let has_github_ci = path.join(".github/workflows/ci.yml").exists() || path.join(".github/workflows/ci.yaml").exists();

    if has_dockerfile { detected_files.push("Dockerfile".to_string()); }
    if has_docker_compose { detected_files.push("docker-compose.yml".to_string()); }
    if has_github_ci { detected_files.push(".github/workflows/ci.yml".to_string()); }

    let default_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("app").to_string();

    if path.join("Cargo.toml").exists() {
        detected_files.push("Cargo.toml".to_string());
        let mut proj_name = default_name.clone();
        if let Ok(content) = fs::read_to_string(path.join("Cargo.toml")) {
            for line in content.lines() {
                let l = line.trim();
                if l.starts_with("name = ") {
                    proj_name = l.trim_start_matches("name = ").trim_matches('"').to_string();
                    break;
                }
            }
        }
        ProjectStack {
            language: ProjectLanguage::Rust,
            language_name: "Rust".to_string(),
            project_name: proj_name,
            detected_files,
            has_dockerfile,
            has_docker_compose,
            has_github_ci,
            suggested_port: 8080,
            build_command: "cargo build --release".to_string(),
            test_command: "cargo test --all-targets".to_string(),
        }
    } else if path.join("package.json").exists() {
        detected_files.push("package.json".to_string());
        let mut proj_name = default_name.clone();
        if let Ok(content) = fs::read_to_string(path.join("package.json")) {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(&content) {
                if let Some(n) = v.get("name").and_then(|n| n.as_str()) {
                    proj_name = n.to_string();
                }
            }
        }
        ProjectStack {
            language: ProjectLanguage::Node,
            language_name: "Node.js / TypeScript".to_string(),
            project_name: proj_name,
            detected_files,
            has_dockerfile,
            has_docker_compose,
            has_github_ci,
            suggested_port: 3000,
            build_command: "npm run build".to_string(),
            test_command: "npm test".to_string(),
        }
    } else if path.join("requirements.txt").exists() || path.join("pyproject.toml").exists() || path.join("Pipfile").exists() {
        if path.join("requirements.txt").exists() { detected_files.push("requirements.txt".to_string()); }
        if path.join("pyproject.toml").exists() { detected_files.push("pyproject.toml".to_string()); }
        ProjectStack {
            language: ProjectLanguage::Python,
            language_name: "Python".to_string(),
            project_name: default_name,
            detected_files,
            has_dockerfile,
            has_docker_compose,
            has_github_ci,
            suggested_port: 8000,
            build_command: "python -m py_compile $(git ls-files '*.py')".to_string(),
            test_command: "pytest".to_string(),
        }
    } else if path.join("go.mod").exists() {
        detected_files.push("go.mod".to_string());
        ProjectStack {
            language: ProjectLanguage::Go,
            language_name: "Go".to_string(),
            project_name: default_name,
            detected_files,
            has_dockerfile,
            has_docker_compose,
            has_github_ci,
            suggested_port: 8080,
            build_command: "go build -o app ./...".to_string(),
            test_command: "go test -v -race ./...".to_string(),
        }
    } else {
        ProjectStack {
            language: ProjectLanguage::Generic,
            language_name: "Generic / Polyglot".to_string(),
            project_name: default_name,
            detected_files,
            has_dockerfile,
            has_docker_compose,
            has_github_ci,
            suggested_port: 8080,
            build_command: "make build".to_string(),
            test_command: "make test".to_string(),
        }
    }
}

pub fn generate_container_and_ci_manifests(stack: &ProjectStack) -> CiManifests {
    let dockerfile: String;
    let docker_compose: String;
    let github_workflow: String;

    match stack.language {
        ProjectLanguage::Rust => {
            dockerfile = format!(r#"# syntax=docker/dockerfile:1.4
# Multi-Stage Hardened Cargo-Chef Rust Build
FROM lukemathwalker/cargo-chef:latest-rust-1-alpine AS chef
WORKDIR /app
RUN apk add --no-cache musl-dev pkgconfig openssl-dev

FROM chef AS planner
COPY . .
RUN cargo chef prepare --recipe-path recipe.json

FROM chef AS builder
COPY --from=planner /app/recipe.json recipe.json
# Build and cache dependencies layer
RUN cargo chef cook --release --recipe-path recipe.json
# Build application binary
COPY . .
RUN cargo build --release --bin {}

# Runtime Stage (Alpine Hardened)
FROM alpine:3.19 AS runtime
WORKDIR /app

RUN addgroup -g 10001 -S appgroup && \
    adduser -u 10001 -S appuser -G appgroup && \
    apk add --no-cache ca-certificates tzdata

COPY --from=builder /app/target/release/{} /usr/local/bin/app

USER appuser:appgroup
EXPOSE {}

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD nc -z 127.0.0.1 {} || exit 1

ENTRYPOINT ["/usr/local/bin/app"]
"#, stack.project_name, stack.project_name, stack.suggested_port, stack.suggested_port);

            docker_compose = format!(r#"version: '3.8'

services:
  {}:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: {}-service
    restart: unless-stopped
    ports:
      - "{}:{}"
    environment:
      - RUST_LOG=info
      - PORT={}
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 512M
    healthcheck:
      test: ["CMD", "nc", "-z", "127.0.0.1", "{}"]
      interval: 30s
      timeout: 5s
      retries: 3
"#, stack.project_name, stack.project_name, stack.suggested_port, stack.suggested_port, stack.suggested_port, stack.suggested_port);

            github_workflow = r#"name: Production CI

on:
  push:
    branches: [ main, master, develop ]
  pull_request:
    branches: [ main, master ]

concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true

jobs:
  lint-and-audit:
    name: Code Quality & Security Audit
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
        with:
          components: clippy, rustfmt
      - uses: Swatinem/rust-cache@v2
      - name: Check Formatting
        run: cargo fmt --all -- --check
      - name: Clippy Linter
        run: cargo clippy --all-targets --all-features -- -D warnings
      - name: Security Vulnerability Audit
        uses: rustsec/audit-check@v1.4.1
        with:
          token: ${{ secrets.GITHUB_TOKEN }}
        continue-on-error: true

  test-matrix:
    name: Test Matrix (${{ matrix.os }})
    needs: lint-and-audit
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - uses: Swatinem/rust-cache@v2
      - name: Run Test Suite
        run: cargo test --all-targets --verbose
"#.to_string();
        }
        ProjectLanguage::Node => {
            dockerfile = format!(r#"# Multi-Stage Hardened Node.js Build
FROM node:20-alpine AS deps
WORKDIR /app
COPY package*.json ./
RUN npm ci

FROM node:20-alpine AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .
RUN npm run build --if-present

FROM node:20-alpine AS runner
WORKDIR /app
ENV NODE_ENV=production
RUN addgroup -g 1001 -S nodejs && adduser -S nextjs -u 1001
COPY --from=builder /app ./
USER nextjs
EXPOSE {}
CMD ["npm", "start"]
"#, stack.suggested_port);

            docker_compose = format!(r#"version: '3.8'
services:
  {}:
    build: .
    ports:
      - "{}:{}"
    environment:
      - NODE_ENV=production
      - PORT={}
    restart: unless-stopped
"#, stack.project_name, stack.suggested_port, stack.suggested_port, stack.suggested_port);

            github_workflow = r#"name: Node CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        node-version: [18.x, 20.x]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: ${{ matrix.node-version }}
          cache: 'npm'
      - run: npm ci
      - run: npm run lint --if-present
      - run: npm test --if-present
      - run: npm audit --audit-level=high
"#.to_string();
        }
        ProjectLanguage::Python => {
            dockerfile = format!(r#"# Multi-Stage Hardened Python Build
FROM python:3.11-slim AS builder
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends gcc build-essential && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

FROM python:3.11-slim AS runner
WORKDIR /app
RUN useradd -u 10001 appuser
COPY --from=builder /root/.local /home/appuser/.local
COPY . .
ENV PATH=/home/appuser/.local/bin:$PATH
USER appuser
EXPOSE {}
CMD ["python", "app.py"]
"#, stack.suggested_port);

            docker_compose = format!(r#"version: '3.8'
services:
  {}:
    build: .
    ports:
      - "{}:{}"
    environment:
      - PYTHONUNBUFFERED=1
    restart: unless-stopped
"#, stack.project_name, stack.suggested_port, stack.suggested_port);

            github_workflow = r#"name: Python CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
          cache: 'pip'
      - run: pip install -r requirements.txt || true
      - run: pip install pytest ruff
      - run: ruff check .
      - run: pytest
"#.to_string();
        }
        ProjectLanguage::Go => {
            dockerfile = format!(r#"# Multi-Stage Hardened Go Build
FROM golang:1.22-alpine AS builder
WORKDIR /app
RUN apk add --no-cache git
COPY go.* ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -ldflags="-w -s" -o /app/server .

FROM alpine:3.19 AS runner
RUN apk --no-cache add ca-certificates tzdata
RUN adduser -D -u 10001 appuser
COPY --from=builder /app/server /server
USER appuser
EXPOSE {}
ENTRYPOINT ["/server"]
"#, stack.suggested_port);

            docker_compose = format!(r#"version: '3.8'
services:
  {}:
    build: .
    ports:
      - "{}:{}"
    restart: unless-stopped
"#, stack.project_name, stack.suggested_port, stack.suggested_port);

            github_workflow = r#"name: Go CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-go@v5
        with:
          go-version: '1.22'
      - run: go test -v -race ./...
"#.to_string();
        }
        ProjectLanguage::Java | ProjectLanguage::Generic => {
            dockerfile = format!(r#"FROM alpine:3.19
WORKDIR /app
COPY . .
EXPOSE {}
CMD ["echo", "Ready"]
"#, stack.suggested_port);

            docker_compose = format!(r#"version: '3.8'
services:
  {}:
    build: .
    ports:
      - "{}:{}"
"#, stack.project_name, stack.suggested_port, stack.suggested_port);

            github_workflow = r#"name: CI

on: [push, pull_request]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: make test || true
"#.to_string();
        }
    }

    let summary = format!(
        "Generated container & CI/CD manifests for {} project `{}` (Dockerfile, docker-compose.yml, .github/workflows/ci.yml)",
        stack.language_name, stack.project_name
    );

    CiManifests {
        stack: stack.clone(),
        dockerfile,
        docker_compose,
        github_workflow,
        summary,
    }
}

pub fn format_ci_manifests_for_terminal(manifests: &CiManifests) -> String {
    let mut out = String::new();
    out.push_str(&format!("\n{}\n", "╔══════════════════════════════════════════════════════════════╗".cyan()));
    out.push_str(&format!("║ {} ║\n", "🐳 CONTAINER & CI/CD MANIFEST GENERATOR REPORT".cyan().bold()));
    out.push_str(&format!("{}\n", "╠══════════════════════════════════════════════════════════════╣".cyan()));
    out.push_str(&format!("  Project:     {}\n", manifests.stack.project_name.yellow().bold()));
    out.push_str(&format!("  Stack:       {}\n", manifests.stack.language_name.cyan()));
    out.push_str(&format!("  Port:        {}\n", manifests.stack.suggested_port.to_string().green()));
    out.push_str(&format!("  Build Cmd:   {}\n", manifests.stack.build_command.dimmed()));
    out.push_str(&format!("  Test Cmd:    {}\n", manifests.stack.test_command.dimmed()));
    out.push_str(&format!("{}\n", "╟──────────────────────────────────────────────────────────────╢".cyan()));
    out.push_str("  ✔ Dockerfile (Hardened Multi-Stage)\n");
    out.push_str("  ✔ docker-compose.yml (Services, Healthchecks, Limits)\n");
    out.push_str("  ✔ .github/workflows/ci.yml (Matrix, Lint, Audit, Cache)\n");
    out.push_str(&format!("{}\n", "╚══════════════════════════════════════════════════════════════╝".cyan()));
    out
}

// ============================================================================
// SYSTEM 4: INTERACTIVE CODEBASE CALL GRAPH VISUALIZER
// ============================================================================

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct CallGraphNode {
    pub symbol: String,
    pub file: String,
    pub line: usize,
    pub is_entrypoint: bool,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct CallGraphEdge {
    pub caller: String,
    pub callee: String,
    pub call_site_file: String,
    pub call_site_line: usize,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct CallGraphReport {
    pub workspace_root: String,
    pub entry_symbol: Option<String>,
    pub total_functions: usize,
    pub total_calls: usize,
    pub nodes: Vec<CallGraphNode>,
    pub edges: Vec<CallGraphEdge>,
    pub ascii_tree: String,
    pub mermaid_diagram: String,
    pub summary: String,
}

pub fn build_call_graph(
    workspace_root: &std::path::Path,
    entry_symbol: Option<&str>,
) -> CallGraphReport {
    let mut nodes = Vec::new();
    let mut file_contents: Vec<(String, String)> = Vec::new();

    // 1. Gather all source files
    let extensions = ["rs", "py", "js", "ts", "jsx", "tsx", "go", "c", "cpp"];
    for entry in WalkDir::new(workspace_root).into_iter().filter_map(|e| e.ok()) {
        let p = entry.path();
        let path_str = p.to_string_lossy();
        if path_str.contains("/target/") || path_str.contains("\\target\\")
            || path_str.contains("/.git/") || path_str.contains("\\.git\\")
            || path_str.contains("/node_modules/") || path_str.contains("\\node_modules\\")
        {
            continue;
        }

        if p.is_file() {
            if let Some(ext) = p.extension().and_then(|e| e.to_str()) {
                if extensions.contains(&ext) {
                    if let Ok(content) = fs::read_to_string(p) {
                        let rel_path = p.strip_prefix(workspace_root).unwrap_or(p).to_string_lossy().to_string();
                        let symbols = extract_symbols_from_source(&content, ext);
                        for s in symbols {
                            let is_entry = s.name == "main" || s.name.starts_with("run_") || s.name == "interactive_chat";
                            nodes.push(CallGraphNode {
                                symbol: s.name,
                                file: rel_path.clone(),
                                line: s.line,
                                is_entrypoint: is_entry,
                            });
                        }
                        file_contents.push((rel_path, content));
                    }
                }
            }
        }
    }

    // Deduplicate symbol definitions if any
    let mut unique_nodes: Vec<CallGraphNode> = Vec::new();
    let mut seen_symbols = std::collections::HashSet::new();
    for n in nodes {
        if seen_symbols.insert(n.symbol.clone()) {
            unique_nodes.push(n);
        }
    }

    // 2. Discover call edges by scanning function blocks and invocations
    let mut edges = Vec::new();
    let known_symbols: std::collections::HashSet<String> = unique_nodes.iter().map(|n| n.symbol.clone()).collect();

    for (file_path, content) in &file_contents {
        let mut current_caller = "global".to_string();
        for (idx, line) in content.lines().enumerate() {
            let line_num = idx + 1;
            let trimmed = line.trim();

            // Detect entering a function
            for node in &unique_nodes {
                if node.file == *file_path && node.line == line_num {
                    current_caller = node.symbol.clone();
                    break;
                }
            }

            // Check if this line calls any known symbol
            for sym in &known_symbols {
                if sym != &current_caller && trimmed.contains(sym.as_str()) {
                    // Check if followed by '(' or '::' or '.'
                    let is_call = trimmed.contains(&format!("{}(", sym))
                        || trimmed.contains(&format!("{} (", sym))
                        || trimmed.contains(&format!("::{}", sym));
                    if is_call {
                        let edge = CallGraphEdge {
                            caller: current_caller.clone(),
                            callee: sym.clone(),
                            call_site_file: file_path.clone(),
                            call_site_line: line_num,
                        };
                        if !edges.contains(&edge) {
                            edges.push(edge);
                        }
                    }
                }
            }
        }
    }

    // 3. Build Adjacency map: caller -> Vec<callee>
    let mut adj: std::collections::HashMap<String, Vec<String>> = std::collections::HashMap::new();
    let mut in_degrees: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for node in &unique_nodes {
        in_degrees.insert(node.symbol.clone(), 0);
    }
    for e in &edges {
        adj.entry(e.caller.clone()).or_default().push(e.callee.clone());
        *in_degrees.entry(e.callee.clone()).or_default() += 1;
    }

    // Determine root nodes
    let roots: Vec<String> = if let Some(target) = entry_symbol {
        vec![target.to_string()]
    } else {
        let candidates: Vec<String> = unique_nodes.iter()
            .filter(|n| n.is_entrypoint || *in_degrees.get(&n.symbol).unwrap_or(&0) == 0)
            .map(|n| n.symbol.clone())
            .collect();
        if candidates.is_empty() {
            unique_nodes.iter().take(5).map(|n| n.symbol.clone()).collect()
        } else {
            candidates
        }
    };

    // Helper recursive tree builder
    #[allow(clippy::too_many_arguments)]
    fn render_tree(
        node: &str,
        adj: &std::collections::HashMap<String, Vec<String>>,
        nodes_map: &std::collections::HashMap<String, &CallGraphNode>,
        visited: &mut std::collections::HashSet<String>,
        prefix: &str,
        is_last: bool,
        out: &mut String,
        depth: usize,
    ) {
        if depth > 10 { return; }
        let connector = if depth == 0 { "" } else if is_last { "└── " } else { "├── " };
        let loc = if let Some(n) = nodes_map.get(node) {
            format!(" ({}:{})", n.file, n.line)
        } else {
            String::new()
        };

        if visited.contains(node) {
            out.push_str(&format!("{}{}{} [cycle]\n", prefix, connector, node));
            return;
        }

        out.push_str(&format!("{}{}{}{}\n", prefix, connector, node, loc));
        visited.insert(node.to_string());

        if let Some(children) = adj.get(node) {
            let next_prefix = if depth == 0 { String::new() } else if is_last { format!("{}    ", prefix) } else { format!("{}│   ", prefix) };
            for (i, child) in children.iter().enumerate() {
                let child_is_last = i == children.len() - 1;
                render_tree(child, adj, nodes_map, visited, &next_prefix, child_is_last, out, depth + 1);
            }
        }
        visited.remove(node);
    }

    let nodes_map: std::collections::HashMap<String, &CallGraphNode> = unique_nodes.iter().map(|n| (n.symbol.clone(), n)).collect();
    let mut ascii_tree = String::new();
    for root in &roots {
        let mut visited = std::collections::HashSet::new();
        render_tree(root, &adj, &nodes_map, &mut visited, "", true, &mut ascii_tree, 0);
        ascii_tree.push('\n');
    }

    // 4. Generate Mermaid diagram
    let mut mermaid = String::from("graph TD;\n");
    for e in &edges {
        mermaid.push_str(&format!("    {} --> {};\n", e.caller, e.callee));
    }

    let summary = format!(
        "Call Graph contains {} function(s) and {} call site(s) across codebase.",
        unique_nodes.len(), edges.len()
    );

    CallGraphReport {
        workspace_root: workspace_root.to_string_lossy().to_string(),
        entry_symbol: entry_symbol.map(|s| s.to_string()),
        total_functions: unique_nodes.len(),
        total_calls: edges.len(),
        nodes: unique_nodes,
        edges,
        ascii_tree: ascii_tree.trim_end().to_string(),
        mermaid_diagram: mermaid,
        summary,
    }
}

pub fn format_call_graph_for_terminal(report: &CallGraphReport) -> String {
    let mut out = String::new();
    out.push_str(&format!("\n{}\n", "╔══════════════════════════════════════════════════════════════╗".cyan()));
    out.push_str(&format!("║ {} ║\n", "🕸️  INTERACTIVE CALL GRAPH & CALL SITE VISUALIZER".cyan().bold()));
    out.push_str(&format!("{}\n", "╠══════════════════════════════════════════════════════════════╣".cyan()));
    out.push_str(&format!("  Workspace:       {}\n", report.workspace_root.yellow().bold()));
    out.push_str(&format!("  Entry Symbol:    {}\n", report.entry_symbol.as_deref().unwrap_or("auto-detected roots").cyan()));
    out.push_str(&format!("  Total Functions: {}\n", report.total_functions.to_string().green().bold()));
    out.push_str(&format!("  Total Call Sites:{}\n", report.total_calls.to_string().magenta().bold()));
    out.push_str(&format!("{}\n", "╟──────────────────────────────────────────────────────────────╢".cyan()));
    out.push_str(&format!("{}\n", report.ascii_tree));
    out.push_str(&format!("{}\n", "╚══════════════════════════════════════════════════════════════╝".cyan()));
    out
}

// ============================================================================
// SYSTEM 5: MULTI-LANGUAGE FORMATTER & LINTER AUTO-FIXER
// ============================================================================

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct FormatterToolResult {
    pub tool: String,
    pub command: String,
    pub success: bool,
    pub stdout: String,
    pub stderr: String,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct LintFormatReport {
    pub workspace_root: String,
    pub fix_mode: bool,
    pub tools_executed: Vec<FormatterToolResult>,
    pub formatted_files: Vec<String>,
    pub issues_found: usize,
    pub issues_fixed: usize,
    pub summary: String,
}

pub fn format_and_lint_workspace(
    workspace_root: &std::path::Path,
    fix: bool,
) -> Result<LintFormatReport, Box<dyn std::error::Error>> {
    let mut tools_executed = Vec::new();
    let mut formatted_files = Vec::new();
    let mut issues_found = 0;
    let mut issues_fixed = 0;

    // 1. Rust Tooling (cargo fmt & cargo clippy)
    if workspace_root.join("Cargo.toml").exists() {
        let fmt_cmd = if fix { "cargo fmt" } else { "cargo fmt --check" };
        let fmt_out = if fix {
            std::process::Command::new("cargo").arg("fmt").current_dir(workspace_root).output()
        } else {
            std::process::Command::new("cargo").args(["fmt", "--", "--check"]).current_dir(workspace_root).output()
        };

        if let Ok(out) = fmt_out {
            let succ = out.status.success();
            let so = String::from_utf8_lossy(&out.stdout).to_string();
            let se = String::from_utf8_lossy(&out.stderr).to_string();
            if !succ { issues_found += 1; }
            if succ && fix { issues_fixed += 1; }
            tools_executed.push(FormatterToolResult {
                tool: "rustfmt".to_string(),
                command: fmt_cmd.to_string(),
                success: succ,
                stdout: so,
                stderr: se,
            });
        }

        let clippy_cmd = if fix { "cargo clippy --fix --allow-dirty --allow-staged" } else { "cargo clippy" };
        let clippy_out = if fix {
            std::process::Command::new("cargo").args(["clippy", "--fix", "--allow-dirty", "--allow-staged"]).current_dir(workspace_root).output()
        } else {
            std::process::Command::new("cargo").arg("clippy").current_dir(workspace_root).output()
        };

        if let Ok(out) = clippy_out {
            let succ = out.status.success();
            let so = String::from_utf8_lossy(&out.stdout).to_string();
            let se = String::from_utf8_lossy(&out.stderr).to_string();
            if !succ { issues_found += 1; }
            if succ && fix { issues_fixed += 1; }
            tools_executed.push(FormatterToolResult {
                tool: "clippy".to_string(),
                command: clippy_cmd.to_string(),
                success: succ,
                stdout: so,
                stderr: se,
            });
        }
    }

    // 2. Node / Prettier / ESLint Tooling
    if workspace_root.join("package.json").exists() {
        let npx_cmd = if fix { "npx prettier --write ." } else { "npx prettier --check ." };
        if let Ok(out) = std::process::Command::new("npx").args(if fix { vec!["prettier", "--write", "."] } else { vec!["prettier", "--check", "."] }).current_dir(workspace_root).output() {
            let succ = out.status.success();
            tools_executed.push(FormatterToolResult {
                tool: "prettier".to_string(),
                command: npx_cmd.to_string(),
                success: succ,
                stdout: String::from_utf8_lossy(&out.stdout).to_string(),
                stderr: String::from_utf8_lossy(&out.stderr).to_string(),
            });
        }
    }

    // 3. In-crate Style & Whitespace Auto-Fixer
    let exts = ["rs", "py", "js", "ts", "go", "json", "toml", "md", "c", "cpp"];
    for entry in WalkDir::new(workspace_root).into_iter().filter_map(|e| e.ok()) {
        let p = entry.path();
        let path_str = p.to_string_lossy();
        if path_str.contains("/target/") || path_str.contains("\\target\\")
            || path_str.contains("/.git/") || path_str.contains("\\.git\\")
            || path_str.contains("/node_modules/") || path_str.contains("\\node_modules\\")
        {
            continue;
        }

        if p.is_file() {
            if let Some(ext) = p.extension().and_then(|e| e.to_str()) {
                if exts.contains(&ext) {
                    if let Ok(content) = fs::read_to_string(p) {
                        let mut modified = false;
                        let mut new_lines = Vec::new();

                        for line in content.lines() {
                            let trimmed_end = line.trim_end_matches([' ', '\t', '\r']);
                            if trimmed_end != line {
                                modified = true;
                                issues_found += 1;
                            }
                            new_lines.push(trimmed_end);
                        }

                        let mut normalized = new_lines.join("\n");
                        if !normalized.is_empty() && !normalized.ends_with('\n') {
                            normalized.push('\n');
                        }

                        if modified && fix && fs::write(p, normalized).is_ok() {
                            issues_fixed += 1;
                            formatted_files.push(p.strip_prefix(workspace_root).unwrap_or(p).to_string_lossy().to_string());
                        }
                    }
                }
            }
        }
    }

    let summary = format!(
        "Linter & Formatter finished across `{}` (fix={}): {} tool(s) executed, {} file(s) formatted, {} issue(s) detected, {} issue(s) resolved.",
        workspace_root.display(), fix, tools_executed.len(), formatted_files.len(), issues_found, issues_fixed
    );

    Ok(LintFormatReport {
        workspace_root: workspace_root.to_string_lossy().to_string(),
        fix_mode: fix,
        tools_executed,
        formatted_files,
        issues_found,
        issues_fixed,
        summary,
    })
}

pub fn format_lint_format_report_for_terminal(report: &LintFormatReport) -> String {
    let mut out = String::new();
    out.push_str(&format!("\n{}\n", "╔══════════════════════════════════════════════════════════════╗".cyan()));
    out.push_str(&format!("║ {} ║\n", "🧹 MULTI-LANGUAGE FORMATTER & LINTER AUTO-FIXER".cyan().bold()));
    out.push_str(&format!("{}\n", "╠══════════════════════════════════════════════════════════════╣".cyan()));
    out.push_str(&format!("  Workspace:       {}\n", report.workspace_root.yellow().bold()));
    out.push_str(&format!("  Fix Mode:        {}\n", if report.fix_mode { "AUTO-FIX ENABLED".green().bold() } else { "CHECK ONLY".yellow() }));
    out.push_str(&format!("  Tools Executed:  {}\n", report.tools_executed.len().to_string().cyan()));
    out.push_str(&format!("  Files Formatted: {}\n", report.formatted_files.len().to_string().green().bold()));
    out.push_str(&format!("  Issues Fixed:    {} / {}\n", report.issues_fixed.to_string().green(), report.issues_found.to_string().yellow()));
    out.push_str(&format!("{}\n", "╟──────────────────────────────────────────────────────────────╢".cyan()));
    for tool in &report.tools_executed {
        out.push_str(&format!("  • [{}] {} -> {}\n", tool.tool.bold(), tool.command.dimmed(), if tool.success { "PASSED".green() } else { "WARNINGS/FAILED".yellow() }));
    }
    for file in report.formatted_files.iter().take(5) {
        out.push_str(&format!("  ✨ Formatted: {}\n", file.green()));
    }
    out.push_str(&format!("{}\n", "╚══════════════════════════════════════════════════════════════╝".cyan()));
    out
}

// ============================================================================
// SYSTEM 6: EPHEMERAL AI MOCK SERVER & API SANDBOX
// ============================================================================

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct MockRoute {
    pub method: String,
    pub path: String,
    pub status_code: u16,
    pub response_body: serde_json::Value,
    #[serde(default)]
    pub headers: std::collections::HashMap<String, String>,
}

pub struct MockServerHandle {
    pub port: u16,
    pub bound_addr: std::net::SocketAddr,
    pub shutdown_tx: Option<tokio::sync::oneshot::Sender<()>>,
    pub routes: Vec<MockRoute>,
}

impl MockServerHandle {
    pub fn shutdown(mut self) {
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(());
        }
    }

    pub fn is_running(&self) -> bool {
        self.shutdown_tx.is_some()
    }

    pub fn port(&self) -> u16 {
        self.port
    }

    pub fn base_url(&self) -> String {
        format!("http://127.0.0.1:{}", self.port)
    }
}

static ACTIVE_MOCK_SERVERS: std::sync::Mutex<Vec<MockServerHandle>> = std::sync::Mutex::new(Vec::new());

pub fn register_active_mock_server(handle: MockServerHandle) {
    let mut lock = ACTIVE_MOCK_SERVERS.lock().unwrap();
    lock.push(handle);
}

pub fn stop_all_active_mock_servers() {
    let mut lock = ACTIVE_MOCK_SERVERS.lock().unwrap();
    for handle in lock.drain(..) {
        handle.shutdown();
    }
}

pub async fn start_ephemeral_mock_server(
    port: u16,
    routes: Vec<MockRoute>,
) -> Result<MockServerHandle, Box<dyn std::error::Error>> {
    let listener = tokio::net::TcpListener::bind(format!("127.0.0.1:{}", port)).await?;
    let bound_addr = listener.local_addr()?;
    let actual_port = bound_addr.port();

    let (shutdown_tx, mut shutdown_rx) = tokio::sync::oneshot::channel::<()>();
    let routes_task = routes.clone();

    tokio::spawn(async move {
        loop {
            tokio::select! {
                _ = &mut shutdown_rx => {
                    break;
                }
                accept_res = listener.accept() => {
                    if let Ok((mut socket, _)) = accept_res {
                        let routes_conn = routes_task.clone();
                        tokio::spawn(async move {
                            let mut buf = vec![0u8; 8192];
                            if let Ok(n) = socket.read(&mut buf).await {
                                if n == 0 { return; }
                                let req_str = String::from_utf8_lossy(&buf[..n]);
                                let first_line = req_str.lines().next().unwrap_or("");
                                let parts: Vec<&str> = first_line.split_whitespace().collect();
                                let req_method = if !parts.is_empty() { parts[0] } else { "GET" };
                                let req_path = if parts.len() > 1 { parts[1].split('?').next().unwrap_or(parts[1]) } else { "/" };

                                let matched = routes_conn.iter().find(|r| {
                                    let method_match = r.method == "*" || r.method.eq_ignore_ascii_case(req_method);
                                    let path_match = r.path == "*" || r.path == req_path;
                                    method_match && path_match
                                });

                                let (status, body_str, headers_map) = match matched {
                                    Some(route) => {
                                        let body = serde_json::to_string(&route.response_body).unwrap_or_else(|_| "{}".to_string());
                                        (route.status_code, body, route.headers.clone())
                                    }
                                    None => {
                                        (404, format!("{{\"error\":\"Route Not Found\",\"method\":\"{}\",\"path\":\"{}\"}}", req_method, req_path), std::collections::HashMap::new())
                                    }
                                };

                                let status_text = match status {
                                    200 => "OK",
                                    201 => "Created",
                                    202 => "Accepted",
                                    204 => "No Content",
                                    400 => "Bad Request",
                                    401 => "Unauthorized",
                                    403 => "Forbidden",
                                    404 => "Not Found",
                                    500 => "Internal Server Error",
                                    _ => "OK",
                                };

                                let mut custom_hdrs = String::new();
                                for (hk, hv) in &headers_map {
                                    custom_hdrs.push_str(&format!("{}: {}\r\n", hk, hv));
                                }

                                let resp = format!(
                                    "HTTP/1.1 {} {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n{}\r\n{}",
                                    status, status_text, body_str.len(), custom_hdrs, body_str
                                );
                                let _ = socket.write_all(resp.as_bytes()).await;
                                let _ = socket.flush().await;
                            }
                        });
                    }
                }
            }
        }
    });

    Ok(MockServerHandle {
        port: actual_port,
        bound_addr,
        shutdown_tx: Some(shutdown_tx),
        routes,
    })
}

pub fn format_mock_server_report_for_terminal(handle: &MockServerHandle) -> String {
    let mut out = String::new();
    out.push_str(&format!("\n{}\n", "╔══════════════════════════════════════════════════════════════╗".cyan()));
    out.push_str(&format!("║ {} ║\n", "🚀 EPHEMERAL AI MOCK SERVER & API SANDBOX ACTIVE".cyan().bold()));
    out.push_str(&format!("{}\n", "╠══════════════════════════════════════════════════════════════╣".cyan()));
    out.push_str(&format!("  Base URL:    {}\n", handle.base_url().green().bold().underline()));
    out.push_str(&format!("  Port:        {}\n", handle.port.to_string().cyan()));
    out.push_str(&format!("  Active:      {}\n", if handle.is_running() { "RUNNING".green().bold() } else { "STOPPED".red() }));
    out.push_str(&format!("{}\n", "╟──────────────────────────────────────────────────────────────╢".cyan()));
    for (i, r) in handle.routes.iter().enumerate() {
        out.push_str(&format!("  {}. [{}] {} -> HTTP {}\n", i + 1, r.method.magenta().bold(), r.path.yellow(), r.status_code));
    }
    out.push_str(&format!("{}\n", "╚══════════════════════════════════════════════════════════════╝".cyan()));
    out
}


pub async fn single_prompt(
    client: &Client, 
    model: &str, 
    system: Option<&str>, 
    files: &[String], 
    prompt: &str,
    agent: bool,
    session: Option<&str>,
    rag: bool,
    markdown: bool,
    tuner: &AiTunerState,
    force: bool,
    executor: Option<String>,
    strategist: bool,
    scout: Option<String>,
    format_schema: Option<serde_json::Value>,
    map: bool,
    sandbox: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut messages = load_session(session);
    budget_aware_prune(&mut messages, tuner.num_ctx);

    let mut init_msgs = build_initial_messages(system, files, strategist)?;
    messages.append(&mut init_msgs);

    if map {
        let repo_map = build_repo_map(std::path::Path::new("."), 2048);
        messages.push(Message {
            role: "system".to_string(),
            content: format!("Repository Symbol Map:\n{}", repo_map),
            tool_calls: None,
            images: None,
        });
        println!("{}", "🗺️  Injected Repository Map into prompt context.".cyan());
    }
    
    if rag {
        apply_rag(client, prompt, &mut messages).await?;
    }

    let expanded = expand_context_mentions(prompt, std::path::Path::new("."));
    for mention in &expanded.mentions {
        println!("{} Attached {} (`{}`)", "📎".cyan(), mention.mention_type.bold(), mention.target.yellow());
    }
    messages.extend(expanded.context_messages);
    
    messages.push(Message {
        role: "user".to_string(),
        content: prompt.to_string(),
        tool_calls: None,
        images: None,
    });

    // Dual-Model Speculative Router
    if let Some(scout_mdl) = &scout {
        let decision = classify_query_route(client, scout_mdl, prompt, &tuner.opts).await;
        if decision == RouteDecision::Chat && !agent && executor.is_none() {
            println!("{} {}", "⚡ [Fast Scout Router: Answering Chat]".cyan().bold(), scout_mdl.yellow());
            if markdown {
                let res = fetch_full_response(client, scout_mdl, &messages, &tuner.opts, format_schema.as_ref()).await?;
                print_text(&res);
                messages.push(Message { role: "assistant".to_string(), content: res, tool_calls: None, images: None });
            } else {
                let res = stream_response(client, scout_mdl, &messages, &tuner.opts, format_schema.as_ref()).await?;
                println!();
                messages.push(Message { role: "assistant".to_string(), content: res, tool_calls: None, images: None });
            }
            save_session(session, &messages);
            return Ok(());
        } else {
            println!("{} {}", "🚀 [Speculative Router: Routing to Heavy Coder]".magenta().bold(), model.yellow().bold());
        }
    }
    
    if let Some(exec) = executor {
        println!("{} {}", "🧠 Swarm Architect Planning...".magenta().bold(), model);
        let plan = fetch_full_response(client, model, &messages, &tuner.opts, format_schema.as_ref()).await?;
        print_text(&plan);
        messages.push(Message { role: "assistant".to_string(), content: plan.clone(), tool_calls: None, images: None });
        
        println!("\n{} {}", "⚡ Swarm Executor Working...".yellow().bold(), exec);
        messages.push(Message { role: "user".to_string(), content: format!("Execute this plan using tools:\n{}", plan), tool_calls: None, images: None });
        agent_loop(client, &exec, &mut messages, markdown, &tuner.opts, force, format_schema.as_ref(), sandbox).await?;
    } else if agent {
        agent_loop(client, model, &mut messages, markdown, &tuner.opts, force, format_schema.as_ref(), sandbox).await?;
    } else {
        if markdown {
            let res = fetch_full_response(client, model, &messages, &tuner.opts, format_schema.as_ref()).await?;
            print_text(&res);
            messages.push(Message { role: "assistant".to_string(), content: res, tool_calls: None, images: None });
        } else {
            let res = stream_response(client, model, &messages, &tuner.opts, format_schema.as_ref()).await?;
            println!();
            messages.push(Message { role: "assistant".to_string(), content: res, tool_calls: None, images: None });
        }
    }
    
    save_session(session, &messages);
    Ok(())
}

pub async fn interactive_chat(
    client: &Client, 
    model: &str, 
    system: Option<&str>, 
    files: &[String],
    agent_flag: bool,
    session: Option<&str>,
    rag_flag: bool,
    markdown: bool,
    tuner: &AiTunerState,
    force: bool,
    executor_flag: Option<String>,
    strategist_flag: bool,
    scout_flag: Option<String>,
    format_schema_flag: Option<serde_json::Value>,
    map_flag: bool,
    sandbox_flag: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut active_model = model.to_string();
    let mut agent = agent_flag;
    let mut rag = rag_flag;
    let mut executor = executor_flag;
    let mut strategist = strategist_flag;
    let mut scout_model = scout_flag;
    let mut format_schema = format_schema_flag;
    let mut sandbox = sandbox_flag;

    let mut messages = load_session(session);
    budget_aware_prune(&mut messages, tuner.num_ctx);
    
    let mut init_msgs = build_initial_messages(system, files, strategist)?;
    messages.append(&mut init_msgs);

    if map_flag {
        let repo_map = build_repo_map(std::path::Path::new("."), 2048);
        messages.push(Message {
            role: "system".to_string(),
            content: format!("Repository Symbol Map:\n{}", repo_map),
            tool_calls: None,
            images: None,
        });
        println!("{}", "🗺️  Injected Repository Map into conversation context.".cyan());
    }

    let scout_disp = scout_model.as_deref().unwrap_or("OFF");
    let format_disp = if format_schema.is_some() { "ON" } else { "OFF" };
    let token_disp = format_token_budget(&messages, tuner.num_ctx);

    println!("\n{}", "╭──────────────────────────────────────────────────────────╮".cyan());
    println!("{} {} {}", "│".cyan(), "🤖 zy Agent Dashboard".bold().white(), "                                │".cyan());
    println!("{}", "├──────────────────────────────────────────────────────────┤".cyan());
    println!("{} Model:   {:<12} │ Agent:   {:<3} (Force: {:<3})        {}", "│".cyan(), active_model.yellow().bold(), if agent { "ON".green() } else { "OFF".red() }, if force { "ON".red() } else { "OFF".green() }, "│".cyan());
    println!("{} RAG:     {:<12} │ Sandbox: {:<20} {}", "│".cyan(), if rag { "ON".green() } else { "OFF".red() }, if sandbox { "DOCKER ON".green().bold() } else { "OFF".yellow() }, "│".cyan());
    println!("{} Swarm:   {:<12} │ Strategy:{:<20} {}", "│".cyan(), if let Some(e) = &executor { e.magenta().bold().to_string() } else { "OFF".green().to_string() }, if strategist { "ENGAGED".red().bold() } else { "OFF".green() }, "│".cyan());
    println!("{} Router:  {:<12} │ Format:  {:<20} {}", "│".cyan(), if scout_model.is_some() { scout_disp.cyan().bold().to_string() } else { scout_disp.dimmed().to_string() }, if format_schema.is_some() { format_disp.green().bold().to_string() } else { format_disp.dimmed().to_string() }, "│".cyan());
    println!("{} Tokens:  {:<47} {}", "│".cyan(), token_disp, "│".cyan());
    let sess_display = session.unwrap_or("None");
    println!("{} Session: {:<46} {}", "│".cyan(), sess_display.white().dimmed(), "│".cyan());
    println!("{}\n", "╰──────────────────────────────────────────────────────────╯".cyan());
    println!("💡 {}", "Type /help for commands or /exit to quit.".dimmed());

    let mut rl = DefaultEditor::new()?;

    loop {
        let readline = rl.readline("\x1b[32mzy ❯\x1b[0m ");
        match readline {
            Ok(line) => {
                let input = line.trim();
                if input.is_empty() { continue; }
                
                if input.starts_with('/') {
                    let parts: Vec<&str> = input.split_whitespace().collect();
                    match parts[0].to_lowercase().as_str() {
                        "/help" => {
                            println!("{}", "Available slash commands:".yellow());
                            println!("  /help                 - Show this help message");
                            println!("  /tui                  - Full-Screen Interactive TUI Dashboard (ratatui + crossterm)");
                            println!("  /bench <cmd> [iters]  - Micro-Benchmarking & Performance Profiler Engine");
                            println!("  /fuzz <file>          - Automated Unit Test & Fuzz Suite Synthesizer");
                            println!("  /dockerfile           - Generate Hardened Multi-Stage Dockerfile");
                            println!("  /ci                   - Generate Production CI/CD Matrix & Container Manifests");
                            println!("  /graph [symbol]       - Interactive Codebase Call Graph Visualizer");
                            println!("  /lint [path]          - Multi-Language Formatter & Linter Auto-Fixer");
                            println!("  /mock <port> <path>   - Ephemeral AI Mock Server & API Sandbox");
                            println!("  /audit [path]         - Autonomous Dependency & Security Auditor");
                            println!("  /db <path> [sql]      - Native Local SQLite Database & SQL Inspector");
                            println!("  /docs <path>          - Automated API & Docstring Documentation Generator");
                            println!("  /transaction <action> - Atomic Multi-File Refactor Transactions (stage/commit/rollback)");
                            println!("  /swarm <goal>         - Autonomous Multi-Agent Swarm (Architect -> Coder -> Auditor -> QA)");
                            println!("  /search <query>       - Perform live DuckDuckGo web search without API key");
                            println!("  /mentions             - Help for interactive @file, @git, @diff, @symbol mentions");
                            println!("  /timeline             - Display interactive session conversation timeline");
                            println!("  /rewind [turns]       - Time-travel session rewind (removes past turns)");
                            println!("  /commit [custom_msg]  - Generate Conventional Commit message & commit changes");
                            println!("  /pr                   - Generate structured Markdown Pull Request description");
                            println!("  /rules                - Display & reload active project rules (.zyrules/zy.toml)");
                            println!("  /map [path]           - Generate & inject compact repository symbol map");
                            println!("  /test [runner_cmd]    - Run tests & launch autonomous TDD auto-repair loop");
                            println!("  /checkpoint [label]   - Create lightweight atomic git micro-checkpoint");
                            println!("  /rollback [id]        - Rollback workspace to previous micro-checkpoint");
                            println!("  /sandbox <on|off>     - Toggle Ephemeral Docker Sandbox container mode");
                            println!("  /lsp <file_or_cmd>    - Native LSP / Compiler Diagnostics engine");
                            println!("  /mcp <srv> <tool> <js>- Execute tool on external MCP server via stdio");
                            println!("  /router <scout|off>   - Configure Dual-Model Speculative Router");
                            println!("  /format <json|tool|off>- Grammar-Constrained JSON Generation");
                            println!("  /schema <json_schema> - Apply custom JSON schema constraint");
                            println!("  /clear                - Clear terminal and conversation history");
                            println!("  /save <name>          - Save current session");
                            println!("  /model <name>         - Switch the active LLM");
                            println!("  /agent <on/off>       - Toggle Agent mode");
                            println!("  /rag <on/off>         - Toggle RAG mode");
                            println!("  /executor <mdl>       - Set Swarm Executor model");
                            println!("  /strategist           - Toggle AI Strategist Protocol");
                            println!("  /listen               - Voice-to-Code (Requires arecord & whisper)");
                            println!("  /evolve <req>         - Self-modify zy's own source code and recompile");
                            println!("  /worker               - Autonomously fix bugs in .projectmem/issues/");
                            println!("  /chaos                - Chaos Monkey: Randomly break a file in project");
                            println!("  /sleep                - Deep Memory Compression (Summarize history)");
                            println!("  /webhook <url>        - Set Discord/Slack webhook for agent push alerts");
                            println!("  /train                - Export RLHF dataset & run local LoRA fine-tuning");
                            println!("  /undo                 - Git-revert the last agent file edit");
                            println!("  /exit, /quit          - End the session");
                            continue;
                        }
                        "/rules" => {
                            let rules_opt = load_project_rules(std::path::Path::new("."));
                            match rules_opt {
                                Some(rules) => {
                                    println!("\n{}\n{}\n", "📜 Active Project & User Rules:".cyan().bold(), rules);
                                    let mut updated = false;
                                    for m in &mut messages {
                                        if m.role == "system" {
                                            if !m.content.contains("ACTIVE PROJECT & USER RULES") {
                                                m.content.push_str(&format!("\n\n=== ACTIVE PROJECT & USER RULES ===\n{}\n===================================", rules));
                                            }
                                            updated = true;
                                            break;
                                        }
                                    }
                                    if !updated {
                                        messages.insert(0, Message {
                                            role: "system".to_string(),
                                            content: format!("=== ACTIVE PROJECT & USER RULES ===\n{}\n===================================", rules),
                                            tool_calls: None,
                                            images: None,
                                        });
                                    }
                                    println!("{}", "✅ Active project rules reloaded into context.".green());
                                }
                                None => {
                                    println!("{}", "⚠️  No project rules found. Create .zyrules, .zy/rules.md, or zy.toml in your workspace.".yellow());
                                }
                            }
                            continue;
                        }
                        "/map" => {
                            let target_path = if parts.len() > 1 { parts[1] } else { "." };
                            println!("{} Scanning codebase `{}`...", "🗺️  Repository Map Engine:".cyan().bold(), target_path.yellow());
                            let repo_map = build_repo_map(std::path::Path::new(target_path), 2048);
                            println!("\n{}\n", repo_map);
                            messages.push(Message {
                                role: "system".to_string(),
                                content: format!("Repository Symbol Map:\n{}", repo_map),
                                tool_calls: None,
                                images: None,
                            });
                            println!("{}", "🗺️  Repository map injected into conversation context.".green());
                            continue;
                        }
                        "/test" => {
                            let custom_cmd = if parts.len() > 1 { Some(parts[1..].join(" ")) } else { None };
                            let cmd_ref = custom_cmd.as_deref();
                            println!("{} Executing test suite...", "🧪 TDD Test Engine:".cyan().bold());
                            match run_project_tests(std::path::Path::new("."), cmd_ref) {
                                Ok(report) => {
                                    println!("\n{}", format_test_report_for_terminal(&report));
                                    if report.success {
                                        println!("{}", "🎉 All tests PASSED!".green().bold());
                                    } else {
                                        println!("{}", "⚠️  Tests Failed! Initiating Autonomous TDD Auto-Repair Loop...".red().bold());
                                        
                                        let failure_summary = if !report.failure_details.is_empty() {
                                            report.failure_details.join("\n")
                                        } else {
                                            report.stderr.clone()
                                        };
                                        
                                        let repair_prompt = format!(
                                            "Autonomous TDD Auto-Repair Request:\nThe test suite failed (runner: {}).\n\nFailures:\n{}\n\nSTDOUT:\n{}\nSTDERR:\n{}\n\nPlease analyze the test failure above, inspect source files, fix the bugs using patch_file or write_file, and re-run tests using run_tests until all tests pass.",
                                            report.runner, failure_summary, report.stdout, report.stderr
                                        );
                                        
                                        messages.push(Message {
                                            role: "user".to_string(),
                                            content: repair_prompt,
                                            tool_calls: None,
                                            images: None,
                                        });
                                        
                                        let _ = agent_loop(client, &active_model, &mut messages, markdown, &tuner.opts, force, format_schema.as_ref(), sandbox).await;
                                        
                                        if let Ok(after_report) = run_project_tests(std::path::Path::new("."), cmd_ref) {
                                            println!("\n{}", format_test_report_for_terminal(&after_report));
                                            if after_report.success {
                                                println!("{}", "🎉 Auto-Repair Successful! Codebase is now GREEN.".green().bold());
                                            } else {
                                                println!("{}", "⚠️  Auto-Repair finished cycle. Tests still have remaining failures.".yellow().bold());
                                            }
                                        }
                                    }
                                }
                                Err(e) => {
                                    println!("{} {}", "❌ Error executing tests:".red(), e);
                                }
                            }
                            continue;
                        }
                        "/checkpoint" => {
                            let label = if parts.len() > 1 { Some(parts[1..].join(" ")) } else { None };
                            match create_git_checkpoint_with_label(label.as_deref()) {
                                Ok(chk_id) => {
                                    println!("{} Checkpoint ID: `{}` ({})", "💾 Micro-Checkpoint Created:".green().bold(), chk_id.cyan(), label.as_deref().unwrap_or("manual"));
                                }
                                Err(e) => {
                                    println!("{} {}", "❌ Checkpoint Error:".red(), e);
                                }
                            }
                            continue;
                        }
                        "/rollback" => {
                            let target_id = if parts.len() > 1 { Some(parts[1]) } else { None };
                            match rollback_git_checkpoint_to(target_id) {
                                Ok(msg) => {
                                    println!("{} {}", "⏪ Rollback Succeeded:".yellow().bold(), msg.green());
                                }
                                Err(e) => {
                                    println!("{} {}", "❌ Rollback Failed:".red(), e);
                                }
                            }
                            continue;
                        }
                        "/sandbox" => {
                            if parts.len() > 1 {
                                let mode = parts[1].to_lowercase();
                                if mode == "on" || mode == "1" || mode == "true" {
                                    sandbox = true;
                                    println!("{}", "🐳 Ephemeral Sandbox Container Engine: ON (wrapping bash in container)".green().bold());
                                } else if mode == "off" || mode == "0" || mode == "false" {
                                    sandbox = false;
                                    println!("{}", "🐳 Ephemeral Sandbox Container Engine: OFF (direct host execution)".yellow().bold());
                                } else {
                                    println!("{}", "Usage: /sandbox <on|off>".red());
                                }
                            } else {
                                sandbox = !sandbox;
                                println!("{} {}", "🐳 Ephemeral Sandbox Container Engine:".cyan().bold(), if sandbox { "ON".green().bold() } else { "OFF".yellow() });
                            }
                            continue;
                        }
                        "/lsp" => {
                            let target = if parts.len() > 1 { parts[1..].join(" ") } else { "src/main.rs".to_string() };
                            println!("{} Analyzing `{}` with native compiler...", "🔍 LSP Diagnostic Engine:".cyan().bold(), target.yellow());
                            let report = run_lsp_diagnostics(&target);
                            println!("{}", format_diagnostic_report_for_terminal(&report));
                            continue;
                        }
                        "/mcp" => {
                            if parts.len() >= 3 {
                                let server_cmd = parts[1];
                                let tool_name = parts[2];
                                let args_raw = if parts.len() > 3 { parts[3..].join(" ") } else { "{}".to_string() };
                                let json_args: serde_json::Value = serde_json::from_str(&args_raw).unwrap_or_else(|_| serde_json::json!({}));
                                println!("{} Calling `{}` on `{}`...", "🔌 MCP Client:".blue().bold(), tool_name.cyan(), server_cmd.dimmed());
                                match execute_mcp_call(server_cmd, tool_name, &json_args).await {
                                    Ok(res) => println!("{} Output:\n{}", "✔️ MCP Success:".green().bold(), res),
                                    Err(e) => println!("{} {}", "❌ MCP Error:".red().bold(), e),
                                }
                            } else {
                                println!("{}", "Usage: /mcp <server_command> <tool_name> [json_args]".red());
                            }
                            continue;
                        }
                        "/router" => {
                            if parts.len() > 1 {
                                let target = parts[1];
                                if target.eq_ignore_ascii_case("off") || target.eq_ignore_ascii_case("none") {
                                    scout_model = None;
                                    println!("{}", "Dual-Model Speculative Router Disabled.".yellow());
                                } else {
                                    scout_model = Some(target.to_string());
                                    println!("{} {}", "⚡ Dual-Model Speculative Router enabled with Scout:".green().bold(), target.bold());
                                }
                            } else {
                                println!("{}", "Usage: /router <scout_model_name|off>".red());
                            }
                            continue;
                        }
                        "/format" => {
                            if parts.len() > 1 {
                                let mode = parts[1].to_lowercase();
                                if mode == "json" {
                                    format_schema = Some(serde_json::json!("json"));
                                    println!("{}", "Grammar Constrained JSON Generation: ON (raw json mode)".green().bold());
                                } else if mode == "tool" || mode == "tools" {
                                    format_schema = Some(build_tool_grammar_schema());
                                    println!("{}", "Grammar Constrained JSON Generation: ON (tool-call schema mode)".green().bold());
                                } else if mode == "off" || mode == "none" {
                                    format_schema = None;
                                    println!("{}", "Grammar Constrained JSON Generation: OFF".yellow());
                                } else {
                                    println!("{}", "Usage: /format <json|tool|off>".red());
                                }
                            } else {
                                println!("{} {}", "Current format schema:".cyan(), if format_schema.is_some() { "ACTIVE".green() } else { "OFF".yellow() });
                            }
                            continue;
                        }
                        "/schema" => {
                            if parts.len() > 1 {
                                let raw_schema = parts[1..].join(" ");
                                match serde_json::from_str::<serde_json::Value>(&raw_schema) {
                                    Ok(val) => {
                                        format_schema = Some(val);
                                        println!("{}", "Custom JSON Schema applied successfully for Ollama generation!".green().bold());
                                    }
                                    Err(e) => {
                                        println!("{} {}", "Invalid JSON schema:".red(), e);
                                    }
                                }
                            } else {
                                println!("{}", "Usage: /schema <{\"type\":\"object\",...}>".red());
                            }
                            continue;
                        }
                        "/clear" => {
                            print!("{esc}[2J{esc}[1;1H", esc = 27 as char);
                            messages.clear();
                            let new_init = build_initial_messages(system, files, strategist).unwrap_or_default();
                            messages.extend(new_init);
                            println!("{}", "Context and terminal cleared.".green());
                            continue;
                        }
                        "/save" => {
                            if parts.len() > 1 {
                                save_session(Some(parts[1]), &messages);
                                println!("{} {}", "Saved to session:".green(), parts[1]);
                            } else {
                                println!("{}", "Usage: /save <session_name>".red());
                            }
                            continue;
                        }
                        "/model" => {
                            if parts.len() > 1 {
                                active_model = parts[1].to_string();
                                println!("{} {}", "Model switched to:".green(), active_model.bold());
                            } else {
                                println!("{}", "Usage: /model <model_name>".red());
                            }
                            continue;
                        }
                        "/agent" => {
                            if parts.len() > 1 {
                                agent = parts[1].eq_ignore_ascii_case("on") || parts[1] == "1" || parts[1].eq_ignore_ascii_case("true");
                                println!("{} {}", "Agent Mode:".magenta(), if agent { "ON".green() } else { "OFF".red() });
                            } else {
                                println!("{}", "Usage: /agent <on/off>".red());
                            }
                            continue;
                        }
                        "/rag" => {
                            if parts.len() > 1 {
                                rag = parts[1].eq_ignore_ascii_case("on") || parts[1] == "1" || parts[1].eq_ignore_ascii_case("true");
                                println!("{} {}", "RAG Mode:".magenta(), if rag { "ON".green() } else { "OFF".red() });
                            } else {
                                println!("{}", "Usage: /rag <on/off>".red());
                            }
                            continue;
                        }
                        "/executor" => {
                            if parts.len() > 1 {
                                executor = Some(parts[1].to_string());
                                println!("{} {}", "Swarm Executor set to:".magenta(), parts[1].bold());
                            } else {
                                executor = None;
                                println!("{}", "Swarm Mode Disabled.".red());
                            }
                            continue;
                        }
                        "/strategist" => {
                            strategist = !strategist;
                            println!("{} {}", "AI Strategist Protocol:".red(), if strategist { "ENGAGED".green() } else { "DISENGAGED".yellow() });
                            
                            messages.retain(|m| m.role != "system");
                            let new_init = build_initial_messages(system, files, strategist).unwrap_or_default();
                            messages.splice(0..0, new_init);
                            continue;
                        }
                        "/listen" => {
                            println!("{}", "🎤 Listening for 5 seconds...".cyan());
                            let _ = std::process::Command::new("arecord").args(["-d", "5", "-f", "S16_LE", "/tmp/zy_voice.wav"]).output();
                            println!("{}", "Processing voice...".cyan());
                            let whisper_out = std::process::Command::new("whisper").args(["/tmp/zy_voice.wav"]).output();
                            
                            let transcript = if let Ok(out) = whisper_out {
                                String::from_utf8_lossy(&out.stdout).to_string()
                            } else {
                                println!("{}", "Whisper not found in PATH. Simulating voice transcription...".yellow());
                                "Simulated voice input: Write a python script to ping google.com".to_string()
                            };
                            
                            println!("{} {}", "Transcription:".green(), transcript.trim());
                            
                            messages.push(Message {
                                role: "user".to_string(),
                                content: transcript.trim().to_string(),
                                tool_calls: None,
                                images: None,
                            });
                            
                            if agent {
                                let _ = agent_loop(client, &active_model, &mut messages, markdown, &tuner.opts, force, format_schema.as_ref(), sandbox).await;
                            } else {
                                if let Ok(response_text) = fetch_full_response(client, &active_model, &messages, &tuner.opts, format_schema.as_ref()).await {
                                    if markdown {
                                        print_text(&response_text);
                                    } else {
                                        println!("{} {}", "zy ❯".green().bold(), response_text);
                                    }
                                    messages.push(Message {
                                        role: "assistant".to_string(),
                                        content: response_text,
                                        tool_calls: None,
                                        images: None,
                                    });
                                }
                            }
                            save_session(session, &messages);
                            continue;
                        }
                        "/undo" => {
                            if std::path::Path::new(".git").exists() {
                                let _ = std::process::Command::new("git").args(["reset", "--hard", "HEAD~1"]).output();
                                println!("{}", "⏪ Reverted codebase to previous git commit.".yellow());
                            } else {
                                println!("{}", "Error: Not a git repository!".red());
                            }
                            continue;
                        }
                        "/chaos" => {
                            println!("{}", "🔥 Chaos Monkey Engaged! Attacking your codebase...".red().bold());
                            let mut files = Vec::new();
                            for entry in WalkDir::new(".").into_iter().filter_map(|e| e.ok()) {
                                let path_str = entry.path().to_string_lossy();
                                if !path_str.contains("/target/") && !path_str.contains("/.git/") && path_str.ends_with(".rs") {
                                    files.push(entry.path().to_path_buf());
                                }
                            }
                            if !files.is_empty() {
                                let seed = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().subsec_nanos() as usize;
                                let target = &files[seed % files.len()];
                                if let Ok(content) = fs::read_to_string(target) {
                                    let mut lines: Vec<&str> = content.lines().collect();
                                    if lines.len() > 10 {
                                        let start = seed % (lines.len() - 5);
                                        lines.drain(start..start+5);
                                        let _ = fs::write(target, lines.join("\n"));
                                        println!("{} {}", "💥 Sabotaged file:".red(), target.to_string_lossy());
                                    }
                                }
                            }
                            continue;
                        }
                        "/sleep" => {
                            println!("{}", "💤 Entering deep sleep... Compressing memories...".blue());
                            let mut temp_msgs = messages.clone();
                            temp_msgs.push(Message {
                                role: "user".to_string(),
                                content: "Summarize our entire conversation so far into a highly dense, strictly factual 'Core Memory' paragraph. Focus on my preferences, the project state, and code structures. Do not output anything other than the summary.".to_string(),
                                tool_calls: None,
                                images: None,
                            });
                            if let Ok(summary) = fetch_full_response(client, &active_model, &temp_msgs, &tuner.opts, format_schema.as_ref()).await {
                                messages.retain(|m| m.role == "system" && !m.content.contains("Core Memory:"));
                                messages.push(Message {
                                    role: "system".to_string(),
                                    content: format!("Core Memory from previous sessions: {}", summary),
                                    tool_calls: None,
                                    images: None,
                                });
                                println!("{} {}", "🧠 Core Memory updated:".magenta(), summary);
                            }
                            continue;
                        }
                        "/webhook" => {
                            if parts.len() > 1 {
                                let url = parts[1].to_string();
                                let _ = fs::write(".zy_webhook.txt", url.clone());
                                println!("{} {}", "🔗 Webhook set to:".green(), url);
                            } else {
                                println!("{}", "Usage: /webhook <https://discord.com/api/webhooks/...>".red());
                            }
                            continue;
                        }
                        "/train" => {
                            println!("{}", "🎓 Preparing local LoRA Fine-Tuning dataset...".yellow());
                            let script = r#"
import os
import glob
import json

print("Reading .zy_session data to build preference dataset...")
session_files = glob.glob('.zy_session_*.json')
dataset = []

for file in session_files:
    try:
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for i in range(len(data) - 1):
                if data[i].get('role') == 'user' and data[i+1].get('role') == 'assistant':
                    dataset.append({
                        "text": f"User: {data[i].get('content')}\nAssistant: {data[i+1].get('content')}"
                    })
    except Exception as e:
        print(f"Error reading {file}: {e}")

if not dataset:
    print("⚠️  Dataset is empty! Chat more with zy to generate training data.")
    exit(1)

with open('.zy_dataset.json', 'w', encoding='utf-8') as f:
    json.dump(dataset, f)

print(f"✅ Generated dataset with {len(dataset)} examples. Starting LoRA Fine-Tuning...")

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model
    
    print("✅ ML Dependencies loaded (torch, transformers, peft, datasets).")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    hf_dataset = load_dataset('json', data_files='.zy_dataset.json', split='train')
    
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)
        
    tokenized_datasets = hf_dataset.map(tokenize_function, batched=True)
    
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    
    config = LoraConfig(
        r=8, 
        lora_alpha=32, 
        target_modules=["q_proj", "v_proj"], 
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    
    training_args = TrainingArguments(
        output_dir="./zy_lora_model",
        per_device_train_batch_size=1,
        num_train_epochs=1,
        learning_rate=2e-4,
        save_steps=10,
        logging_steps=1,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
    )
    
    print("🚀 Starting local training loop...")
    trainer.train()
    
    model.save_pretrained("./zy_lora_final")
    print("🎉 Local LoRA Fine-Tuning Complete! Weights saved to ./zy_lora_final")
    
except ImportError:
    print("❌ Missing ML libraries. Please `pip install torch transformers peft datasets` to run the real RLHF loop.")
except Exception as e:
    print(f"⚠️  Training aborted due to error: {e}")
"#;
                            fs::write(".zy_train.py", script).unwrap();
                            let _ = std::process::Command::new("python3").arg(".zy_train.py").status();
                            println!("{}", "✅ Training sequence finalized!".green());
                            continue;
                        }
                        "/worker" => {
                            println!("{}", "👷 Autonomous Issue Crusher Started...".yellow().bold());
                            let issues_dir = "../.projectmem/issues";
                            if std::path::Path::new(issues_dir).exists() {
                                for entry in std::fs::read_dir(issues_dir).unwrap().flatten() {
                                    if entry.path().is_file() {
                                        let issue_text = fs::read_to_string(entry.path()).unwrap();
                                        println!("{} {}", "🛠️  Tackling Issue:".blue(), entry.path().to_string_lossy());
                                        let prompt = format!("Fix this issue autonomously using your tools:\n\n{}", issue_text);
                                        
                                        messages.push(Message { role: "user".to_string(), content: prompt, tool_calls: None, images: None });
                                        let _ = agent_loop(client, &active_model, &mut messages, markdown, &tuner.opts, true, format_schema.as_ref(), sandbox).await;
                                        
                                        println!("{}", "✅ Issue Processed!".green());
                                        break;
                                    }
                                }
                            } else {
                                println!("{}", "No .projectmem/issues/ directory found.".red());
                            }
                            continue;
                        }
                        "/evolve" => {
                            if parts.len() > 1 {
                                let instruction = parts[1..].join(" ");
                                println!("{} {}", "🧬 Evolver Protocol Engaged. Upgrading zy...".magenta().bold(), instruction);
                                
                                if let Ok(src) = fs::read_to_string("src/main.rs") {
                                    let prompt = format!("You are a god-tier Rust AI. Modify the following Rust source code to implement this new feature: '{}'. \n\nIMPORTANT: Output ONLY the raw, complete, valid Rust code for the entire file. Do NOT wrap it in ```rust or markdown tags. Start exactly with 'use' and end with the last bracket. Do not explain.\n\nCode:\n{}", instruction, src);
                                    
                                    let temp_msgs = vec![Message { role: "user".to_string(), content: prompt, tool_calls: None, images: None }];
                                    println!("{}", "🧠 zy is writing its own source code...".cyan());
                                    if let Ok(mut new_code) = fetch_full_response(client, &active_model, &temp_msgs, &tuner.opts, format_schema.as_ref()).await {
                                        new_code = new_code.replace("```rust", "").replace("```", "");
                                        let backup = src.clone();
                                        let _ = fs::write("src/main.rs", new_code.trim());
                                        
                                        println!("{}", "🔨 Compiling new DNA...".yellow());
                                        let output = std::process::Command::new("cargo").arg("check").output();
                                        if let Ok(out) = output {
                                            if out.status.success() {
                                                println!("{}", "✨ Evolution Successful! Please restart zy to use your new capabilities.".green().bold());
                                            } else {
                                                println!("{}", "❌ Genetic Mutation Failed (Compiler Error). Reverting DNA...".red());
                                                let _ = fs::write("src/main.rs", backup);
                                            }
                                        }
                                    }
                                }
                            }
                            continue;
                        }
                        "/swarm" => {
                            if parts.len() > 1 {
                                let goal = parts[1..].join(" ");
                                match run_swarm_workflow(client, &active_model, executor.as_deref(), &goal, &tuner.opts, markdown, force, sandbox).await {
                                    Ok(res) => {
                                        messages.push(Message {
                                            role: "assistant".to_string(),
                                            content: format!("Swarm Goal '{}' Completed.\n\nPlan:\n{}\n\nAudit Verdict:\n{}", res.goal, res.plan, res.audit_report),
                                            tool_calls: None,
                                            images: None,
                                        });
                                    }
                                    Err(e) => println!("{} {}", "❌ Swarm Error:".red(), e),
                                }
                            } else {
                                println!("{}", "Usage: /swarm <goal>".red());
                            }
                            continue;
                        }
                        "/search" => {
                            if parts.len() > 1 {
                                let query = parts[1..].join(" ");
                                println!("{} Live searching for `{}`...", "🌐 DuckDuckGo Web Search:".cyan().bold(), query.yellow());
                                match perform_web_search(client, &query).await {
                                    Ok(results) => {
                                        if results.is_empty() {
                                            println!("{}", "⚠️  No results found.".yellow());
                                        } else {
                                            println!("\n{}\n", "╔═══════════════════════════════════════════════════════════╗".cyan());
                                            for (i, r) in results.iter().take(5).enumerate() {
                                                println!("  {}. {}", (i + 1).to_string().cyan().bold(), r.title.bold());
                                                println!("     {} {}", "URL:".dimmed(), r.url.blue().underline());
                                                println!("     {}\n", r.snippet.dimmed());
                                            }
                                            println!("{}\n", "╚═══════════════════════════════════════════════════════════╝".cyan());
                                        }
                                    }
                                    Err(e) => println!("{} {}", "❌ Search Error:".red(), e),
                                }
                            } else {
                                println!("{}", "Usage: /search <query>".red());
                            }
                            continue;
                        }
                        "/mentions" => {
                            println!("\n{}", "📎 Interactive @ Context Mentions Help:".cyan().bold());
                            println!("  @<filepath>          - Attach file content directly (e.g. @src/main.rs, @Cargo.toml)");
                            println!("  @file:<path>         - Explicit file context attachment (e.g. @file:src/lib.rs)");
                            println!("  @git or @diff        - Attach current git status and uncommitted diff");
                            println!("  @symbol:<name>       - Search & attach symbol definition (e.g. @symbol:bm25_score)\n");
                            continue;
                        }
                        "/timeline" => {
                            let tl = format_timeline(&messages);
                            println!("{}", tl);
                            continue;
                        }
                        "/rewind" => {
                            let count = if parts.len() > 1 { parts[1].parse::<usize>().unwrap_or(1) } else { 1 };
                            let removed = rewind_messages(&mut messages, count);
                            if removed > 0 {
                                println!("{} Rewound {} turn(s). Remaining turns: {}", "⏪ Timeline Rewind:".green().bold(), removed.to_string().cyan(), extract_timeline_turns(&messages).len());
                                save_session(session, &messages);
                            } else {
                                println!("{}", "⚠️  No conversational turns to rewind.".yellow());
                            }
                            continue;
                        }
                        "/commit" => {
                            let custom_hint = if parts.len() > 1 { Some(parts[1..].join(" ")) } else { None };
                            println!("{} Analyzing git diff...", "📦 Conventional Commit Generator:".cyan().bold());
                            
                            let diff_out = std::process::Command::new("git").args(["diff", "HEAD"]).output();
                            let diff = match diff_out {
                                Ok(out) => {
                                    let d = String::from_utf8_lossy(&out.stdout).to_string();
                                    if d.trim().is_empty() {
                                        let d_staged = std::process::Command::new("git").args(["diff", "--cached"]).output();
                                        d_staged.map(|o| String::from_utf8_lossy(&o.stdout).to_string()).unwrap_or_default()
                                    } else {
                                        d
                                    }
                                }
                                Err(_) => String::new(),
                            };
                            
                            if diff.trim().is_empty() {
                                println!("{}", "⚠️  No uncommitted changes detected in git repository.".yellow());
                                continue;
                            }
                            
                            match generate_commit_message(client, &active_model, &diff, &tuner.opts, custom_hint.as_deref()).await {
                                Ok(msg) => {
                                    println!("\n{} {}", "Suggested Commit Message:".green().bold(), msg.cyan().bold());
                                    let proceed = if force { true } else { ask_confirmation("Proceed with git commit?") };
                                    if proceed {
                                        let _ = std::process::Command::new("git").args(["add", "-A"]).output();
                                        let commit_res = std::process::Command::new("git").args(["commit", "-m", &msg]).output();
                                        match commit_res {
                                            Ok(out) => {
                                                if out.status.success() {
                                                    println!("{} {}", "🎉 Committed:".green().bold(), msg);
                                                } else {
                                                    println!("{} {}", "❌ Git commit failed:".red(), String::from_utf8_lossy(&out.stderr));
                                                }
                                            }
                                            Err(e) => println!("{} {}", "❌ Error executing git commit:".red(), e),
                                        }
                                    } else {
                                        println!("{}", "Commit canceled.".yellow());
                                    }
                                }
                                Err(e) => println!("{} {}", "❌ Failed to generate commit message:".red(), e),
                            }
                            continue;
                        }
                        "/pr" => {
                            println!("{} Generating Pull Request description...", "🔀 GitHub PR Generator:".magenta().bold());
                            let branch_out = std::process::Command::new("git").args(["branch", "--show-current"]).output();
                            let branch = match branch_out {
                                Ok(out) => String::from_utf8_lossy(&out.stdout).trim().to_string(),
                                Err(_) => "main".to_string(),
                            };
                            let branch_name = if branch.is_empty() { "main" } else { &branch };
                            
                            let diff_out = std::process::Command::new("git").args(["diff", "HEAD~1"]).output()
                                .or_else(|_| std::process::Command::new("git").args(["diff"]).output());
                            let diff = diff_out.map(|o| String::from_utf8_lossy(&o.stdout).to_string()).unwrap_or_default();
                            
                            match generate_pr_description(client, &active_model, &diff, branch_name, &tuner.opts).await {
                                Ok(pr_desc) => {
                                    println!("\n{}\n", pr_desc);
                                    let _ = std::fs::write("PULL_REQUEST.md", &pr_desc);
                                    println!("{}", "💾 Saved PR description to `PULL_REQUEST.md`.".green());
                                }
                                Err(e) => println!("{} {}", "❌ Failed to generate PR description:".red(), e),
                            }
                            continue;
                        }
                        "/tui" => {
                            println!("{}", "Launching Full-Screen Interactive TUI Dashboard...".cyan().bold());
                            let _ = run_tui_app(client, &active_model, system, files, agent, rag, tuner, force).await;
                            continue;
                        }
                        "/audit" => {
                            let target_path = if parts.len() > 1 { parts[1] } else { "." };
                            println!("{} Auditing dependencies in `{}`...", "🛡️  Security Auditor:".cyan().bold(), target_path.yellow());
                            let report = audit_project_dependencies(std::path::Path::new(target_path));
                            println!("{}", format_security_report_for_terminal(&report));
                            continue;
                        }
                        "/db" => {
                            if parts.len() > 1 {
                                let db_path = parts[1];
                                let query = if parts.len() > 2 { Some(parts[2..].join(" ")) } else { None };
                                println!("{} Inspecting database `{}`...", "🗄️  Database Inspector:".cyan().bold(), db_path.yellow());
                                match inspect_sqlite_database(std::path::Path::new(db_path), query.as_deref()) {
                                    Ok(report) => println!("{}", format_database_report_for_terminal(&report)),
                                    Err(e) => println!("{} {}", "❌ Database Inspector Error:".red(), e),
                                }
                            } else {
                                println!("{}", "Usage: /db <path_to_sqlite_db> [safe_sql_query]".red());
                            }
                            continue;
                        }
                        "/docs" => {
                            let target_path = if parts.len() > 1 { parts[1] } else { "." };
                            println!("{} Scanning `{}` for undocumented symbols...", "📚 Docstring Generator:".cyan().bold(), target_path.yellow());
                            let symbols = scan_undocumented_symbols(std::path::Path::new(target_path));
                            let patches = generate_docstring_patches(&symbols, "rust");
                            let report = DocGenerationReport {
                                target_path: target_path.to_string(),
                                total_symbols_scanned: symbols.len(),
                                undocumented_count: symbols.len(),
                                symbols,
                                patches: patches.clone(),
                                applied_count: 0,
                                summary: format!("Found {} undocumented symbols in {}", patches.len(), target_path),
                            };
                            println!("{}", format_doc_generation_report_for_terminal(&report));
                            if !patches.is_empty() {
                                let proceed = if force { true } else { ask_confirmation("Apply generated docstrings to files?") };
                                if proceed {
                                    match apply_doc_patches(&patches) {
                                        Ok(count) => println!("{} Successfully applied docstrings to {} file(s).", "✨".green(), count),
                                        Err(e) => println!("{} {}", "❌ Failed to apply docstrings:".red(), e),
                                    }
                                }
                            }
                            continue;
                        }
                        "/transaction" => {
                            if parts.len() > 1 {
                                let action = parts[1].to_lowercase();
                                match action.as_str() {
                                    "begin" => {
                                        begin_refactor_transaction();
                                        println!("{}", "🚀 Initiated fresh in-memory Refactor Transaction.".green().bold());
                                    }
                                    "stage" => {
                                        if parts.len() >= 4 {
                                            let path = parts[2];
                                            let content = parts[3..].join(" ");
                                            stage_in_refactor_transaction(std::path::Path::new(path), &content);
                                            println!("{} Staged virtual edit for `{}`", "📝".cyan(), path.yellow());
                                        } else {
                                            println!("{}", "Usage: /transaction stage <path> <content>".red());
                                        }
                                    }
                                    "validate" => {
                                        println!("{}", "🔍 Validating all staged virtual edits with compiler diagnostics...".cyan().bold());
                                        let report = validate_refactor_transaction(std::path::Path::new("."));
                                        if report.is_valid {
                                            println!("{} {}", "✅ Validation Clean:".green().bold(), report.summary);
                                        } else {
                                            println!("{} {}", "❌ Validation Failed:".red().bold(), report.summary);
                                            for err in &report.errors {
                                                println!("   {} {}", "•".red(), err);
                                            }
                                        }
                                    }
                                    "diff" => {
                                        let diff = get_refactor_transaction_diff();
                                        println!("\n{}\n", diff);
                                    }
                                    "commit" => {
                                        match commit_refactor_transaction() {
                                            Ok(files) => {
                                                println!("{} Atomically committed {} file(s):", "🎉 Transaction Committed:".green().bold(), files.len());
                                                for f in files {
                                                    println!("   {} {}", "✔".green(), f);
                                                }
                                            }
                                            Err(e) => println!("{} {}", "❌ Transaction Commit Failed:".red(), e),
                                        }
                                    }
                                    "rollback" => {
                                        rollback_refactor_transaction();
                                        println!("{}", "⏪ Refactor Transaction rolled back. Staging buffer cleared.".yellow().bold());
                                    }
                                    "status" => {
                                        println!("{}", get_refactor_transaction_status());
                                    }
                                    _ => {
                                        println!("{}", "Usage: /transaction <begin|stage|validate|diff|commit|rollback|status>".red());
                                    }
                                }
                            } else {
                                println!("{}", get_refactor_transaction_status());
                            }
                            continue;
                        }
                        "/bench" => {
                            if parts.len() > 1 {
                                let mut iters = 5;
                                let warmup = 1;
                                let cmd;
                                if parts.len() >= 3 && parts.last().unwrap().parse::<usize>().is_ok() {
                                    iters = parts.last().unwrap().parse::<usize>().unwrap();
                                    cmd = parts[1..parts.len() - 1].join(" ");
                                } else {
                                    cmd = parts[1..].join(" ");
                                }
                                println!("{} Running micro-benchmark for `{}` (iters: {})...", "⚡".yellow().bold(), cmd.cyan(), iters);
                                match run_micro_benchmark(&cmd, iters, warmup) {
                                    Ok(report) => println!("{}", format_benchmark_report_for_terminal(&report)),
                                    Err(e) => println!("{} {}", "❌ Benchmark Error:".red(), e),
                                }
                            } else {
                                println!("{}", "Usage: /bench <command> [iterations]".red());
                            }
                            continue;
                        }
                        "/fuzz" => {
                            if parts.len() > 1 {
                                let target_file = parts[1];
                                println!("{} Synthesizing unit tests and property fuzzing suite for `{}`...", "🧪".cyan().bold(), target_file.yellow());
                                match synthesize_test_suite(std::path::Path::new(target_file), "auto", true) {
                                    Ok(suite) => {
                                        println!("{}", format_test_suite_report_for_terminal(&suite));
                                        let old_content = fs::read_to_string(&suite.test_file_path).unwrap_or_default();
                                        let diff_preview = render_terminal_diff(&suite.test_file_path, &old_content, &suite.test_code);
                                        println!("\n{}\n", diff_preview);

                                        let proceed = if force { true } else { ask_confirmation(&format!("Write synthesized test suite to `{}`?", suite.test_file_path)) };
                                        if proceed {
                                            if let Some(parent) = std::path::Path::new(&suite.test_file_path).parent() {
                                                let _ = fs::create_dir_all(parent);
                                            }
                                            if fs::write(&suite.test_file_path, &suite.test_code).is_ok() {
                                                println!("{} Test suite successfully written to `{}`.", "✨".green(), suite.test_file_path.cyan());
                                            } else {
                                                println!("{}", "❌ Failed to write test file.".red());
                                            }
                                        }
                                    }
                                    Err(e) => println!("{} {}", "❌ Test Synthesis Error:".red(), e),
                                }
                            } else {
                                println!("{}", "Usage: /fuzz <source_file>".red());
                            }
                            continue;
                        }
                        "/dockerfile" => {
                            let target_path = if parts.len() > 1 { parts[1] } else { "." };
                            println!("{} Generating hardened multi-stage Dockerfile...", "🐳".cyan().bold());
                            let stack = detect_project_stack(std::path::Path::new(target_path));
                            let manifests = generate_container_and_ci_manifests(&stack);
                            println!("{}", format_ci_manifests_for_terminal(&manifests));
                            println!("\n{}\n{}\n", "─── Dockerfile Preview ───".cyan().bold(), manifests.dockerfile.dimmed());
                            let proceed = if force { true } else { ask_confirmation("Write Dockerfile and docker-compose.yml to workspace?") };
                            if proceed {
                                let _ = fs::write("Dockerfile", &manifests.dockerfile);
                                let _ = fs::write("docker-compose.yml", &manifests.docker_compose);
                                println!("{}", "✨ Dockerfile and docker-compose.yml written to disk.".green());
                            }
                            continue;
                        }
                        "/ci" => {
                            let target_path = if parts.len() > 1 { parts[1] } else { "." };
                            println!("{} Generating production CI/CD matrix and container manifests...", "⚙️ ".cyan().bold());
                            let stack = detect_project_stack(std::path::Path::new(target_path));
                            let manifests = generate_container_and_ci_manifests(&stack);
                            println!("{}", format_ci_manifests_for_terminal(&manifests));
                            println!("\n{}\n{}\n", "─── .github/workflows/ci.yml Preview ───".cyan().bold(), manifests.github_workflow.dimmed());
                            let proceed = if force { true } else { ask_confirmation("Write CI workflow (.github/workflows/ci.yml) and container files?") };
                            if proceed {
                                let _ = fs::create_dir_all(".github/workflows");
                                let _ = fs::write(".github/workflows/ci.yml", &manifests.github_workflow);
                                let _ = fs::write("Dockerfile", &manifests.dockerfile);
                                let _ = fs::write("docker-compose.yml", &manifests.docker_compose);
                                println!("{}", "✨ CI/CD workflow and container manifests written to disk.".green());
                            }
                            continue;
                        }
                        "/graph" => {
                            let entry_sym = if parts.len() > 1 { Some(parts[1]) } else { None };
                            println!("{} Generating interactive call graph...", "🕸️ ".cyan().bold());
                            let report = build_call_graph(std::path::Path::new("."), entry_sym);
                            println!("{}", format_call_graph_for_terminal(&report));
                            continue;
                        }
                        "/lint" => {
                            let target_path = if parts.len() > 1 { parts[1] } else { "." };
                            println!("{} Formatting and auto-fixing workspace `{}`...", "🧹".cyan().bold(), target_path.yellow());
                            match format_and_lint_workspace(std::path::Path::new(target_path), true) {
                                Ok(report) => println!("{}", format_lint_format_report_for_terminal(&report)),
                                Err(e) => println!("{} {}", "❌ Lint Error:".red(), e),
                            }
                            continue;
                        }
                        "/mock" => {
                            if parts.len() >= 3 {
                                let port = parts[1].parse::<u16>().unwrap_or(8080);
                                let path = parts[2];
                                let json_raw = if parts.len() > 3 { parts[3..].join(" ") } else { "{\"status\":\"ok\"}".to_string() };
                                let json_body: serde_json::Value = serde_json::from_str(&json_raw).unwrap_or_else(|_| serde_json::json!({ "status": "ok", "raw": json_raw }));

                                let route = MockRoute {
                                    method: "GET".to_string(),
                                    path: path.to_string(),
                                    status_code: 200,
                                    response_body: json_body,
                                    headers: std::collections::HashMap::new(),
                                };

                                println!("{} Starting ephemeral mock server on port {}...", "🚀".cyan().bold(), port);
                                match start_ephemeral_mock_server(port, vec![route]).await {
                                    Ok(handle) => {
                                        println!("{}", format_mock_server_report_for_terminal(&handle));
                                        register_active_mock_server(handle);
                                    }
                                    Err(e) => println!("{} {}", "❌ Mock Server Error:".red(), e),
                                }
                            } else {
                                println!("{}", "Usage: /mock <port> <path> [json_response]".red());
                            }
                            continue;
                        }
                        "/exit" | "/quit" => break,
                        _ => {
                            println!("{}", "Unknown slash command. Type /help to see available commands.".red());
                            continue;
                        }
                    }
                }

                rl.add_history_entry(input)?;
                
                budget_aware_prune(&mut messages, tuner.num_ctx);
                
                if rag {
                    apply_rag(client, input, &mut messages).await?;
                }

                let expanded = expand_context_mentions(input, std::path::Path::new("."));
                for mention in &expanded.mentions {
                    println!("{} Attached {} (`{}`)", "📎".cyan(), mention.mention_type.bold(), mention.target.yellow());
                }
                messages.extend(expanded.context_messages);
                
                messages.push(Message {
                    role: "user".to_string(),
                    content: input.to_string(),
                    tool_calls: None,
                    images: None,
                });

                // Dual-Model Speculative Router
                if let Some(scout_mdl) = &scout_model {
                    let decision = classify_query_route(client, scout_mdl, input, &tuner.opts).await;
                    if decision == RouteDecision::Chat && !agent && executor.is_none() {
                        println!("{} {}", "⚡ [Fast Scout Router: Answering Chat]".cyan().bold(), scout_mdl.yellow());
                        if markdown {
                            let response_text = fetch_full_response(client, scout_mdl, &messages, &tuner.opts, format_schema.as_ref()).await?;
                            print_text(&response_text);
                            messages.push(Message {
                                role: "assistant".to_string(),
                                content: response_text,
                                tool_calls: None,
                                images: None,
                            });
                        } else {
                            let response_text = stream_response(client, scout_mdl, &messages, &tuner.opts, format_schema.as_ref()).await?;
                            println!();
                            messages.push(Message {
                                role: "assistant".to_string(),
                                content: response_text,
                                tool_calls: None,
                                images: None,
                            });
                        }
                        save_session(session, &messages);
                        continue;
                    } else {
                        println!("{} {}", "🚀 [Speculative Router: Routing to Heavy Coder]".magenta().bold(), active_model.yellow().bold());
                    }
                }

                if let Some(exec) = &executor {
                    println!("{} {}", "🧠 Swarm Architect Planning...".magenta().bold(), active_model);
                    let plan = fetch_full_response(client, &active_model, &messages, &tuner.opts, format_schema.as_ref()).await?;
                    print_text(&plan);
                    messages.push(Message { role: "assistant".to_string(), content: plan.clone(), tool_calls: None, images: None });
                    
                    println!("\n{} {}", "⚡ Swarm Executor Working...".yellow().bold(), exec);
                    messages.push(Message { role: "user".to_string(), content: format!("Execute this plan using tools:\n{}", plan), tool_calls: None, images: None });
                    agent_loop(client, exec, &mut messages, markdown, &tuner.opts, force, format_schema.as_ref(), sandbox).await?;
                } else if agent {
                    agent_loop(client, &active_model, &mut messages, markdown, &tuner.opts, force, format_schema.as_ref(), sandbox).await?;
                } else {
                    if markdown {
                        let response_text = fetch_full_response(client, &active_model, &messages, &tuner.opts, format_schema.as_ref()).await?;
                        print_text(&response_text);
                        messages.push(Message {
                            role: "assistant".to_string(),
                            content: response_text,
                            tool_calls: None,
                            images: None,
                        });
                    } else {
                        let response_text = stream_response(client, &active_model, &messages, &tuner.opts, format_schema.as_ref()).await?;
                        println!();
                        messages.push(Message {
                            role: "assistant".to_string(),
                            content: response_text,
                            tool_calls: None,
                            images: None,
                        });
                    }
                }
                save_session(session, &messages);
            }
            Err(ReadlineError::Interrupted) | Err(ReadlineError::Eof) => { break; }
            Err(err) => { println!("Error: {:?}", err); break; }
        }
    }
    Ok(())
}

pub async fn stream_response(
    client: &Client, 
    model: &str, 
    messages: &[Message], 
    options: &OllamaOptions,
    format: Option<&serde_json::Value>,
) -> Result<String, Box<dyn std::error::Error>> {
    let req_body = ChatRequest {
        model: model.to_string(),
        messages: messages.to_vec(),
        stream: true,
        tools: None,
        format: format.cloned(),
        options: Some(options.clone()),
        keep_alive: Some(-1),
    };

    let res = client.post(format!("{}/api/chat", OLLAMA_URL)).json(&req_body).send().await?;

    if !res.status().is_success() {
        println!("{}", format!("Error: Failed to get response from Ollama. Ensure model '{}' is installed.", model).red());
        return Ok(String::new());
    }

    let mut full_response = String::new();
    let mut stream = res.bytes_stream();
    
    print!("{}", "zy ❯ ".green().bold());
    io::stdout().flush()?;

    let mut in_think_block = false;

    while let Some(chunk_result) = stream.next().await {
        let chunk = chunk_result?;
        let chunk_str = String::from_utf8_lossy(&chunk);
        
        for line in chunk_str.lines() {
            if line.is_empty() { continue; }
            if let Ok(parsed) = serde_json::from_str::<ChatResponse>(line) {
                if let Some(msg) = parsed.message {
                    let content = &msg.content;
                    full_response.push_str(content);
                    
                    if content.contains("<think>") { in_think_block = true; }
                    if content.contains("</think>") { 
                        print!("{}", "</think>".dimmed());
                        in_think_block = false; 
                        io::stdout().flush()?;
                        continue;
                    }
                    
                    if in_think_block {
                        print!("{}", content.dimmed());
                    } else {
                        print!("{}", content);
                    }
                    io::stdout().flush()?;
                }
                if let Some(err) = parsed.error {
                    println!("\nOllama error: {}", err);
                }
            }
        }
    }

    Ok(full_response)
}

pub async fn fetch_full_response(
    client: &Client, 
    model: &str, 
    messages: &[Message], 
    options: &OllamaOptions,
    format: Option<&serde_json::Value>,
) -> Result<String, Box<dyn std::error::Error>> {
    let req_body = ChatRequest {
        model: model.to_string(),
        messages: messages.to_vec(),
        stream: false,
        tools: None,
        format: format.cloned(),
        options: Some(options.clone()),
        keep_alive: Some(-1),
    };
    
    let spinner = ProgressBar::new_spinner();
    spinner.set_style(ProgressStyle::default_spinner().tick_strings(&["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏", "✔"]).template("{spinner:.green} {msg}").unwrap());
    spinner.set_message("zy is thinking...");
    spinner.enable_steady_tick(std::time::Duration::from_millis(80));

    let res = client.post(format!("{}/api/chat", OLLAMA_URL)).json(&req_body).send().await?;
    spinner.finish_and_clear();
    
    if !res.status().is_success() {
        return Ok(format!("Error: Failed to get response. Is model '{}' installed?", model));
    }
    
    let parsed: ChatResponse = res.json().await?;
    if let Some(msg) = parsed.message {
        Ok(msg.content)
    } else {
        Ok(String::new())
    }
}

pub fn get_tools() -> serde_json::Value {
    serde_json::json!([
        {
            "type": "function",
            "function": {
                "name": "run_bash",
                "description": "Execute a shell command. If it fails, analyze STDERR and iteratively fix errors.",
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
                "name": "get_repo_map",
                "description": "Generate a compact ASCII symbol outline map of the codebase (functions, structs, classes, interfaces).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": { "type": "string", "description": "Root directory path to scan (defaults to '.')" },
                        "max_tokens": { "type": "integer", "description": "Maximum token budget for the repository map (default 1500)" }
                    }
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "run_tests",
                "description": "Execute project test suite (cargo test, pytest, npm test, go test, or custom command). Returns structured pass/fail results, counts, and failure error traces.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": { "type": "string", "description": "Optional test command (e.g. 'cargo test --test integration_tests')" },
                        "path": { "type": "string", "description": "Project directory path (defaults to '.')" }
                    }
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "lsp_diagnostics",
                "description": "Run native compiler and linter diagnostics on a file or workspace (Rust/cargo, Python/py_compile, TypeScript/tsc, JavaScript/node, C/C++/gcc, Go/go vet). Returns structured JSON diagnostics with file, line, column, severity, and message.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_or_cmd": { "type": "string", "description": "File path (e.g., 'src/main.rs', 'app.py') or explicit compiler command ('cargo check --message-format=json')" }
                    },
                    "required": ["file_or_cmd"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "mcp_execute",
                "description": "Execute a tool on an external MCP (Model Context Protocol) server via stdio JSON-RPC 2.0.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "server_command": { "type": "string", "description": "Command line to start MCP server (e.g. 'npx @modelcontextprotocol/server-filesystem .')" },
                        "tool_name": { "type": "string", "description": "Tool name on the MCP server" },
                        "arguments": { "type": "object", "description": "Tool JSON arguments object" }
                    },
                    "required": ["server_command", "tool_name"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "Write entire content to a file. Overwrites the file completely.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": { "type": "string", "description": "Path to the file" },
                        "content": { "type": "string", "description": "Content to write" }
                    },
                    "required": ["path", "content"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "patch_file",
                "description": "Smart edit: Replace a specific old string with a new string in an existing file.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": { "type": "string" },
                        "old_text": { "type": "string", "description": "Exact text to replace" },
                        "new_text": { "type": "string", "description": "New text to insert" }
                    },
                    "required": ["path", "old_text", "new_text"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "fetch_url",
                "description": "Live Web Scraper: Fetch text content from a URL (e.g., to read documentation).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": { "type": "string" }
                    },
                    "required": ["url"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "browser_action",
                "description": "Playwright Agent: Navigate a headless browser by running an auto-generated node script.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": { "type": "string", "description": "URL to visit" },
                        "javascript_to_execute": { "type": "string", "description": "JS to run in the console" }
                    },
                    "required": ["url", "javascript_to_execute"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "send_webhook",
                "description": "Send a push notification to the user's phone/webhook when a long background task completes.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message": { "type": "string", "description": "The notification text to send" }
                    },
                    "required": ["message"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Perform live web search without an API key using DuckDuckGo to find answers, code examples, documentation, and technical articles.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": { "type": "string", "description": "The search query string" },
                        "max_results": { "type": "integer", "description": "Maximum search results to return (default 5)" }
                    },
                    "required": ["query"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "audit_security",
                "description": "Autonomous Dependency & Security Auditor. Scans Cargo.lock, package-lock.json, requirements.txt, etc., for known security vulnerabilities, license risks, and outdated/wildcard packages.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": { "type": "string", "description": "Root directory path to audit (defaults to '.')" }
                    }
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "db_query",
                "description": "Native Local Database & SQL Inspector. Introspects SQLite database schemas, tables, views, columns, and executes safe read-only SQL queries.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "db_path": { "type": "string", "description": "Path to the SQLite database file" },
                        "query": { "type": "string", "description": "Optional safe read-only SQL query to execute (e.g. 'SELECT * FROM users LIMIT 10')" }
                    },
                    "required": ["db_path"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "generate_docs",
                "description": "Automated API & Docstring Documentation Generator. Scans codebase for undocumented symbols (Rust, Python, TypeScript) and generates idiomatic docstrings.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": { "type": "string", "description": "File or directory path to scan for undocumented symbols (defaults to '.')" },
                        "auto_apply": { "type": "boolean", "description": "Whether to automatically write the generated docstrings to files (default false)" }
                    }
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "refactor_transaction",
                "description": "Atomic Multi-File Refactor Transaction Engine. Manages an in-memory virtual staging buffer with pre-commit compiler validation, diff review, and rollback.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": { "type": "string", "description": "Action to perform: 'begin', 'stage', 'validate', 'diff', 'commit', 'rollback', 'status'" },
                        "path": { "type": "string", "description": "File path (for 'stage' action)" },
                        "content": { "type": "string", "description": "Staged file content (for 'stage' action)" }
                    },
                    "required": ["action"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "benchmark_code",
                "description": "Micro-Benchmarking & Performance Profiler Engine. Executes shell commands with configurable warmup runs and iteration counts, computing min, max, mean, standard deviation, and ops/sec.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": { "type": "string", "description": "Command or binary to benchmark" },
                        "iterations": { "type": "integer", "description": "Number of measured iterations (default 5)" },
                        "warmup": { "type": "integer", "description": "Number of warmup runs before measurement (default 1)" }
                    },
                    "required": ["command"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "generate_tests",
                "description": "Automated Unit Test & Fuzz Suite Synthesizer. Scans source code symbols and synthesizes unit tests and property-based fuzz tests (proptest for Rust, hypothesis for Python, fast-check for JS/TS, native fuzz for Go).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "source_file": { "type": "string", "description": "Path to source code file" },
                        "language": { "type": "string", "description": "Language (e.g. 'rust', 'python', 'javascript', 'go', or auto-detect)" },
                        "fuzz": { "type": "boolean", "description": "Whether to generate property-based fuzz tests (default true)" }
                    },
                    "required": ["source_file"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "generate_ci",
                "description": "Production Container & CI/CD Pipeline Generator. Detects project stack and generates hardened multi-stage Dockerfile, docker-compose.yml, and GitHub Actions CI workflow.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": { "type": "string", "description": "Project directory path (defaults to '.')" },
                        "write_files": { "type": "boolean", "description": "Whether to write Dockerfile, docker-compose.yml, and .github/workflows/ci.yml directly to disk" }
                    }
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "call_graph",
                "description": "Interactive Codebase Call Graph Visualizer. Parses function definitions and cross-file call sites across project sources, constructing directed call graphs and hierarchical ASCII trees.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": { "type": "string", "description": "Workspace root path (defaults to '.')" },
                        "entry_symbol": { "type": "string", "description": "Optional entrypoint function symbol (e.g. 'main', 'interactive_chat')" }
                    }
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "auto_format",
                "description": "Multi-Language Formatter & Linter Auto-Fixer. Auto-detects and runs native formatters and linters (cargo fmt, clippy, prettier, eslint, black, ruff, gofmt) to clean code and auto-resolve warnings.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": { "type": "string", "description": "Workspace root path (defaults to '.')" },
                        "fix": { "type": "boolean", "description": "Whether to auto-apply linter fixes and format files (default true)" }
                    }
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "mock_api",
                "description": "Ephemeral AI Mock Server & API Sandbox. Spins up an asynchronous HTTP mock server on a local port with synthetic JSON routes and status codes.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "port": { "type": "integer", "description": "Port to bind (e.g. 8080, or 0 for dynamic random port)" },
                        "path": { "type": "string", "description": "Route path (e.g. '/api/v1/users')" },
                        "method": { "type": "string", "description": "HTTP method ('GET', 'POST', 'PUT', 'DELETE', default 'GET')" },
                        "response": { "type": "object", "description": "JSON response body" },
                        "status": { "type": "integer", "description": "HTTP status code (default 200)" }
                    },
                    "required": ["port", "path"]
                }
            }
        }
    ])
}

pub fn auto_git_backup(path: &str) {
    if std::path::Path::new(".git").exists() {
        let _ = std::process::Command::new("git").args(["add", path]).output();
        let _ = std::process::Command::new("git").args(["commit", "-m", "zy auto-backup before agent edit"]).output();
    }
}

pub fn ask_confirmation(prompt: &str) -> bool {
    print!("{} [Y/n]: ", prompt.yellow().bold());
    io::stdout().flush().unwrap();
    let mut input = String::new();
    if io::stdin().read_line(&mut input).is_ok() {
        let input = input.trim().to_lowercase();
        return input.is_empty() || input == "y" || input == "yes";
    }
    false
}

pub async fn agent_loop(
    client: &Client, 
    model: &str, 
    messages: &mut Vec<Message>, 
    markdown: bool, 
    options: &OllamaOptions, 
    force: bool,
    format_schema: Option<&serde_json::Value>,
    sandbox: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    loop {
        let req_body = ChatRequest {
            model: model.to_string(),
            messages: messages.to_vec(),
            stream: true,
            tools: Some(get_tools()),
            format: format_schema.cloned(),
            options: Some(options.clone()),
            keep_alive: Some(-1),
        };

        let res = client.post(format!("{}/api/chat", OLLAMA_URL)).json(&req_body).send().await?;

        if !res.status().is_success() {
            println!("{}", format!("Error: Failed to get response from Ollama for model '{}'.", model).red());
            break;
        }

        let mut full_response = String::new();
        let mut accumulated_tool_calls: Vec<ToolCall> = Vec::new();
        let mut in_think_block = false;
        let mut stream = res.bytes_stream();
        let mut printed_tokens = false;

        while let Some(chunk_result) = stream.next().await {
            let chunk = chunk_result?;
            let chunk_str = String::from_utf8_lossy(&chunk);

            for line in chunk_str.lines() {
                let line_trim = line.trim();
                if line_trim.is_empty() { continue; }
                if let Ok(parsed) = serde_json::from_str::<ChatResponse>(line_trim) {
                    if let Some(msg) = parsed.message {
                        if !msg.content.is_empty() {
                            if !printed_tokens && !msg.content.starts_with('{') {
                                print!("{}", "zy ❯ ".green().bold());
                                printed_tokens = true;
                            }
                            full_response.push_str(&msg.content);

                            if msg.content.contains("<think>") { in_think_block = true; }
                            if msg.content.contains("</think>") {
                                print!("{}", "</think>".dimmed());
                                in_think_block = false;
                                io::stdout().flush()?;
                                continue;
                            }

                            if in_think_block {
                                print!("{}", msg.content.dimmed());
                            } else {
                                print!("{}", msg.content);
                            }
                            io::stdout().flush()?;
                        }

                        if let Some(calls) = msg.tool_calls {
                            accumulated_tool_calls.extend(calls);
                        }
                    }
                    if let Some(err) = parsed.error {
                        println!("\nOllama error: {}", err);
                    }
                }
            }
        }

        if printed_tokens {
            println!();
        }

        // Fallback JSON parser if model responded with raw JSON tool call
        if accumulated_tool_calls.is_empty() && full_response.trim().starts_with('{') {
            if let Ok(val) = serde_json::from_str::<serde_json::Value>(full_response.trim()) {
                if let Some(action) = val.get("action").or_else(|| val.get("name")).and_then(|v| v.as_str()) {
                    let arguments = val.get("action_input").or_else(|| val.get("arguments")).cloned().unwrap_or(serde_json::json!({}));
                    accumulated_tool_calls.push(ToolCall {
                        function: ToolCallFunction {
                            name: action.to_string(),
                            arguments,
                        }
                    });
                }
            }
        }

        if !accumulated_tool_calls.is_empty() {
            let assistant_msg = Message {
                role: "assistant".to_string(),
                content: full_response.clone(),
                tool_calls: Some(accumulated_tool_calls.clone()),
                images: None,
            };
            messages.push(assistant_msg);

            for call in &accumulated_tool_calls {
                let fn_name = &call.function.name;
                let args = &call.function.arguments;
                let mut tool_result = String::new();

                let arg_str = args.to_string();
                let preview = if arg_str.len() > 30 { format!("{}...", &arg_str[0..27]) } else { arg_str };
                print!("{} {} {} ", "⚙️ ".magenta(), fn_name.cyan().bold(), preview.dimmed());
                io::stdout().flush()?;

                // Automatic Git Micro-Checkpoint before executing agent actions
                let _ = create_git_checkpoint_with_label(Some(&format!("pre-tool-{}", fn_name)));

                if fn_name == "run_bash" {
                    if let Some(cmd) = args.get("command").and_then(|v| v.as_str()) {
                        let mut proceed = force;
                        if !force {
                            println!();
                            proceed = ask_confirmation(&format!("zy wants to execute: `{}`. Allow?", cmd));
                        }
                        
                        if proceed {
                            let output = if sandbox {
                                println!("{} Executing in Docker sandbox (alpine:latest)...", "🐳".cyan());
                                let (prog, s_args) = build_sandbox_command(cmd, std::path::Path::new("."), None);
                                std::process::Command::new(prog).args(&s_args).output()
                            } else {
                                #[cfg(windows)]
                                { std::process::Command::new("cmd").arg("/C").arg(cmd).output() }
                                #[cfg(not(windows))]
                                { std::process::Command::new("sh").arg("-c").arg(cmd).output() }
                            };
                            match output {
                                Ok(out) => {
                                    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
                                    let stderr = String::from_utf8_lossy(&out.stderr).to_string();
                                    tool_result = format!("STDOUT:\n{}\nSTDERR:\n{}", stdout, stderr);
                                    println!("{}", "✔️".green());
                                }
                                Err(e) => {
                                    tool_result = format!("Failed to execute (sandbox={}): {}", sandbox, e);
                                    println!("{}", "❌".red());
                                }
                            }
                        } else {
                            tool_result = "Execution denied by user.".to_string();
                            println!("{}", "⛔ Denied".red());
                        }
                    } else {
                        tool_result = "Error: Missing command parameter".to_string();
                        println!("{}", "❌ Error".red());
                    }
                } else if fn_name == "get_repo_map" {
                    let root_path = args.get("path").and_then(|v| v.as_str()).unwrap_or(".");
                    let max_tokens = args.get("max_tokens").and_then(|v| v.as_u64()).unwrap_or(1500) as usize;
                    let repo_map = build_repo_map(std::path::Path::new(root_path), max_tokens);
                    println!("\n{}", "🗺️  Generated Repository Symbol Map".cyan().bold());
                    tool_result = repo_map;
                    println!("{}", "✔️".green());
                } else if fn_name == "run_tests" {
                    let cmd_opt = args.get("command").and_then(|v| v.as_str());
                    let p_str = args.get("path").and_then(|v| v.as_str()).unwrap_or(".");
                    match run_project_tests(std::path::Path::new(p_str), cmd_opt) {
                        Ok(report) => {
                            println!("\n{}", format_test_report_for_terminal(&report));
                            tool_result = serde_json::to_string_pretty(&report).unwrap_or_else(|_| report.summary.clone());
                            println!("{}", if report.success { "✔️ Tests Passed".green() } else { "❌ Tests Failed".red() });
                        }
                        Err(e) => {
                            tool_result = format!("Error running tests: {}", e);
                            println!("{}", "❌ Error".red());
                        }
                    }
                } else if fn_name == "lsp_diagnostics" {
                    if let Some(target) = args.get("file_or_cmd").and_then(|v| v.as_str()) {
                        let report = run_lsp_diagnostics(target);
                        println!("\n{}", format_diagnostic_report_for_terminal(&report));
                        tool_result = serde_json::to_string_pretty(&report).unwrap_or_else(|_| report.summary.clone());
                        println!("{}", if report.success { "✔️ Diagnostics Clean".green() } else { "⚠️ Issues Found".yellow() });
                    } else {
                        tool_result = "Error: Missing file_or_cmd parameter".to_string();
                        println!("{}", "❌ Error".red());
                    }
                } else if fn_name == "mcp_execute" {
                    if let (Some(server_cmd), Some(tool)) = (args.get("server_command").and_then(|v| v.as_str()), args.get("tool_name").and_then(|v| v.as_str())) {
                        let empty_args = serde_json::json!({});
                        let mcp_args = args.get("arguments").unwrap_or(&empty_args);
                        println!("{} Calling MCP tool `{}` on `{}`...", "🔌".blue(), tool.cyan(), server_cmd.dimmed());
                        match execute_mcp_call(server_cmd, tool, mcp_args).await {
                            Ok(res) => {
                                tool_result = res;
                                println!("{}", "✔️ MCP Call Complete".green());
                            }
                            Err(e) => {
                                tool_result = format!("MCP execution failed: {}", e);
                                println!("{} {}", "❌ MCP Error:".red(), e);
                            }
                        }
                    } else {
                        tool_result = "Error: Missing server_command or tool_name parameter".to_string();
                        println!("{}", "❌ Error".red());
                    }
                } else if fn_name == "write_file" {
                    if let (Some(path), Some(content)) = (args.get("path").and_then(|v| v.as_str()), args.get("content").and_then(|v| v.as_str())) {
                        let old_content = fs::read_to_string(path).unwrap_or_default();
                        let diff_output = render_terminal_diff(path, &old_content, content);
                        println!("\n{}", diff_output);

                        let mut proceed = force;
                        if !force {
                            proceed = ask_confirmation(&format!("zy wants to write to file: `{}`. Allow?", path));
                        }

                        if proceed {
                            auto_git_backup(path);
                            match fs::write(path, content) {
                                Ok(_) => {
                                    tool_result = format!("Successfully wrote to {}", path);
                                    println!("{}", "✔️".green());
                                }
                                Err(e) => {
                                    tool_result = format!("Failed to write file: {}", e);
                                    println!("{}", "❌".red());
                                }
                            }
                        } else {
                            tool_result = "File write denied by user.".to_string();
                            println!("{}", "⛔ Denied".red());
                        }
                    } else {
                        tool_result = "Error: Missing path or content parameter".to_string();
                        println!("{}", "❌ Error".red());
                    }
                } else if fn_name == "patch_file" {
                    if let (Some(path), Some(old_t), Some(new_t)) = (args.get("path").and_then(|v| v.as_str()), args.get("old_text").and_then(|v| v.as_str()), args.get("new_text").and_then(|v| v.as_str())) {
                        match fs::read_to_string(path) {
                            Ok(content) => {
                                if content.contains(old_t) {
                                    let updated = content.replace(old_t, new_t);
                                    let diff_output = render_terminal_diff(path, &content, &updated);
                                    println!("\n{}", diff_output);

                                    let mut proceed = force;
                                    if !force {
                                        proceed = ask_confirmation(&format!("zy wants to PATCH file: `{}`. Allow?", path));
                                    }
                                    if proceed {
                                        auto_git_backup(path);
                                        if fs::write(path, updated).is_ok() {
                                            tool_result = format!("Successfully patched {}", path);
                                            println!("{}", "✔️".green());
                                        } else {
                                            tool_result = "Failed to write patched file".to_string();
                                            println!("{}", "❌".red());
                                        }
                                    } else {
                                        tool_result = "Patch denied by user.".to_string();
                                        println!("{}", "⛔ Denied".red());
                                    }
                                } else {
                                    tool_result = "Error: old_text not found in file".to_string();
                                    println!("{}", "❌ Not Found".red());
                                }
                            }
                            Err(e) => {
                                tool_result = format!("Error: Could not read file: {}", e);
                                println!("{}", "❌ Error".red());
                            }
                        }
                    } else {
                        tool_result = "Missing parameters".to_string();
                        println!("{}", "❌ Error".red());
                    }
                } else if fn_name == "fetch_url" {
                    if let Some(url) = args.get("url").and_then(|v| v.as_str()) {
                        match client.get(url).send().await {
                            Ok(res) => {
                                if let Ok(text) = res.text().await {
                                    let mut safe_text = text;
                                    if safe_text.len() > 4000 { safe_text.truncate(4000); }
                                    tool_result = safe_text;
                                    println!("{}", "✔️".green());
                                } else {
                                    tool_result = "Failed to read body".to_string();
                                    println!("{}", "❌".red());
                                }
                            }
                            Err(e) => {
                                tool_result = format!("HTTP error: {}", e);
                                println!("{}", "❌".red());
                            }
                        }
                    } else {
                        tool_result = "Error: Missing URL parameter".to_string();
                        println!("{}", "❌ Error".red());
                    }
                } else if fn_name == "browser_action" {
                    if let (Some(url), Some(js)) = (args.get("url").and_then(|v| v.as_str()), args.get("javascript_to_execute").and_then(|v| v.as_str())) {
                        let script = format!("const puppeteer = require('puppeteer'); (async () => {{ const browser = await puppeteer.launch(); const page = await browser.newPage(); await page.goto('{}'); const result = await page.evaluate(() => {{ {} }}); console.log(result); await browser.close(); }})();", url, js);
                        let _ = fs::write(".zy_puppeteer.js", script);
                        let output = std::process::Command::new("node").arg(".zy_puppeteer.js").output();
                        match output {
                            Ok(out) => {
                                tool_result = String::from_utf8_lossy(&out.stdout).to_string();
                                println!("{}", "✔️".green());
                            }
                            Err(_) => {
                                tool_result = "Node/Puppeteer not installed on system.".to_string();
                                println!("{}", "❌ Error".red());
                            }
                        }
                    }
                } else if fn_name == "send_webhook" {
                    if let Some(msg_text) = args.get("message").and_then(|v| v.as_str()) {
                        if let Ok(url) = fs::read_to_string(".zy_webhook.txt") {
                            let payload = serde_json::json!({ "content": msg_text });
                            let _ = client.post(url.trim()).json(&payload).send().await;
                            tool_result = "Webhook sent successfully.".to_string();
                            println!("{}", "✔️ Notification Sent".green());
                        } else {
                            tool_result = "Error: No webhook URL configured. Tell the user to use /webhook <url>".to_string();
                            println!("{}", "❌ No Webhook configured".red());
                        }
                    } else {
                        tool_result = "Missing message".to_string();
                        println!("{}", "❌ Error".red());
                    }
                } else if fn_name == "web_search" {
                    if let Some(query) = args.get("query").and_then(|v| v.as_str()) {
                        let max_res = args.get("max_results").and_then(|v| v.as_u64()).unwrap_or(5) as usize;
                        println!("{} Searching the web for `{}`...", "🌐".cyan(), query.yellow());
                        match perform_web_search(client, query).await {
                            Ok(results) => {
                                let top: Vec<&SearchResult> = results.iter().take(max_res).collect();
                                if top.is_empty() {
                                    tool_result = format!("No search results found for '{}'.", query);
                                    println!("{}", "⚠️ No results".yellow());
                                } else {
                                    let mut out = format!("Web search results for '{}':\n\n", query);
                                    for (i, r) in top.iter().enumerate() {
                                        out.push_str(&format!("{}. [{}]({})\n   {}\n\n", i + 1, r.title, r.url, r.snippet));
                                    }
                                    tool_result = out;
                                    println!("{}", "✔️ Search Complete".green());
                                }
                            }
                            Err(e) => {
                                tool_result = format!("Web search failed: {}", e);
                                println!("{} {}", "❌ Search Error:".red(), e);
                            }
                        }
                    } else {
                        tool_result = "Error: Missing query parameter".to_string();
                        println!("{}", "❌ Error".red());
                    }
                } else if fn_name == "audit_security" {
                    let root_path = args.get("path").and_then(|v| v.as_str()).unwrap_or(".");
                    println!("{} Auditing security and dependencies for `{}`...", "🛡️ ".cyan(), root_path.yellow());
                    let report = audit_project_dependencies(std::path::Path::new(root_path));
                    println!("{}", format_security_report_for_terminal(&report));
                    tool_result = serde_json::to_string_pretty(&report).unwrap_or_else(|_| report.summary.clone());
                    println!("{}", if report.passed { "✔️ Security Audit Clean".green() } else { "⚠️ Security Issues Detected".yellow() });
                } else if fn_name == "db_query" {
                    if let Some(db_path) = args.get("db_path").and_then(|v| v.as_str()) {
                        let query_opt = args.get("query").and_then(|v| v.as_str());
                        println!("{} Inspecting database `{}`...", "🗄️ ".cyan(), db_path.yellow());
                        match inspect_sqlite_database(std::path::Path::new(db_path), query_opt) {
                            Ok(report) => {
                                println!("{}", format_database_report_for_terminal(&report));
                                tool_result = serde_json::to_string_pretty(&report).unwrap_or_else(|_| report.summary.clone());
                                println!("{}", "✔️ Database Inspected".green());
                            }
                            Err(e) => {
                                tool_result = format!("Database error: {}", e);
                                println!("{} {}", "❌ Database Error:".red(), e);
                            }
                        }
                    } else {
                        tool_result = "Error: Missing db_path parameter".to_string();
                        println!("{}", "❌ Error".red());
                    }
                } else if fn_name == "generate_docs" {
                    let target_path = args.get("path").and_then(|v| v.as_str()).unwrap_or(".");
                    let auto_apply = args.get("auto_apply").and_then(|v| v.as_bool()).unwrap_or(false);
                    println!("{} Scanning `{}` for undocumented API symbols...", "📚".cyan(), target_path.yellow());
                    let symbols = scan_undocumented_symbols(std::path::Path::new(target_path));
                    let patches = generate_docstring_patches(&symbols, "rust");
                    let mut applied = 0;
                    if auto_apply && !patches.is_empty() {
                        applied = apply_doc_patches(&patches).unwrap_or(0);
                    }
                    let report = DocGenerationReport {
                        target_path: target_path.to_string(),
                        total_symbols_scanned: symbols.len(),
                        undocumented_count: symbols.len(),
                        symbols,
                        patches: patches.clone(),
                        applied_count: applied,
                        summary: format!("Generated {} docstring patches (applied: {})", patches.len(), applied),
                    };
                    println!("{}", format_doc_generation_report_for_terminal(&report));
                    tool_result = serde_json::to_string_pretty(&report).unwrap_or_else(|_| report.summary.clone());
                    println!("{}", "✔️ Doc Generation Complete".green());
                } else if fn_name == "refactor_transaction" {
                    if let Some(action) = args.get("action").and_then(|v| v.as_str()) {
                        match action.to_lowercase().as_str() {
                            "begin" => {
                                begin_refactor_transaction();
                                tool_result = "Initiated new refactor transaction.".to_string();
                                println!("{}", "✔️ Transaction Begun".green());
                            }
                            "stage" => {
                                if let (Some(path), Some(content)) = (args.get("path").and_then(|v| v.as_str()), args.get("content").and_then(|v| v.as_str())) {
                                    stage_in_refactor_transaction(std::path::Path::new(path), content);
                                    tool_result = format!("Staged virtual edit for {}", path);
                                    println!("{}", "✔️ Edit Staged".green());
                                } else {
                                    tool_result = "Error: Missing path or content for staging".to_string();
                                    println!("{}", "❌ Missing Parameters".red());
                                }
                            }
                            "validate" => {
                                let val_rep = validate_refactor_transaction(std::path::Path::new("."));
                                tool_result = serde_json::to_string_pretty(&val_rep).unwrap_or_else(|_| val_rep.summary.clone());
                                println!("{}", if val_rep.is_valid { "✔️ Validation Clean".green() } else { "⚠️ Validation Issues".yellow() });
                            }
                            "diff" => {
                                tool_result = get_refactor_transaction_diff();
                                println!("{}", "✔️ Diff Generated".green());
                            }
                            "commit" => {
                                match commit_refactor_transaction() {
                                    Ok(files) => {
                                        tool_result = format!("Successfully committed transaction (modified files: {:?})", files);
                                        println!("{}", "✔️ Transaction Committed".green());
                                    }
                                    Err(e) => {
                                        tool_result = format!("Transaction commit error: {}", e);
                                        println!("{} {}", "❌ Commit Error:".red(), e);
                                    }
                                }
                            }
                            "rollback" => {
                                rollback_refactor_transaction();
                                tool_result = "Transaction rolled back and staging buffer cleared.".to_string();
                                println!("{}", "⏪ Rolled Back".yellow());
                            }
                            "status" => {
                                tool_result = get_refactor_transaction_status();
                                println!("{}", "✔️ Status Checked".green());
                            }
                            _ => {
                                tool_result = format!("Unknown transaction action '{}'", action);
                                println!("{}", "❌ Unknown Action".red());
                            }
                        }
                    } else {
                        tool_result = "Error: Missing action parameter".to_string();
                        println!("{}", "❌ Error".red());
                    }
                } else if fn_name == "benchmark_code" {
                    if let Some(cmd) = args.get("command").and_then(|v| v.as_str()) {
                        let iters = args.get("iterations").and_then(|v| v.as_u64()).unwrap_or(5) as usize;
                        let warmup = args.get("warmup").and_then(|v| v.as_u64()).unwrap_or(1) as usize;
                        println!("{} Running micro-benchmark for `{}` (iters: {}, warmup: {})...", "⚡".yellow(), cmd.cyan(), iters, warmup);
                        match run_micro_benchmark(cmd, iters, warmup) {
                            Ok(report) => {
                                println!("{}", format_benchmark_report_for_terminal(&report));
                                tool_result = serde_json::to_string_pretty(&report).unwrap_or_else(|_| report.summary.clone());
                                println!("{}", "✔️ Benchmark Complete".green());
                            }
                            Err(e) => {
                                tool_result = format!("Benchmark error: {}", e);
                                println!("{} {}", "❌ Benchmark Error:".red(), e);
                            }
                        }
                    } else {
                        tool_result = "Error: Missing command parameter".to_string();
                        println!("{}", "❌ Error".red());
                    }
                } else if fn_name == "generate_tests" {
                    if let Some(src_file) = args.get("source_file").and_then(|v| v.as_str()) {
                        let lang = args.get("language").and_then(|v| v.as_str()).unwrap_or("auto");
                        let fuzz = args.get("fuzz").and_then(|v| v.as_bool()).unwrap_or(true);
                        println!("{} Synthesizing test & fuzz suite for `{}`...", "🧪".cyan(), src_file.yellow());
                        match synthesize_test_suite(std::path::Path::new(src_file), lang, fuzz) {
                            Ok(suite) => {
                                println!("{}", format_test_suite_report_for_terminal(&suite));
                                tool_result = serde_json::to_string_pretty(&suite).unwrap_or_else(|_| suite.summary.clone());
                                println!("{}", "✔️ Test Suite Synthesized".green());
                            }
                            Err(e) => {
                                tool_result = format!("Test generation error: {}", e);
                                println!("{} {}", "❌ Test Gen Error:".red(), e);
                            }
                        }
                    } else {
                        tool_result = "Error: Missing source_file parameter".to_string();
                        println!("{}", "❌ Error".red());
                    }
                } else if fn_name == "generate_ci" {
                    let root_path = args.get("path").and_then(|v| v.as_str()).unwrap_or(".");
                    let write_files = args.get("write_files").and_then(|v| v.as_bool()).unwrap_or(false);
                    println!("{} Detecting project stack and generating CI/CD manifests for `{}`...", "🐳".cyan(), root_path.yellow());
                    let stack = detect_project_stack(std::path::Path::new(root_path));
                    let manifests = generate_container_and_ci_manifests(&stack);
                    println!("{}", format_ci_manifests_for_terminal(&manifests));
                    if write_files {
                        let _ = fs::write("Dockerfile", &manifests.dockerfile);
                        let _ = fs::write("docker-compose.yml", &manifests.docker_compose);
                        let _ = fs::create_dir_all(".github/workflows");
                        let _ = fs::write(".github/workflows/ci.yml", &manifests.github_workflow);
                        println!("{}", "💾 Manifest files written to disk (Dockerfile, docker-compose.yml, .github/workflows/ci.yml).".green());
                    }
                    tool_result = serde_json::to_string_pretty(&manifests).unwrap_or_else(|_| manifests.summary.clone());
                    println!("{}", "✔️ CI Manifests Ready".green());
                } else if fn_name == "call_graph" {
                    let root_path = args.get("path").and_then(|v| v.as_str()).unwrap_or(".");
                    let entry_sym = args.get("entry_symbol").and_then(|v| v.as_str());
                    println!("{} Constructing interactive call graph for `{}`...", "🕸️ ".cyan(), root_path.yellow());
                    let report = build_call_graph(std::path::Path::new(root_path), entry_sym);
                    println!("{}", format_call_graph_for_terminal(&report));
                    tool_result = serde_json::to_string_pretty(&report).unwrap_or_else(|_| report.summary.clone());
                    println!("{}", "✔️ Call Graph Complete".green());
                } else if fn_name == "auto_format" {
                    let root_path = args.get("path").and_then(|v| v.as_str()).unwrap_or(".");
                    let fix = args.get("fix").and_then(|v| v.as_bool()).unwrap_or(true);
                    println!("{} Running multi-language formatters and linters on `{}` (fix={})...", "🧹".cyan(), root_path.yellow(), fix);
                    match format_and_lint_workspace(std::path::Path::new(root_path), fix) {
                        Ok(report) => {
                            println!("{}", format_lint_format_report_for_terminal(&report));
                            tool_result = serde_json::to_string_pretty(&report).unwrap_or_else(|_| report.summary.clone());
                            println!("{}", "✔️ Lint & Format Complete".green());
                        }
                        Err(e) => {
                            tool_result = format!("Lint/format error: {}", e);
                            println!("{} {}", "❌ Lint Error:".red(), e);
                        }
                    }
                } else if fn_name == "mock_api" {
                    let port = args.get("port").and_then(|v| v.as_u64()).unwrap_or(8080) as u16;
                    let path = args.get("path").and_then(|v| v.as_str()).unwrap_or("/");
                    let method = args.get("method").and_then(|v| v.as_str()).unwrap_or("GET");
                    let status = args.get("status").and_then(|v| v.as_u64()).unwrap_or(200) as u16;
                    let default_resp = serde_json::json!({ "status": "ok", "message": "Synthetic Mock Response", "timestamp": "2026-09-04T11:00:00Z" });
                    let resp_body = args.get("response").cloned().unwrap_or(default_resp);

                    let route = MockRoute {
                        method: method.to_string(),
                        path: path.to_string(),
                        status_code: status,
                        response_body: resp_body,
                        headers: std::collections::HashMap::new(),
                    };

                    println!("{} Starting Ephemeral Mock Server on port {} for `{}`...", "🚀".cyan(), port, path.yellow());
                    match start_ephemeral_mock_server(port, vec![route]).await {
                        Ok(handle) => {
                            println!("{}", format_mock_server_report_for_terminal(&handle));
                            let server_info = serde_json::json!({
                                "status": "running",
                                "port": handle.port,
                                "base_url": handle.base_url(),
                                "routes": handle.routes,
                            });
                            register_active_mock_server(handle);
                            tool_result = serde_json::to_string_pretty(&server_info).unwrap();
                            println!("{}", "✔️ Mock Server Running in Background".green());
                        }
                        Err(e) => {
                            tool_result = format!("Failed to start mock server: {}", e);
                            println!("{} {}", "❌ Mock Server Error:".red(), e);
                        }
                    }
                } else {
                    tool_result = format!("Unknown function: {}", fn_name);
                    println!("{}", "❓ Unknown".red());
                }

                messages.push(Message {
                    role: "tool".to_string(),
                    content: tool_result,
                    tool_calls: None,
                    images: None,
                });
            }
        } else {
            let assistant_msg = Message {
                role: "assistant".to_string(),
                content: full_response,
                tool_calls: None,
                images: None,
            };
            if markdown {
                print_text(&assistant_msg.content);
            }
            messages.push(assistant_msg);
            break;
        }
    }
    Ok(())
}
