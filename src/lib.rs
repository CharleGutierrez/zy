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
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
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

#[derive(Serialize, Deserialize, Debug)]
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

pub async fn apply_rag(client: &Client, prompt: &str, messages: &mut Vec<Message>) -> Result<(), Box<dyn std::error::Error>> {
    let index_file = ".zy_rag_index.json";
    if let Ok(data) = tokio::fs::read_to_string(index_file).await {
        if let Ok(chunks) = serde_json::from_str::<Vec<RagChunk>>(&data) {
            if chunks.is_empty() { return Ok(()); }
            
            print!("{} ", "🔍 Searching local codebase (RAG)...".magenta());
            io::stdout().flush()?;
            
            let query_vec = embed_text(client, prompt).await?;
            let mut scored: Vec<(f32, &RagChunk)> = chunks.iter()
                .map(|c| (dot_product(&query_vec, &c.vector), c))
                .collect();
                
            scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            
            let mut context_text = String::from("Relevant codebase context via RAG:\n");
            for (score, chunk) in scored.iter().take(3) {
                if *score > 10.0 {
                    context_text.push_str(&format!("--- FILE: {} ---\n{}\n\n", chunk.file, chunk.text));
                }
            }
            
            messages.push(Message {
                role: "system".to_string(),
                content: context_text,
                tool_calls: None,
                images: None,
            });
            println!("{}", "Done".green());
        }
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
    println!("{} {}", "Indexing directory (Smart Chunking):".bold(), path.cyan());
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
    
    let json_data = serde_json::to_string(&chunks)?;
    tokio::fs::write(".zy_rag_index.json", json_data).await?;
    println!("{} {} chunks", "Indexed & saved".green().bold(), chunks.len());
    Ok(())
}

pub async fn vella_reindex_file(client: &Client, file_path: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
    let index_file = ".zy_rag_index.json";
    let path_str = file_path.to_string_lossy().to_string();
    
    let mut chunks: Vec<RagChunk> = if let Ok(data) = tokio::fs::read_to_string(index_file).await {
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
    
    let json_data = serde_json::to_string(&chunks)?;
    tokio::fs::write(index_file, json_data).await?;
    println!("{} {}", "✔️  Vella Sync Complete:".green(), path_str);
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
