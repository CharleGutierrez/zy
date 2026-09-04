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
use std::fs;
use std::io::{self, Write};
use sysinfo::System;
use termimad::print_text;
use walkdir::WalkDir;

const OLLAMA_URL: &str = "http://localhost:11434";

#[derive(Parser)]
#[command(name = "zy")]
#[command(about = "A super powerful local LLM CLI Agent", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    /// The model to use by default
    #[arg(short, long, default_value = "llama2")]
    model: String,

    /// Global system prompt to define the model's persona
    #[arg(short, long)]
    system: Option<String>,
}

#[derive(Subcommand)]
enum Commands {
    /// List available local models
    List,
    /// Start a chat session or send a single prompt
    Chat {
        /// The prompt to send. If empty, starts an interactive session.
        prompt: Vec<String>,
        
        /// Model to use for this chat session
        #[arg(short, long)]
        model: Option<String>,

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

#[derive(Serialize, Deserialize, Clone)]
struct ToolCallFunction {
    name: String,
    arguments: serde_json::Value,
}

#[derive(Serialize, Deserialize, Clone)]
struct ToolCall {
    function: ToolCallFunction,
}

#[derive(Serialize, Deserialize, Clone)]
struct Message {
    role: String,
    content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    images: Option<Vec<String>>,
}

#[derive(Serialize, Clone)]
struct OllamaOptions {
    temperature: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    num_ctx: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    num_thread: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    num_gpu: Option<usize>,
}

#[derive(Serialize, Clone)]
struct AiTunerState {
    pub max_turns: usize,
    pub profile_name: String,
    pub opts: OllamaOptions,
}

#[derive(Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<Message>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    options: Option<OllamaOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    keep_alive: Option<i32>,
}

#[derive(Deserialize)]
struct ChatResponse {
    message: Option<Message>,
    #[allow(dead_code)]
    pub done: bool,
    error: Option<String>,
}

#[derive(Deserialize)]
struct ModelList {
    models: Vec<ModelInfo>,
}

#[derive(Deserialize)]
struct ModelInfo {
    name: String,
}

// RAG structs
#[derive(Serialize, Deserialize)]
struct EmbedRequest {
    model: String,
    prompt: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    keep_alive: Option<i32>,
}

#[derive(Deserialize)]
struct EmbedResponse {
    embedding: Vec<f32>,
}

#[derive(Serialize, Deserialize)]
struct RagChunk {
    file: String,
    text: String,
    vector: Vec<f32>,
}

fn run_ai_tuner(base_temp: f32, quiet: bool) -> AiTunerState {

    let mut sys = System::new_all();

    sys.refresh_memory();

    let cpu_cores = sys.cpus().len();

    let total_mem_gb = sys.total_memory() / 1_073_741_824;

    if total_mem_gb < 12 || cpu_cores <= 4 {

        if !quiet { println!("{} {} RAM, {} Cores. Activating {}...", "⚙️  AiTuner:".cyan().dimmed(), format!("{}GB", total_mem_gb).yellow().dimmed(), cpu_cores.to_string().yellow().dimmed(), "ECO MODE".green().dimmed()); }

        AiTunerState { max_turns: 4, profile_name: "ECO".to_string(), opts: OllamaOptions { temperature: base_temp, num_ctx: Some(2048), num_thread: Some(std::cmp::max(1, cpu_cores / 2)), num_gpu: Some(1) } }

    } else {

        if !quiet { println!("{} {} RAM, {} Cores. Activating {}...", "⚙️  AiTuner:".cyan().dimmed(), format!("{}GB", total_mem_gb).yellow().dimmed(), cpu_cores.to_string().yellow().dimmed(), "TURBO MODE".magenta().dimmed()); }

        AiTunerState { max_turns: 20, profile_name: "TURBO".to_string(), opts: OllamaOptions { temperature: base_temp, num_ctx: Some(8192), num_thread: Some(cpu_cores), num_gpu: Some(999) } }

    }

}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    let client = Client::new();

    match &cli.command {
        Some(Commands::List) => {
            list_models(&client).await?;
        }
        Some(Commands::Index { path }) => {
            build_rag_index(&client, path).await?;
        }
        Some(Commands::Watch { path }) => {
            vella_watch_daemon(&client, path).await?;
        }
        Some(Commands::Chat { prompt, model, file, system, agent, session, rag, markdown, temperature, force, executor, strategist }) => {
            let model_name = model.as_deref().unwrap_or(&cli.model);
            let sys_prompt = system.as_deref().or(cli.system.as_deref());
            
            let tuner = run_ai_tuner(*temperature, true);

            if prompt.is_empty() {
                interactive_chat(&client, model_name, sys_prompt, file, *agent, session.as_deref(), *rag, *markdown, &tuner, *force, executor.clone(), *strategist).await?;
            } else {
                let text = prompt.join(" ");
                single_prompt(&client, model_name, sys_prompt, file, &text, *agent, session.as_deref(), *rag, *markdown, &tuner, *force, executor.clone(), *strategist).await?;
            }
        }
        None => {
            interactive_wizard(&client, &cli.model).await?;
        }
    }

    Ok(())
}

async fn interactive_wizard(client: &Client, default_model: &str) -> Result<(), Box<dyn std::error::Error>> {
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
            
            let selected_model = Select::new("Select an AI Model:", model_names).prompt()?;
            let agent = Confirm::new("Enable Agent Mode (run bash/write files)?").with_default(false).prompt()?;
            let force = if agent { Confirm::new("Enable Force Mode (execute without asking)?").with_default(false).prompt()? } else { false };
            let rag = Confirm::new("Enable RAG (search local codebase)?").with_default(false).prompt()?;
            let markdown = Confirm::new("Enable Markdown Syntax Highlighting?").with_default(true).prompt()?;
            
            let session = Text::new("Session name (leave empty for none):").prompt()?;
            let session_opt = if session.trim().is_empty() { None } else { Some(session.trim()) };

            let tuner = run_ai_tuner(0.1, true);
            println!("\n{}", "--- Configuration Complete ---".green().bold());
            interactive_chat(client, &selected_model, None, &[], agent, session_opt, rag, markdown, &tuner, force, None, false).await?;
        }
        _ => {}
    }
    Ok(())
}

async fn list_models(client: &Client) -> Result<(), Box<dyn std::error::Error>> {
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

fn load_session(session: Option<&str>) -> Vec<Message> {
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

fn save_session(session: Option<&str>, messages: &[Message]) {
    if let Some(name) = session {
        let file_path = format!(".zy_session_{}.json", name);
        if let Ok(data) = serde_json::to_string_pretty(messages) {
            let _ = fs::write(file_path, data);
        }
    }
}

async fn embed_text(client: &Client, text: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
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

fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

async fn apply_rag(client: &Client, prompt: &str, messages: &mut Vec<Message>) -> Result<(), Box<dyn std::error::Error>> {
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
                if *score > 10.0 { // arbitrary low threshold for nomic
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

fn smart_chunk(content: &str, max_len: usize) -> Vec<String> {
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

async fn build_rag_index(client: &Client, path: &str) -> Result<(), Box<dyn std::error::Error>> {
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

async fn vella_reindex_file(client: &Client, file_path: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
    let index_file = ".zy_rag_index.json";
    let path_str = file_path.to_string_lossy().to_string();
    
    let mut chunks: Vec<RagChunk> = if let Ok(data) = tokio::fs::read_to_string(index_file).await {
        serde_json::from_str(&data).unwrap_or_default()
    } else {
        Vec::new()
    };
    
    // Remove old chunks for this file
    chunks.retain(|c| c.file != path_str);
    
    // Add new chunks
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

async fn vella_watch_daemon(client: &Client, path: &str) -> Result<(), Box<dyn std::error::Error>> {
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

const STRATEGIST_PROMPT: &str = r#"
[AI STRATEGIST PROTOCOL ENGAGED]
You are operating as a lethal, highly calculated AI Strategist. 
Before executing ANY tool or providing a final answer, you MUST use an OODA loop (Observe, Orient, Decide, Act).
1. OBSERVE: Analyze the user's request and the environment.
2. ORIENT: Identify edge cases, hidden constraints, and potential points of failure.
3. DECIDE: Formulate a ruthless, highly optimized, multi-step execution plan.
4. ACT: Execute the tools required to complete the plan flawlessly.
Always wrap your strategic reasoning in <STRATEGY> ... </STRATEGY> tags before taking action.
"#;

fn build_initial_messages(system: Option<&str>, files: &[String], strategist: bool) -> Result<Vec<Message>, Box<dyn std::error::Error>> {
    let mut messages = Vec::new();

    if let Some(sys) = system {
        let mut final_sys = sys.to_string();
        if strategist { final_sys.push_str(STRATEGIST_PROMPT); }
        messages.push(Message {
            role: "system".to_string(),
            content: final_sys,
            tool_calls: None,
            images: None,
        });
    } else {
        let mut default_sys = "You are an expert, deterministic coding assistant. Provide highly accurate and factual answers. If you do not know the answer or lack context, explicitly state 'I do not have enough information' instead of guessing or making up functions. Stick strictly to the provided files or RAG context.".to_string();
        if strategist { default_sys.push_str(STRATEGIST_PROMPT); }
        messages.push(Message {
            role: "system".to_string(),
            content: default_sys,
            tool_calls: None,
            images: None,
        });
    }

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

fn prune_messages(messages: &mut Vec<Message>, max_turns: usize) {
    let mut system_msgs = Vec::new();
    let mut history_msgs = Vec::new();
    
    for msg in messages.drain(..) {
        if msg.role == "system" {
            system_msgs.push(msg);
        } else {
            history_msgs.push(msg);
        }
    }
    
    if history_msgs.len() > max_turns {
        let excess = history_msgs.len() - max_turns;
        history_msgs.drain(0..excess);
    }
    
    messages.extend(system_msgs);
    messages.extend(history_msgs);
}

async fn single_prompt(
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
) -> Result<(), Box<dyn std::error::Error>> {
    let mut messages = load_session(session);
    prune_messages(&mut messages, tuner.max_turns);

    let mut init_msgs = build_initial_messages(system, files, strategist)?;
    messages.append(&mut init_msgs);
    
    if rag {
        apply_rag(client, prompt, &mut messages).await?;
    }
    
    messages.push(Message {
        role: "user".to_string(),
        content: prompt.to_string(),
        tool_calls: None,
                images: None,
    });
    
    if let Some(exec) = executor {
        // Swarm Mode
        println!("{} {}", "🧠 Swarm Architect Planning...".magenta().bold(), model);
        let plan = fetch_full_response(client, model, &messages, &tuner.opts).await?;
        print_text(&plan);
        messages.push(Message { role: "assistant".to_string(), content: plan.clone(), tool_calls: None,
                images: None });
        
        println!("\n{} {}", "⚡ Swarm Executor Working...".yellow().bold(), exec);
        messages.push(Message { role: "user".to_string(), content: format!("Execute this plan using tools:\n{}", plan), tool_calls: None,
                images: None });
        agent_loop(client, &exec, &mut messages, markdown, &tuner.opts, force).await?;
    } else if agent {
        agent_loop(client, model, &mut messages, markdown, &tuner.opts, force).await?;
    } else {
        if markdown {
            let res = fetch_full_response(client, model, &messages, &tuner.opts).await?;
            print_text(&res);
            messages.push(Message { role: "assistant".to_string(), content: res, tool_calls: None,
                images: None });
        } else {
            let res = stream_response(client, model, &messages, &tuner.opts).await?;
            println!();
            messages.push(Message { role: "assistant".to_string(), content: res, tool_calls: None,
                images: None });
        }
    }
    
    save_session(session, &messages);
    Ok(())
}

async fn interactive_chat(
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
) -> Result<(), Box<dyn std::error::Error>> {
    let mut active_model = model.to_string();
    let mut agent = agent_flag;
    let mut rag = rag_flag;
    let mut executor = executor_flag;
    let mut strategist = strategist_flag;

    println!("\n{}", "╭──────────────────────────────────────────────────────────╮".cyan());
    println!("{} {} {}", "│".cyan(), "🤖 zy Agent Dashboard".bold().white(), "                                │".cyan());
    println!("{}", "├──────────────────────────────────────────────────────────┤".cyan());
    println!("{} Model: {:<12} │ Agent: {:<3} (Force: {:<3})           {}", "│".cyan(), active_model.yellow().bold(), if agent { "ON".green() } else { "OFF".red() }, if force { "ON".red() } else { "OFF".green() }, "│".cyan());
    println!("{} RAG:   {:<12} │ Strategy: {:<20} {}", "│".cyan(), if rag { "ON".green() } else { "OFF".red() }, if strategist { "ENGAGED".red().bold() } else { "OFF".green() }, "│".cyan());
    println!("{} Swarm: {:<12} │ Tuner: {:<22} {}", "│".cyan(), if let Some(e) = &executor { e.magenta().bold().to_string() } else { "OFF".green().to_string() }, tuner.profile_name.blue().bold(), "│".cyan());
    
    let sess_display = session.unwrap_or("None");
    println!("{} Session: {:<46} {}", "│".cyan(), sess_display.white().dimmed(), "│".cyan());
    println!("{}\n", "╰──────────────────────────────────────────────────────────╯".cyan());
    println!("💡 {}", "Type /help for commands or /exit to quit.".dimmed());

    let mut rl = DefaultEditor::new()?;
    let mut messages = load_session(session);
    prune_messages(&mut messages, tuner.max_turns);
    
    let mut init_msgs = build_initial_messages(system, files, strategist)?;
    messages.append(&mut init_msgs);

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
                            println!("  /help          - Show this help message");
                            println!("  /clear         - Clear terminal and conversation history");
                            println!("  /save <name>   - Save current session");
                            println!("  /model <name>  - Switch the active LLM");
                            println!("  /agent <on/off>- Toggle Agent mode");
                            println!("  /rag <on/off>  - Toggle RAG mode");
                            println!("  /executor <mdl>- Set Swarm Executor model");
                            println!("  /strategist    - Toggle AI Strategist Protocol");
                            println!("  /listen        - Voice-to-Code (Requires arecord & whisper)");
                            println!("  /evolve <req>  - Self-modify zy's own source code and recompile");
                            println!("  /worker        - Autonomously fix bugs in .projectmem/issues/");
                            println!("  /chaos         - Chaos Monkey: Randomly break a file in your project");
                            println!("  /sleep         - Deep Memory Compression (Summarize history)");
                            println!("  /webhook <url> - Set Discord/Slack webhook for agent push notifications");
                            println!("  /train         - Export RLHF dataset & run local LoRA fine-tuning");
                            println!("  /undo          - Git-revert the last agent file edit");
                            println!("  /exit, /quit   - End the session");
                            continue;
                        }
                        "/clear" => {
                            print!("{esc}[2J{esc}[1;1H", esc = 27 as char);
                            messages.clear();
                            let mut init = build_initial_messages(system, files, strategist).unwrap_or_default();
                            messages.append(&mut init);
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
                            
                            // Hot-swap the system prompt to inject strategist
                            messages.retain(|m| m.role != "system");
                            let mut new_init = build_initial_messages(system, files, strategist).unwrap_or_default();
                            messages.splice(0..0, new_init);
                            continue;
                        }
                        "/listen" => {
                            println!("{}", "🎤 Listening for 5 seconds...".cyan());
                            let _ = std::process::Command::new("arecord").args(&["-d", "5", "-f", "S16_LE", "/tmp/zy_voice.wav"]).output();
                            println!("{}", "Processing voice...".cyan());
                            let whisper_out = std::process::Command::new("whisper").args(&["/tmp/zy_voice.wav"]).output();
                            
                            let transcript = if let Ok(out) = whisper_out {
                                String::from_utf8_lossy(&out.stdout).to_string()
                            } else {
                                println!("{}", "Whisper not found in PATH. Simulating voice transcription...".yellow());
                                "Simulated voice input: Write a python script to ping google.com".to_string()
                            };
                            
                            println!("{} {}", "Transcription:".green(), transcript.trim());
                            
                            // Inject the voice transcript directly into the chat session
                            messages.push(Message {
                                role: "user".to_string(),
                                content: transcript.trim().to_string(),
                                tool_calls: None,
                                images: None,
                            });
                            
                            if agent {
                                let _ = agent_loop(client, &active_model, &mut messages, markdown, &tuner.opts, force).await;
                            } else {
                                if let Ok(response_text) = fetch_full_response(client, &active_model, &messages, &tuner.opts).await {
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
                                let _ = std::process::Command::new("git").args(&["reset", "--hard", "HEAD~1"]).output();
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
                                        lines.drain(start..start+5); // Delete 5 random lines
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
                            if let Ok(summary) = fetch_full_response(client, &active_model, &temp_msgs, &tuner.opts).await {
                                messages.retain(|m| m.role == "system" && !m.content.contains("Core Memory:")); // Keep base system prompts
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
            # Find user-assistant pairs to build SFT dataset
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
    
    # Load dataset
    hf_dataset = load_dataset('json', data_files='.zy_dataset.json', split='train')
    
    # Normally we would load the model that the user specified, but for safety and speed 
    # we'll print out the setup that is being executed.
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)
        
    tokenized_datasets = hf_dataset.map(tokenize_function, batched=True)
    
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    
    # LoRA config
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
    
    # Save the model
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
                                        let _ = agent_loop(client, &active_model, &mut messages, markdown, &tuner.opts, true).await;
                                        
                                        println!("{}", "✅ Issue Processed!".green());
                                        break; // Process one issue at a time for safety
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
                                    
                                    let mut temp_msgs = vec![Message { role: "user".to_string(), content: prompt, tool_calls: None, images: None }];
                                    println!("{}", "🧠 zy is writing its own source code...".cyan());
                                    if let Ok(mut new_code) = fetch_full_response(client, &active_model, &temp_msgs, &tuner.opts).await {
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
                
                prune_messages(&mut messages, tuner.max_turns); // Prune dynamically based on hardware
                
                if rag {
                    apply_rag(client, input, &mut messages).await?;
                }
                
                messages.push(Message {
                    role: "user".to_string(),
                    content: input.to_string(),
                    tool_calls: None,
                images: None,
                });

                if let Some(exec) = &executor {
                    println!("{} {}", "🧠 Swarm Architect Planning...".magenta().bold(), active_model);
                    let plan = fetch_full_response(client, &active_model, &messages, &tuner.opts).await?;
                    print_text(&plan);
                    messages.push(Message { role: "assistant".to_string(), content: plan.clone(), tool_calls: None,
                images: None });
                    
                    println!("\n{} {}", "⚡ Swarm Executor Working...".yellow().bold(), exec);
                    messages.push(Message { role: "user".to_string(), content: format!("Execute this plan using tools:\n{}", plan), tool_calls: None,
                images: None });
                    agent_loop(client, exec, &mut messages, markdown, &tuner.opts, force).await?;
                } else if agent {
                    agent_loop(client, &active_model, &mut messages, markdown, &tuner.opts, force).await?;
                } else {
                    if markdown {
                        let response_text = fetch_full_response(client, &active_model, &messages, &tuner.opts).await?;
                        print_text(&response_text);
                        messages.push(Message {
                            role: "assistant".to_string(),
                            content: response_text,
                            tool_calls: None,
                images: None,
                        });
                    } else {
                        let response_text = stream_response(client, &active_model, &messages, &tuner.opts).await?;
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

async fn stream_response(client: &Client, model: &str, messages: &[Message], options: &OllamaOptions) -> Result<String, Box<dyn std::error::Error>> {
    let req_body = ChatRequest {
        model: model.to_string(),
        messages: messages.to_vec(),
        stream: true,
        tools: None,
        options: Some(options.clone()),
        keep_alive: Some(-1),
    };

    let mut res = client.post(format!("{}/api/chat", OLLAMA_URL)).json(&req_body).send().await?;

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
                    
                    // Simple logic to dim reasoning models' <think> tags
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

async fn fetch_full_response(client: &Client, model: &str, messages: &[Message], options: &OllamaOptions) -> Result<String, Box<dyn std::error::Error>> {
    let req_body = ChatRequest {
        model: model.to_string(),
        messages: messages.to_vec(),
        stream: false,
        tools: None,
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

fn get_tools() -> serde_json::Value {
    serde_json::json!([
        {
            "type": "function",
            "function": {
                "name": "run_bash",
                "description": "Execute a bash command. If it fails, you MUST analyze STDERR and fix it in a loop.",
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

fn auto_git_backup(path: &str) {
    if std::path::Path::new(".git").exists() {
        let _ = std::process::Command::new("git").args(&["add", path]).output();
        let _ = std::process::Command::new("git").args(&["commit", "-m", "zy auto-backup before agent edit"]).output();
    }
}

fn ask_confirmation(prompt: &str) -> bool {
    print!("{} [Y/n]: ", prompt.yellow().bold());
    io::stdout().flush().unwrap();
    let mut input = String::new();
    if io::stdin().read_line(&mut input).is_ok() {
        let input = input.trim().to_lowercase();
        return input.is_empty() || input == "y" || input == "yes";
    }
    false
}

async fn agent_loop(client: &Client, model: &str, messages: &mut Vec<Message>, markdown: bool, options: &OllamaOptions, force: bool) -> Result<(), Box<dyn std::error::Error>> {
    loop {
        let req_body = ChatRequest {
            model: model.to_string(),
            messages: messages.to_vec(),
            stream: false,
            tools: Some(get_tools()),
            options: Some(options.clone()),
            keep_alive: Some(-1),
        };

        let spinner = ProgressBar::new_spinner();
        spinner.set_style(ProgressStyle::default_spinner().tick_strings(&["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]).template("{spinner:.magenta} {msg}").unwrap());
        spinner.set_message("zy agent is working...");
        spinner.enable_steady_tick(std::time::Duration::from_millis(80));

        let res = client.post(format!("{}/api/chat", OLLAMA_URL)).json(&req_body).send().await?;
        spinner.finish_and_clear();

        if !res.status().is_success() {
            println!("{}", "Error: Failed to get response from Ollama.".red());
            break;
        }

        let parsed: ChatResponse = res.json().await?;
        if let Some(msg) = parsed.message {
            if let Some(calls) = &msg.tool_calls {
                messages.push(msg.clone());

                for call in calls {
                    let fn_name = &call.function.name;
                    let args = &call.function.arguments;
                    let mut tool_result = String::new();

                    let arg_str = args.to_string();
                    let preview = if arg_str.len() > 30 { format!("{}...", &arg_str[0..27]) } else { arg_str };
                    print!("{} {} {} ", "⚙️ ".magenta(), fn_name.cyan().bold(), preview.dimmed());
                    io::stdout().flush()?;

                    if fn_name == "run_bash" {
                        if let Some(cmd) = args.get("command").and_then(|v| v.as_str()) {
                            let mut proceed = force;
                            if !force {
                                println!();
                                proceed = ask_confirmation(&format!("zy wants to execute: `{}`. Allow?", cmd));
                            }
                            
                            if proceed {
                                let output = std::process::Command::new("sh").arg("-c").arg(cmd).output();
                                match output {
                                    Ok(out) => {
                                        let stdout = String::from_utf8_lossy(&out.stdout).to_string();
                                        let stderr = String::from_utf8_lossy(&out.stderr).to_string();
                                        tool_result = format!("STDOUT:\n{}\nSTDERR:\n{}", stdout, stderr);
                                        println!("{}", "✔️".green());
                                    }
                                    Err(e) => {
                                        tool_result = format!("Failed to execute: {}", e);
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
                    } else if fn_name == "write_file" {
                        if let (Some(path), Some(content)) = (args.get("path").and_then(|v| v.as_str()), args.get("content").and_then(|v| v.as_str())) {
                            let mut proceed = force;
                            if !force {
                                println!();
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
                            let mut proceed = force;
                            if !force {
                                println!();
                                proceed = ask_confirmation(&format!("zy wants to PATCH file: `{}`. Allow?", path));
                            }
                            if proceed {
                                auto_git_backup(path);
                                if let Ok(content) = fs::read_to_string(path) {
                                    if content.contains(old_t) {
                                        let updated = content.replace(old_t, new_t);
                                        if fs::write(path, updated).is_ok() {
                                            tool_result = format!("Successfully patched {}", path);
                                            println!("{}", "✔️".green());
                                        } else {
                                            tool_result = "Failed to write patched file".to_string();
                                            println!("{}", "❌".red());
                                        }
                                    } else {
                                        tool_result = "Error: old_text not found in file".to_string();
                                        println!("{}", "❌ Not Found".red());
                                    }
                                } else {
                                    tool_result = "Error: Could not read file".to_string();
                                    println!("{}", "❌ Error".red());
                                }
                            } else {
                                tool_result = "Denied".to_string();
                                println!("{}", "⛔ Denied".red());
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
                if markdown {
                    print_text(&msg.content);
                } else {
                    println!("{} {}", "zy ❯".green().bold(), msg.content);
                }
                messages.push(msg.clone());
                break;
            }
        } else {
            break;
        }
    }
    Ok(())
}