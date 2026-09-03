#!/bin/bash

# Insert sysinfo and structs at the top
sed -i 's/use termimad::print_text;/use sysinfo::System;\nuse termimad::print_text;/g' src/main.rs

sed -i 's/struct OllamaOptions {/struct OllamaOptions {\n    #[serde(skip_serializing_if = "Option::is_none")]\n    num_ctx: Option<usize>,\n    #[serde(skip_serializing_if = "Option::is_none")]\n    num_thread: Option<usize>,/g' src/main.rs

sed -i '/struct OllamaOptions/i #[derive(Clone)]\nstruct AiTunerState {\n    pub max_turns: usize,\n    pub profile_name: String,\n    pub opts: OllamaOptions,\n}\n' src/main.rs

# Insert run_ai_tuner function before main
sed -i '/#\[tokio::main\]/i \
fn run_ai_tuner(base_temp: f32) -> AiTunerState {\n\
    let mut sys = System::new_all();\n\
    sys.refresh_memory();\n\
    let cpu_cores = sys.cpus().len();\n\
    let total_mem_gb = sys.total_memory() / 1_073_741_824;\n\
    if total_mem_gb < 12 || cpu_cores <= 4 {\n\
        println!("{} {} RAM, {} Cores. Activating {}...", "⚙️  AiTuner:".cyan(), format!("{}GB", total_mem_gb).yellow(), cpu_cores.to_string().yellow(), "ECO MODE (Low-End Optimizer)".green().bold());\n\
        AiTunerState { max_turns: 4, profile_name: "ECO".to_string(), opts: OllamaOptions { temperature: base_temp, num_ctx: Some(2048), num_thread: Some(std::cmp::max(1, cpu_cores / 2)) } }\n\
    } else {\n\
        println!("{} {} RAM, {} Cores. Activating {}...", "⚙️  AiTuner:".cyan(), format!("{}GB", total_mem_gb).yellow(), cpu_cores.to_string().yellow(), "TURBO MODE (Unrestricted)".magenta().bold());\n\
        AiTunerState { max_turns: 20, profile_name: "TURBO".to_string(), opts: OllamaOptions { temperature: base_temp, num_ctx: Some(8192), num_thread: Some(cpu_cores) } }\n\
    }\n\
}\n' src/main.rs

# Replace options usage in main
sed -i 's/let opts = OllamaOptions { temperature: \*temperature };/let tuner = run_ai_tuner(\*temperature);/g' src/main.rs
sed -i 's/let opts = OllamaOptions { temperature: 0.1 };/let tuner = run_ai_tuner(0.1);/g' src/main.rs
sed -i 's/interactive_chat(&client, model_name, sys_prompt, file, \*agent, session.as_deref(), \*rag, \*markdown, &opts, \*force)/interactive_chat(\&client, model_name, sys_prompt, file, \*agent, session.as_deref(), \*rag, \*markdown, \&tuner, \*force)/g' src/main.rs
sed -i 's/single_prompt(&client, model_name, sys_prompt, file, &text, \*agent, session.as_deref(), \*rag, \*markdown, &opts, \*force)/single_prompt(\&client, model_name, sys_prompt, file, \&text, \*agent, session.as_deref(), \*rag, \*markdown, \&tuner, \*force)/g' src/main.rs
sed -i 's/interactive_chat(&client, &cli.model, cli.system.as_deref(), &\[\], false, None, false, false, &opts, false)/interactive_chat(\&client, \&cli.model, cli.system.as_deref(), \&\[\], false, None, false, false, \&tuner, false)/g' src/main.rs
sed -i 's/interactive_chat(client, &selected_model, None, &\[\], agent, session_opt, rag, markdown, &opts, force)/interactive_chat(client, \&selected_model, None, \&\[\], agent, session_opt, rag, markdown, \&tuner, force)/g' src/main.rs

# Replace in single_prompt definition
sed -i 's/options: &OllamaOptions/tuner: \&AiTunerState/g' src/main.rs

# Inside single_prompt replace options with tuner.opts and fix pruning
sed -i 's/prune_messages(&mut messages, 20); \/\/ Keep max 20 conversational turns to prevent overflow/prune_messages(\&mut messages, tuner.max_turns);/g' src/main.rs
sed -i 's/agent_loop(client, model, &mut messages, markdown, options, force)/agent_loop(client, model, \&mut messages, markdown, \&tuner.opts, force)/g' src/main.rs
sed -i 's/fetch_full_response(client, model, &messages, options)/fetch_full_response(client, model, \&messages, \&tuner.opts)/g' src/main.rs
sed -i 's/stream_response(client, model, &messages, options)/stream_response(client, model, \&messages, \&tuner.opts)/g' src/main.rs

# Inside interactive_chat fix pruning
sed -i 's/prune_messages(&mut messages, 20); \/\/ Keep sliding window of 20 turns/prune_messages(\&mut messages, tuner.max_turns);/g' src/main.rs
sed -i 's/prune_messages(&mut messages, 20); \/\/ Prune before adding new to prevent overflow/prune_messages(\&mut messages, tuner.max_turns);/g' src/main.rs

# Also replace options in interactive_chat
sed -i 's/agent_loop(client, &active_model, &mut messages, markdown, options, force)/agent_loop(client, \&active_model, \&mut messages, markdown, \&tuner.opts, force)/g' src/main.rs
sed -i 's/fetch_full_response(client, &active_model, &messages, options)/fetch_full_response(client, \&active_model, \&messages, \&tuner.opts)/g' src/main.rs
sed -i 's/stream_response(client, &active_model, &messages, options)/stream_response(client, \&active_model, \&messages, \&tuner.opts)/g' src/main.rs

# The stream_response, fetch_full_response, and agent_loop signatures should STILL use OllamaOptions.
# We fixed them globally with `s/options: &OllamaOptions/tuner: &AiTunerState/g` by accident! Let's revert the signatures.
sed -i 's/async fn stream_response(.*, tuner: &AiTunerState)/async fn stream_response(client: \&Client, model: \&str, messages: \&\[Message\], options: \&OllamaOptions)/g' src/main.rs
sed -i 's/async fn fetch_full_response(.*, tuner: &AiTunerState)/async fn fetch_full_response(client: \&Client, model: \&str, messages: \&\[Message\], options: \&OllamaOptions)/g' src/main.rs
sed -i 's/async fn agent_loop(.*, tuner: &AiTunerState, force: bool)/async fn agent_loop(client: \&Client, model: \&str, messages: \&mut Vec<Message>, markdown: bool, options: \&OllamaOptions, force: bool)/g' src/main.rs

