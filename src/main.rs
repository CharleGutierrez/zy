use clap::Parser;
use reqwest::Client;
use zy::*;

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
        Some(Commands::Chat { prompt, model, scout, file, system, agent, session, rag, markdown, temperature, force, executor, strategist, format, map, sandbox }) => {
            let model_name = model.as_deref().unwrap_or(&cli.model);
            let sys_prompt = system.as_deref().or(cli.system.as_deref());
            let scout_model = scout.clone().or_else(|| cli.scout.clone());
            let map_flag = *map || cli.map;
            let sandbox_flag = *sandbox || cli.sandbox;
            
            let tuner = run_ai_tuner(*temperature, true);
            let format_schema = format.as_deref().map(|f| {
                if f.eq_ignore_ascii_case("json") {
                    serde_json::json!("json")
                } else if f.eq_ignore_ascii_case("tool") {
                    build_tool_grammar_schema()
                } else {
                    serde_json::from_str::<serde_json::Value>(f).unwrap_or_else(|_| serde_json::json!(f))
                }
            });

            if prompt.is_empty() {
                interactive_chat(&client, model_name, sys_prompt, file, *agent, session.as_deref(), *rag, *markdown, &tuner, *force, executor.clone(), *strategist, scout_model, format_schema, map_flag, sandbox_flag).await?;
            } else {
                let text = prompt.join(" ");
                single_prompt(&client, model_name, sys_prompt, file, &text, *agent, session.as_deref(), *rag, *markdown, &tuner, *force, executor.clone(), *strategist, scout_model, format_schema, map_flag, sandbox_flag).await?;
            }
        }
        None => {
            interactive_wizard(&client, &cli.model, cli.scout.clone()).await?;
        }
    }

    Ok(())
}