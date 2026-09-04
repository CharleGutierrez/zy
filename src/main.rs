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
        Some(Commands::Chat { prompt, model, scout, file, system, agent, session, rag, markdown, temperature, force, executor, strategist, format, map, sandbox, swarm, tui }) => {
            let model_name = model.as_deref().unwrap_or(&cli.model);
            let sys_prompt = system.as_deref().or(cli.system.as_deref());
            let scout_model = scout.clone().or_else(|| cli.scout.clone());
            let map_flag = *map || cli.map;
            let sandbox_flag = *sandbox || cli.sandbox;
            let swarm_goal = swarm.clone().or_else(|| cli.swarm.clone());
            let tui_flag = *tui || cli.tui;
            
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

            if tui_flag {
                run_tui_app(&client, model_name, sys_prompt, file, *agent, *rag, &tuner, *force).await?;
                return Ok(());
            }

            if let Some(goal) = swarm_goal {
                run_swarm_workflow(&client, model_name, executor.as_deref(), &goal, &tuner.opts, *markdown, *force, sandbox_flag).await?;
                return Ok(());
            }

            if prompt.is_empty() {
                interactive_chat(&client, model_name, sys_prompt, file, *agent, session.as_deref(), *rag, *markdown, &tuner, *force, executor.clone(), *strategist, scout_model, format_schema, map_flag, sandbox_flag).await?;
            } else {
                let text = prompt.join(" ");
                single_prompt(&client, model_name, sys_prompt, file, &text, *agent, session.as_deref(), *rag, *markdown, &tuner, *force, executor.clone(), *strategist, scout_model, format_schema, map_flag, sandbox_flag).await?;
            }
        }
        None => {
            if cli.tui {
                let tuner = run_ai_tuner(0.1, true);
                run_tui_app(&client, &cli.model, cli.system.as_deref(), &[], false, false, &tuner, false).await?;
            } else if let Some(goal) = &cli.swarm {
                let tuner = run_ai_tuner(0.1, true);
                run_swarm_workflow(&client, &cli.model, None, goal, &tuner.opts, true, false, cli.sandbox).await?;
            } else {
                interactive_wizard(&client, &cli.model, cli.scout.clone()).await?;
            }
        }
    }

    Ok(())
}