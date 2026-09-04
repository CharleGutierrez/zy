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
        Some(Commands::Worktree { action, task_id, command }) => {
            let id = task_id.as_deref().unwrap_or("default-task");
            match action.to_lowercase().as_str() {
                "create" => {
                    let handle = create_task_worktree(std::path::Path::new("."), id, None)?;
                    println!("{}", format_worktree_report_for_terminal(&handle));
                }
                "execute" => {
                    if let Some(cmd) = command {
                        let handle = WorktreeHandle {
                            task_id: id.to_string(),
                            branch_name: format!("zy-task-{}", id),
                            worktree_path: std::path::Path::new(".").join(".zy").join("worktrees").join(id),
                            workspace_root: std::path::PathBuf::from("."),
                            created_at: "active".to_string(),
                        };
                        let res = execute_in_worktree(&handle, cmd)?;
                        println!("STDOUT:\n{}\nSTDERR:\n{}", res.stdout, res.stderr);
                    } else {
                        eprintln!("Error: Missing command for execute");
                    }
                }
                "merge" => {
                    let handle = WorktreeHandle {
                        task_id: id.to_string(),
                        branch_name: format!("zy-task-{}", id),
                        worktree_path: std::path::Path::new(".").join(".zy").join("worktrees").join(id),
                        workspace_root: std::path::PathBuf::from("."),
                        created_at: "active".to_string(),
                    };
                    let res = merge_worktree_back(&handle, None)?;
                    println!("{}", res.summary);
                }
                "cleanup" => {
                    let handle = WorktreeHandle {
                        task_id: id.to_string(),
                        branch_name: format!("zy-task-{}", id),
                        worktree_path: std::path::Path::new(".").join(".zy").join("worktrees").join(id),
                        workspace_root: std::path::PathBuf::from("."),
                        created_at: "active".to_string(),
                    };
                    let cleaned = cleanup_worktree(&handle, true)?;
                    println!("Worktree cleanup status: {}", cleaned);
                }
                _ => {
                    let list = list_task_worktrees(std::path::Path::new("."))?;
                    println!("{}", format_worktree_list_for_terminal(&list));
                }
            }
        }
        Some(Commands::Review { path }) => {
            let report = perform_code_review(std::path::Path::new("."), Some(path.as_str()))?;
            println!("{}", format_code_review_for_terminal(&report));
        }
        Some(Commands::Resolve { path }) => {
            let p = std::path::Path::new(path);
            if p.is_file() {
                let res = resolve_merge_conflict(p)?;
                println!("{}", format_conflict_resolution_for_terminal(&res));
            } else {
                let conflicts = find_merge_conflicts(p);
                for cf in &conflicts {
                    let res = resolve_merge_conflict(cf)?;
                    println!("{}", format_conflict_resolution_for_terminal(&res));
                }
            }
        }
        Some(Commands::AstGrep { pattern, replacement, path }) => {
            let res = execute_structural_search(std::path::Path::new(path), pattern, replacement.as_deref())?;
            println!("{}", format_structural_search_for_terminal(&res));
        }
        Some(Commands::Release { bump, path }) => {
            let bump_override = match bump.to_lowercase().as_str() {
                "major" => Some(BumpType::Major),
                "minor" => Some(BumpType::Minor),
                "patch" => Some(BumpType::Patch),
                _ => None,
            };
            let plan = execute_release(std::path::Path::new(path), bump_override, false, true)?;
            println!("{}", format_release_plan_for_terminal(&plan));
        }
        Some(Commands::Remote { action, port, token }) => {
            match action.to_lowercase().as_str() {
                "start" => {
                    let handle = start_remote_pair_bridge(*port, token.as_deref()).await?;
                    println!("{}", format_remote_bridge_report_for_terminal(&handle));
                    // Keep running until Ctrl+C
                    tokio::signal::ctrl_c().await?;
                    handle.stop();
                }
                "stop" => {
                    stop_active_bridge();
                    println!("Remote bridge stopped.");
                }
                _ => {
                    if let Some(h) = get_active_bridge() {
                        println!("{}", format_remote_bridge_report_for_terminal(&h));
                    } else {
                        println!("No remote bridge active.");
                    }
                }
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