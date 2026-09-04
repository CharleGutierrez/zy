use clap::Parser;
use colored::Colorize;
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
        Some(Commands::Quantize { model_path, output_name, quant_type, system, path }) => {
            let rep = quantize_and_import_model(std::path::Path::new(path), std::path::Path::new(model_path), output_name, quant_type, system.as_deref())?;
            println!("{}", format_quantize_report_for_terminal(&rep));
        }
        Some(Commands::Prune { path, apply }) => {
            let rep = find_dead_code_symbols(std::path::Path::new(path))?;
            if *apply && !rep.patches.is_empty() {
                let pruned = apply_dead_code_pruning(&rep.patches)?;
                println!("Auto-applied {} pruning patch(es).", pruned);
            }
            println!("{}", format_dead_code_report_for_terminal(&rep));
        }
        Some(Commands::Env { file, path, apply }) => {
            let rep = sanitize_workspace_environment(std::path::Path::new(path), file.as_deref())?;
            if *apply {
                let _ = write_env_example_and_update_gitignore(&rep, std::path::Path::new(path));
            }
            println!("{}", format_env_sanitize_report_for_terminal(&rep));
        }
        Some(Commands::Sdk { spec, lang, package }) => {
            let spec_content = if std::path::Path::new(spec).is_file() {
                std::fs::read_to_string(spec)?
            } else {
                spec.clone()
            };
            let sdk = generate_openapi_sdk(&spec_content, lang, package)?;
            println!("{}", format_sdk_report_for_terminal(&sdk));
        }
        Some(Commands::Eval { engine, query, data }) => {
            let res = evaluate_scratchpad_query(engine, query, data)?;
            println!("{}", format_eval_result_for_terminal(&res));
        }
        Some(Commands::Rebase { base, path, execute }) => {
            let plan = plan_smart_rebase(std::path::Path::new(path), Some(base.as_str()))?;
            if *execute {
                let _ = execute_smart_rebase(std::path::Path::new(path), &plan, true)?;
            }
            println!("{}", format_rebase_plan_for_terminal(&plan));
        }
        Some(Commands::Migrate { old_schema, new_schema, name, dialect, write_files }) => {
            let res = generate_schema_migration(old_schema, new_schema, name, dialect)?;
            if *write_files {
                let dir = std::path::Path::new("migrations");
                let _ = std::fs::create_dir_all(dir);
                let up_file = dir.join(format!("{}_up.sql", name));
                let down_file = dir.join(format!("{}_down.sql", name));
                std::fs::write(&up_file, &res.up_sql)?;
                std::fs::write(&down_file, &res.down_sql)?;
                println!("Saved migration files:\n  - {}\n  - {}", up_file.display(), down_file.display());
            }
            println!("{}", format_migration_report_for_terminal(&res));
        }
        Some(Commands::Translate { source, target_lang, source_lang, output }) => {
            let src_code = if std::path::Path::new(source).is_file() {
                std::fs::read_to_string(source)?
            } else {
                source.clone()
            };
            let s_lang = source_lang.as_deref().unwrap_or_else(|| detect_source_language(source));
            let res = transpile_code_snippet(&src_code, s_lang, target_lang, Some(&client), Some(&cli.model), None).await?;
            if let Some(out_p) = output {
                std::fs::write(out_p, &res.transpiled_code)?;
                println!("Saved transpiled code to {}", out_p);
            }
            println!("{}", format_transpile_report_for_terminal(&res));
        }
        Some(Commands::Adr { title, context, decision, consequences, status, path }) => {
            let res = create_architecture_decision_record(std::path::Path::new(path), title, context, decision, consequences, Some(status.as_str()))?;
            println!("{}", format_adr_report_for_terminal(&res));
        }
        Some(Commands::Pkg { ecosystem, package }) => {
            let info = query_package_registry(ecosystem, package, &client).await?;
            println!("{}", format_package_info_for_terminal(&info));
        }
        Some(Commands::A11y { target, path }) => {
            let rep = audit_workspace_accessibility(std::path::Path::new(path), target.as_deref())?;
            println!("{}", format_a11y_report_for_terminal(&rep));
        }
        Some(Commands::Stats { path, reset }) => {
            if *reset {
                reset_analytics(std::path::Path::new(path))?;
                println!("Analytics usage metrics reset successfully.");
            } else {
                let rep = generate_analytics_report(std::path::Path::new(path));
                println!("{}", format_analytics_dashboard_for_terminal(&rep));
            }
        }
        Some(Commands::Graphic { path, protocol, max_width, max_height }) => {
            let proto = protocol.as_deref().unwrap_or("auto");
            let max_w = max_width.unwrap_or(60);
            let max_h = max_height.unwrap_or(28);
            let rendered = render_diagram_or_image(path, proto, max_w, max_h)?;
            let report = TerminalGraphicReport {
                format: "auto".to_string(),
                protocol: proto.to_string(),
                dimensions: (max_w, max_h),
                payload_size: path.len(),
                rendered_output: rendered,
                summary: format!("Rendered graphic '{}' using protocol '{}'", path, proto),
            };
            println!("{}", format_graphic_report_for_terminal(&report));
        }
        Some(Commands::Gui { action, port, open_browser }) => {
            match action.to_lowercase().as_str() {
                "start" => {
                    let handle = launch_desktop_companion_gui(*port, *open_browser).await?;
                    register_active_gui(handle.clone());
                    println!("{}", format_gui_report_for_terminal(&handle));
                    println!("Press Ctrl+C to stop the GUI server...");
                    tokio::signal::ctrl_c().await?;
                    handle.stop();
                    stop_active_gui();
                    println!("Desktop Companion GUI server stopped.");
                }
                "stop" => {
                    stop_active_gui();
                    println!("Desktop Companion GUI server stopped.");
                }
                _ => {
                    if let Some(h) = get_active_gui() {
                        println!("{}", format_gui_report_for_terminal(&h));
                    } else {
                        println!("No active Desktop Companion GUI server.");
                    }
                }
            }
        }
        Some(Commands::Studio { action, port }) => {
            match action.to_lowercase().as_str() {
                "start" => {
                    let handle = start_swarm_studio_server(*port).await?;
                    register_active_studio(handle.clone());
                    println!("{}", format_studio_report_for_terminal(&handle));
                    println!("Press Ctrl+C to stop the Swarm Studio server...");
                    tokio::signal::ctrl_c().await?;
                    handle.stop();
                    stop_active_studio();
                    println!("Visual Swarm Studio server stopped.");
                }
                "stop" => {
                    stop_active_studio();
                    println!("Visual Swarm Studio server stopped.");
                }
                _ => {
                    if let Some(h) = get_active_studio() {
                        println!("{}", format_studio_report_for_terminal(&h));
                    } else {
                        println!("No active Visual Swarm Studio server.");
                    }
                }
            }
        }
        Some(Commands::Theme { name, list, preview }) => {
            if *list {
                println!("\n{}", "Available Built-in TrueColor Themes:".cyan().bold());
                for t in ThemeManager::list_themes() {
                    let pal = ThemeManager::get_theme(t).unwrap();
                    println!("  • {:<18} {}", t.bold(), pal.primary_accent.paint("████████"));
                }
                println!();
            } else if let Some(n) = name {
                let pal = set_active_theme(n)?;
                println!("Active theme switched to '{}'.", pal.name.green().bold());
                println!("{}", format_theme_report_for_terminal(&pal));
            } else if *preview {
                let active = ThemeManager::get_active_theme();
                println!("{}", format_theme_report_for_terminal(&active));
            } else {
                let active = ThemeManager::get_active_theme();
                println!("{}", format_theme_report_for_terminal(&active));
            }
        }
        Some(Commands::Palette { query, category }) => {
            let items = FuzzyCommandPalette::build_default_items(std::path::Path::new("."), &[]);
            let filtered: Vec<PaletteItem> = if let Some(cat) = category {
                let cat_lower = cat.to_lowercase();
                items.into_iter().filter(|i| match i.category {
                    PaletteCategory::SlashCommand => cat_lower == "command" || cat_lower == "cmd" || cat_lower == "slash",
                    PaletteCategory::File => cat_lower == "file",
                    PaletteCategory::Tool => cat_lower == "tool",
                    PaletteCategory::SessionHistory => cat_lower == "history" || cat_lower == "hist",
                    PaletteCategory::Action => cat_lower == "action",
                }).collect()
            } else {
                items
            };
            let results = FuzzyCommandPalette::search_palette(query, &filtered);
            println!("{}", format_palette_results_for_terminal(query, &results));
        }
        Some(Commands::Sound { action, cue }) => {
            match action.to_lowercase().as_str() {
                "on" | "enable" => {
                    AudioCueEngine::set_enabled(true);
                    println!("{}", format_audio_engine_status_for_terminal(true, None));
                }
                "off" | "disable" | "mute" => {
                    AudioCueEngine::set_enabled(false);
                    println!("{}", format_audio_engine_status_for_terminal(false, None));
                }
                "toggle" => {
                    let enabled = AudioCueEngine::toggle_enabled();
                    println!("{}", format_audio_engine_status_for_terminal(enabled, None));
                }
                "test" => {
                    let results = AudioCueEngine::test_all_cues();
                    println!("\n{}", "Synthesized Audio Sensory Feedback Cues:".cyan().bold());
                    for r in results {
                        println!("  🔊 {}", r);
                    }
                    let _ = play_sound_cue("task_completed");
                }
                "status" => {
                    let enabled = AudioCueEngine::is_enabled();
                    println!("{}", format_audio_engine_status_for_terminal(enabled, cue.as_deref()));
                }
                _ => {
                    let cue_name = cue.as_deref().unwrap_or(action.as_str());
                    let _ = play_sound_cue(cue_name)?;
                    println!("Played audio cue: {}", cue_name.cyan().bold());
                }
            }
        }
        Some(Commands::Stage { path, indices, apply, split }) => {
            let hunk_indices: Vec<usize> = indices.as_deref()
                .map(|s| s.split(',').filter_map(|x| x.trim().parse::<usize>().ok()).collect())
                .unwrap_or_default();
            let diff_content = if std::path::Path::new(path).is_file() {
                let out = std::process::Command::new("git").args(["diff", path]).output().ok();
                out.and_then(|o| if !o.stdout.is_empty() { String::from_utf8(o.stdout).ok() } else { None })
                    .unwrap_or_else(|| std::fs::read_to_string(path).unwrap_or_default())
            } else {
                path.clone()
            };
            let mut hunks = parse_diff_into_hunks(&diff_content);
            if *split {
                let mut split_hunks = Vec::new();
                for h in &hunks {
                    split_hunks.extend(split_hunk_into_lines(h));
                }
                for (i, h) in split_hunks.iter_mut().enumerate() {
                    h.index = i;
                }
                hunks = split_hunks;
            }
            if *apply && std::path::Path::new(path).is_file() {
                let orig = std::fs::read_to_string(path)?;
                let staged = apply_selected_hunks(&orig, &hunks, &hunk_indices)?;
                std::fs::write(path, staged)?;
                println!("Applied {} staged hunk(s) to {}", hunk_indices.len(), path);
            }
            println!("{}", format_hunk_staging_report_for_terminal(path, &hunks, &hunk_indices));
        }
        Some(Commands::Heatmap { max_ctx, session }) => {
            let messages = load_session(session.as_deref());
            let ctx = max_ctx.unwrap_or(8192);
            let rep = inspect_token_heatmap(&messages, ctx);
            println!("{}", format_token_heatmap_for_terminal(&rep));
        }
        Some(Commands::Slides { path, slide, width, height }) => {
            let content = if std::path::Path::new(path).is_file() {
                std::fs::read_to_string(path)?
            } else {
                path.clone()
            };
            let slides = parse_markdown_into_slides(&content);
            if let Some(idx) = slide {
                let s_idx = (*idx).min(slides.len().saturating_sub(1));
                if let Some(s) = slides.get(s_idx) {
                    let w = width.unwrap_or(80);
                    let h = height.unwrap_or(24);
                    println!("{}", render_slide_to_terminal(s, s_idx, slides.len(), w, h));
                }
            } else {
                run_interactive_presentation(&slides)?;
            }
        }
        Some(Commands::Widgets { action, widget }) => {
            let mut state = TuiWidgetBarState::new();
            state.update_git_metrics(std::path::Path::new("."));
            state.update_hardware_metrics();
            match action.to_lowercase().as_str() {
                "toggle" => {
                    if let Some(w_name) = widget {
                        if let Some(w_type) = parse_widget_type_name(w_name) {
                            state.toggle_widget(w_type);
                            println!("Toggled widget {:?}", w_type);
                        }
                    }
                    println!("{}", render_dockable_widget_bar(&state, 80));
                }
                "list" | "status" => {
                    println!("\n{}", "Modular TUI Widgets Status:".cyan().bold());
                    for w in &[WidgetType::GitStream, WidgetType::DockerMonitor, WidgetType::DatabaseTailer, WidgetType::HardwareSparklines] {
                        let enabled = state.is_widget_enabled(*w);
                        println!("  • {:<20} [{}]", format!("{:?}", w).bold(), if enabled { "ENABLED".green().bold() } else { "DISABLED".dimmed() });
                    }
                    println!();
                }
                _ => {
                    println!("{}", render_dockable_widget_bar(&state, 80));
                }
            }
        }
        Some(Commands::Speak { text, speed, pitch, background }) => {
            let text_to_speak = if text.is_empty() { "zy intelligent voice engine ready.".to_string() } else { text.join(" ") };
            if *background {
                speak_in_background(&text_to_speak, *speed, *pitch)?;
                println!("Synthesizing speech in background: \"{}\"", text_to_speak.cyan());
            } else {
                synthesize_speech(&text_to_speak, *speed, *pitch)?;
                println!("Spoken: \"{}\"", text_to_speak.cyan());
            }
        }
        Some(Commands::Debug { trace_or_cmd, execute }) => {
            let input = trace_or_cmd.join(" ");
            let trace_content = if *execute && !input.is_empty() {
                println!("{} Executing command to capture crash: `{}`", "🐛".cyan(), input.yellow());
                let parts: Vec<&str> = input.split_whitespace().collect();
                if let Some(cmd) = parts.first() {
                    let out = std::process::Command::new(cmd).args(&parts[1..]).output();
                    match out {
                        Ok(o) => {
                            let mut full = String::from_utf8_lossy(&o.stdout).to_string();
                            full.push_str("\n");
                            full.push_str(&String::from_utf8_lossy(&o.stderr));
                            full
                        }
                        Err(e) => format!("Execution failure: {}", e),
                    }
                } else {
                    input
                }
            } else if std::path::Path::new(&input).is_file() {
                std::fs::read_to_string(&input)?
            } else {
                input
            };
            let parsed = parse_crash_stack_trace(&trace_content)?;
            println!("{}", format_stack_trace_report_for_terminal(&parsed));
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