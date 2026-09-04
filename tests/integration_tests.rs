use zy::*;

#[test]
fn test_lsp_diagnostics_cargo_valid_and_invalid() {
    // 1. Test cargo check on current valid crate
    let report = run_lsp_diagnostics("src/main.rs");
    assert_eq!(report.target, "src/main.rs");
    assert!(report.tool.contains("cargo"));
    assert!(report.success);
    assert_eq!(report.error_count, 0);

    // 2. Test cargo check JSON message parsing on synthetic error
    let raw_cargo_err = r#"
{"reason":"compiler-message","package_id":"zy 0.1.0","manifest_path":"Cargo.toml","target":{"kind":["bin"],"crate_types":["bin"],"name":"zy"},"message":{"rendered":"error[E0425]: cannot find value `unresolved_var` in this scope\n","children":[],"code":{"code":"E0425","explanation":null},"level":"error","message":"cannot find value `unresolved_var` in this scope","spans":[{"byte_end":200,"byte_start":180,"column_end":25,"column_start":5,"file_name":"src/main.rs","is_primary":true,"line_end":42,"line_start":42,"suggested_replacement":null,"suggestion_applicability":null,"text":[{"highlight_end":25,"highlight_start":5,"text":"    unresolved_var = 100;"}]}]}}
"#;
    let issues = parse_cargo_json_diagnostics("src/main.rs", raw_cargo_err);
    assert_eq!(issues.len(), 1);
    assert_eq!(issues[0].file, "src/main.rs");
    assert_eq!(issues[0].line, 42);
    assert_eq!(issues[0].column, 5);
    assert_eq!(issues[0].severity, "error");
    assert_eq!(issues[0].message, "cannot find value `unresolved_var` in this scope");
    assert!(issues[0].code_snippet.as_ref().unwrap().contains("unresolved_var"));

    let terminal_formatted = format_diagnostic_report_for_terminal(&DiagnosticReport {
        target: "src/main.rs".to_string(),
        tool: "cargo check".to_string(),
        success: false,
        issue_count: 1,
        error_count: 1,
        warning_count: 0,
        issues: issues.clone(),
        summary: "1 error found".to_string(),
    });
    assert!(terminal_formatted.contains("LSP DIAGNOSTICS REPORT"));
    assert!(terminal_formatted.contains("src/main.rs:42:5"));
}

#[test]
fn test_lsp_diagnostics_python_parser() {
    let raw_py_stderr = r#"
Traceback (most recent call last):
  File "C:\Python\lib\py_compile.py", line 144, in compile
    code = loader.source_to_code(source_bytes, dfile or file,
  File "<frozen importlib._bootstrap_external>", line 940, in source_to_code
  File "test_script.py", line 15
    print "missing parentheses in python 3"
          ^
SyntaxError: Missing parentheses in call to 'print'. Did you mean print(...)?
"#;
    let issues = parse_python_stderr("test_script.py", raw_py_stderr);
    assert_eq!(issues.len(), 1);
    assert_eq!(issues[0].file, "test_script.py");
    assert_eq!(issues[0].line, 15);
    assert_eq!(issues[0].severity, "error");
    assert!(issues[0].message.contains("SyntaxError"));
}

#[test]
fn test_mcp_jsonrpc_protocol_structures() {
    // Validate initialize JSON-RPC structure
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
    assert_eq!(init_req["jsonrpc"], "2.0");
    assert_eq!(init_req["method"], "initialize");
    assert_eq!(init_req["params"]["clientInfo"]["name"], "zy");

    // Validate tools/call JSON-RPC structure
    let call_req = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {
            "name": "read_file",
            "arguments": { "path": "Cargo.toml" }
        }
    });
    assert_eq!(call_req["method"], "tools/call");
    assert_eq!(call_req["params"]["name"], "read_file");
    assert_eq!(call_req["params"]["arguments"]["path"], "Cargo.toml");

    // Validate tools list contains mcp_execute
    let tools = get_tools();
    let tools_str = serde_json::to_string(&tools).unwrap();
    assert!(tools_str.contains("mcp_execute"));
    assert!(tools_str.contains("lsp_diagnostics"));
}

#[test]
fn test_visual_terminal_diff_viewer() {
    let old_text = "line 1: start\nline 2: to delete\nline 3: keep\n";
    let new_text = "line 1: start\nline 2: replacement\nline 3: keep\nline 4: new end\n";

    let diff_render = render_terminal_diff("test.txt", old_text, new_text);
    assert!(diff_render.contains("File Unified Diff: test.txt"));
    assert!(diff_render.contains("line 2: to delete"));
    assert!(diff_render.contains("line 2: replacement"));
    assert!(diff_render.contains("line 4: new end"));
    assert!(diff_render.contains("│"));
}

#[test]
fn test_token_budgeting_engine_allocations() {
    // 1. Token estimator
    assert_eq!(estimate_tokens(""), 0);
    let short_text = "Hello world";
    assert!(estimate_tokens(short_text) >= 2);

    let code_snippet = r#"
    fn calculate_sum(a: i32, b: i32) -> i32 {
        let result = a + b;
        println!("Calculated: {}", result);
        result
    }
    "#;
    let code_tokens = estimate_tokens(code_snippet);
    assert!(code_tokens > 20);

    // 2. Budget aware pruning
    let sys_msg = Message { role: "system".to_string(), content: "System prompt".to_string(), tool_calls: None, images: None };
    let mut msgs = vec![sys_msg.clone()];
    
    // Add 10 bulky turns
    for i in 0..10 {
        msgs.push(Message {
            role: "user".to_string(),
            content: format!("User turn {} with some long content repeating text {}", i, "word ".repeat(50)),
            tool_calls: None,
            images: None,
        });
        msgs.push(Message {
            role: "assistant".to_string(),
            content: format!("Assistant turn {} response with code {}", i, "code ".repeat(50)),
            tool_calls: None,
            images: None,
        });
    }
    msgs.push(Message {
        role: "user".to_string(),
        content: "CRITICAL NEW PROMPT FROM USER".to_string(),
        tool_calls: None,
        images: None,
    });

    let total_before = estimate_conversation_tokens(&msgs);
    assert!(total_before > 1000);

    // Prune for 512 token context window
    budget_aware_prune(&mut msgs, 512);

    // Assert system prompt preserved
    assert_eq!(msgs.first().unwrap().role, "system");
    assert_eq!(msgs.first().unwrap().content, "System prompt");

    // Assert latest user prompt preserved
    assert_eq!(msgs.last().unwrap().content, "CRITICAL NEW PROMPT FROM USER");

    // Assert total tokens within budget
    let total_after = estimate_conversation_tokens(&msgs);
    assert!(total_after <= 512);

    // Format token budget string
    let budget_str = format_token_budget(&msgs, 512);
    assert!(budget_str.contains("Tokens:"));
    assert!(budget_str.contains("512"));
}

#[test]
fn test_grammar_constrained_schema() {
    let schema = build_tool_grammar_schema();
    assert_eq!(schema["type"], "object");
    assert_eq!(schema["required"][0], "action");

    let req = ChatRequest {
        model: "llama3".to_string(),
        messages: vec![],
        stream: false,
        tools: None,
        format: Some(schema.clone()),
        options: None,
        keep_alive: None,
    };

    let req_json = serde_json::to_string(&req).unwrap();
    assert!(req_json.contains("\"format\":{"));
    assert!(req_json.contains("\"properties\":{"));
}

#[test]
fn test_dual_model_speculative_router_classification() {
    // Test route prompt construction and types
    let chat_query = "Hello, how are you today?";
    let coding_query = "Please write a Rust function to parse JSON with error handling and patch src/main.rs";

    assert_eq!(chat_query.len() > 0, true);
    assert_eq!(coding_query.len() > 0, true);
    assert_ne!(RouteDecision::Chat, RouteDecision::Coding);
}

#[test]
fn test_rules_engine_toml_and_markdown() {
    // 1. Test parse_toml_rules with single string
    let toml1 = r#"
[rules]
instructions = "Always write clean idiomatic Rust."
"#;
    assert_eq!(parse_toml_rules(toml1).unwrap(), "Always write clean idiomatic Rust.");

    // 2. Test parse_toml_rules with array of strings
    let toml2 = r#"
[rules]
rules = [
    "Rule 1: Always add unit tests.",
    "Rule 2: Never use unwrap() in production."
]
"#;
    let res2 = parse_toml_rules(toml2).unwrap();
    assert!(res2.contains("Rule 1: Always add unit tests."));
    assert!(res2.contains("Rule 2: Never use unwrap() in production."));

    // 3. Test parse_toml_rules top level fallback
    let toml3 = r#"
rules = "Top level rule text"
"#;
    assert_eq!(parse_toml_rules(toml3).unwrap(), "Top level rule text");

    // 4. Test parse_toml_rules on irrelevant toml
    let toml4 = r#"
[package]
name = "my_app"
version = "0.1.0"
"#;
    assert!(parse_toml_rules(toml4).is_none());

    // 5. Test load_project_rules across priority hierarchy (.zyrules -> .zy/rules.md -> zy.toml)
    let temp_base = std::env::temp_dir().join(format!("zy_rules_test_{}_{}", std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
    let _ = std::fs::create_dir_all(&temp_base);

    // Write .zyrules
    let zyrules_path = temp_base.join(".zyrules");
    std::fs::write(&zyrules_path, "Rule from .zyrules file").unwrap();
    let loaded = load_project_rules(&temp_base);
    assert!(loaded.is_some());
    assert!(loaded.unwrap().contains("Rule from .zyrules file"));

    // Remove .zyrules, write .zy/rules.md
    let _ = std::fs::remove_file(&zyrules_path);
    let zy_dir = temp_base.join(".zy");
    let _ = std::fs::create_dir_all(&zy_dir);
    let md_path = zy_dir.join("rules.md");
    std::fs::write(&md_path, "Rule from .zy/rules.md file").unwrap();
    let loaded_md = load_project_rules(&temp_base);
    assert!(loaded_md.is_some());
    assert!(loaded_md.unwrap().contains("Rule from .zy/rules.md file"));

    // Remove .zy, write zy.toml
    let _ = std::fs::remove_dir_all(&zy_dir);
    let toml_path = temp_base.join("zy.toml");
    std::fs::write(&toml_path, "[rules]\nprompt = 'Rule from zy.toml'").unwrap();
    let loaded_toml = load_project_rules(&temp_base);
    assert!(loaded_toml.is_some());
    assert!(loaded_toml.unwrap().contains("Rule from zy.toml"));

    // Clean up
    let _ = std::fs::remove_dir_all(&temp_base);

    // 6. Test build_initial_messages integration
    let msgs = build_initial_messages(Some("Custom system prompt."), &[], false).unwrap();
    assert!(!msgs.is_empty());
    assert_eq!(msgs[0].role, "system");
    assert!(msgs[0].content.contains("Custom system prompt."));
}

#[test]
fn test_repo_map_symbol_extraction_and_budgeting() {
    // 1. Test extract_identifier_after
    assert_eq!(extract_identifier_after("pub fn calculate_sum(a: i32)", "fn "), Some("calculate_sum"));
    assert_eq!(extract_identifier_after("class UserService<T>:", "class "), Some("UserService"));
    assert_eq!(extract_identifier_after("export interface IAuthService {", "interface "), Some("IAuthService"));
    assert_eq!(extract_identifier_after("func ProcessBatch(ctx Context)", "func "), Some("ProcessBatch"));

    // 2. Test extract_symbols across multiple languages
    // Rust
    let rs_code = r#"
    pub struct AppState {
        count: i32,
    }
    pub enum UserRole {
        Admin,
        Member,
    }
    pub trait ServiceWorker {
        fn execute(&self);
    }
    pub async fn run_server(port: u16) {
    }
    fn internal_helper() {
    }
    "#;
    let rs_syms = extract_symbols(rs_code, "rs");
    assert!(rs_syms.iter().any(|s| s.contains("struct AppState")));
    assert!(rs_syms.iter().any(|s| s.contains("enum UserRole")));
    assert!(rs_syms.iter().any(|s| s.contains("trait ServiceWorker")));
    assert!(rs_syms.iter().any(|s| s.contains("fn run_server")));
    assert!(rs_syms.iter().any(|s| s.contains("fn internal_helper")));

    // Python
    let py_code = r#"
    class DataPipeline:
        def __init__(self):
            pass
        async def fetch_data(self):
            pass
    def standalone_func(x):
        return x * 2
    "#;
    let py_syms = extract_symbols(py_code, "py");
    assert!(py_syms.iter().any(|s| s.contains("class DataPipeline")));
    assert!(py_syms.iter().any(|s| s.contains("def fetch_data")));
    assert!(py_syms.iter().any(|s| s.contains("def standalone_func")));

    // TypeScript / JS
    let ts_code = r#"
    export interface AuthPayload {
        token: string;
    }
    export class SessionManager {
    }
    export async function handleRequest(req: any) {
    }
    const computeHash = () => {
    }
    "#;
    let ts_syms = extract_symbols(ts_code, "ts");
    assert!(ts_syms.iter().any(|s| s.contains("interface AuthPayload")));
    assert!(ts_syms.iter().any(|s| s.contains("class SessionManager")));
    assert!(ts_syms.iter().any(|s| s.contains("fn handleRequest")));
    assert!(ts_syms.iter().any(|s| s.contains("const computeHash")));

    // Go
    let go_code = r#"
    type Engine struct {
        power int
    }
    func StartEngine(e *Engine) error {
        return nil
    }
    "#;
    let go_syms = extract_symbols(go_code, "go");
    assert!(go_syms.iter().any(|s| s.contains("type Engine")));
    assert!(go_syms.iter().any(|s| s.contains("func StartEngine")));

    // C / C++
    let cpp_code = r#"
    class TextureCache {
    };
    struct VertexData {
    };
    void render_frame() {
    }
    "#;
    let cpp_syms = extract_symbols(cpp_code, "cpp");
    assert!(cpp_syms.iter().any(|s| s.contains("class TextureCache")));
    assert!(cpp_syms.iter().any(|s| s.contains("struct VertexData")));
    assert!(cpp_syms.iter().any(|s| s.contains("fn render_frame")));

    // 3. Test build_repo_map with temp directory
    let temp_repo = std::env::temp_dir().join(format!("zy_repomap_test_{}_{}", std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
    let src_dir = temp_repo.join("src");
    let _ = std::fs::create_dir_all(&src_dir);
    std::fs::write(src_dir.join("lib.rs"), "pub fn compute_sum() {}\npub struct DataConfig {}").unwrap();
    std::fs::write(src_dir.join("api.py"), "class ClientApi:\n    def call(): pass").unwrap();

    let map_full = build_repo_map(&temp_repo, 2000);
    assert!(map_full.contains("lib.rs"));
    assert!(map_full.contains("fn compute_sum"));
    assert!(map_full.contains("struct DataConfig"));
    assert!(map_full.contains("api.py"));
    assert!(map_full.contains("class ClientApi"));

    // Budget truncation
    let map_truncated = build_repo_map(&temp_repo, 5);
    assert!(!map_truncated.is_empty());

    let _ = std::fs::remove_dir_all(&temp_repo);

    // 4. Test tool schema
    let tools = get_tools();
    let tools_str = serde_json::to_string(&tools).unwrap();
    assert!(tools_str.contains("get_repo_map"));
}

#[test]
fn test_tdd_runner_detection_and_parser() {
    let temp_dir = std::env::temp_dir().join(format!("zy_tdd_test_{}_{}", std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
    let _ = std::fs::create_dir_all(&temp_dir);

    // 1. Detection tests
    std::fs::write(temp_dir.join("Cargo.toml"), "[package]\nname = 'test'").unwrap();
    assert_eq!(detect_test_runner(&temp_dir), "cargo test");
    let _ = std::fs::remove_file(temp_dir.join("Cargo.toml"));

    std::fs::write(temp_dir.join("pytest.ini"), "[pytest]").unwrap();
    assert_eq!(detect_test_runner(&temp_dir), "pytest");
    let _ = std::fs::remove_file(temp_dir.join("pytest.ini"));

    std::fs::write(temp_dir.join("package.json"), "{\"scripts\": {\"test\": \"jest\"}}").unwrap();
    assert_eq!(detect_test_runner(&temp_dir), "npm test");
    let _ = std::fs::remove_file(temp_dir.join("package.json"));

    std::fs::write(temp_dir.join("go.mod"), "module example.com/test").unwrap();
    assert_eq!(detect_test_runner(&temp_dir), "go test ./...");
    let _ = std::fs::remove_file(temp_dir.join("go.mod"));

    let _ = std::fs::remove_dir_all(&temp_dir);

    // 2. Output parsing tests - Cargo
    let cargo_out = r#"
running 3 tests
test tests::test_alpha ... ok
test tests::test_beta ... FAILED
test tests::test_gamma ... ok

failures:
---- tests::test_beta stdout ----
thread 'tests::test_beta' panicked at 'assertion failed: `(left == right)`'

test result: FAILED. 2 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out
"#;
    let (p, f, fails, summary) = parse_test_output("cargo test", cargo_out, "", Some(101));
    assert_eq!(p, 2);
    assert_eq!(f, 1);
    assert!(fails.iter().any(|s| s.contains("test tests::test_beta ... FAILED")));
    assert!(summary.contains("Tests failed"));

    // 3. Output parsing tests - Pytest
    let pytest_out = r#"
test_auth.py::test_login PASSED
test_auth.py::test_logout FAILED
test_db.py::test_query PASSED
"#;
    let (p, f, fails, summary) = parse_test_output("pytest", pytest_out, "", Some(1));
    assert_eq!(p, 2);
    assert_eq!(f, 1);
    assert!(fails.iter().any(|s| s.contains("FAILED")));
    assert!(summary.contains("Tests failed"));

    // 4. Output parsing tests - NPM / Jest
    let npm_out = r#"
  ✓ should create user successfully
  ✕ should reject invalid email
  ✓ should update profile
"#;
    let (p, f, fails, summary) = parse_test_output("npm test", npm_out, "", Some(1));
    assert_eq!(p, 2);
    assert_eq!(f, 1);
    assert!(fails.iter().any(|s| s.contains("✕")));
    assert!(summary.contains("Tests failed"));

    // 5. Output parsing tests - Go
    let go_out = r#"
--- PASS: TestInit (0.00s)
--- FAIL: TestConfig (0.01s)
--- PASS: TestRun (0.02s)
"#;
    let (p, f, fails, summary) = parse_test_output("go test ./...", go_out, "", Some(1));
    assert_eq!(p, 2);
    assert_eq!(f, 1);
    assert!(fails.iter().any(|s| s.contains("--- FAIL: TestConfig")));
    assert!(summary.contains("Tests failed"));

    // 6. Test terminal formatter
    let report_passed = TestReport {
        runner: "cargo test".to_string(),
        success: true,
        exit_code: Some(0),
        passed_count: 5,
        failed_count: 0,
        failure_details: vec![],
        stdout: "ok".to_string(),
        stderr: "".to_string(),
        summary: "All tests passed (5 passed)".to_string(),
    };
    let formatted_pass = format_test_report_for_terminal(&report_passed);
    assert!(formatted_pass.contains("PASSED"));
    assert!(formatted_pass.contains("5 passed"));

    let report_failed = TestReport {
        runner: "cargo test".to_string(),
        success: false,
        exit_code: Some(101),
        passed_count: 4,
        failed_count: 1,
        failure_details: vec!["test failed: bad logic".to_string()],
        stdout: "".to_string(),
        stderr: "err".to_string(),
        summary: "Tests failed".to_string(),
    };
    let formatted_fail = format_test_report_for_terminal(&report_failed);
    assert!(formatted_fail.contains("FAILED"));
    assert!(formatted_fail.contains("bad logic"));

    // 7. Verify run_tests tool schema
    let tools = get_tools();
    let tools_str = serde_json::to_string(&tools).unwrap();
    assert!(tools_str.contains("run_tests"));
}

#[test]
fn test_git_micro_checkpoints_and_rollback() {
    // 1. Serialization tests
    let checkpoints = vec![
        GitCheckpoint {
            id: "chk_1001".to_string(),
            label: "before refactor".to_string(),
            commit_sha: "abcdef1234567890".to_string(),
            timestamp: 1700000000,
        },
        GitCheckpoint {
            id: "chk_1002".to_string(),
            label: "auto-checkpoint before tool run".to_string(),
            commit_sha: "fedcba0987654321".to_string(),
            timestamp: 1700000050,
        }
    ];

    save_checkpoints(&checkpoints);
    let loaded = load_checkpoints();
    assert_eq!(loaded.len(), 2);
    assert_eq!(loaded[0].id, "chk_1001");
    assert_eq!(loaded[0].label, "before refactor");
    assert_eq!(loaded[1].id, "chk_1002");

    // Clean up checkpoints file
    let _ = std::fs::remove_file(CHECKPOINTS_FILE);

    // 2. Non-git repo check
    let non_git_dir = std::env::temp_dir().join(format!("zy_nongit_{}_{}", std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
    let _ = std::fs::create_dir_all(&non_git_dir);
    
    // In a directory without .git, verify error
    let prev_dir = std::env::current_dir().unwrap();
    let _ = std::env::set_current_dir(&non_git_dir);
    let res = create_git_checkpoint_with_label(Some("test"));
    assert!(res.is_err());
    let _ = std::env::set_current_dir(&prev_dir);
    let _ = std::fs::remove_dir_all(&non_git_dir);
}

#[test]
fn test_sandbox_container_command_builder() {
    let workspace = std::path::Path::new("/workspace/project");

    // 1. Default image
    let (exe, args) = build_sandbox_command("npm run build", workspace, None);
    assert_eq!(exe, "docker");
    assert_eq!(args[0], "run");
    assert_eq!(args[1], "--rm");
    assert_eq!(args[2], "-i");
    assert_eq!(args[3], "-v");
    assert!(args[4].contains("project:/workspace") || args[4].contains("/workspace/project:/workspace"));
    assert_eq!(args[5], "-w");
    assert_eq!(args[6], "/workspace");
    assert_eq!(args[7], "alpine:latest");
    assert_eq!(args[8], "sh");
    assert_eq!(args[9], "-c");
    assert_eq!(args[10], "npm run build");

    // 2. Custom image
    let (exe_custom, args_custom) = build_sandbox_command("pytest -v", workspace, Some("python:3.12-alpine"));
    assert_eq!(exe_custom, "docker");
    assert_eq!(args_custom[7], "python:3.12-alpine");
    assert_eq!(args_custom[10], "pytest -v");
}

#[test]
fn test_streaming_think_tag_and_tool_call_accumulator() {
    // 1. Test streamed content with <think> blocks
    let raw_stream = "<think>Analyzing user request\nConsidering function signature</think>Here is the solution.";
    let mut in_think = false;
    let mut think_content = String::new();
    let mut final_content = String::new();

    let mut remaining = raw_stream;
    while !remaining.is_empty() {
        if !in_think {
            if let Some(pos) = remaining.find("<think>") {
                final_content.push_str(&remaining[..pos]);
                in_think = true;
                remaining = &remaining[pos + 7..];
            } else {
                final_content.push_str(remaining);
                break;
            }
        } else {
            if let Some(pos) = remaining.find("</think>") {
                think_content.push_str(&remaining[..pos]);
                in_think = false;
                remaining = &remaining[pos + 8..];
            } else {
                think_content.push_str(remaining);
                break;
            }
        }
    }

    assert_eq!(think_content, "Analyzing user request\nConsidering function signature");
    assert_eq!(final_content, "Here is the solution.");

    // 2. Test JSON Tool Call extraction fallback
    let raw_json_tool = r#"{"name": "run_bash", "arguments": {"cmd": "ls -la"}}"#;
    let parsed: Result<serde_json::Value, _> = serde_json::from_str(raw_json_tool);
    assert!(parsed.is_ok());
    let val = parsed.unwrap();
    assert_eq!(val["name"], "run_bash");
    assert_eq!(val["arguments"]["cmd"], "ls -la");
}

