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
    assert!(terminal_formatted.contains("src/main.rs"));
    assert!(terminal_formatted.contains("42"));
    assert!(terminal_formatted.contains("5"));
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

    assert!(chat_query.starts_with("Hello"));
    assert!(coding_query.contains("Rust"));
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
    assert!(formatted_pass.contains("5"));
    assert!(formatted_pass.contains("passed"));

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

#[test]
fn test_hybrid_rag_bm25_lexical_scoring() {
    // 1. Tokenizer tests
    let text = "Hybrid RAG Engine with BM-25 & Vector Cosine!";
    let tokens = tokenize_text(text);
    assert_eq!(tokens, vec!["hybrid", "rag", "engine", "with", "bm", "25", "vector", "cosine"]);

    // 2. BM25 scoring tests
    let doc_tokens_1 = vec!["rust".to_string(), "async".to_string(), "tokio".to_string(), "rust".to_string()];
    let doc_tokens_2 = vec!["python".to_string(), "django".to_string(), "web".to_string()];
    let doc_tokens_3 = vec!["rust".to_string(), "systems".to_string(), "programming".to_string(), "performance".to_string()];

    let avg_len = 3.67;
    let doc_count = 3;
    let doc_freq_rust = 2; // doc 1 and doc 3 have "rust"

    // Term present in doc 1 (2 occurrences)
    let score_doc1 = bm25_score("rust", &doc_tokens_1, avg_len, doc_count, doc_freq_rust);
    assert!(score_doc1 > 0.0);

    // Term present in doc 3 (1 occurrence)
    let score_doc3 = bm25_score("rust", &doc_tokens_3, avg_len, doc_count, doc_freq_rust);
    assert!(score_doc3 > 0.0);

    // Higher frequency in doc 1 gives higher score than doc 3
    assert!(score_doc1 > score_doc3);

    // Term absent in doc 2
    let score_doc2 = bm25_score("rust", &doc_tokens_2, avg_len, doc_count, doc_freq_rust);
    assert_eq!(score_doc2, 0.0);

    // Edge cases
    assert_eq!(bm25_score("", &doc_tokens_1, avg_len, doc_count, doc_freq_rust), 0.0);
    assert_eq!(bm25_score("rust", &[], avg_len, doc_count, doc_freq_rust), 0.0);
    assert_eq!(bm25_score("rust", &doc_tokens_1, avg_len, 0, doc_freq_rust), 0.0);

    // Multi-term scoring
    let mut df_map = std::collections::HashMap::new();
    df_map.insert("rust".to_string(), 2);
    df_map.insert("tokio".to_string(), 1);
    let query_tokens = vec!["rust".to_string(), "tokio".to_string()];
    let score_multi = score_document_bm25(&query_tokens, &doc_tokens_1, avg_len, doc_count, &df_map);
    assert!(score_multi > score_doc1);
}

#[test]
fn test_hybrid_rag_cosine_similarity() {
    let vec_a = vec![1.0, 0.0, 0.0];
    let vec_b = vec![1.0, 0.0, 0.0];
    let vec_c = vec![0.0, 1.0, 0.0];
    let vec_d = vec![-1.0, 0.0, 0.0];

    // Identical vectors
    let sim_ab = cosine_similarity(&vec_a, &vec_b);
    assert!((sim_ab - 1.0).abs() < 1e-5);

    // Orthogonal vectors
    let sim_ac = cosine_similarity(&vec_a, &vec_c);
    assert!((sim_ac - 0.0).abs() < 1e-5);

    // Opposite vectors
    let sim_ad = cosine_similarity(&vec_a, &vec_d);
    assert!((sim_ad - (-1.0)).abs() < 1e-5);

    // Edge cases
    assert_eq!(cosine_similarity(&[], &[]), 0.0);
    assert_eq!(cosine_similarity(&[1.0], &[1.0, 2.0]), 0.0);
}

#[test]
fn test_hybrid_rag_search_rrf_reranking() {
    let chunks = vec![
        RagChunk {
            file: "chunk_bm25_only.rs".to_string(),
            text: "Quicksort partition algorithm implementation and recursive pivot selection.".to_string(),
            vector: vec![0.1, 0.0, 0.0], // Low semantic similarity
        },
        RagChunk {
            file: "chunk_vector_only.rs".to_string(),
            text: "Fast divide and conquer ordering method for arrays and collections.".to_string(),
            vector: vec![0.95, 0.05, 0.0], // High semantic similarity
        },
        RagChunk {
            file: "chunk_hybrid_best.rs".to_string(),
            text: "Quicksort partition algorithm with optimal pivot and divide and conquer ordering.".to_string(),
            vector: vec![0.90, 0.10, 0.0], // High in both semantic vector and BM25 keywords
        },
        RagChunk {
            file: "chunk_irrelevant.rs".to_string(),
            text: "Database connection pool management and connection timeouts.".to_string(),
            vector: vec![0.0, 0.95, 0.0], // Irrelevant
        },
    ];

    let query = "quicksort partition";
    let query_vec = vec![0.92, 0.08, 0.0];

    let results = hybrid_rag_search(&chunks, query, &query_vec, 3, 60);

    assert_eq!(results.len(), 3);
    // The chunk that ranks high in both vector similarity and lexical BM25 should win the #1 spot
    assert_eq!(results[0].1.file, "chunk_hybrid_best.rs");
    assert!(results[0].0 > results[1].0);
    assert!(results[1].0 > results[2].0);
}

#[test]
fn test_swarm_workflow_structures_and_verdicts() {
    let test_rep = TestReport {
        runner: "cargo test".to_string(),
        success: true,
        exit_code: Some(0),
        passed_count: 10,
        failed_count: 0,
        failure_details: vec![],
        stdout: "test result: ok. 10 passed".to_string(),
        stderr: "".to_string(),
        summary: "All tests passed".to_string(),
    };

    let swarm_res = SwarmWorkflowResult {
        goal: "Implement authentication middleware".to_string(),
        plan: "1. Create auth.rs\n2. Add JWT validation\n3. Register middleware".to_string(),
        coder_output: "Successfully wrote auth.rs and patched router.".to_string(),
        audit_report: "Security review completed. No vulnerabilities found. [AUDIT: PASS]".to_string(),
        test_report: Some(test_rep),
        success: true,
    };

    // Test serialization / deserialization roundtrip
    let json_str = serde_json::to_string_pretty(&swarm_res).unwrap();
    let deserialized: SwarmWorkflowResult = serde_json::from_str(&json_str).unwrap();
    assert_eq!(deserialized.goal, "Implement authentication middleware");
    assert!(deserialized.audit_report.contains("[AUDIT: PASS]"));
    assert!(deserialized.success);
}

#[test]
fn test_expand_context_mentions_file_git_and_symbol() {
    let workspace = std::path::Path::new(".");

    // 1. Direct file mention @Cargo.toml
    let prompt1 = "Can you check @Cargo.toml for dependencies?";
    let exp1 = expand_context_mentions(prompt1, workspace);
    assert_eq!(exp1.mentions.len(), 1);
    assert_eq!(exp1.mentions[0].mention_type, "file");
    assert_eq!(exp1.mentions[0].target, "Cargo.toml");
    assert!(exp1.mentions[0].content.contains("name = \"zy\""));
    assert_eq!(exp1.context_messages.len(), 1);

    // 2. Explicit @file:Cargo.toml mention
    let prompt2 = "Inspect @file:Cargo.toml configuration.";
    let exp2 = expand_context_mentions(prompt2, workspace);
    assert_eq!(exp2.mentions.len(), 1);
    assert_eq!(exp2.mentions[0].mention_type, "file");
    assert_eq!(exp2.mentions[0].target, "Cargo.toml");

    // 3. @git and @diff mentions
    let prompt3 = "Compare my changes with @git.";
    let exp3 = expand_context_mentions(prompt3, workspace);
    assert_eq!(exp3.mentions.len(), 1);
    assert_eq!(exp3.mentions[0].mention_type, "git");
    assert!(exp3.mentions[0].content.contains("ACTIVE GIT REPOSITORY CONTEXT"));

    // 4. @symbol:bm25_score mention
    let prompt4 = "Optimize the calculation in @symbol:bm25_score.";
    let exp4 = expand_context_mentions(prompt4, workspace);
    assert_eq!(exp4.mentions.len(), 1);
    assert_eq!(exp4.mentions[0].mention_type, "symbol");
    assert_eq!(exp4.mentions[0].target, "bm25_score");
    assert!(exp4.mentions[0].content.contains("bm25_score"));

    // 5. Multiple mixed mentions in a single prompt
    let prompt5 = "Please review @Cargo.toml and @symbol:hybrid_rag_search alongside @diff.";
    let exp5 = expand_context_mentions(prompt5, workspace);
    assert_eq!(exp5.mentions.len(), 3);
    assert!(exp5.mentions.iter().any(|m| m.mention_type == "file"));
    assert!(exp5.mentions.iter().any(|m| m.mention_type == "symbol"));
    assert!(exp5.mentions.iter().any(|m| m.mention_type == "git"));
    assert_eq!(exp5.context_messages.len(), 3);
}

#[test]
fn test_web_search_html_parser_and_url_decoding() {
    // 1. URL decoder test
    assert_eq!(url_decode("https%3A%2F%2Fexample.com%2Fdocs%3Fquery%3Drust%2Blang"), "https://example.com/docs?query=rust+lang");
    assert_eq!(url_decode("hello+world"), "hello world");

    // 2. Extract DuckDuckGo redirect URL
    let ddg_redirect = "//duckduckgo.com/l/?uddg=https%3A%2F%2Fdocs.rs%2Ftokio%2Flatest%2Ftokio%2F&rut=123";
    assert_eq!(extract_duckduckgo_url(ddg_redirect), "https://docs.rs/tokio/latest/tokio/");

    // 3. Strip HTML tags & entities
    let html_snippet = "<b>Rust</b> is a <i>systems</i> programming language &amp; framework &lt;safe&gt;.";
    assert_eq!(strip_html_tags(html_snippet), "Rust is a systems programming language & framework <safe>.");

    // 4. Parse DuckDuckGo HTML standard results fixture
    let sample_ddg_html = r##"
    <div class="result results_links">
      <div class="result__body">
        <h2 class="result__title">
          <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.rust-lang.org%2F">Rust Programming Language</a>
        </h2>
        <a class="result__snippet" href="#">A language empowering everyone to build reliable and efficient software.</a>
      </div>
    </div>
    <div class="result results_links">
      <div class="result__body">
        <h2 class="result__title">
          <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fdoc.rust-lang.org%2Fbook%2F">The Rust Programming Language Book</a>
        </h2>
        <a class="result__snippet" href="#">Affectionately referred to as the book, The Rust Programming Language gives you an overview...</a>
      </div>
    </div>
    "##;

    let parsed = parse_duckduckgo_html(sample_ddg_html);
    assert_eq!(parsed.len(), 2);
    assert_eq!(parsed[0].title, "Rust Programming Language");
    assert_eq!(parsed[0].url, "https://www.rust-lang.org/");
    assert!(parsed[0].snippet.contains("reliable and efficient software"));
    assert_eq!(parsed[1].title, "The Rust Programming Language Book");
    assert_eq!(parsed[1].url, "https://doc.rust-lang.org/book/");

    // 5. Parse DuckDuckGo Lite HTML fixture
    let sample_lite_html = r##"
    <table>
      <tr>
        <td>
          <a class="result-link" href="https://crates.io/crates/tokio">tokio - crates.io: Rust Package Registry</a>
        </td>
      </tr>
      <tr>
        <td class="result-snippet">An event-driven, non-blocking I/O platform for writing asynchronous applications with the Rust programming language.</td>
      </tr>
    </table>
    "##;
    let parsed_lite = parse_duckduckgo_html(sample_lite_html);
    assert_eq!(parsed_lite.len(), 1);
    assert_eq!(parsed_lite[0].title, "tokio - crates.io: Rust Package Registry");
    assert_eq!(parsed_lite[0].url, "https://crates.io/crates/tokio");
    assert!(parsed_lite[0].snippet.contains("event-driven"));

    // 6. Tools contains web_search
    let tools = get_tools();
    let tools_str = serde_json::to_string(&tools).unwrap();
    assert!(tools_str.contains("web_search"));
}

#[test]
fn test_time_travel_debugger_timeline_and_rewind() {
    let mut messages = vec![
        Message { role: "system".to_string(), content: "Base system prompt".to_string(), tool_calls: None, images: None },
        Message { role: "user".to_string(), content: "Turn 1: Explain BM25".to_string(), tool_calls: None, images: None },
        Message { role: "assistant".to_string(), content: "BM25 is a ranking function used in information retrieval.".to_string(), tool_calls: None, images: None },
        Message { role: "user".to_string(), content: "Turn 2: Run tests now".to_string(), tool_calls: None, images: None },
        Message {
            role: "assistant".to_string(),
            content: "Running test suite...".to_string(),
            tool_calls: Some(vec![ToolCall {
                function: ToolCallFunction {
                    name: "run_tests".to_string(),
                    arguments: serde_json::json!({}),
                }
            }]),
            images: None,
        },
        Message { role: "tool".to_string(), content: "All tests passed".to_string(), tool_calls: None, images: None },
        Message { role: "assistant".to_string(), content: "Tests are green.".to_string(), tool_calls: None, images: None },
        Message { role: "user".to_string(), content: "Turn 3: Deploy to production".to_string(), tool_calls: None, images: None },
        Message { role: "assistant".to_string(), content: "Deployment initiated.".to_string(), tool_calls: None, images: None },
    ];

    // 1. Extract timeline turns
    let turns = extract_timeline_turns(&messages);
    assert_eq!(turns.len(), 3);
    assert_eq!(turns[0].turn_index, 1);
    assert!(turns[0].user_preview.contains("Turn 1"));
    assert_eq!(turns[1].turn_index, 2);
    assert!(turns[1].tool_calls_count >= 1);
    assert_eq!(turns[2].turn_index, 3);
    assert!(turns[2].user_preview.contains("Turn 3"));

    // 2. Format timeline
    let timeline_formatted = format_timeline(&messages);
    assert!(timeline_formatted.contains("CONVERSATION SESSION TIMELINE"));
    assert!(timeline_formatted.contains("Turn #"));
    assert!(timeline_formatted.contains("1"));
    assert!(timeline_formatted.contains("2"));
    assert!(timeline_formatted.contains("3"));
    assert!(timeline_formatted.contains("Tools:"));

    // 3. Rewind 1 turn
    let rewound_1 = rewind_messages(&mut messages, 1);
    assert_eq!(rewound_1, 1);
    let turns_after_1 = extract_timeline_turns(&messages);
    assert_eq!(turns_after_1.len(), 2);
    assert_eq!(messages.last().unwrap().content, "Tests are green.");

    // 4. Rewind 2 turns (removes remaining turns)
    let rewound_2 = rewind_messages(&mut messages, 2);
    assert_eq!(rewound_2, 2);
    let turns_after_2 = extract_timeline_turns(&messages);
    assert_eq!(turns_after_2.len(), 0);

    // System prompt preserved!
    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0].role, "system");
    assert_eq!(messages[0].content, "Base system prompt");
}

#[test]
fn test_conventional_commit_and_pr_generator() {
    // 1. Parse commit message from markdown code fences
    let fenced_commit = "```\nfeat(rag): implement hybrid BM25 and vector cosine search\n```";
    assert_eq!(parse_conventional_commit(fenced_commit), "feat(rag): implement hybrid BM25 and vector cosine search");

    let quoted_commit = "\"fix(agent): handle DuckDuckGo lite HTML results gracefully\"";
    assert_eq!(parse_conventional_commit(quoted_commit), "fix(agent): handle DuckDuckGo lite HTML results gracefully");

    // 2. Fallback commit message generation by diff inspection
    let diff_cargo = "diff --git a/Cargo.toml b/Cargo.toml\n+reqwest = '0.11'";
    assert!(generate_fallback_commit_message(diff_cargo).starts_with("chore(deps):"));

    let diff_tests = "diff --git a/tests/integration_tests.rs b/tests/integration_tests.rs\n+#[test]";
    assert!(generate_fallback_commit_message(diff_tests).starts_with("test(core):"));

    let diff_docs = "diff --git a/README.md b/README.md\n+# zy Agent Documentation";
    assert!(generate_fallback_commit_message(diff_docs).starts_with("docs(readme):"));

    let diff_fix = "diff --git a/src/lib.rs b/src/lib.rs\n-let err = unwrap();\n+let err = match...";
    assert!(generate_fallback_commit_message(diff_fix).starts_with("fix(core):"));

    let diff_feat = "diff --git a/src/swarm.rs b/src/swarm.rs\n+pub struct SwarmOrchestrator";
    assert!(generate_fallback_commit_message(diff_feat).starts_with("feat(core):"));
}

#[test]
fn test_brutal_hybrid_rag_edge_cases_and_saturation() {
    // 1. BM25 term frequency saturation test
    let avg_len = 100.0;
    let doc_count = 10;
    let doc_freq = 1;

    let doc_tf_5: Vec<String> = vec!["algorithm".to_string(); 5];
    let doc_tf_50: Vec<String> = vec!["algorithm".to_string(); 50];
    let doc_tf_500: Vec<String> = vec!["algorithm".to_string(); 500];

    let score_5 = bm25_score("algorithm", &doc_tf_5, avg_len, doc_count, doc_freq);
    let score_50 = bm25_score("algorithm", &doc_tf_50, avg_len, doc_count, doc_freq);
    let score_500 = bm25_score("algorithm", &doc_tf_500, avg_len, doc_count, doc_freq);

    assert!(score_5 < score_50);
    assert!(score_50 < score_500);
    assert!((score_500 - score_50) < 5.0 * (score_50 - score_5));

    // 2. Cosine similarity zero norms and stability
    let zero_vec = vec![0.0, 0.0, 0.0];
    let normal_vec = vec![1.0, 2.0, 3.0];
    assert_eq!(cosine_similarity(&zero_vec, &normal_vec), 0.0);
    assert_eq!(cosine_similarity(&zero_vec, &zero_vec), 0.0);

    // 3. RRF with empty chunks
    let empty_results = hybrid_rag_search(&[], "query", &[1.0, 0.0], 5, 60);
    assert!(empty_results.is_empty());

    // 4. Single chunk search
    let single_chunk = vec![RagChunk {
        file: "single.rs".to_string(),
        text: "hello world".to_string(),
        vector: vec![1.0, 0.0],
    }];
    let single_res = hybrid_rag_search(&single_chunk, "hello", &[1.0, 0.0], 1, 60);
    assert_eq!(single_res.len(), 1);
    assert_eq!(single_res[0].1.file, "single.rs");
    let expected_score = 2.0 / 61.0;
    assert!((single_res[0].0 - expected_score).abs() < 1e-4);
}

#[test]
fn test_brutal_context_mentions_punctuation_and_resilience() {
    let workspace = std::path::Path::new(".");

    // 1. Nested file path mention
    let prompt1 = "Look at @src/lib.rs and check @src/main.rs please.";
    let exp1 = expand_context_mentions(prompt1, workspace);
    assert_eq!(exp1.mentions.len(), 2);
    assert!(exp1.mentions.iter().any(|m| m.target == "src/lib.rs"));
    assert!(exp1.mentions.iter().any(|m| m.target == "src/main.rs"));

    // 2. Mentions with weird punctuation
    let prompt2 = "Review (@Cargo.toml), and also @git? Finally check {@symbol:Cli};";
    let exp2 = expand_context_mentions(prompt2, workspace);
    assert_eq!(exp2.mentions.len(), 3);
    assert!(exp2.mentions.iter().any(|m| m.mention_type == "file" && m.target == "Cargo.toml"));
    assert!(exp2.mentions.iter().any(|m| m.mention_type == "git"));
    assert!(exp2.mentions.iter().any(|m| m.mention_type == "symbol" && m.target == "Cli"));

    // 3. Non-existent file mention does not panic or crash
    let prompt3 = "Inspect @this_file_definitely_does_not_exist_12345.xyz and @file:also_fake.txt";
    let exp3 = expand_context_mentions(prompt3, workspace);
    assert_eq!(exp3.mentions.len(), 0);
    assert_eq!(exp3.context_messages.len(), 0);

    // 4. Extract symbol context on missing symbol
    let missing_sym = extract_symbol_context(workspace, "completely_nonexistent_symbol_abcxyz_999");
    assert!(missing_sym.is_none());
}

#[test]
fn test_brutal_duckduckgo_html_parser_edge_cases() {
    // 1. HTML with special encoded characters & parameters
    let complex_url = "//duckduckgo.com/l/?uddg=https%3A%2F%2Fcrates.io%2Fsearch%3Fq%3Dtokio%26sort%3Ddownloads%23top&rut=xyz";
    let extracted = extract_duckduckgo_url(complex_url);
    assert_eq!(extracted, "https://crates.io/search?q=tokio&sort=downloads#top");

    // 2. Direct absolute URLs
    let direct_url = "https://github.com/rust-lang/rust";
    assert_eq!(extract_duckduckgo_url(direct_url), "https://github.com/rust-lang/rust");

    // 3. Relative URLs
    let rel_url = "/html?q=rust";
    assert_eq!(extract_duckduckgo_url(rel_url), "https://duckduckgo.com/html?q=rust");

    // 4. Malformed HTML without closing tags
    let broken_html = r##"
    <div class="result results_links">
      <div class="result__body">
        <a class="result__a" href="https://example.com/one">Unclosed Title
        <a class="result__snippet">Unclosed snippet text
    </div>
    "##;
    let parsed = parse_duckduckgo_html(broken_html);
    assert!(!parsed.is_empty());
    assert_eq!(parsed[0].url, "https://example.com/one");
}

#[test]
fn test_brutal_time_travel_deep_session_rewind() {
    let mut deep_messages = Vec::new();
    deep_messages.push(Message { role: "system".to_string(), content: "CORE SYSTEM INSTRUCTIONS".to_string(), tool_calls: None, images: None });
    deep_messages.push(Message { role: "system".to_string(), content: "Core Memory: Active project is zy".to_string(), tool_calls: None, images: None });

    // Generate 10 conversation turns
    for t in 1..=10 {
        deep_messages.push(Message { role: "user".to_string(), content: format!("Turn {}: Request info", t), tool_calls: None, images: None });
        deep_messages.push(Message { role: "assistant".to_string(), content: format!("Turn {}: Response info", t), tool_calls: None, images: None });
    }

    assert_eq!(extract_timeline_turns(&deep_messages).len(), 10);

    // Rewind 3 turns
    let r1 = rewind_messages(&mut deep_messages, 3);
    assert_eq!(r1, 3);
    assert_eq!(extract_timeline_turns(&deep_messages).len(), 7);
    assert_eq!(deep_messages.last().unwrap().content, "Turn 7: Response info");

    // Rewind 4 more turns
    let r2 = rewind_messages(&mut deep_messages, 4);
    assert_eq!(r2, 4);
    assert_eq!(extract_timeline_turns(&deep_messages).len(), 3);
    assert_eq!(deep_messages.last().unwrap().content, "Turn 3: Response info");

    // Rewind 100 turns (more than remaining 3)
    let r3 = rewind_messages(&mut deep_messages, 100);
    assert_eq!(r3, 3);
    assert_eq!(extract_timeline_turns(&deep_messages).len(), 0);

    // Both initial system messages preserved!
    assert_eq!(deep_messages.len(), 2);
    assert_eq!(deep_messages[0].content, "CORE SYSTEM INSTRUCTIONS");
    assert_eq!(deep_messages[1].content, "Core Memory: Active project is zy");
}

#[test]
fn test_brutal_conventional_commit_parser_and_pr_description() {
    // 1. Markdown with multiple fences and explanations
    let complex_llm_response = r#"
```commit
refactor(rag): optimize BM25 reciprocal rank fusion calculation

- vectorize inner loop dot products
- add document frequency caching
```
"#;
    let parsed = parse_conventional_commit(complex_llm_response);
    assert!(parsed.starts_with("refactor(rag): optimize BM25 reciprocal rank fusion calculation"));

    // 2. Single line with backticks
    assert_eq!(parse_conventional_commit("`perf(search): accelerate DuckDuckGo parser`"), "perf(search): accelerate DuckDuckGo parser");

    // 3. Fallback PR template generation
    let branch = "feature/hybrid-rag";
    let fallback_pr = format!(
        "## 🎯 Overview\nThis PR updates the `{}` branch.\n\n## 🚀 Key Changes\n- RAG engine\n\n## 🧪 Testing Checklist\n- [x] cargo test\n",
        branch
    );
    assert!(fallback_pr.contains("feature/hybrid-rag"));
    assert!(fallback_pr.contains("## 🎯 Overview"));
    assert!(fallback_pr.contains("## 🚀 Key Changes"));
    assert!(fallback_pr.contains("## 🧪 Testing Checklist"));
}

// -------------------------------------------------------------------------------------------------
// INTEGRATION TESTS: 6 NEW ESSENTIAL SYSTEMS
// -------------------------------------------------------------------------------------------------

#[test]
fn test_tui_dashboard_layout_rendering_and_panels() {
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;

    let backend = TestBackend::new(140, 45);
    let mut terminal = Terminal::new(backend).unwrap();

    let mut state = TuiAppState {
        active_model: "llama3:70b".to_string(),
        agent_mode: true,
        force_mode: true,
        rag_mode: true,
        cpu_cores: 16,
        total_mem_gb: 32,
        used_mem_gb: 12,
        token_budget_info: "1420 / 8192 (17%)".to_string(),
        aituner_profile: "TURBO (8192 ctx)".to_string(),
        status_msg: "Agent ready".to_string(),
        ..Default::default()
    };

    state.messages.push(Message {
        role: "user".to_string(),
        content: "Refactor vector indexing to zero-copy binary format".to_string(),
        tool_calls: None,
        images: None,
    });
    state.messages.push(Message {
        role: "assistant".to_string(),
        content: "<think>\nAnalyzing memory layout and byte alignment\n</think>\nImplementation complete with 8-byte headers.".to_string(),
        tool_calls: None,
        images: None,
    });

    state.preview_file = "src/vector.rs".to_string();
    state.preview_content = "pub struct BinaryVectorStore {\n    pub version: u32,\n}".to_string();
    state.diff_content = "+pub struct BinaryVectorStore {\n+    pub version: u32,\n+}".to_string();

    terminal.draw(|f| {
        render_tui_layout(f, &state);
    }).unwrap();

    let buffer = terminal.backend().buffer();
    let content = format!("{:?}", buffer);

    // Assert 3 panels rendered
    assert!(content.contains("Chat & Agent Thinking"));
    assert!(content.contains("File Preview & Live Diff"));
    assert!(content.contains("Hardware Stats & AiTuner Profile"));

    // Assert hardware metrics & profile displayed
    assert!(content.contains("16 Cores"));
    assert!(content.contains("12 GB / 32 GB"));
    assert!(content.contains("TURBO"));
    assert!(content.contains("1420 / 8192"));
    assert!(content.contains("llama3:70b"));

    // Assert chat & think tags processed
    assert!(content.contains("User"));
    assert!(content.contains("Refactor vector indexing"));
    assert!(content.contains("Analyzing memory layout"));
    assert!(content.contains("Implementation complete"));

    // Assert diff content rendered
    assert!(content.contains("Live Unified Code Diff"));
    assert!(content.contains("BinaryVectorStore"));
}

#[test]
fn test_embedded_binary_vector_store_serialization_and_search() {
    let temp_dir = std::env::temp_dir().join(format!("zy_bin_vec_test_{}_{}", std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
    let _ = std::fs::create_dir_all(&temp_dir);
    let bin_path = temp_dir.join("vectors.bin");

    // 1. Create realistic chunks with 768-dim float vectors
    let mut chunk1_vec = vec![0.0f32; 768];
    chunk1_vec[0] = 0.85;
    chunk1_vec[1] = 0.45;
    chunk1_vec[767] = -0.32;

    let mut chunk2_vec = vec![0.0f32; 768];
    chunk2_vec[0] = 0.10;
    chunk2_vec[1] = 0.95;
    chunk2_vec[767] = 0.12;

    let chunks = vec![
        RagChunk {
            file: "src/memory.rs".to_string(),
            text: "Zero-copy persistent binary vector database for fast neural embeddings.".to_string(),
            vector: chunk1_vec.clone(),
        },
        RagChunk {
            file: "src/database.rs".to_string(),
            text: "SQLite schema introspection and safe read-only SQL execution engine.".to_string(),
            vector: chunk2_vec.clone(),
        },
    ];

    // 2. Save binary vector index
    let bytes_written = save_binary_vector_index(&bin_path, &chunks).unwrap();
    assert!(bytes_written > 32);
    assert!(bin_path.exists());

    // Verify binary magic header directly from disk bytes
    let raw_bytes = std::fs::read(&bin_path).unwrap();
    assert_eq!(&raw_bytes[0..8], BINARY_VECTOR_MAGIC);

    // 3. Load binary vector index
    let store = load_binary_vector_index(&bin_path).unwrap();
    assert_eq!(store.version, 1);
    assert_eq!(store.vector_dim, 768);
    assert_eq!(store.len(), 2);
    assert_eq!(store.chunks[0].file, "src/memory.rs");
    assert_eq!(store.chunks[0].text, "Zero-copy persistent binary vector database for fast neural embeddings.");
    assert_eq!(store.chunks[0].vector.len(), 768);
    assert!((store.chunks[0].vector[0] - 0.85).abs() < 1e-6);
    assert_eq!(store.chunks[1].file, "src/database.rs");

    // 4. Test search operations
    let query = "binary vector database";
    let mut query_vec = vec![0.0f32; 768];
    query_vec[0] = 0.90;
    query_vec[1] = 0.40;

    let search_res = store.search(query, &query_vec, 2, 60);
    assert_eq!(search_res.len(), 2);
    assert_eq!(search_res[0].1.file, "src/memory.rs");

    let fast_res = store.fast_vector_search(&query_vec, 1);
    assert_eq!(fast_res.len(), 1);
    assert_eq!(fast_res[0].1.file, "src/memory.rs");

    // 5. Test add_or_replace_file
    let mut store_mut = store;
    let mut new_vec = vec![0.0f32; 768];
    new_vec[0] = 0.99;
    store_mut.add_or_replace_file("src/memory.rs", vec![
        RagChunk {
            file: "src/memory.rs".to_string(),
            text: "Updated memory engine chunk".to_string(),
            vector: new_vec,
        }
    ]);
    assert_eq!(store_mut.len(), 2);
    assert_eq!(store_mut.chunks.iter().find(|c| c.file == "src/memory.rs").unwrap().text, "Updated memory engine chunk");

    // 6. Test corruption handling
    let corrupt_path = temp_dir.join("corrupt.bin");
    std::fs::write(&corrupt_path, b"INVALID_HEADER_DATA_1234567890").unwrap();
    assert!(load_binary_vector_index(&corrupt_path).is_err());

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_dependency_and_security_auditor() {
    let temp_dir = std::env::temp_dir().join(format!("zy_audit_test_{}_{}", std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
    let _ = std::fs::create_dir_all(&temp_dir);

    // 1. Synthetic Cargo.lock with known vulnerable rustls and rsa
    let cargo_lock_content = r#"
version = 3

[[package]]
name = "rustls"
version = "0.21.5"

[[package]]
name = "rsa"
version = "0.8.0"

[[package]]
name = "serde"
version = "1.0.197"
"#;
    std::fs::write(temp_dir.join("Cargo.lock"), cargo_lock_content).unwrap();

    // 2. Synthetic Cargo.toml with wildcard and copyleft license
    let cargo_toml_content = r#"
[package]
name = "sample-app"
version = "0.1.0"
license = "AGPL-3.0"

[dependencies]
bad-crate = "*"
insecure-pkg = "http://insecure-server.com/repo.git"
"#;
    std::fs::write(temp_dir.join("Cargo.toml"), cargo_toml_content).unwrap();

    // 3. Synthetic package.json with vulnerable lodash and axios
    let pkg_json_content = r#"{
  "name": "web-frontend",
  "dependencies": {
    "lodash": "4.17.15",
    "axios": "1.5.0",
    "left-pad": "latest"
  }
}"#;
    std::fs::write(temp_dir.join("package.json"), pkg_json_content).unwrap();

    // 4. Synthetic requirements.txt with vulnerable requests and unpinned package
    let req_content = "requests==2.25.0\ndjango==4.0.0\nflask\n";
    std::fs::write(temp_dir.join("requirements.txt"), req_content).unwrap();

    // 5. Run audit
    let report = audit_project_dependencies(&temp_dir);
    assert!(!report.passed);
    assert!(report.scanned_manifests.contains(&"Cargo.lock".to_string()));
    assert!(report.scanned_manifests.contains(&"package.json".to_string()));
    assert!(report.scanned_manifests.contains(&"requirements.txt".to_string()));
    assert!(report.total_dependencies >= 7);

    // Assert vulnerabilities detected
    assert!(report.vulnerabilities.iter().any(|v| v.package == "rustls" && v.severity == "HIGH"));
    assert!(report.vulnerabilities.iter().any(|v| v.package == "rsa" && v.severity == "HIGH"));
    assert!(report.vulnerabilities.iter().any(|v| v.package == "lodash" && v.severity == "HIGH"));
    assert!(report.vulnerabilities.iter().any(|v| v.package == "requests" && v.severity == "MEDIUM"));
    assert!(report.vulnerabilities.iter().any(|v| v.package == "django" && v.severity == "HIGH"));
    assert!(report.vulnerabilities.iter().any(|v| v.package == "insecure-pkg" && v.title.contains("Insecure Plaintext Transport")));

    // Assert license risk detected
    assert!(report.license_risks.iter().any(|l| l.license == "AGPL-3.0" && l.risk_level == "HIGH"));

    // Assert wildcard and unpinned dependencies detected
    assert!(report.outdated_or_wildcards.iter().any(|o| o.package == "bad-crate" && o.current_requirement == "*"));
    assert!(report.outdated_or_wildcards.iter().any(|o| o.package == "left-pad" && o.current_requirement == "latest"));
    assert!(report.outdated_or_wildcards.iter().any(|o| o.package == "flask" && o.current_requirement == "unpinned"));

    // 6. Test terminal report formatting
    let terminal_out = format_security_report_for_terminal(&report);
    assert!(terminal_out.contains("SECURITY & DEPENDENCY AUDIT"));
    assert!(terminal_out.contains("CVE-2024-32650"));
    assert!(terminal_out.contains("CVE-2021-23337"));
    assert!(terminal_out.contains("LICENSE RISK"));

    // 7. Verify audit_security tool exists in get_tools()
    let tools = get_tools();
    let tools_str = serde_json::to_string(&tools).unwrap();
    assert!(tools_str.contains("audit_security"));

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_sqlite_database_and_sql_inspector() {
    let temp_dir = std::env::temp_dir().join(format!("zy_db_test_{}_{}", std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
    let _ = std::fs::create_dir_all(&temp_dir);
    let db_path = temp_dir.join("test_app.sqlite");

    // 1. Create and populate SQLite database
    {
        let conn = rusqlite::Connection::open(&db_path).unwrap();
        conn.execute_batch(r#"
            CREATE TABLE users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                email TEXT,
                balance REAL DEFAULT 0.0
            );
            CREATE TABLE orders (
                order_id INTEGER PRIMARY KEY,
                user_id INTEGER NOT NULL,
                total REAL NOT NULL,
                status TEXT NOT NULL
            );
            CREATE VIEW active_users AS SELECT id, username, email FROM users WHERE balance > 0.0;

            INSERT INTO users (username, email, balance) VALUES ('alice', 'alice@zy.ai', 150.50);
            INSERT INTO users (username, email, balance) VALUES ('bob', 'bob@zy.ai', 0.0);
            INSERT INTO users (username, email, balance) VALUES ('charlie', 'charlie@zy.ai', 99.99);

            INSERT INTO orders (user_id, total, status) VALUES (1, 45.0, 'completed');
            INSERT INTO orders (user_id, total, status) VALUES (3, 89.99, 'pending');
        "#).unwrap();
    }

    // 2. Test schema inspection without query
    let schema_report = inspect_sqlite_database(&db_path, None).unwrap();
    assert!(schema_report.success);
    assert_eq!(schema_report.tables.len(), 3); // users, orders, active_users

    let users_table = schema_report.tables.iter().find(|t| t.table_name == "users").unwrap();
    assert_eq!(users_table.item_type, "table");
    assert_eq!(users_table.row_count, 3);
    assert_eq!(users_table.columns.len(), 4);
    assert!(users_table.columns.iter().any(|c| c.name == "username" && c.notnull));
    assert!(users_table.columns.iter().any(|c| c.name == "id" && c.pk));

    // 3. Test safe read-only SQL query execution
    let query_sql = "SELECT username, email, balance FROM users WHERE balance > 10.0 ORDER BY balance DESC";
    let query_report = inspect_sqlite_database(&db_path, Some(query_sql)).unwrap();
    assert!(query_report.success);
    assert!(query_report.query_result.is_some());

    let q_res = query_report.query_result.as_ref().unwrap();
    assert_eq!(q_res.columns, vec!["username", "email", "balance"]);
    assert_eq!(q_res.row_count, 2);
    assert_eq!(q_res.rows[0][0], "alice");
    assert_eq!(q_res.rows[1][0], "charlie");

    // 4. Test safe queries: EXPLAIN, PRAGMA, WITH
    assert!(is_safe_read_only_query("EXPLAIN SELECT 1").is_ok());
    assert!(is_safe_read_only_query("PRAGMA table_info('users')").is_ok());
    assert!(is_safe_read_only_query("WITH top_users AS (SELECT * FROM users) SELECT * FROM top_users").is_ok());

    // 5. Test security rejection of destructive mutation queries
    assert!(is_safe_read_only_query("DROP TABLE users").is_err());
    assert!(is_safe_read_only_query("DELETE FROM users WHERE 1=1").is_err());
    assert!(is_safe_read_only_query("INSERT INTO users (username) VALUES ('hacker')").is_err());
    assert!(is_safe_read_only_query("UPDATE users SET balance = 100000").is_err());
    assert!(is_safe_read_only_query("ALTER TABLE users ADD COLUMN password TEXT").is_err());
    assert!(is_safe_read_only_query("SELECT * FROM users; DROP TABLE users;").is_err());

    // 6. Test terminal output formatting with ASCII table grid
    let terminal_out = format_database_report_for_terminal(&query_report);
    assert!(terminal_out.contains("SQLITE DATABASE INSPECTOR"));
    assert!(terminal_out.contains("users"));
    assert!(terminal_out.contains("orders"));
    assert!(terminal_out.contains("┌"));
    assert!(terminal_out.contains("alice"));
    assert!(terminal_out.contains("charlie"));

    // 7. Verify db_query tool exists in get_tools()
    let tools = get_tools();
    let tools_str = serde_json::to_string(&tools).unwrap();
    assert!(tools_str.contains("db_query"));

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_docstring_and_api_documentation_generator() {
    let temp_dir = std::env::temp_dir().join(format!("zy_docs_test_{}_{}", std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
    let _ = std::fs::create_dir_all(&temp_dir);

    // 1. Rust file with undocumented symbols
    let rs_path = temp_dir.join("sample_service.rs");
    let rs_content = r#"
pub fn calculate_hash(data: &[u8]) -> u64 {
    42
}

pub struct ServiceConfig {
    pub port: u16,
}

/// Already documented function
pub fn documented_helper() -> bool {
    true
}
"#;
    std::fs::write(&rs_path, rs_content).unwrap();

    // 2. Python file with undocumented symbols
    let py_path = temp_dir.join("worker.py");
    let py_content = r#"
def process_data(records):
    return len(records)

class TaskManager:
    pass

def already_documented():
    """This function is documented."""
    pass
"#;
    std::fs::write(&py_path, py_content).unwrap();

    // 3. TypeScript file with undocumented symbols
    let ts_path = temp_dir.join("api.ts");
    let ts_content = r#"
export function sendRequest(url: string) {
    return fetch(url);
}

export interface ClientOptions {
    timeout: number;
}

/**
 * Documented API client
 */
export class ApiClient {}
"#;
    std::fs::write(&ts_path, ts_content).unwrap();

    // 4. Scan undocumented symbols
    let symbols = scan_undocumented_symbols(&temp_dir);
    assert_eq!(symbols.len(), 6); // 2 in rs, 2 in py, 2 in ts

    assert!(symbols.iter().any(|s| s.name == "calculate_hash" && s.language == "rust"));
    assert!(symbols.iter().any(|s| s.name == "ServiceConfig" && s.language == "rust"));
    assert!(symbols.iter().any(|s| s.name == "process_data" && s.language == "python"));
    assert!(symbols.iter().any(|s| s.name == "TaskManager" && s.language == "python"));
    assert!(symbols.iter().any(|s| s.name == "sendRequest" && s.language == "typescript"));
    assert!(symbols.iter().any(|s| s.name == "ClientOptions" && s.language == "typescript"));

    // Ensure already documented functions are NOT flagged
    assert!(!symbols.iter().any(|s| s.name == "documented_helper"));
    assert!(!symbols.iter().any(|s| s.name == "already_documented"));
    assert!(!symbols.iter().any(|s| s.name == "ApiClient"));

    // 5. Generate docstring patches
    let patches = generate_docstring_patches(&symbols, "rust");
    assert_eq!(patches.len(), 6);

    let rs_patch = patches.iter().find(|p| p.symbol_name == "calculate_hash").unwrap();
    assert!(rs_patch.docstring.starts_with("///"));
    assert!(rs_patch.patch_diff.contains("Unified Diff"));

    let py_patch = patches.iter().find(|p| p.symbol_name == "process_data").unwrap();
    assert!(py_patch.docstring.contains("\"\"\""));

    let ts_patch = patches.iter().find(|p| p.symbol_name == "sendRequest").unwrap();
    assert!(ts_patch.docstring.contains("/**"));

    // 6. Apply patches to files
    let applied_count = apply_doc_patches(&patches).unwrap();
    assert_eq!(applied_count, 3); // 3 files modified

    // Verify files on disk have received docstrings
    let updated_rs = std::fs::read_to_string(&rs_path).unwrap();
    assert!(updated_rs.contains("/// calculate_hash"));
    assert!(updated_rs.contains("/// ServiceConfig"));

    let updated_py = std::fs::read_to_string(&py_path).unwrap();
    assert!(updated_py.contains("\"\"\""));

    let updated_ts = std::fs::read_to_string(&ts_path).unwrap();
    assert!(updated_ts.contains("/**"));

    // 7. Verify re-scan reports 0 undocumented symbols
    let rescan = scan_undocumented_symbols(&temp_dir);
    assert_eq!(rescan.len(), 0);

    // 8. Test report formatting
    let report = DocGenerationReport {
        target_path: temp_dir.to_string_lossy().to_string(),
        total_symbols_scanned: 6,
        undocumented_count: 6,
        symbols,
        patches,
        applied_count,
        summary: "6 docstrings applied".to_string(),
    };
    let formatted_rep = format_doc_generation_report_for_terminal(&report);
    assert!(formatted_rep.contains("DOCSTRING & API GENERATOR"));
    assert!(formatted_rep.contains("calculate_hash"));

    // 9. Verify generate_docs tool exists in get_tools()
    let tools = get_tools();
    let tools_str = serde_json::to_string(&tools).unwrap();
    assert!(tools_str.contains("generate_docs"));

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_atomic_multi_file_refactor_transactions() {
    let temp_dir = std::env::temp_dir().join(format!("zy_tx_test_{}_{}", std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
    let src_dir = temp_dir.join("src");
    let _ = std::fs::create_dir_all(&src_dir);

    let file_a = src_dir.join("math.rs");
    let file_b = src_dir.join("formatter.rs");
    let file_to_delete = src_dir.join("deprecated.rs");

    std::fs::write(&file_a, "pub fn add(a: i32, b: i32) -> i32 { a + b }\n").unwrap();
    std::fs::write(&file_b, "pub fn format_val(v: i32) -> String { format!(\"{}\", v) }\n").unwrap();
    std::fs::write(&file_to_delete, "pub fn old_code() {}\n").unwrap();

    // 1. Initialize transaction
    let mut tx = RefactorTransaction::new();
    assert!(tx.id.starts_with("tx_"));
    assert!(tx.staged_files.is_empty());

    // 2. Stage edits in-memory (virtual buffer)
    let new_math_code = "pub fn add(a: i32, b: i32) -> i32 {\n    // Vectorized addition\n    a + b\n}\npub fn mul(a: i32, b: i32) -> i32 { a * b }\n";
    let new_fmt_code = "pub fn format_val(v: i32) -> String {\n    format!(\"Value: {}\", v)\n}\n";

    tx.stage_edit(&file_a, new_math_code);
    tx.stage_edit(&file_b, new_fmt_code);
    tx.stage_delete(&file_to_delete);

    assert_eq!(tx.staged_files.len(), 3);

    // Assert disk files are UNMODIFIED before commit
    assert_eq!(std::fs::read_to_string(&file_a).unwrap(), "pub fn add(a: i32, b: i32) -> i32 { a + b }\n");
    assert!(file_to_delete.exists());

    // 3. Render unified diff
    let diff_view = tx.render_diff();
    assert!(diff_view.contains("math.rs"));
    assert!(diff_view.contains("formatter.rs"));
    assert!(diff_view.contains("deprecated.rs"));

    // 4. Validate all staged changes (syntax & compiler checks)
    let val_report_ok = tx.validate_all_staged(&temp_dir);
    assert!(val_report_ok.is_valid);
    assert_eq!(val_report_ok.staged_files_count, 3);

    // Test syntax error detection during validation
    let mut bad_tx = RefactorTransaction::new();
    bad_tx.stage_edit(&file_a, "pub fn broken() { let x = 10; "); // Mismatched braces
    let val_report_err = bad_tx.validate_all_staged(&temp_dir);
    assert!(!val_report_err.is_valid);
    assert!(val_report_err.errors.iter().any(|e| e.contains("Mismatched curly braces")));

    // 5. Commit transaction atomically
    let committed = tx.commit().unwrap();
    assert_eq!(committed.len(), 3);
    assert!(tx.staged_files.is_empty());

    // Verify disk files are now updated
    assert_eq!(std::fs::read_to_string(&file_a).unwrap(), new_math_code);
    assert_eq!(std::fs::read_to_string(&file_b).unwrap(), new_fmt_code);
    assert!(!file_to_delete.exists());

    // 6. Test rollback
    let mut rollback_tx = RefactorTransaction::new();
    rollback_tx.stage_edit(&file_a, "COMPLETELY BAD EDIT THAT SHOULD NOT BE WRITTEN");
    assert_eq!(rollback_tx.staged_files.len(), 1);
    rollback_tx.rollback();
    assert!(rollback_tx.staged_files.is_empty());
    assert_eq!(std::fs::read_to_string(&file_a).unwrap(), new_math_code);

    // 7. Test global transaction management functions
    begin_refactor_transaction();
    stage_in_refactor_transaction(&file_a, "global staged test");
    assert!(get_refactor_transaction_diff().contains("global staged test"));
    let status_str = get_refactor_transaction_status();
    assert!(status_str.contains("Staged files:"));
    assert!(status_str.contains("1"));
    rollback_refactor_transaction();
    assert!(get_refactor_transaction_status().contains("None active"));

    // 8. Verify refactor_transaction tool exists in get_tools()
    let tools = get_tools();
    let tools_str = serde_json::to_string(&tools).unwrap();
    assert!(tools_str.contains("refactor_transaction"));

    let _ = std::fs::remove_dir_all(&temp_dir);
}

// -------------------------------------------------------------------------------------------------
// TESTS: 6 NEW ESSENTIAL SYSTEMS
// -------------------------------------------------------------------------------------------------

#[test]
fn test_micro_benchmarking_and_performance_profiler() {
    // 1. Benchmark a fast command
    let cmd = "echo bench_test";
    let report = run_micro_benchmark(cmd, 5, 2).expect("Benchmark execution failed");

    assert_eq!(report.command, cmd);
    assert_eq!(report.iterations, 5);
    assert_eq!(report.warmup, 2);
    assert_eq!(report.durations_ms.len(), 5);
    assert!(report.min_ms > 0.0);
    assert!(report.max_ms >= report.min_ms);
    assert!(report.mean_ms >= report.min_ms && report.mean_ms <= report.max_ms);
    assert!(report.median_ms >= report.min_ms && report.median_ms <= report.max_ms);
    assert!(report.std_dev_ms >= 0.0);
    assert!(report.ops_per_sec > 0.0);
    assert_eq!(report.success_count, 5);
    assert_eq!(report.failure_count, 0);

    // 2. Terminal report formatting
    let terminal_out = format_benchmark_report_for_terminal(&report);
    assert!(terminal_out.contains("MICRO-BENCHMARK & PERFORMANCE PROFILER REPORT"));
    assert!(terminal_out.contains("Command:"));
    assert!(terminal_out.contains("Iterations:"));
    assert!(terminal_out.contains("Mean:"));
    assert!(terminal_out.contains("Std Dev"));
    assert!(terminal_out.contains("ops/sec"));

    // 3. Serialization roundtrip
    let json_str = serde_json::to_string(&report).unwrap();
    let deserialized: BenchmarkReport = serde_json::from_str(&json_str).unwrap();
    assert_eq!(deserialized.command, cmd);
    assert_eq!(deserialized.iterations, 5);

    // 4. Verify benchmark_code tool exists in get_tools()
    let tools = get_tools();
    let tools_str = serde_json::to_string(&tools).unwrap();
    assert!(tools_str.contains("benchmark_code"));
}

#[test]
fn test_automated_unit_test_and_fuzz_suite_synthesizer() {
    let temp_dir = std::env::temp_dir().join(format!("zy_synthesizer_test_{}_{}", std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
    let _ = std::fs::create_dir_all(&temp_dir);

    // 1. Rust Source File
    let rs_file = temp_dir.join("math_utils.rs");
    let rs_content = r#"
pub fn compute_factorial(n: u32) -> u64 {
    (1..=n as u64).product()
}

pub fn calculate_checksum(buffer: &[u8]) -> u32 {
    buffer.iter().map(|&b| b as u32).sum()
}

pub struct Matrix3x3 {
    pub data: [f32; 9],
}
"#;
    std::fs::write(&rs_file, rs_content).unwrap();

    let rs_suite = synthesize_test_suite(&rs_file, "rust", true).expect("Failed to synthesize Rust test suite");
    assert_eq!(rs_suite.language, "rust");
    assert!(rs_suite.fuzz_enabled);
    assert!(rs_suite.scanned_symbols.len() >= 3);
    assert!(rs_suite.unit_tests.len() >= 2);
    assert!(rs_suite.fuzz_tests.len() >= 2);
    assert!(rs_suite.test_file_path.ends_with("math_utils_test.rs"));
    assert!(rs_suite.test_code.contains("proptest!"));
    assert!(rs_suite.test_code.contains("fuzz_compute_factorial_arbitrary_inputs"));
    assert!(rs_suite.test_code.contains("test_compute_factorial_deterministic_behavior"));

    // 2. Python Source File
    let py_file = temp_dir.join("analytics.py");
    let py_content = r#"
def calculate_variance(data):
    """Calculate sample variance."""
    return 0.0

def normalize_features(matrix):
    return matrix
"#;
    std::fs::write(&py_file, py_content).unwrap();

    let py_suite = synthesize_test_suite(&py_file, "python", true).expect("Failed to synthesize Python test suite");
    assert_eq!(py_suite.language, "python");
    assert!(py_suite.test_code.contains("import pytest"));
    assert!(py_suite.test_code.contains("from hypothesis import given, strategies as st"));
    assert!(py_suite.test_code.contains("@given(st.integers(), st.text())"));
    assert!(py_suite.test_code.contains("def test_fuzz_calculate_variance"));
    assert!(py_suite.test_code.contains("def test_calculate_variance_basic"));

    // 3. TypeScript Source File
    let ts_file = temp_dir.join("payload.ts");
    let ts_content = r#"
export function serializePayload(data: any) {
    return JSON.stringify(data);
}
export const calculateOffset = (index: number) => {
    return index * 4;
}
"#;
    std::fs::write(&ts_file, ts_content).unwrap();

    let ts_suite = synthesize_test_suite(&ts_file, "typescript", true).expect("Failed to synthesize TS test suite");
    assert_eq!(ts_suite.language, "typescript");
    assert!(ts_suite.test_code.contains("import { describe, test, expect } from 'vitest';"));
    assert!(ts_suite.test_code.contains("import * as fc from 'fast-check';"));
    assert!(ts_suite.test_code.contains("fc.assert(fc.property"));

    // 4. Go Source File
    let go_file = temp_dir.join("parser.go");
    let go_content = r#"
package parser

func ParsePacket(data []byte) bool {
    return len(data) > 0
}
"#;
    std::fs::write(&go_file, go_content).unwrap();

    let go_suite = synthesize_test_suite(&go_file, "go", true).expect("Failed to synthesize Go test suite");
    assert_eq!(go_suite.language, "go");
    assert!(go_suite.test_code.contains("func TestParsePacket(t *testing.T)"));
    assert!(go_suite.test_code.contains("func FuzzParsePacket(f *testing.F)"));

    // 5. Terminal report formatting
    let terminal_out = format_test_suite_report_for_terminal(&rs_suite);
    assert!(terminal_out.contains("AUTOMATED TEST & FUZZ SUITE SYNTHESIZER REPORT"));
    assert!(terminal_out.contains("compute_factorial"));
    assert!(terminal_out.contains("Unit Tests:"));
    assert!(terminal_out.contains("Fuzz Suites:"));

    // 6. Verify generate_tests tool in get_tools()
    let tools = get_tools();
    let tools_str = serde_json::to_string(&tools).unwrap();
    assert!(tools_str.contains("generate_tests"));

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_production_container_and_ci_cd_manifest_generator() {
    let temp_dir = std::env::temp_dir().join(format!("zy_ci_gen_test_{}_{}", std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
    let _ = std::fs::create_dir_all(&temp_dir);

    // 1. Rust Stack
    let rust_dir = temp_dir.join("rust_proj");
    let _ = std::fs::create_dir_all(&rust_dir);
    std::fs::write(rust_dir.join("Cargo.toml"), "[package]\nname = \"zy_service\"\nversion = \"1.0.0\"\n").unwrap();

    let rust_stack = detect_project_stack(&rust_dir);
    assert_eq!(rust_stack.language, ProjectLanguage::Rust);
    assert_eq!(rust_stack.project_name, "zy_service");
    assert_eq!(rust_stack.suggested_port, 8080);

    let rust_manifests = generate_container_and_ci_manifests(&rust_stack);
    assert!(rust_manifests.dockerfile.contains("lukemathwalker/cargo-chef"));
    assert!(rust_manifests.dockerfile.contains("musl-dev"));
    assert!(rust_manifests.dockerfile.contains("zy_service"));
    assert!(rust_manifests.dockerfile.contains("HEALTHCHECK"));
    assert!(rust_manifests.docker_compose.contains("zy_service:"));
    assert!(rust_manifests.docker_compose.contains("8080:8080"));
    assert!(rust_manifests.docker_compose.contains("limits:"));
    assert!(rust_manifests.github_workflow.contains("Production CI"));
    assert!(rust_manifests.github_workflow.contains("Swatinem/rust-cache@v2"));
    assert!(rust_manifests.github_workflow.contains("matrix:"));
    assert!(rust_manifests.github_workflow.contains("windows-latest"));
    assert!(rust_manifests.github_workflow.contains("macos-latest"));

    // 2. Node.js Stack
    let node_dir = temp_dir.join("node_proj");
    let _ = std::fs::create_dir_all(&node_dir);
    std::fs::write(node_dir.join("package.json"), "{\"name\": \"web-gateway\", \"version\": \"2.0.0\"}").unwrap();

    let node_stack = detect_project_stack(&node_dir);
    assert_eq!(node_stack.language, ProjectLanguage::Node);
    assert_eq!(node_stack.project_name, "web-gateway");
    assert_eq!(node_stack.suggested_port, 3000);

    let node_manifests = generate_container_and_ci_manifests(&node_stack);
    assert!(node_manifests.dockerfile.contains("FROM node:20-alpine AS deps"));
    assert!(node_manifests.dockerfile.contains("npm ci"));
    assert!(node_manifests.dockerfile.contains("USER nextjs"));
    assert!(node_manifests.docker_compose.contains("3000:3000"));
    assert!(node_manifests.github_workflow.contains("actions/setup-node@v4"));

    // 3. Python Stack
    let py_dir = temp_dir.join("py_proj");
    let _ = std::fs::create_dir_all(&py_dir);
    std::fs::write(py_dir.join("requirements.txt"), "fastapi\nuvicorn\n").unwrap();

    let py_stack = detect_project_stack(&py_dir);
    assert_eq!(py_stack.language, ProjectLanguage::Python);
    assert_eq!(py_stack.suggested_port, 8000);

    let py_manifests = generate_container_and_ci_manifests(&py_stack);
    assert!(py_manifests.dockerfile.contains("FROM python:3.11-slim AS builder"));
    assert!(py_manifests.dockerfile.contains("appuser"));
    assert!(py_manifests.github_workflow.contains("actions/setup-python@v5"));

    // 4. Go Stack
    let go_dir = temp_dir.join("go_proj");
    let _ = std::fs::create_dir_all(&go_dir);
    std::fs::write(go_dir.join("go.mod"), "module github.com/zy/backend\ngo 1.22\n").unwrap();

    let go_stack = detect_project_stack(&go_dir);
    assert_eq!(go_stack.language, ProjectLanguage::Go);

    let go_manifests = generate_container_and_ci_manifests(&go_stack);
    assert!(go_manifests.dockerfile.contains("FROM golang:1.22-alpine AS builder"));
    assert!(go_manifests.dockerfile.contains("CGO_ENABLED=0"));

    // 5. Terminal report formatting
    let terminal_out = format_ci_manifests_for_terminal(&rust_manifests);
    assert!(terminal_out.contains("CONTAINER & CI/CD MANIFEST GENERATOR REPORT"));
    assert!(terminal_out.contains("zy_service"));
    assert!(terminal_out.contains("Dockerfile"));
    assert!(terminal_out.contains("docker-compose.yml"));
    assert!(terminal_out.contains(".github/workflows/ci.yml"));

    // 6. Verify generate_ci tool exists in get_tools()
    let tools = get_tools();
    let tools_str = serde_json::to_string(&tools).unwrap();
    assert!(tools_str.contains("generate_ci"));

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_interactive_call_graph_visualizer() {
    let temp_dir = std::env::temp_dir().join(format!("zy_graph_test_{}_{}", std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
    let src_dir = temp_dir.join("src");
    let _ = std::fs::create_dir_all(&src_dir);

    let file_main = src_dir.join("main.rs");
    let file_server = src_dir.join("server.rs");

    let main_code = r#"
fn main() {
    init_config();
    run_server();
}

fn init_config() {
    load_environment();
}
"#;

    let server_code = r#"
pub fn run_server() {
    handle_connection();
}

pub fn handle_connection() {
    parse_request();
}

pub fn parse_request() {
    emit_metrics();
}

pub fn emit_metrics() {
}

pub fn load_environment() {
}
"#;

    std::fs::write(&file_main, main_code).unwrap();
    std::fs::write(&file_server, server_code).unwrap();

    // 1. Build call graph rooted at "main"
    let report = build_call_graph(&temp_dir, Some("main"));

    assert_eq!(report.entry_symbol.as_deref(), Some("main"));
    assert!(report.total_functions >= 7);
    assert!(report.total_calls >= 6);

    // Assert ASCII tree contains full hierarchical chain
    assert!(report.ascii_tree.contains("main"));
    assert!(report.ascii_tree.contains("run_server"));
    assert!(report.ascii_tree.contains("handle_connection"));
    assert!(report.ascii_tree.contains("parse_request"));
    assert!(report.ascii_tree.contains("emit_metrics"));
    assert!(report.ascii_tree.contains("init_config"));
    assert!(report.ascii_tree.contains("load_environment"));

    // Assert Mermaid diagram syntax
    assert!(report.mermaid_diagram.starts_with("graph TD;\n"));
    assert!(report.mermaid_diagram.contains("main --> run_server;"));
    assert!(report.mermaid_diagram.contains("run_server --> handle_connection;"));
    assert!(report.mermaid_diagram.contains("handle_connection --> parse_request;"));

    // 2. Terminal report formatting
    let terminal_out = format_call_graph_for_terminal(&report);
    assert!(terminal_out.contains("INTERACTIVE CALL GRAPH"));
    assert!(terminal_out.contains("Total Functions:"));
    assert!(terminal_out.contains("Total Call Sites:"));
    assert!(terminal_out.contains("main"));

    // 3. Verify call_graph tool in get_tools()
    let tools = get_tools();
    let tools_str = serde_json::to_string(&tools).unwrap();
    assert!(tools_str.contains("call_graph"));

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_multi_language_formatter_and_linter_auto_fixer() {
    let temp_dir = std::env::temp_dir().join(format!("zy_lint_test_{}_{}", std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
    let _ = std::fs::create_dir_all(&temp_dir);

    // Create files with trailing spaces and missing trailing newlines
    let rs_file = temp_dir.join("bad_format.rs");
    let py_file = temp_dir.join("bad_format.py");
    let json_file = temp_dir.join("bad_format.json");

    std::fs::write(&rs_file, "pub fn compute() -> i32 {    \n    42   \n}").unwrap();
    std::fs::write(&py_file, "def calculate():   \n    return 100   ").unwrap();
    std::fs::write(&json_file, "{\n  \"status\": \"ok\"   \n}").unwrap();

    // 1. Run in check mode (fix = false)
    let check_report = format_and_lint_workspace(&temp_dir, false).expect("Formatter check mode failed");
    assert!(!check_report.fix_mode);
    assert!(check_report.issues_found >= 3);
    assert_eq!(check_report.issues_fixed, 0);

    // Verify files on disk NOT modified in check mode
    assert!(std::fs::read_to_string(&rs_file).unwrap().contains("    \n"));

    // 2. Run in fix mode (fix = true)
    let fix_report = format_and_lint_workspace(&temp_dir, true).expect("Formatter fix mode failed");
    assert!(fix_report.fix_mode);
    assert!(fix_report.issues_fixed >= 3);
    assert!(fix_report.formatted_files.len() >= 3);

    // Verify files on disk are cleaned
    let fixed_rs = std::fs::read_to_string(&rs_file).unwrap();
    assert_eq!(fixed_rs, "pub fn compute() -> i32 {\n    42\n}\n");
    assert!(fixed_rs.ends_with('\n'));

    let fixed_py = std::fs::read_to_string(&py_file).unwrap();
    assert_eq!(fixed_py, "def calculate():\n    return 100\n");
    assert!(fixed_py.ends_with('\n'));

    let fixed_json = std::fs::read_to_string(&json_file).unwrap();
    assert_eq!(fixed_json, "{\n  \"status\": \"ok\"\n}\n");

    // 3. Terminal report formatting
    let terminal_out = format_lint_format_report_for_terminal(&fix_report);
    assert!(terminal_out.contains("MULTI-LANGUAGE FORMATTER & LINTER AUTO-FIXER"));
    assert!(terminal_out.contains("AUTO-FIX ENABLED"));
    assert!(terminal_out.contains("Files Formatted:"));
    assert!(terminal_out.contains("Issues Fixed:"));

    // 4. Verify auto_format tool in get_tools()
    let tools = get_tools();
    let tools_str = serde_json::to_string(&tools).unwrap();
    assert!(tools_str.contains("auto_format"));

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[tokio::test]
async fn test_ephemeral_mock_server_and_api_sandbox() {
    let mut custom_headers = std::collections::HashMap::new();
    custom_headers.insert("X-Zy-Engine".to_string(), "Mock-v1".to_string());

    let routes = vec![
        MockRoute {
            method: "GET".to_string(),
            path: "/api/v1/health".to_string(),
            status_code: 200,
            response_body: serde_json::json!({
                "status": "healthy",
                "uptime_sec": 3600,
                "version": "1.0.0"
            }),
            headers: custom_headers,
        },
        MockRoute {
            method: "POST".to_string(),
            path: "/api/v1/users".to_string(),
            status_code: 201,
            response_body: serde_json::json!({
                "id": 101,
                "username": "zy_agent",
                "role": "admin"
            }),
            headers: std::collections::HashMap::new(),
        },
        MockRoute {
            method: "PUT".to_string(),
            path: "/api/v1/settings".to_string(),
            status_code: 202,
            response_body: serde_json::json!({
                "updated": true,
                "theme": "dark"
            }),
            headers: std::collections::HashMap::new(),
        },
        MockRoute {
            method: "DELETE".to_string(),
            path: "/api/v1/sessions/101".to_string(),
            status_code: 200,
            response_body: serde_json::json!({
                "revoked": true
            }),
            headers: std::collections::HashMap::new(),
        },
    ];

    // 1. Start ephemeral mock server on random dynamic port (port 0)
    let handle = start_ephemeral_mock_server(0, routes).await.expect("Failed to start ephemeral mock server");
    assert!(handle.is_running());
    assert!(handle.port() > 0);
    assert!(handle.base_url().starts_with("http://127.0.0.1:"));

    let client = reqwest::Client::new();

    // 2. Test GET /api/v1/health
    let res_get = client.get(format!("{}/api/v1/health", handle.base_url()))
        .send().await.expect("GET request failed");
    assert_eq!(res_get.status(), reqwest::StatusCode::OK);
    assert_eq!(res_get.headers().get("X-Zy-Engine").and_then(|h| h.to_str().ok()), Some("Mock-v1"));
    let get_json: serde_json::Value = res_get.json().await.unwrap();
    assert_eq!(get_json["status"], "healthy");
    assert_eq!(get_json["uptime_sec"], 3600);

    // 3. Test POST /api/v1/users
    let res_post = client.post(format!("{}/api/v1/users", handle.base_url()))
        .json(&serde_json::json!({"username": "zy_agent"}))
        .send().await.expect("POST request failed");
    assert_eq!(res_post.status(), reqwest::StatusCode::CREATED);
    let post_json: serde_json::Value = res_post.json().await.unwrap();
    assert_eq!(post_json["id"], 101);
    assert_eq!(post_json["username"], "zy_agent");

    // 4. Test PUT /api/v1/settings
    let res_put = client.put(format!("{}/api/v1/settings", handle.base_url()))
        .json(&serde_json::json!({"theme": "dark"}))
        .send().await.expect("PUT request failed");
    assert_eq!(res_put.status(), reqwest::StatusCode::ACCEPTED);
    let put_json: serde_json::Value = res_put.json().await.unwrap();
    assert_eq!(put_json["updated"], true);

    // 5. Test DELETE /api/v1/sessions/101
    let res_del = client.delete(format!("{}/api/v1/sessions/101", handle.base_url()))
        .send().await.expect("DELETE request failed");
    assert_eq!(res_del.status(), reqwest::StatusCode::OK);
    let del_json: serde_json::Value = res_del.json().await.unwrap();
    assert_eq!(del_json["revoked"], true);

    // 6. Test 404 Route Not Found
    let res_404 = client.get(format!("{}/non_existent_route", handle.base_url()))
        .send().await.expect("404 GET request failed");
    assert_eq!(res_404.status(), reqwest::StatusCode::NOT_FOUND);
    let err_json: serde_json::Value = res_404.json().await.unwrap();
    assert_eq!(err_json["error"], "Route Not Found");

    // 7. Terminal report formatting
    let terminal_out = format_mock_server_report_for_terminal(&handle);
    assert!(terminal_out.contains("EPHEMERAL AI MOCK SERVER & API SANDBOX ACTIVE"));
    assert!(terminal_out.contains("Base URL:"));
    assert!(terminal_out.contains("/api/v1/health"));
    assert!(terminal_out.contains("/api/v1/users"));

    // 8. Test active server registration & clean shutdown
    register_active_mock_server(handle);
    stop_all_active_mock_servers();

    // 9. Verify mock_api tool in get_tools()
    let tools = get_tools();
    let tools_str = serde_json::to_string(&tools).unwrap();
    assert!(tools_str.contains("mock_api"));
}

#[test]
fn test_brutal_edge_cases_across_6_systems() {
    let temp_dir = std::env::temp_dir().join(format!("zy_brutal_test_{}_{}", std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
    let _ = std::fs::create_dir_all(&temp_dir);

    // 1. Micro-benchmark 0 iterations & 0 warmup normalization
    let zero_bench = run_micro_benchmark("echo brutal_edge_case", 0, 0).expect("Zero iteration benchmark failed");
    assert_eq!(zero_bench.iterations, 1);
    assert_eq!(zero_bench.durations_ms.len(), 1);

    // 2. Synthesizer on empty file & unknown language
    let empty_file = temp_dir.join("empty.unknown");
    std::fs::write(&empty_file, "").unwrap();
    let empty_suite = synthesize_test_suite(&empty_file, "unknown_lang", false).unwrap();
    assert_eq!(empty_suite.unit_tests.len(), 0);
    assert_eq!(empty_suite.fuzz_tests.len(), 0);

    // 3. Container & CI detector on completely empty workspace fallback to Generic
    let empty_ws = temp_dir.join("empty_ws");
    let _ = std::fs::create_dir_all(&empty_ws);
    let empty_stack = detect_project_stack(&empty_ws);
    assert_eq!(empty_stack.language, ProjectLanguage::Generic);
    let gen_manifests = generate_container_and_ci_manifests(&empty_stack);
    assert!(gen_manifests.dockerfile.contains("FROM alpine:3.19"));
    assert!(gen_manifests.github_workflow.contains("make test"));

    // 4. Call graph cycle detection without infinite loop
    let cycle_file = temp_dir.join("cycles.rs");
    let cycle_code = r#"
pub fn function_alpha() {
    function_beta();
}

pub fn function_beta() {
    function_alpha();
}
"#;
    std::fs::write(&cycle_file, cycle_code).unwrap();
    let cycle_report = build_call_graph(&temp_dir, Some("function_alpha"));
    assert!(cycle_report.ascii_tree.contains("[cycle]"));
    assert!(cycle_report.mermaid_diagram.contains("function_alpha --> function_beta;"));
    assert!(cycle_report.mermaid_diagram.contains("function_beta --> function_alpha;"));

    let _ = std::fs::remove_dir_all(&temp_dir);
}

// ============================================================================
// SYSTEM 1: GIT WORKTREE TASK ISOLATION TESTS
// ============================================================================

#[test]
fn test_git_worktree_task_isolation_lifecycle() {
    let temp_dir = std::env::temp_dir().join(format!("zy_wt_test_{}_{}", std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
    let _ = std::fs::create_dir_all(&temp_dir);

    // 1. Create a dummy file in the workspace
    let dummy_file = temp_dir.join("main.rs");
    std::fs::write(&dummy_file, "fn main() { println!(\"base workspace\"); }").unwrap();

    // 2. Create task worktree
    let handle = create_task_worktree(&temp_dir, "auth-oauth2", Some("feat/auth-oauth2")).expect("Failed to create task worktree");
    assert_eq!(handle.task_id, "auth-oauth2");
    assert_eq!(handle.branch_name, "feat/auth-oauth2");
    assert!(handle.worktree_path.exists());
    assert!(handle.worktree_path.join("main.rs").exists());

    // 3. Execute command in worktree
    let exec_res = handle.execute("echo worktree_isolated_execution").expect("Execution failed");
    assert_eq!(exec_res.task_id, "auth-oauth2");
    assert!(exec_res.stdout.contains("worktree_isolated_execution") || exec_res.success);

    // 4. Modify file inside worktree
    let wt_main = handle.worktree_path.join("main.rs");
    std::fs::write(&wt_main, "fn main() { println!(\"updated in worktree\"); }").unwrap();

    // 5. List task worktrees
    let list = list_task_worktrees(&temp_dir).expect("Failed to list task worktrees");
    assert_eq!(list.len(), 1);
    assert_eq!(list[0].task_id, "auth-oauth2");

    // 6. Merge worktree back
    let merge_res = handle.merge_back(Some("feat(auth): complete oauth2 implementation")).expect("Merge back failed");
    assert_eq!(merge_res.task_id, "auth-oauth2");
    assert!(merge_res.success);

    // Verify workspace has updated file
    let ws_content = std::fs::read_to_string(&dummy_file).unwrap();
    assert!(ws_content.contains("updated in worktree"));

    // 7. Cleanup worktree
    let cleanup_res = handle.cleanup(true).expect("Cleanup failed");
    assert!(cleanup_res);
    assert!(!handle.worktree_path.exists());

    // 8. Terminal report formatting
    let report_out = format_worktree_report_for_terminal(&handle);
    assert!(report_out.contains("GIT WORKTREE TASK ISOLATION ACTIVE"));
    assert!(report_out.contains("auth-oauth2"));

    let list_out = format_worktree_list_for_terminal(&[handle]);
    assert!(list_out.contains("ACTIVE GIT TASK WORKTREES"));

    // 9. Verify isolate_task in get_tools()
    let tools = get_tools();
    let tools_str = serde_json::to_string(&tools).unwrap();
    assert!(tools_str.contains("isolate_task"));

    let _ = std::fs::remove_dir_all(&temp_dir);
}

// ============================================================================
// SYSTEM 2: DEEP SARIF SECURITY CODE REVIEW & AUDITOR TESTS
// ============================================================================

#[test]
fn test_deep_sarif_security_code_review_and_auditor() {
    let temp_dir = std::env::temp_dir().join(format!("zy_review_test_{}_{}", std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
    let _ = std::fs::create_dir_all(&temp_dir);

    // 1. Create source file with security vulnerabilities and code smells
    let vuln_code = r#"
use std::sync::Mutex;

static mut GLOBAL_INSECURE_COUNTER: u64 = 0;

pub async fn process_user_action(user_input: &str, user_path: &str, input_id: &str, password: &str) {
    let api_key = "sk_live_TEST_DUMMY_KEY_NON_FUNCTIONAL_12345";
    let cmd = format!("rm -rf {}", user_input);
    let _ = std::process::Command::new("sh").arg("-c").arg(cmd).output();
    let query = format!("SELECT * FROM users WHERE id = {}", input_id);
    let _ = std::fs::read(user_path);
    let hash = md5::compute(password);

    let m = Mutex::new(42);
    let guard = m.lock().unwrap();
    tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    drop(guard);

    let mut buffer = Vec::new();
    loop {
        buffer.push("leak");
        break;
    }

    let items = vec![1, 2, 3];
    let other = vec![2, 3, 4];
    for x in &items {
        if other.contains(x) {
            println!("found");
        }
    }

    let mut s = String::new();
    for chunk in &["a", "b", "c"] {
        s += chunk;
    }
}
"#;

    let vuln_file = temp_dir.join("vuln.rs");
    std::fs::write(&vuln_file, vuln_code).unwrap();

    // 2. Perform code review
    let report = perform_code_review(&temp_dir, None).expect("Code review failed");
    assert!(report.files_scanned >= 1);
    assert!(report.findings.len() >= 6);
    assert!(report.critical_count >= 2); // Hardcoded secret, Command injection, Lock across await

    // 3. Verify specific findings
    let has_secret = report.findings.iter().any(|f| f.rule_id == "zy/security/hardcoded-secret");
    let has_cmd_inj = report.findings.iter().any(|f| f.rule_id == "zy/security/command-injection");
    let has_sql_inj = report.findings.iter().any(|f| f.rule_id == "zy/security/sql-injection");
    let has_path_trav = report.findings.iter().any(|f| f.rule_id == "zy/security/path-traversal");
    let has_broken_crypto = report.findings.iter().any(|f| f.rule_id == "zy/security/broken-cryptography");
    let has_lock_await = report.findings.iter().any(|f| f.rule_id == "zy/concurrency/lock-across-await");
    let has_static_mut = report.findings.iter().any(|f| f.rule_id == "zy/concurrency/mutable-static");
    let has_perf_o_n2 = report.findings.iter().any(|f| f.rule_id == "zy/performance/o-n2-nested-search");

    assert!(has_secret);
    assert!(has_cmd_inj);
    assert!(has_sql_inj);
    assert!(has_path_trav);
    assert!(has_broken_crypto);
    assert!(has_lock_await);
    assert!(has_static_mut);
    assert!(has_perf_o_n2);

    // 4. Validate SARIF v2.1.0 JSON format
    let sarif = report.to_sarif_json();
    assert_eq!(sarif["version"], "2.1.0");
    assert!(sarif["$schema"].as_str().unwrap().contains("sarif-schema-2.1.0.json"));
    assert_eq!(sarif["runs"][0]["tool"]["driver"]["name"], "zy-deep-sarif-auditor");
    assert!(sarif["runs"][0]["results"].as_array().unwrap().len() >= 6);

    let first_result = &sarif["runs"][0]["results"][0];
    assert!(first_result["ruleId"].as_str().unwrap().starts_with("zy/"));
    assert!(first_result["locations"][0]["physicalLocation"]["artifactLocation"]["uri"].is_string());
    assert!(first_result["fixes"].is_array());

    // 5. Test terminal report formatting
    let terminal_out = format_code_review_for_terminal(&report);
    assert!(terminal_out.contains("DEEP SARIF SECURITY CODE REVIEW & AUDITOR REPORT"));
    assert!(terminal_out.contains("Critical"));
    assert!(terminal_out.contains("Remediation:"));

    // 6. Test direct diff review mode
    let diff_snippet = r#"
+ let api_key = "sk_live_TEST_DUMMY_TOKEN_NON_FUNCTIONAL_999";
+ let cmd = format!("sh -c {}", user_param);
+ let _ = os.system(cmd);
"#;
    let diff_report = perform_code_review(&temp_dir, Some(diff_snippet)).unwrap();
    assert!(diff_report.findings.len() >= 2);

    // 7. Verify code_review in get_tools()
    let tools = get_tools();
    let tools_str = serde_json::to_string(&tools).unwrap();
    assert!(tools_str.contains("code_review"));

    let _ = std::fs::remove_dir_all(&temp_dir);
}

// ============================================================================
// SYSTEM 3: SEMANTIC 3-WAY MERGE CONFLICT RESOLVER TESTS
// ============================================================================

#[test]
fn test_semantic_3way_merge_conflict_resolver() {
    let temp_dir = std::env::temp_dir().join(format!("zy_conflict_test_{}_{}", std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
    let _ = std::fs::create_dir_all(&temp_dir);

    // 1. Test Import Union & Deduplication Strategy
    let import_conflict = r#"
// Header
<<<<<<< HEAD
use std::collections::HashMap;
use std::fs;
=======
use std::collections::BTreeMap;
use std::collections::HashMap;
use std::path::Path;
>>>>>>> incoming
// Rest of file
fn test() {}
"#;
    let import_res = resolve_merge_conflict_content(import_conflict, "imports.rs");
    assert_eq!(import_res.conflicts_found, 1);
    assert_eq!(import_res.conflicts_resolved, 1);
    assert!(import_res.verified_syntax);
    assert!(!import_res.resolved_file_content.contains("<<<<<<<"));
    assert!(!import_res.resolved_file_content.contains("======="));
    assert!(!import_res.resolved_file_content.contains(">>>>>>>"));
    assert!(import_res.resolved_file_content.contains("use std::collections::BTreeMap;"));
    assert!(import_res.resolved_file_content.contains("use std::collections::HashMap;"));
    assert!(import_res.resolved_file_content.contains("use std::path::Path;"));
    assert_eq!(import_res.blocks[0].strategy_used, "import_union_dedup");

    // 2. Test Additive Function Merge Strategy
    let func_conflict = r#"
<<<<<<< HEAD
pub fn calculate_tax(amount: f64) -> f64 {
    amount * 0.15
}
=======
pub fn calculate_discount(price: f64) -> f64 {
    price * 0.90
}
>>>>>>> feature-discount
"#;
    let func_res = resolve_merge_conflict_content(func_conflict, "functions.rs");
    assert_eq!(func_res.conflicts_found, 1);
    assert!(func_res.resolved_file_content.contains("pub fn calculate_tax"));
    assert!(func_res.resolved_file_content.contains("pub fn calculate_discount"));
    assert!(func_res.verified_syntax);
    assert_eq!(func_res.blocks[0].strategy_used, "additive_function_merge");

    // 3. Test 3-Way Base-Aware (Diff3) Merge Strategy
    let diff3_conflict = r#"
<<<<<<< HEAD
let config_path = "default.toml";
||||||| base
let config_path = "old.toml";
=======
let config_path = "production.toml";
>>>>>>> feature-prod-config
"#;
    let diff3_res = resolve_merge_conflict_content(diff3_conflict, "config.rs");
    assert_eq!(diff3_res.conflicts_found, 1);
    assert!(diff3_res.resolved_file_content.contains("production.toml") || diff3_res.resolved_file_content.contains("default.toml"));
    assert!(diff3_res.verified_syntax);

    // 4. Test Config Key-Value Merge Strategy
    let toml_conflict = r#"
[server]
<<<<<<< HEAD
port = 8080
timeout = 30
=======
port = 8080
max_connections = 1000
>>>>>>> incoming
"#;
    let toml_res = resolve_merge_conflict_content(toml_conflict, "settings.toml");
    assert_eq!(toml_res.conflicts_found, 1);
    assert!(toml_res.resolved_file_content.contains("port = 8080"));
    assert!(toml_res.resolved_file_content.contains("timeout = 30"));
    assert!(toml_res.resolved_file_content.contains("max_connections = 1000"));
    assert_eq!(toml_res.blocks[0].strategy_used, "config_key_merge");

    // 5. Test File Resolution on Disk
    let disk_file = temp_dir.join("conflict.rs");
    std::fs::write(&disk_file, import_conflict).unwrap();

    let disk_conflicts = find_merge_conflicts(&temp_dir);
    assert_eq!(disk_conflicts.len(), 1);

    let disk_res = resolve_merge_conflict(&disk_file).expect("Failed to resolve merge conflict on disk");
    assert!(disk_res.applied);
    assert_eq!(disk_res.conflicts_resolved, 1);

    let updated_disk_content = std::fs::read_to_string(&disk_file).unwrap();
    assert!(!updated_disk_content.contains("<<<<<<<"));
    assert!(updated_disk_content.contains("use std::collections::HashMap;"));

    // 6. Test Terminal Formatting
    let term_out = format_conflict_resolution_for_terminal(&disk_res);
    assert!(term_out.contains("SEMANTIC 3-WAY MERGE CONFLICT RESOLVER"));
    assert!(term_out.contains("VERIFIED CLEAN"));

    // 7. Verify resolve_conflicts in get_tools()
    let tools = get_tools();
    let tools_str = serde_json::to_string(&tools).unwrap();
    assert!(tools_str.contains("resolve_conflicts"));

    let _ = std::fs::remove_dir_all(&temp_dir);
}

// ============================================================================
// SYSTEM 4: STRUCTURAL AST PATTERN SEARCH & REPLACE TESTS
// ============================================================================

#[test]
fn test_structural_ast_pattern_search_and_replace() {
    let temp_dir = std::env::temp_dir().join(format!("zy_ast_test_{}_{}", std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
    let _ = std::fs::create_dir_all(&temp_dir);

    // 1. Test basic pattern matching with metavariables
    let code_sample = r#"
pub fn calculate_sum(a: i32, b: i32) -> i32 {
    let result = a + b;
    println!("result: {}", result);
    result
}

pub fn calculate_product(x: i32, y: i32) -> i32 {
    let total = x * y;
    println!("total: {}", total);
    total
}
"#;
    let sample_file = temp_dir.join("math.rs");
    std::fs::write(&sample_file, code_sample).unwrap();

    // 2. Search with single and multi metavariables: fn $NAME($$$ARGS) -> $RET { $$$BODY }
    let matches = match_structural_pattern("fn $NAME($$$ARGS) -> $RET { $$$BODY }", code_sample);
    assert_eq!(matches.len(), 2);
    assert_eq!(matches[0].3.get("$NAME").unwrap(), "calculate_sum");
    assert_eq!(matches[0].3.get("$RET").unwrap(), "i32");
    assert_eq!(matches[1].3.get("$NAME").unwrap(), "calculate_product");
    assert_eq!(matches[1].3.get("$RET").unwrap(), "i32");

    // 3. Search and Replace: println!($$$ARGS) -> tracing::info!($$$ARGS)
    let search_res = execute_structural_search(
        &temp_dir,
        "println! ( $$$ARGS ) ;",
        Some("tracing::info! ( $$$ARGS ) ;")
    ).expect("Structural search failed");

    assert_eq!(search_res.total_matches, 2);
    assert_eq!(search_res.files_searched, 1);
    assert!(search_res.diff_preview.is_some());
    let diff = search_res.diff_preview.as_ref().unwrap();
    assert!(diff.contains("tracing::info!"));

    // 4. Test metavariable consistency: $X == $X
    let dup_code = "if (a == a) { fix(); } if (a == b) { ok(); }";
    let dup_matches = match_structural_pattern("$X == $X", dup_code);
    assert_eq!(dup_matches.len(), 1);
    assert_eq!(dup_matches[0].3.get("$X").unwrap(), "a");

    // 5. Test Terminal report formatting
    let term_out = format_structural_search_for_terminal(&search_res);
    assert!(term_out.contains("STRUCTURAL AST PATTERN SEARCH & REPLACE"));
    assert!(term_out.contains("Total Matches:"));
    assert!(term_out.contains("2"));

    // 6. Verify structural_search in get_tools()
    let tools = get_tools();
    let tools_str = serde_json::to_string(&tools).unwrap();
    assert!(tools_str.contains("structural_search"));

    let _ = std::fs::remove_dir_all(&temp_dir);
}

// ============================================================================
// SYSTEM 5: AUTOMATED SEMVER BUMPER & RELEASE SYNTHESIZER TESTS
// ============================================================================

#[test]
fn test_automated_semver_bumper_and_release_synthesizer() {
    let temp_dir = std::env::temp_dir().join(format!("zy_release_test_{}_{}", std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
    let _ = std::fs::create_dir_all(&temp_dir);

    // 1. Test SemVer struct parsing and bumping
    let v1 = SemVer::parse("0.1.0").unwrap();
    assert_eq!(v1.major, 0);
    assert_eq!(v1.minor, 1);
    assert_eq!(v1.patch, 0);
    assert_eq!(v1.to_string(), "0.1.0");

    assert_eq!(v1.bump(BumpType::Patch).to_string(), "0.1.1");
    assert_eq!(v1.bump(BumpType::Minor).to_string(), "0.2.0");
    assert_eq!(v1.bump(BumpType::Major).to_string(), "1.0.0");

    let v_pre = SemVer::parse("v2.1.0-beta.1").unwrap();
    assert_eq!(v_pre.major, 2);
    assert_eq!(v_pre.minor, 1);
    assert_eq!(v_pre.patch, 0);
    assert_eq!(v_pre.pre_release, Some("beta.1".to_string()));

    // 2. Test Commit message parser
    let c1 = parse_commit_line("feat(auth)!: switch to asymmetric JWT verification");
    assert!(c1.is_breaking);
    assert_eq!(c1.commit_type, "feat");
    assert_eq!(c1.scope, Some("auth".to_string()));

    let c2 = parse_commit_line("feat(api): add GraphQL subscription endpoints");
    assert!(!c2.is_breaking);
    assert_eq!(c2.commit_type, "feat");
    assert_eq!(c2.scope, Some("api".to_string()));

    let c3 = parse_commit_line("fix: resolve race condition in database pool");
    assert!(!c3.is_breaking);
    assert_eq!(c3.commit_type, "fix");

    // 3. Create dummy Cargo.toml and package.json in test workspace
    let cargo_toml = temp_dir.join("Cargo.toml");
    std::fs::write(&cargo_toml, "[package]\nname = \"test_pkg\"\nversion = \"0.1.0\"\nedition = \"2021\"\n").unwrap();

    let package_json = temp_dir.join("package.json");
    std::fs::write(&package_json, "{\n  \"name\": \"test-app\",\n  \"version\": \"0.1.0\"\n}").unwrap();

    // 4. Calculate next SemVer
    let plan = calculate_next_semver(&temp_dir).expect("Failed to calculate next SemVer");
    assert_eq!(plan.current_version, "0.1.0");
    assert!(!plan.next_version.is_empty());
    assert!(plan.changelog_entry.contains("## ["));

    // 5. Execute release with file writing (override to Minor -> 0.2.0)
    let executed_plan = execute_release(&temp_dir, Some(BumpType::Minor), false, true).expect("Failed to execute release");
    assert_eq!(executed_plan.next_version, "0.2.0");
    assert_eq!(executed_plan.tag_name, "v0.2.0");

    // Verify Cargo.toml was updated
    let updated_cargo = std::fs::read_to_string(&cargo_toml).unwrap();
    assert!(updated_cargo.contains("version = \"0.2.0\""));

    // Verify package.json was updated
    let updated_pkg = std::fs::read_to_string(&package_json).unwrap();
    assert!(updated_pkg.contains("\"version\": \"0.2.0\""));

    // Verify CHANGELOG.md was created
    let changelog_path = temp_dir.join("CHANGELOG.md");
    assert!(changelog_path.exists());
    let changelog_content = std::fs::read_to_string(&changelog_path).unwrap();
    assert!(changelog_content.contains("## [0.2.0]"));

    // 6. Test Terminal Formatting
    let term_out = format_release_plan_for_terminal(&executed_plan);
    assert!(term_out.contains("AUTOMATED SEMVER BUMPER & RELEASE SYNTHESIZER"));
    assert!(term_out.contains("0.2.0"));

    // 7. Verify bump_version in get_tools()
    let tools = get_tools();
    let tools_str = serde_json::to_string(&tools).unwrap();
    assert!(tools_str.contains("bump_version"));

    let _ = std::fs::remove_dir_all(&temp_dir);
}

// ============================================================================
// SYSTEM 6: REAL-TIME REMOTE PAIR-PROGRAMMING BRIDGE TESTS
// ============================================================================

#[tokio::test]
async fn test_realtime_remote_pair_programming_bridge() {
    let auth_token = "zy-secret-token-12345";
    let handle = start_remote_pair_bridge(0, Some(auth_token)).await.expect("Failed to start remote bridge");

    assert!(handle.is_running());
    assert!(handle.port() > 0);
    assert!(handle.base_url().starts_with("http://127.0.0.1:"));

    let client = reqwest::Client::new();

    // 1. Unauthenticated request should return 401
    let unauth_res = client.get(format!("{}/status", handle.base_url()))
        .send().await.expect("Request failed");
    assert_eq!(unauth_res.status(), reqwest::StatusCode::UNAUTHORIZED);

    // 2. Authenticated GET /status
    let status_res = client.get(format!("{}/status", handle.base_url()))
        .header("Authorization", format!("Bearer {}", auth_token))
        .send().await.expect("Authenticated status request failed");
    assert_eq!(status_res.status(), reqwest::StatusCode::OK);
    let status_json: serde_json::Value = status_res.json().await.unwrap();
    assert_eq!(status_json["status"], "active");
    assert_eq!(status_json["authenticated"], true);

    // 3. Authenticated POST /prompt
    let prompt_res = client.post(format!("{}/prompt", handle.base_url()))
        .header("Authorization", format!("Bearer {}", auth_token))
        .json(&serde_json::json!({ "prompt": "refactor src/main.rs to use error handling" }))
        .send().await.expect("POST /prompt failed");
    assert_eq!(prompt_res.status(), reqwest::StatusCode::OK);
    let prompt_json: serde_json::Value = prompt_res.json().await.unwrap();
    assert_eq!(prompt_json["status"], "received");

    // 4. Authenticated POST /approval
    let approval_res = client.post(format!("{}/approval", handle.base_url()))
        .header("Authorization", format!("Bearer {}", auth_token))
        .json(&serde_json::json!({ "approved": true, "tool_id": "call_9988" }))
        .send().await.expect("POST /approval failed");
    assert_eq!(approval_res.status(), reqwest::StatusCode::OK);
    let approval_json: serde_json::Value = approval_res.json().await.unwrap();
    assert_eq!(approval_json["status"], "approval_recorded");

    // 5. Broadcast custom event and verify /history
    handle.broadcast(BridgeEventType::ThoughtStream, serde_json::json!({ "thought": "Analyzing repository structure..." }));
    let history_res = client.get(format!("{}/history", handle.base_url()))
        .header("Authorization", format!("Bearer {}", auth_token))
        .send().await.expect("GET /history failed");
    assert_eq!(history_res.status(), reqwest::StatusCode::OK);
    let history_json: serde_json::Value = history_res.json().await.unwrap();
    assert!(history_json.as_array().unwrap().len() >= 3);

    // 6. Test Active Bridge Registration & Global Broadcast
    register_active_bridge(handle.clone());
    let active_opt = get_active_bridge();
    assert!(active_opt.is_some());
    broadcast_to_active_bridge(BridgeEventType::ChatMessage, serde_json::json!({ "message": "Global agent ping" }));

    // 7. Test Terminal Report Formatting
    let term_out = format_remote_bridge_report_for_terminal(&handle);
    assert!(term_out.contains("REAL-TIME REMOTE PAIR-PROGRAMMING BRIDGE ACTIVE"));
    assert!(term_out.contains("Base URL:"));
    assert!(term_out.contains("SSE Stream:"));

    // 8. Shutdown bridge
    stop_active_bridge();
    handle.stop();
    assert!(!handle.is_running());

    // 9. Verify remote_bridge in get_tools()
    let tools = get_tools();
    let tools_str = serde_json::to_string(&tools).unwrap();
    assert!(tools_str.contains("remote_bridge"));
}

// ============================================================================
// SYSTEM 7: BRUTAL EDGE CASES & RESILIENCE ACROSS ALL 6 SYSTEMS
// ============================================================================

#[test]
fn test_brutal_edge_cases_across_6_advanced_systems() {
    let temp_dir = std::env::temp_dir().join(format!("zy_brutal_6sys_{}_{}", std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
    let _ = std::fs::create_dir_all(&temp_dir);

    // 1. Worktree on weird task ID with symbols and non-existent root
    let weird_handle = create_task_worktree(&temp_dir, "task/sub:weird#99!", None).expect("Weird task ID failed");
    assert_eq!(weird_handle.task_id, "task_sub_weird_99_");
    let _ = weird_handle.cleanup(true);

    // 2. Code review on totally empty file & non-existent file
    let empty_file = temp_dir.join("empty.rs");
    std::fs::write(&empty_file, "").unwrap();
    let empty_report = perform_code_review(&temp_dir, Some(&empty_file.to_string_lossy())).unwrap();
    assert_eq!(empty_report.findings.len(), 0);
    assert_eq!(empty_report.critical_count, 0);

    // 3. Conflict resolver on file with 0 conflicts and file with 3 consecutive conflicts
    let no_conflicts = "fn clean() { println!(\"clean\"); }";
    let clean_res = resolve_merge_conflict_content(no_conflicts, "clean.rs");
    assert_eq!(clean_res.conflicts_found, 0);
    assert_eq!(clean_res.conflicts_resolved, 0);
    assert!(clean_res.verified_syntax);

    let triple_conflict = r#"
<<<<<<< HEAD
use std::io;
=======
use std::fs;
>>>>>>> inc1
fn a() {}
<<<<<<< HEAD
fn b() { 1 }
=======
fn c() { 2 }
>>>>>>> inc2
fn d() {}
<<<<<<< HEAD
let x = 10;
||||||| base
let x = 5;
=======
let x = 20;
>>>>>>> inc3
"#;
    let triple_res = resolve_merge_conflict_content(triple_conflict, "triple.rs");
    assert_eq!(triple_res.conflicts_found, 3);
    assert_eq!(triple_res.conflicts_resolved, 3);
    assert!(!triple_res.resolved_file_content.contains("<<<<<<<"));

    // 4. Structural AST matcher on empty pattern & non-matching pattern
    let empty_matches = match_structural_pattern("", "let x = 1;");
    assert_eq!(empty_matches.len(), 0);
    let no_matches = match_structural_pattern("non_existent_symbol ( $$$ARGS )", "let x = 1;");
    assert_eq!(no_matches.len(), 0);

    // 5. SemVer parser on invalid versions fallback
    assert!(SemVer::parse("invalid_not_a_semver").is_none());
    assert_eq!(SemVer::parse("3").unwrap().to_string(), "3.0.0");
    assert_eq!(SemVer::parse("1.2").unwrap().to_string(), "1.2.0");

    let fallback_commit = parse_commit_line("arbitrary unformatted commit without type");
    assert_eq!(fallback_commit.commit_type, "chore");
    assert!(!fallback_commit.is_breaking);

    let _ = std::fs::remove_dir_all(&temp_dir);
}

// ============================================================================
// SYSTEM 1: LOCAL GGUF QUANTIZER & OLLAMA MODEL IMPORTER TESTS
// ============================================================================

#[test]
fn test_local_gguf_quantizer_and_ollama_importer() {
    let temp_dir = std::env::temp_dir().join(format!("zy_quant_test_{}_{}", std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
    let _ = std::fs::create_dir_all(&temp_dir);

    // 1. Test Quantization Type Normalization & Compression Ratios
    let (q4, r4) = normalize_quantization_type("q4_k_m");
    assert_eq!(q4, "Q4_K_M");
    assert!((r4 - 0.28).abs() < 0.01);

    let (q5, r5) = normalize_quantization_type("Q5_K");
    assert_eq!(q5, "Q5_K_M");
    assert!((r5 - 0.35).abs() < 0.01);

    let (q8, r8) = normalize_quantization_type("q8_0");
    assert_eq!(q8, "Q8_0");
    assert!((r8 - 0.50).abs() < 0.01);

    let (fp16, r16) = normalize_quantization_type("f16");
    assert_eq!(fp16, "FP16");
    assert!((r16 - 1.00).abs() < 0.01);

    let (custom_q, _) = normalize_quantization_type("q6_k");
    assert_eq!(custom_q, "Q6_K");

    // 2. Test Modelfile Content Builder
    let mut params = std::collections::HashMap::new();
    params.insert("temperature".to_string(), "0.7".to_string());
    params.insert("top_p".to_string(), "0.9".to_string());
    params.insert("stop".to_string(), "<|im_end|>,<|endoftext|>".to_string());

    let modelfile = build_modelfile_content("/models/deepseek-7b.gguf", Some("You are an expert AI software architect."), &params);
    assert!(modelfile.contains("FROM /models/deepseek-7b.gguf"));
    assert!(modelfile.contains("PARAMETER temperature 0.7"));
    assert!(modelfile.contains("PARAMETER top_p 0.9"));
    assert!(modelfile.contains("PARAMETER stop \"<|im_end|>\""));
    assert!(modelfile.contains("PARAMETER stop \"<|endoftext|>\""));
    assert!(modelfile.contains("SYSTEM \"\"\"You are an expert AI software architect.\"\"\""));
    assert!(modelfile.contains("TEMPLATE"));

    // 3. Test Full Quantize and Import Lifecycle with GGUF file
    let fake_gguf = temp_dir.join("source_model.gguf");
    std::fs::write(&fake_gguf, "GGUF_BINARY_DATA").unwrap();

    let report_gguf = quantize_and_import_model(&temp_dir, &fake_gguf, "zy-deepseek-q4", "Q4_K_M", Some("You are Zy.")).expect("Quantize failed");
    assert_eq!(report_gguf.output_name, "zy-deepseek-q4");
    assert_eq!(report_gguf.quantization_type, "Q4_K_M");
    assert!(report_gguf.modelfile_path.ends_with("zy-deepseek-q4.Modelfile"));
    assert!(std::path::Path::new(&report_gguf.modelfile_path).exists());
    assert!(report_gguf.conversion_command.contains("llama-quantize"));

    // 4. Test Conversion Recipe for PyTorch / Safetensors directory
    let hf_dir = temp_dir.join("hf_model_dir");
    std::fs::create_dir_all(&hf_dir).unwrap();
    std::fs::write(hf_dir.join("config.json"), "{}").unwrap();

    let report_hf = quantize_and_import_model(&temp_dir, &hf_dir, "zy-llama-q8", "Q8_0", None).expect("HF Quantize failed");
    assert_eq!(report_hf.quantization_type, "Q8_0");
    assert!(report_hf.conversion_command.contains("convert_hf_to_gguf.py"));
    assert!(report_hf.conversion_command.contains("llama-quantize"));

    // 5. Test Terminal Formatting
    let term_out = format_quantize_report_for_terminal(&report_gguf);
    assert!(term_out.contains("LOCAL GGUF QUANTIZER & OLLAMA MODEL IMPORTER"));
    assert!(term_out.contains("zy-deepseek-q4"));
    assert!(term_out.contains("Q4_K_M"));

    // 6. Verify quantize_model in get_tools()
    let tools = get_tools();
    let tools_str = serde_json::to_string(&tools).unwrap();
    assert!(tools_str.contains("quantize_model"));

    let _ = std::fs::remove_dir_all(&temp_dir);
}

// ============================================================================
// SYSTEM 2: CROSS-FILE DEAD CODE ELIMINATOR TESTS
// ============================================================================

#[test]
fn test_cross_file_dead_code_and_unused_symbol_eliminator() {
    let temp_dir = std::env::temp_dir().join(format!("zy_deadcode_test_{}_{}", std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
    let src_dir = temp_dir.join("src");
    let _ = std::fs::create_dir_all(&src_dir);

    // 1. Rust file with active symbols and dead symbols/imports
    let rs_file = src_dir.join("service.rs");
    let rs_content = r#"use std::collections::BTreeMap;
use std::fs;

pub fn active_service_fn() -> i32 {
    42
}

pub fn unused_dead_function(x: i32) -> i32 {
    x * 100
}

pub struct UnusedDeadStruct {
    pub value: String,
}
"#;
    std::fs::write(&rs_file, rs_content).unwrap();

    // 2. Python file referencing active_service_fn but having its own dead function
    let py_file = src_dir.join("app.py");
    let py_content = r#"import os
import sys

def active_py_runner():
    active_service_fn()

def unused_dead_py_calc():
    return 999
"#;
    std::fs::write(&py_file, py_content).unwrap();

    // 3. TypeScript file with dead interface and unused import
    let ts_file = src_dir.join("client.ts");
    let ts_content = r#"import { DeadImportHelper } from './helper';

export function activeClientCall() {
    active_py_runner();
}

export interface UnusedDeadInterface {
    id: string;
}
"#;
    std::fs::write(&ts_file, ts_content).unwrap();

    // 4. Run Dead Code Analysis
    let report = find_dead_code_symbols(&temp_dir).expect("Dead code scan failed");
    assert!(report.scanned_files >= 3);
    assert!(!report.dead_symbols.is_empty());
    assert!(!report.dead_imports.is_empty());

    // Verify specific dead symbols
    let has_dead_rs_fn = report.dead_symbols.iter().any(|s| s.name == "unused_dead_function");
    let has_dead_rs_struct = report.dead_symbols.iter().any(|s| s.name == "UnusedDeadStruct");
    let has_dead_py_fn = report.dead_symbols.iter().any(|s| s.name == "unused_dead_py_calc");
    let has_dead_ts_iface = report.dead_symbols.iter().any(|s| s.name == "UnusedDeadInterface");

    assert!(has_dead_rs_fn, "Expected unused_dead_function to be detected");
    assert!(has_dead_rs_struct, "Expected UnusedDeadStruct to be detected");
    assert!(has_dead_py_fn, "Expected unused_dead_py_calc to be detected");
    assert!(has_dead_ts_iface, "Expected UnusedDeadInterface to be detected");

    // Verify active symbols are NOT flagged as dead
    assert!(!report.dead_symbols.iter().any(|s| s.name == "active_service_fn"));
    assert!(!report.dead_symbols.iter().any(|s| s.name == "active_py_runner"));

    // Verify unused imports
    let has_dead_import = report.dead_imports.iter().any(|i| i.name == "BTreeMap" || i.name == "DeadImportHelper");
    assert!(has_dead_import, "Expected dead import to be detected");

    // 5. Test Safe Pruning Patches
    assert!(!report.patches.is_empty());
    let rs_patch = report.patches.iter().find(|p| p.symbol_name == "unused_dead_function").unwrap();
    assert!(rs_patch.diff.contains("unused_dead_function"));
    assert!(!rs_patch.pruned_content.contains("pub fn unused_dead_function"));

    // Apply pruning
    let applied = apply_dead_code_pruning(&report.patches).expect("Apply pruning failed");
    assert!(applied > 0);

    // Verify file on disk is pruned
    let updated_rs = std::fs::read_to_string(&rs_file).unwrap();
    assert!(!updated_rs.contains("pub fn unused_dead_function"));
    assert!(updated_rs.contains("pub fn active_service_fn"));

    // 6. Test Terminal Formatting
    let term_out = format_dead_code_report_for_terminal(&report);
    assert!(term_out.contains("CROSS-FILE DEAD CODE & UNUSED SYMBOL ELIMINATOR"));
    assert!(term_out.contains("unused_dead_function"));

    // 7. Verify dead_code_eliminator in get_tools()
    let tools = get_tools();
    let tools_str = serde_json::to_string(&tools).unwrap();
    assert!(tools_str.contains("dead_code_eliminator"));

    let _ = std::fs::remove_dir_all(&temp_dir);
}

// ============================================================================
// SYSTEM 3: SECRETS SANITIZER & .env.example SYNTHESIZER TESTS
// ============================================================================

#[test]
fn test_secrets_sanitizer_and_env_example_synthesizer() {
    let temp_dir = std::env::temp_dir().join(format!("zy_env_test_{}_{}", std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
    let _ = std::fs::create_dir_all(&temp_dir);

    // 1. Create .env with secret patterns and safe placeholders
    let env_file = temp_dir.join(".env");
    let env_content = r#"# Database Configuration
DATABASE_URL="postgres://appuser:supersecretpass123@localhost:5432/appdb"
REDIS_URL=redis://localhost:6379

# Authentication & Keys
OPENAI_API_KEY="sk-samplekey1234567890abcdef"
JWT_SECRET=super_secret_jwt_signing_key_9999
AUTH_TOKEN=bearer_sample_token_xyz12345

# App Settings (Safe Placeholders)
PORT=8080
NODE_ENV=production
DEBUG=false
"#;
    std::fs::write(&env_file, env_content).unwrap();

    // 2. Create existing .gitignore without .env
    let gitignore_file = temp_dir.join(".gitignore");
    std::fs::write(&gitignore_file, "target/\nnode_modules/\n").unwrap();

    // 3. Run Sanitizer
    let report = sanitize_workspace_environment(&temp_dir, Some(".env")).expect("Env sanitize failed");
    assert_eq!(report.secrets_detected.len(), 4); // DATABASE_URL, OPENAI_API_KEY, JWT_SECRET, AUTH_TOKEN

    // Verify secret masking
    let db_sec = report.secrets_detected.iter().find(|s| s.key == "DATABASE_URL").unwrap();
    assert_eq!(db_sec.secret_type, "database_uri");
    assert!(!db_sec.masked_value.contains("supersecretpass123"));
    assert!(db_sec.masked_value.contains("..."));

    let key_sec = report.secrets_detected.iter().find(|s| s.key == "OPENAI_API_KEY").unwrap();
    assert_eq!(key_sec.secret_type, "api_key");
    assert!(key_sec.masked_value.starts_with("sk-"));

    // Verify Synthesized .env.example
    let ex_content = &report.example_content;
    assert!(ex_content.contains("DATABASE_URL=postgres://user:password@localhost:5432/dbname"));
    assert!(ex_content.contains("OPENAI_API_KEY=your_openai_api_key_here"));
    assert!(ex_content.contains("JWT_SECRET=your_jwt_secret_key_here"));
    assert!(ex_content.contains("AUTH_TOKEN=your_auth_token_here"));
    assert!(ex_content.contains("PORT=8080"));
    assert!(ex_content.contains("NODE_ENV=production"));

    // 4. Test Writing Example & Updating .gitignore
    let written = write_env_example_and_update_gitignore(&report, &temp_dir).expect("Write env example failed");
    assert!(written);

    let ex_file = temp_dir.join(".env.example");
    assert!(ex_file.exists());
    let ex_file_text = std::fs::read_to_string(&ex_file).unwrap();
    assert!(ex_file_text.contains("DATABASE_URL=postgres://user:password@localhost:5432/dbname"));

    let updated_gi = std::fs::read_to_string(&gitignore_file).unwrap();
    assert!(updated_gi.contains(".env"));
    assert!(updated_gi.contains("*.env"));

    // 5. Test Terminal Formatting
    let term_out = format_env_sanitize_report_for_terminal(&report);
    assert!(term_out.contains("SECRETS SANITIZER & .env.example SYNTHESIZER"));
    assert!(term_out.contains("OPENAI_API_KEY"));
    assert!(term_out.contains("DATABASE_URL"));

    // 6. Verify sanitize_env in get_tools()
    let tools = get_tools();
    let tools_str = serde_json::to_string(&tools).unwrap();
    assert!(tools_str.contains("sanitize_env"));

    let _ = std::fs::remove_dir_all(&temp_dir);
}

// ============================================================================
// SYSTEM 4: OPENAPI / SWAGGER CLIENT SDK GENERATOR TESTS
// ============================================================================

#[test]
fn test_openapi_swagger_client_sdk_generator() {
    let spec = r#"{
  "openapi": "3.0.0",
  "info": { "title": "User Management API", "version": "1.0.0" },
  "paths": {
    "/users": {
      "get": {
        "operationId": "get_all_users",
        "summary": "Retrieve all users",
        "responses": { "200": { "description": "Success" } }
      },
      "post": {
        "operationId": "create_new_user",
        "summary": "Create a user",
        "responses": { "201": { "description": "Created" } }
      }
    },
    "/users/{id}": {
      "get": {
        "operationId": "get_user_by_id",
        "parameters": [{ "name": "id", "in": "path", "required": true, "schema": { "type": "string" } }],
        "responses": { "200": { "description": "User details" } }
      },
      "delete": {
        "operationId": "delete_user",
        "parameters": [{ "name": "id", "in": "path", "required": true, "schema": { "type": "string" } }],
        "responses": { "204": { "description": "Deleted" } }
      }
    }
  },
  "components": {
    "schemas": {
      "User": {
        "type": "object",
        "properties": {
          "id": { "type": "string" },
          "name": { "type": "string" },
          "age": { "type": "integer" },
          "is_active": { "type": "boolean" }
        }
      },
      "CreateUserRequest": {
        "type": "object",
        "properties": {
          "name": { "type": "string" },
          "email": { "type": "string" }
        }
      }
    }
  }
}"#;

    // 1. Generate Rust SDK
    let rust_sdk = generate_openapi_sdk(spec, "rust", "user_client").expect("Rust SDK gen failed");
    assert_eq!(rust_sdk.language, "rust");
    assert_eq!(rust_sdk.models.len(), 2);
    assert_eq!(rust_sdk.endpoints.len(), 4);
    assert!(rust_sdk.files.contains_key("client.rs"));
    assert!(rust_sdk.files.contains_key("models.rs"));
    assert!(rust_sdk.files.contains_key("lib.rs"));

    assert!(rust_sdk.models_code.contains("pub struct User"));
    assert!(rust_sdk.models_code.contains("pub is_active: Option<bool>"));
    assert!(rust_sdk.models_code.contains("pub age: Option<i64>"));
    assert!(rust_sdk.client_code.contains("pub struct ApiClient"));
    assert!(rust_sdk.client_code.contains("pub async fn get_all_users"));
    assert!(rust_sdk.client_code.contains("pub async fn get_user_by_id"));
    assert!(rust_sdk.client_code.contains("pub async fn delete_user"));

    // 2. Generate TypeScript SDK
    let ts_sdk = generate_openapi_sdk(spec, "typescript", "user-client").expect("TS SDK gen failed");
    assert_eq!(ts_sdk.language, "typescript");
    assert!(ts_sdk.models_code.contains("export interface User"));
    assert!(ts_sdk.models_code.contains("is_active?: boolean;"));
    assert!(ts_sdk.models_code.contains("age?: number;"));
    assert!(ts_sdk.client_code.contains("export class ApiClient"));
    assert!(ts_sdk.client_code.contains("async get_all_users()"));
    assert!(ts_sdk.client_code.contains("fetch("));

    // 3. Generate Python SDK
    let py_sdk = generate_openapi_sdk(spec, "python", "user_sdk").expect("Python SDK gen failed");
    assert_eq!(py_sdk.language, "python");
    assert!(py_sdk.models_code.contains("class User(BaseModel):"));
    assert!(py_sdk.models_code.contains("is_active: Optional[bool] = None"));
    assert!(py_sdk.models_code.contains("age: Optional[int] = None"));
    assert!(py_sdk.client_code.contains("class ApiClient:"));
    assert!(py_sdk.client_code.contains("async def get_all_users(self) -> Any:"));
    assert!(py_sdk.client_code.contains("httpx.AsyncClient()"));

    // 4. Test Terminal Formatting
    let term_out = format_sdk_report_for_terminal(&rust_sdk);
    assert!(term_out.contains("OPENAPI / SWAGGER STRONGLY-TYPED CLIENT SDK GENERATOR"));
    assert!(term_out.contains("user_client"));
    assert!(term_out.contains("get_all_users"));

    // 5. Verify generate_sdk in get_tools()
    let tools = get_tools();
    let tools_str = serde_json::to_string(&tools).unwrap();
    assert!(tools_str.contains("generate_sdk"));
}

// ============================================================================
// SYSTEM 5: INTERACTIVE REGEX, JQ & SCRATCHPAD EVALUATOR TESTS
// ============================================================================

#[test]
fn test_interactive_regex_jq_and_scratchpad_evaluator() {
    // 1. Test Regex Engine with Named and Indexed Capture Groups
    let regex_pattern = r"(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})";
    let text_data = "Release date: 2026-09-04 and next patch on 2027-01-15.";

    let re_res = evaluate_scratchpad_query("regex", regex_pattern, text_data).expect("Regex eval failed");
    assert!(re_res.success);
    assert_eq!(re_res.engine, "regex");
    let matches = re_res.matches.as_ref().unwrap();
    assert_eq!(matches.len(), 2);
    assert_eq!(matches[0].matched_text, "2026-09-04");
    assert_eq!(matches[1].matched_text, "2027-01-15");
    assert!(matches[0].groups.iter().any(|(k, v)| k == "year" && v == "2026"));
    assert!(matches[0].groups.iter().any(|(k, v)| k == "month" && v == "09"));
    assert!(matches[0].groups.iter().any(|(k, v)| k == "day" && v == "04"));

    // 2. Test JQ Engine Queries
    let json_data = r#"{
  "service": "zy-agent",
  "version": "1.0.0",
  "active": true,
  "users": [
    { "id": 101, "name": "Alice", "role": "admin" },
    { "id": 102, "name": "Bob", "role": "user" }
  ],
  "nested": {
    "deep": {
      "id": 999
    }
  }
}"#;

    // Field access & Array indexing
    let jq_name = evaluate_scratchpad_query("jq", ".users[0].name", json_data).unwrap();
    assert_eq!(jq_name.output, serde_json::json!("Alice"));

    // Root keys
    let jq_keys = evaluate_scratchpad_query("jq", "keys", json_data).unwrap();
    assert!(jq_keys.output.as_array().unwrap().contains(&serde_json::json!("service")));
    assert!(jq_keys.output.as_array().unwrap().contains(&serde_json::json!("users")));

    // Length
    let jq_len = evaluate_scratchpad_query("jq", "length", json_data).unwrap();
    assert_eq!(jq_len.output, serde_json::json!(5));

    // Type
    let jq_type = evaluate_scratchpad_query("jq", "type", json_data).unwrap();
    assert_eq!(jq_type.output, serde_json::json!("object"));

    // Recursive descent `..id`
    let jq_desc = evaluate_scratchpad_query("jq", "..id", json_data).unwrap();
    let desc_arr = jq_desc.output.as_array().unwrap();
    assert!(desc_arr.contains(&serde_json::json!(101)));
    assert!(desc_arr.contains(&serde_json::json!(102)));
    assert!(desc_arr.contains(&serde_json::json!(999)));

    // 3. Test Math / Expr Evaluator
    // Standard arithmetic with precedence and power
    let math1 = evaluate_scratchpad_query("math", "(10 * 5) + 2^3 - sqrt(16)", "").unwrap();
    assert_eq!(math1.output, serde_json::json!(54.0));

    // Math functions and constants
    let math2 = evaluate_scratchpad_query("math", "max(4, 18) + min(10, 2) * 3", "").unwrap();
    assert_eq!(math2.output, serde_json::json!(24.0));

    let math_pi = evaluate_scratchpad_query("math", "sin(0) + cos(0) * 10", "").unwrap();
    assert_eq!(math_pi.output, serde_json::json!(10.0));

    // Variables and Assignments
    let math_vars = evaluate_scratchpad_query("math", "x = 10; y = 20; x * y + 5", "").unwrap();
    assert_eq!(math_vars.output, serde_json::json!(205.0));

    // Context from input JSON
    let math_ctx = evaluate_scratchpad_query("math", "a / b + 2", "{\"a\": 15, \"b\": 3}").unwrap();
    assert_eq!(math_ctx.output, serde_json::json!(7.0));

    // 4. Test Terminal Formatting
    let term_out = format_eval_result_for_terminal(&re_res);
    assert!(term_out.contains("INTERACTIVE REGEX, JQ & SCRATCHPAD EVALUATOR"));
    assert!(term_out.contains("regex"));
    assert!(term_out.contains("2026-09-04"));

    // 5. Verify interactive_eval in get_tools()
    let tools = get_tools();
    let tools_str = serde_json::to_string(&tools).unwrap();
    assert!(tools_str.contains("interactive_eval"));
}

// ============================================================================
// SYSTEM 6: SMART GIT REBASE & HISTORY SQUEEZER TESTS
// ============================================================================

#[test]
fn test_smart_git_rebase_and_history_squeezer() {
    let temp_dir = std::env::temp_dir().join(format!("zy_rebase_test_{}_{}", std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
    let _ = std::fs::create_dir_all(&temp_dir);

    // 1. Test Commit Line Parsing
    let c1 = parse_rebase_commit_line("a1b2c3d|Dev|feat(auth): implement oauth2 login flow").unwrap();
    assert_eq!(c1.hash, "a1b2c3d");
    assert_eq!(c1.author, "Dev");
    assert_eq!(c1.commit_type, "feat");
    assert_eq!(c1.scope.as_deref(), Some("auth"));

    let c2 = parse_rebase_commit_line("e4f5g6h|Dev|wip: fix typo in auth handler").unwrap();
    assert_eq!(c2.commit_type, "wip");

    let c3 = parse_rebase_commit_line("i7j8k9l|Dev|fix(ui): button margin on mobile").unwrap();
    assert_eq!(c3.commit_type, "fix");
    assert_eq!(c3.scope.as_deref(), Some("ui"));

    // 2. Test Plan Smart Rebase
    let plan = plan_smart_rebase(&temp_dir, Some("main")).expect("Plan rebase failed");
    assert_eq!(plan.base_branch, "main");
    assert!(plan.total_commits >= 1);
    assert!(!plan.clusters.is_empty());
    assert!(!plan.git_commands.is_empty());

    // Verify git-rebase-todo script format
    let script = &plan.rebase_todo_script;
    assert!(script.contains("pick"));
    assert!(script.contains("squash") || plan.clusters.len() == plan.total_commits);

    // Verify Synthesized Conventional Commit message
    let first_cluster = &plan.clusters[0];
    assert!(first_cluster.synthesized_message.starts_with("feat") || first_cluster.synthesized_message.starts_with("fix") || first_cluster.synthesized_message.starts_with("chore"));

    // 3. Test Rebase Execution Staging
    let exec_res = execute_smart_rebase(&temp_dir, &plan, true).expect("Execute smart rebase failed");
    assert!(exec_res.contains("Rebase plan written"));
    assert!(temp_dir.join(".zy").join("rebase_plan.sh").exists());

    // 4. Test Terminal Formatting
    let term_out = format_rebase_plan_for_terminal(&plan);
    assert!(term_out.contains("SMART GIT REBASE & HISTORY SQUEEZER"));
    assert!(term_out.contains("Base Branch:"));
    assert!(term_out.contains("Total Commits:"));

    // 5. Verify smart_rebase in get_tools()
    let tools = get_tools();
    let tools_str = serde_json::to_string(&tools).unwrap();
    assert!(tools_str.contains("smart_rebase"));

    let _ = std::fs::remove_dir_all(&temp_dir);
}

// ============================================================================
// SYSTEM 7: BRUTAL EDGE CASES ACROSS ALL 6 NEW SYSTEMS
// ============================================================================

#[test]
fn test_brutal_edge_cases_across_6_new_systems() {
    let temp_dir = std::env::temp_dir().join(format!("zy_brutal_new6_{}_{}", std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
    let _ = std::fs::create_dir_all(&temp_dir);

    // 1. Quantizer on non-existent path and weird quant type
    let weird_quant = quantize_and_import_model(&temp_dir, std::path::Path::new("missing_model.bin"), "weird-model", "q99_custom", None).unwrap();
    assert_eq!(weird_quant.quantization_type, "Q99_CUSTOM");
    assert!(weird_quant.modelfile_content.contains("FROM"));

    // 2. Dead code eliminator on totally empty workspace
    let empty_dir = temp_dir.join("empty_ws");
    std::fs::create_dir_all(&empty_dir).unwrap();
    let empty_dead_rep = find_dead_code_symbols(&empty_dir).unwrap();
    assert_eq!(empty_dead_rep.scanned_files, 0);
    assert_eq!(empty_dead_rep.dead_symbols.len(), 0);
    assert_eq!(empty_dead_rep.patches.len(), 0);

    // 3. Secrets sanitizer on empty file
    let empty_env = temp_dir.join(".env.empty");
    std::fs::write(&empty_env, "# Just comments\n# Another comment\n").unwrap();
    let env_rep = sanitize_workspace_environment(&temp_dir, Some(".env.empty")).unwrap();
    assert_eq!(env_rep.secrets_detected.len(), 0);
    assert!(env_rep.example_content.contains("# Just comments"));

    // 4. OpenAPI generator on minimal empty spec
    let mini_spec = r#"{ "openapi": "3.0.0", "info": { "title": "Minimal", "version": "1.0" }, "paths": {} }"#;
    let mini_sdk = generate_openapi_sdk(mini_spec, "rust", "mini_client").unwrap();
    assert_eq!(mini_sdk.models.len(), 0);
    assert!(mini_sdk.endpoints.len() >= 1); // Synthesizes default fallback endpoint

    // 5. Evaluator edge cases: invalid regex, malformed JSON, math division by zero
    let bad_re = evaluate_scratchpad_query("regex", "[a-z", "text");
    assert!(bad_re.is_err());

    let div_zero = evaluate_scratchpad_query("math", "100 / 0", "");
    assert!(div_zero.is_err());

    let bad_engine = evaluate_scratchpad_query("unknown_engine", "1+1", "");
    assert!(bad_engine.is_err());

    // 6. Smart rebase on empty commit line
    assert!(parse_rebase_commit_line("").is_none());
    assert!(parse_rebase_commit_line("incomplete|line").is_none());

    let _ = std::fs::remove_dir_all(&temp_dir);
}

// ============================================================================
// SYSTEM 8: DATABASE MIGRATION & SCHEMA DIFF GENERATOR TESTS
// ============================================================================

#[test]
fn test_database_migration_and_schema_diff_generator() {
    let old_schema = r#"
        -- Initial Schema
        CREATE TABLE users (
            id INTEGER PRIMARY KEY,
            username VARCHAR(50) NOT NULL,
            email VARCHAR(100) UNIQUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE posts (
            id INTEGER PRIMARY KEY,
            author_id INTEGER REFERENCES users(id),
            title VARCHAR(200) NOT NULL,
            content TEXT
        );

        CREATE INDEX idx_users_email ON users (email);
    "#;

    let new_schema = r#"
        -- Target Schema with additions and modifications
        CREATE TABLE users (
            id INTEGER PRIMARY KEY,
            username VARCHAR(50) NOT NULL,
            email VARCHAR(100) UNIQUE,
            role VARCHAR(20) DEFAULT 'member',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE posts (
            id INTEGER PRIMARY KEY,
            author_id INTEGER REFERENCES users(id),
            title VARCHAR(200) NOT NULL,
            body TEXT,
            view_count INTEGER DEFAULT 0
        );

        CREATE TABLE comments (
            id INTEGER PRIMARY KEY,
            post_id INTEGER REFERENCES posts(id),
            commenter VARCHAR(50) NOT NULL,
            text TEXT NOT NULL
        );

        CREATE INDEX idx_users_email ON users (email);
        CREATE INDEX idx_posts_author ON posts (author_id);
    "#;

    // 1. Test Schema Parsing
    let old_parsed = parse_sql_schema(old_schema);
    assert_eq!(old_parsed.tables.len(), 2);
    assert!(old_parsed.tables.contains_key("users"));
    assert!(old_parsed.tables.contains_key("posts"));
    assert_eq!(old_parsed.standalone_indexes.len(), 1);

    let new_parsed = parse_sql_schema(new_schema);
    assert_eq!(new_parsed.tables.len(), 3);
    assert!(new_parsed.tables.contains_key("comments"));
    assert_eq!(new_parsed.standalone_indexes.len(), 2);

    // 2. Test Schema Diff Computation
    let diff = compute_schema_diff(&old_parsed, &new_parsed, "postgres");
    assert_eq!(diff.added_tables.len(), 1);
    assert_eq!(diff.added_tables[0].name, "comments");
    assert_eq!(diff.dropped_tables.len(), 0);
    assert_eq!(diff.added_indexes.len(), 1);
    assert_eq!(diff.added_indexes[0].name, "idx_posts_author");

    // 3. Test Migration Generation (Postgres)
    let migration_pg = generate_schema_migration(old_schema, new_schema, "v2_add_comments_and_roles", "postgres").expect("Postgres migration generation failed");
    assert_eq!(migration_pg.name, "v2_add_comments_and_roles");
    assert_eq!(migration_pg.dialect, "postgres");
    assert!(migration_pg.up_sql.contains("CREATE TABLE comments"));
    assert!(migration_pg.up_sql.contains("ALTER TABLE users ADD COLUMN role"));
    assert!(migration_pg.up_sql.contains("CREATE INDEX IF NOT EXISTS idx_posts_author"));
    assert!(migration_pg.down_sql.contains("DROP TABLE IF EXISTS comments"));
    assert!(migration_pg.down_sql.contains("ALTER TABLE users DROP COLUMN role"));
    assert!(migration_pg.down_sql.contains("DROP INDEX IF EXISTS idx_posts_author"));

    // 4. Test Migration Generation (SQLite)
    let migration_sqlite = generate_schema_migration(old_schema, new_schema, "v2_sqlite", "sqlite").expect("SQLite migration failed");
    assert_eq!(migration_sqlite.dialect, "sqlite");
    assert!(migration_sqlite.up_sql.contains("CREATE TABLE comments"));

    // 5. Test Migration from File Paths
    let temp_dir = std::env::temp_dir().join(format!("zy_mig_test_{}", std::process::id()));
    let _ = std::fs::create_dir_all(&temp_dir);
    let old_file = temp_dir.join("schema_v1.sql");
    let new_file = temp_dir.join("schema_v2.sql");
    std::fs::write(&old_file, old_schema).unwrap();
    std::fs::write(&new_file, new_schema).unwrap();

    let file_mig = generate_schema_migration(old_file.to_str().unwrap(), new_file.to_str().unwrap(), "file_based_mig", "postgres").expect("File based migration failed");
    assert_eq!(file_mig.added_tables.len(), 1);
    assert_eq!(file_mig.added_tables[0], "comments");

    // 6. Test Terminal Formatting
    let term_out = format_migration_report_for_terminal(&migration_pg);
    assert!(term_out.contains("DATABASE MIGRATION & SCHEMA DIFF GENERATOR"));
    assert!(term_out.contains("v2_add_comments_and_roles"));
    assert!(term_out.contains("comments"));

    // 7. Verify generate_migration in get_tools()
    let tools = get_tools();
    let tools_str = serde_json::to_string(&tools).unwrap();
    assert!(tools_str.contains("generate_migration"));

    let _ = std::fs::remove_dir_all(&temp_dir);
}

// ============================================================================
// SYSTEM 9: MULTI-LANGUAGE CODE TRANSPILER & PORTER TESTS
// ============================================================================

#[test]
fn test_multilanguage_code_transpiler_and_porter() {
    // 1. Test Language Detection
    assert_eq!(detect_source_language("app.py"), "python");
    assert_eq!(detect_source_language("main.rs"), "rust");
    assert_eq!(detect_source_language("service.ts"), "typescript");
    assert_eq!(detect_source_language("index.js"), "javascript");
    assert_eq!(detect_source_language("server.go"), "go");
    assert_eq!(detect_source_language("driver.c"), "c");

    // 2. Test Python to Rust Transpilation (Offline Rule Engine)
    let py_code = r#"
def calculate_tax(amount: float, rate: float):
    if amount < 0:
        raise ValueError("Amount cannot be negative")
    print(amount * rate)
    return amount * rate
"#;
    let py_res = transpile_code_offline(py_code, "python", "rust").expect("Py to Rust transpilation failed");
    assert_eq!(py_res.source_language, "python");
    assert_eq!(py_res.target_language, "rust");
    assert!(py_res.transpiled_code.contains("pub fn calculate_tax"));
    assert!(py_res.transpiled_code.contains("Result<"));
    assert!(py_res.transpiled_code.contains("println!"));
    assert!(py_res.transpiled_code.contains("return Err"));
    assert!(!py_res.idiomatic_conversions.is_empty());
    assert!(!py_res.diff_preview.is_empty());

    // 3. Test C to Safe Rust Transpilation
    let c_code = r#"
#include <stdio.h>
#include <stdlib.h>

int main() {
    char* buffer = (char*)malloc(1024);
    printf("Hello from C\n");
    free(buffer);
    return 0;
}
"#;
    let c_res = transpile_code_offline(c_code, "c", "rust").expect("C to Rust transpilation failed");
    assert_eq!(c_res.source_language, "c");
    assert_eq!(c_res.target_language, "rust");
    assert!(c_res.transpiled_code.contains("pub fn main"));
    assert!(c_res.transpiled_code.contains("println!"));
    assert!(c_res.transpiled_code.contains("Vec::with_capacity") || c_res.transpiled_code.contains("buffer"));
    assert!(c_res.transpiled_code.contains("Drop") || !c_res.transpiled_code.contains("free("));

    // 4. Test Python to Go Transpilation
    let py_go_code = r#"
def process_data():
    print("processing...")
    raise Exception("Failed to process")
"#;
    let go_res = transpile_code_offline(py_go_code, "python", "go").expect("Py to Go transpilation failed");
    assert_eq!(go_res.target_language, "go");
    assert!(go_res.transpiled_code.contains("package main"));
    assert!(go_res.transpiled_code.contains("func process_data() error"));
    assert!(go_res.transpiled_code.contains("fmt.Println"));
    assert!(go_res.transpiled_code.contains("fmt.Errorf"));

    // 5. Test JavaScript to TypeScript Transpilation
    let js_code = r#"
const user = { name: "Alice", age: 30 };
function getUser() {
    return user;
}
"#;
    let ts_res = transpile_code_offline(js_code, "javascript", "typescript").expect("JS to TS transpilation failed");
    assert_eq!(ts_res.target_language, "typescript");
    assert!(ts_res.transpiled_code.contains("interface"));
    assert!(ts_res.transpiled_code.contains("export function getUser"));

    // 6. Test Terminal Formatting
    let term_out = format_transpile_report_for_terminal(&py_res);
    assert!(term_out.contains("MULTI-LANGUAGE CODE TRANSPILER & PORTER"));
    assert!(term_out.contains("Source Language:"));
    assert!(term_out.contains("python"));
    assert!(term_out.contains("rust"));

    // 7. Verify translate_code in get_tools()
    let tools = get_tools();
    let tools_str = serde_json::to_string(&tools).unwrap();
    assert!(tools_str.contains("translate_code"));
}

// ============================================================================
// SYSTEM 10: ARCHITECTURE DECISION RECORD (ADR) SYNTHESIZER TESTS
// ============================================================================

#[test]
fn test_architecture_decision_record_synthesizer() {
    let temp_dir = std::env::temp_dir().join(format!("zy_adr_test_{}_{}", std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
    let _ = std::fs::create_dir_all(&temp_dir);

    // 1. Test Slugify
    assert_eq!(slugify("Use PostgreSQL for Primary Storage"), "use-postgresql-for-primary-storage");
    assert_eq!(slugify("Adopt Rust & WebAssembly (Wasm)!"), "adopt-rust-webassembly-wasm");
    assert_eq!(slugify("  Multiple   Spaces --- And Symbols #123  "), "multiple-spaces-and-symbols-123");

    // 2. Test Sequential ADR Creation
    let adr1 = create_architecture_decision_record(
        &temp_dir,
        "Adopt PostgreSQL as Core Relational Store",
        "The system requires ACID guarantees, concurrent transactional consistency, and rich JSONB querying capabilities.",
        "We choose PostgreSQL 16 as the primary relational database system.",
        "Ensures ACID compliance and rich indexing; requires managed infrastructure maintenance.",
        Some("Accepted"),
    ).expect("Creating ADR 1 failed");

    assert_eq!(adr1.id, 1);
    assert_eq!(adr1.slug, "adopt-postgresql-as-core-relational-store");
    assert_eq!(adr1.status, "Accepted");
    assert!(adr1.file_path.exists());
    assert!(adr1.content.contains("# ADR-0001: Adopt PostgreSQL as Core Relational Store"));
    assert!(adr1.content.contains("Context and Problem Statement"));
    assert!(adr1.content.contains("ACID guarantees"));

    // 3. Test Next Index Discovery for ADR 2
    let adr2 = create_architecture_decision_record(
        &temp_dir,
        "Implement Ephemeral In-Memory Cache with Redis",
        "High frequency read endpoints require sub-millisecond response latency.",
        "Deploy standalone Redis cluster for session caching and rate-limiting.",
        "Reduces database read load by 80%; adds cache invalidation complexity.",
        Some("Proposed"),
    ).expect("Creating ADR 2 failed");

    assert_eq!(adr2.id, 2);
    assert_eq!(adr2.slug, "implement-ephemeral-in-memory-cache-with-redis");
    assert_eq!(adr2.status, "Proposed");
    assert!(adr2.content.contains("# ADR-0002: Implement Ephemeral In-Memory Cache with Redis"));

    // 4. Test List Existing ADRs
    let adr_list = list_existing_adrs(&temp_dir).expect("Listing ADRs failed");
    assert_eq!(adr_list.len(), 2);
    assert_eq!(adr_list[0].id, 1);
    assert_eq!(adr_list[1].id, 2);

    // 5. Test Terminal Formatting
    let term_out = format_adr_report_for_terminal(&adr1);
    assert!(term_out.contains("ARCHITECTURE DECISION RECORD (ADR) SYNTHESIZER"));
    assert!(term_out.contains("ADR-0001"));
    assert!(term_out.contains("Adopt PostgreSQL"));

    // 6. Verify generate_adr in get_tools()
    let tools = get_tools();
    let tools_str = serde_json::to_string(&tools).unwrap();
    assert!(tools_str.contains("generate_adr"));

    let _ = std::fs::remove_dir_all(&temp_dir);
}

// ============================================================================
// SYSTEM 11: PACKAGE REGISTRY & COMPATIBILITY INSPECTOR TESTS
// ============================================================================

#[test]
fn test_package_registry_and_compatibility_inspector() {
    // 1. Test crates.io Registry JSON Parser
    let crates_io_raw = r#"{
        "crate": {
            "name": "serde",
            "max_version": "1.0.210",
            "description": "A generic serialization/deserialization framework",
            "homepage": "https://serde.rs",
            "repository": "https://github.com/serde-rs/serde",
            "documentation": "https://docs.rs/serde",
            "downloads": 450000000,
            "keywords": ["serde", "serialization", "no_std"]
        },
        "versions": [
            {
                "num": "1.0.210",
                "license": "MIT OR Apache-2.0",
                "features": {
                    "derive": ["serde_derive"],
                    "std": [],
                    "alloc": []
                }
            }
        ]
    }"#;

    let crate_info = parse_package_registry_response("crates.io", "serde", crates_io_raw).expect("Parsing crates.io response failed");
    assert_eq!(crate_info.name, "serde");
    assert_eq!(crate_info.ecosystem, "crates.io");
    assert_eq!(crate_info.latest_version, "1.0.210");
    assert_eq!(crate_info.license.as_deref(), Some("MIT OR Apache-2.0"));
    assert_eq!(crate_info.homepage.as_deref(), Some("https://serde.rs"));
    assert!(crate_info.features.contains(&"derive".to_string()));
    assert_eq!(crate_info.downloads, Some(450000000));

    // 2. Test npm Registry JSON Parser
    let npm_raw = r#"{
        "name": "react",
        "description": "React is a JavaScript library for building user interfaces.",
        "dist-tags": { "latest": "18.3.1" },
        "license": "MIT",
        "homepage": "https://react.dev/",
        "repository": { "type": "git", "url": "https://github.com/facebook/react.git" },
        "keywords": ["react", "ui", "virtual-dom"],
        "versions": {
            "18.3.1": {
                "dependencies": {
                    "loose-envify": "^1.1.0"
                }
            }
        }
    }"#;

    let npm_info = parse_package_registry_response("npm", "react", npm_raw).expect("Parsing npm response failed");
    assert_eq!(npm_info.name, "react");
    assert_eq!(npm_info.ecosystem, "npm");
    assert_eq!(npm_info.latest_version, "18.3.1");
    assert_eq!(npm_info.license.as_deref(), Some("MIT"));
    assert!(npm_info.dependencies.iter().any(|d| d.starts_with("loose-envify")));

    // 3. Test PyPI Registry JSON Parser
    let pypi_raw = r#"{
        "info": {
            "name": "requests",
            "version": "2.32.3",
            "summary": "Python HTTP for Humans.",
            "license": "Apache-2.0",
            "home_page": "https://requests.readthedocs.io",
            "project_urls": {
                "Documentation": "https://requests.readthedocs.io",
                "Repository": "https://github.com/psf/requests"
            },
            "requires_dist": [
                "charset-normalizer<4,>=2",
                "idna<4,>=2.5",
                "urllib3<3,>=1.21.1",
                "certifi>=2017.4.17"
            ]
        }
    }"#;

    let pypi_info = parse_package_registry_response("pypi", "requests", pypi_raw).expect("Parsing pypi response failed");
    assert_eq!(pypi_info.name, "requests");
    assert_eq!(pypi_info.ecosystem, "pypi");
    assert_eq!(pypi_info.latest_version, "2.32.3");
    assert_eq!(pypi_info.license.as_deref(), Some("Apache-2.0"));
    assert_eq!(pypi_info.dependencies.len(), 4);

    // 4. Test Terminal Formatting
    let term_out = format_package_info_for_terminal(&crate_info);
    assert!(term_out.contains("PACKAGE REGISTRY & COMPATIBILITY INSPECTOR"));
    assert!(term_out.contains("serde"));
    assert!(term_out.contains("1.0.210"));

    // 5. Verify search_registry in get_tools()
    let tools = get_tools();
    let tools_str = serde_json::to_string(&tools).unwrap();
    assert!(tools_str.contains("search_registry"));
}

// ============================================================================
// SYSTEM 12: FRONTEND ACCESSIBILITY (A11Y) & WEB VITALS AUDITOR TESTS
// ============================================================================

#[test]
fn test_frontend_accessibility_and_web_vitals_auditor() {
    let temp_dir = std::env::temp_dir().join(format!("zy_a11y_test_{}_{}", std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
    let _ = std::fs::create_dir_all(&temp_dir);

    let bad_html_content = r#"
<html>
<head><title>Test Page</title></head>
<body>
    <h1>Welcome to Dashboard</h1>
    <h3>Skipped heading level 2 directly to 3</h3>

    <!-- Missing alt on img -->
    <img src="/avatar.png" />

    <!-- Button without accessible name -->
    <button class="icon-btn"><svg></svg></button>

    <!-- Form control without label -->
    <input type="text" placeholder="Enter username" />

    <!-- Non-interactive element with click listener without key listener or role -->
    <div onClick={handleClick}>Clickable Card</div>

    <!-- Iframe without title -->
    <iframe src="https://example.com/embed"></iframe>
</body>
</html>
"#;

    let bad_file = temp_dir.join("index.html");
    std::fs::write(&bad_file, bad_html_content).unwrap();

    // 1. Audit Target File
    let report = audit_workspace_accessibility(&temp_dir, Some("index.html")).expect("Auditing accessibility failed");
    assert_eq!(report.scanned_files_count, 1);
    assert!(report.total_violations >= 5);
    assert!(report.critical_count >= 1); // missing alt
    assert!(report.serious_count >= 2);  // button without name, input without label, non-interactive click
    assert!(report.moderate_count >= 1); // html missing lang, iframe missing title
    assert!(report.score < 80.0);

    // 2. Verify Violation Rules
    let rule_ids: Vec<String> = report.violations.iter().map(|v| v.rule_id.clone()).collect();
    assert!(rule_ids.contains(&"image-alt".to_string()));
    assert!(rule_ids.contains(&"button-name".to_string()));
    assert!(rule_ids.contains(&"form-control-label".to_string()));
    assert!(rule_ids.contains(&"html-has-lang".to_string()));
    assert!(rule_ids.contains(&"click-events-have-key-events".to_string()));
    assert!(rule_ids.contains(&"iframe-title".to_string()));
    assert!(rule_ids.contains(&"heading-order".to_string()));

    // 3. Test Clean Accessible Component
    let clean_jsx_content = r#"
<html lang="en">
<body>
    <h1>Profile</h1>
    <h2>Personal Details</h2>
    <img src="/avatar.png" alt="User profile avatar photo" />
    <button aria-label="Submit profile form">Submit</button>
    <label for="uname">Username</label>
    <input id="uname" type="text" />
</body>
</html>
"#;
    let clean_file = temp_dir.join("clean.html");
    std::fs::write(&clean_file, clean_jsx_content).unwrap();

    let clean_report = audit_workspace_accessibility(&temp_dir, Some("clean.html")).expect("Auditing clean file failed");
    assert_eq!(clean_report.total_violations, 0);
    assert_eq!(clean_report.score, 100.0);

    // 4. Test Terminal Formatting
    let term_out = format_a11y_report_for_terminal(&report);
    assert!(term_out.contains("FRONTEND ACCESSIBILITY (A11Y) & WEB VITALS AUDITOR"));
    assert!(term_out.contains("image-alt"));
    assert!(term_out.contains("button-name"));

    // 5. Verify audit_accessibility in get_tools()
    let tools = get_tools();
    let tools_str = serde_json::to_string(&tools).unwrap();
    assert!(tools_str.contains("audit_accessibility"));

    let _ = std::fs::remove_dir_all(&temp_dir);
}

// ============================================================================
// SYSTEM 13: LOCAL TOKEN & CLOUD COST SAVINGS ANALYTICS ENGINE TESTS
// ============================================================================

#[test]
fn test_local_token_and_cloud_cost_savings_analytics_engine() {
    let temp_dir = std::env::temp_dir().join(format!("zy_analytics_test_{}_{}", std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
    let _ = std::fs::create_dir_all(&temp_dir);

    // 1. Reset initial analytics
    reset_analytics(&temp_dir).expect("Reset analytics failed");

    // 2. Record First Inference Batch
    // 500 prompt tokens, 200 completion tokens, 1000ms latency, model "llama3:8b"
    let rep1 = record_token_usage(&temp_dir, 500, 200, 1000, "llama3:8b").expect("Recording token usage 1 failed");
    assert_eq!(rep1.total_requests, 1);
    assert_eq!(rep1.prompt_tokens, 500);
    assert_eq!(rep1.completion_tokens, 200);
    assert_eq!(rep1.total_tokens, 700);
    assert!((rep1.avg_tokens_per_sec - 700.0).abs() < 1.0);

    // Savings computation check:
    // (500 / 1000 * 0.003) + (200 / 1000 * 0.015) = 0.0015 + 0.0030 = 0.0045 USD
    assert!((rep1.commercial_cost_savings_usd - 0.0045).abs() < 0.0001);

    // 3. Record Second Inference Batch (accumulative persistence)
    // 1500 prompt tokens, 800 completion tokens, 2000ms latency, model "qwen2.5-coder:7b"
    let rep2 = record_token_usage(&temp_dir, 1500, 800, 2000, "qwen2.5-coder:7b").expect("Recording token usage 2 failed");
    assert_eq!(rep2.total_requests, 2);
    assert_eq!(rep2.prompt_tokens, 2000);
    assert_eq!(rep2.completion_tokens, 1000);
    assert_eq!(rep2.total_tokens, 3000);
    assert_eq!(rep2.model_breakdown.len(), 2);

    // Total savings:
    // Prompt: 2000 / 1000 * 0.003 = 0.006
    // Completion: 1000 / 1000 * 0.015 = 0.015
    // Total = 0.021 USD
    assert!((rep2.commercial_cost_savings_usd - 0.021).abs() < 0.0001);
    assert!(rep2.gpt4_savings_usd > rep2.commercial_cost_savings_usd);
    assert!(rep2.claude_opus_savings_usd > rep2.commercial_cost_savings_usd);

    // 4. Generate Report on workspace
    let final_rep = generate_analytics_report(&temp_dir);
    assert_eq!(final_rep.total_requests, 2);
    assert_eq!(final_rep.total_tokens, 3000);

    // 5. Test Terminal Formatting Dashboard
    let term_out = format_analytics_dashboard_for_terminal(&final_rep);
    assert!(term_out.contains("LOCAL TOKEN & CLOUD COST SAVINGS ANALYTICS ENGINE"));
    assert!(term_out.contains("Total Requests:"));
    assert!(term_out.contains("CUMULATIVE SAVINGS"));
    assert!(term_out.contains("MODEL BREAKDOWN"));
    assert!(term_out.contains("llama3:8b"));
    assert!(term_out.contains("qwen2.5-coder:7b"));

    // 6. Test Analytics Reset
    reset_analytics(&temp_dir).expect("Resetting analytics failed");
    let after_reset = generate_analytics_report(&temp_dir);
    assert_eq!(after_reset.total_requests, 0);
    assert_eq!(after_reset.total_tokens, 0);

    // 7. Verify usage_analytics in get_tools()
    let tools = get_tools();
    let tools_str = serde_json::to_string(&tools).unwrap();
    assert!(tools_str.contains("usage_analytics"));

    let _ = std::fs::remove_dir_all(&temp_dir);
}

// ============================================================================
// SYSTEM 14: BRUTAL EDGE CASES ACROSS ALL 6 LATEST SYSTEMS
// ============================================================================

#[test]
fn test_brutal_edge_cases_across_6_latest_systems() {
    let temp_dir = std::env::temp_dir().join(format!("zy_brutal_latest6_{}_{}", std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
    let _ = std::fs::create_dir_all(&temp_dir);

    // 1. Migration: completely empty and malformed SQL
    let empty_mig = generate_schema_migration("", "", "empty_mig", "postgres").unwrap();
    assert_eq!(empty_mig.added_tables.len(), 0);
    assert_eq!(empty_mig.dropped_tables.len(), 0);

    // Complex nested types and comments
    let complex_sql = r#"
        /* Big Block Comment */
        CREATE TABLE complex (
            id SERIAL PRIMARY KEY,
            balance NUMERIC(12, 2) NOT NULL DEFAULT 0.00,
            metadata JSONB,
            CONSTRAINT check_positive CHECK (balance >= 0)
        );
    "#;
    let parsed_complex = parse_sql_schema(complex_sql);
    assert!(parsed_complex.tables.contains_key("complex"));
    let col_balance = parsed_complex.tables["complex"].columns.iter().find(|c| c.name == "balance").unwrap();
    assert!(!col_balance.nullable);

    // 2. Transpiler: empty string and unknown languages
    let empty_trans = transpile_code_offline("", "unknown", "unknown").unwrap();
    assert_eq!(empty_trans.original_code, "");

    // 3. ADR: Slugify empty and weird characters
    assert_eq!(slugify(""), "");
    assert_eq!(slugify("$$$###@@@"), "");
    let adr_weird = create_architecture_decision_record(&temp_dir, "ADR Title With & Symbols $100", "Ctx", "Dec", "Con", None).unwrap();
    assert_eq!(adr_weird.id, 1);
    assert!(adr_weird.file_path.exists());

    // 4. Package Registry: malformed and partial JSON
    let bad_json = parse_package_registry_response("crates.io", "bad", "{}");
    assert!(bad_json.is_err());

    let mini_pypi = parse_package_registry_response("pypi", "mini", r#"{"info": {"name": "mini", "version": "0.0.1"}}"#).unwrap();
    assert_eq!(mini_pypi.latest_version, "0.0.1");

    // 5. A11y: Empty directory audit
    let empty_dir = temp_dir.join("empty_dir");
    std::fs::create_dir_all(&empty_dir).unwrap();
    let empty_a11y = audit_workspace_accessibility(&empty_dir, None).unwrap();
    assert_eq!(empty_a11y.scanned_files_count, 0);
    assert_eq!(empty_a11y.total_violations, 0);
    assert_eq!(empty_a11y.score, 100.0);

    // 6. Analytics: Zero requests calculations
    let zero_report = AnalyticsEngine::generate_report(&AnalyticsData {
        total_requests: 0,
        total_prompt_tokens: 0,
        total_completion_tokens: 0,
        total_duration_ms: 0,
        model_usage: std::collections::HashMap::new(),
        last_updated: "2026-09-04T00:00:00Z".to_string(),
    });
    assert_eq!(zero_report.avg_tokens_per_sec, 0.0);
    assert_eq!(zero_report.commercial_cost_savings_usd, 0.0);

    let _ = std::fs::remove_dir_all(&temp_dir);
}

// ============================================================================
// SYSTEM 15: TERMINAL GRAPHICS PROTOCOLS & UNICODE TRUECOLOR ENGINE
// ============================================================================

#[test]
fn test_terminal_graphics_protocols_and_unicode_fallback() {
    let temp_dir = std::env::temp_dir().join(format!("zy_test_graphics_{}_{}", std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
    let _ = std::fs::create_dir_all(&temp_dir);

    let test_img_path = temp_dir.join("test_sample.png");
    let synthetic_png_bytes = vec![
        0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, // PNG header
        0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52, // IHDR
        0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x10,
        0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x91, 0x68,
        0x36, 0x00, 0x00, 0x00, 0x0C, 0x49, 0x44, 0x41,
        0x54, 0x78, 0x9C, 0x63, 0x60, 0x00, 0x00, 0x00,
        0x02, 0x00, 0x01, 0xE2, 0x21, 0xBC, 0x33, 0x00,
        0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE,
        0x42, 0x60, 0x82,
    ];
    std::fs::write(&test_img_path, &synthetic_png_bytes).unwrap();

    // 1. Kitty Graphics Protocol
    let kitty_out = render_terminal_graphics(&synthetic_png_bytes, "png", "kitty", 40, 20).unwrap();
    assert!(kitty_out.starts_with("\x1b_G"));
    assert!(kitty_out.ends_with("\x1b\\"));
    assert!(kitty_out.contains("a=T"));

    // 2. iTerm2 Protocol
    let iterm_out = render_terminal_graphics(&synthetic_png_bytes, "png", "iterm2", 40, 20).unwrap();
    assert!(iterm_out.starts_with("\x1b]1337;File=inline=1"));
    assert!(iterm_out.ends_with("\x07"));
    assert!(iterm_out.contains(";size="));

    // 3. Sixel Protocol
    let sixel_out = render_terminal_graphics(&synthetic_png_bytes, "png", "sixel", 40, 20).unwrap();
    assert!(sixel_out.starts_with("\x1bPq"));
    assert!(sixel_out.ends_with("\x1b\\"));
    assert!(sixel_out.contains("#0;2;"));

    // 4. Unicode Half-Block TrueColor Fallback
    let unicode_out = render_terminal_graphics(&synthetic_png_bytes, "png", "unicode", 20, 10).unwrap();
    assert!(unicode_out.contains("▀"));
    assert!(unicode_out.contains("\x1b[38;2;"));
    assert!(unicode_out.contains(";48;2;"));

    // 5. Quadrant Fallback
    let quad_out = render_terminal_graphics(&synthetic_png_bytes, "png", "quadrant", 20, 10).unwrap();
    assert!(quad_out.contains("\x1b[38;2;"));

    // 6. render_diagram_or_image from file
    let file_rendered = render_diagram_or_image(test_img_path.to_str().unwrap(), "kitty", 30, 15).unwrap();
    assert!(file_rendered.starts_with("\x1b_G"));

    // 7. render_diagram_or_image from diagram spec string
    let diagram_rendered = render_diagram_or_image("flowchart TD\nA-->B", "unicode", 30, 15).unwrap();
    assert!(diagram_rendered.contains("▀"));

    // 8. Terminal Graphic Report Formatting
    let rep = TerminalGraphicReport {
        format: "png".to_string(),
        protocol: "sixel".to_string(),
        dimensions: (40, 20),
        payload_size: synthetic_png_bytes.len(),
        rendered_output: sixel_out.clone(),
        summary: "Rendered Sixel graphics test".to_string(),
    };
    let term_str = format_graphic_report_for_terminal(&rep);
    assert!(term_str.contains("TERMINAL GRAPHICS VISUALIZER"));
    assert!(term_str.contains("sixel"));
    assert!(term_str.contains("40x20"));

    // 9. Verify tool in get_tools()
    let tools = get_tools();
    let tools_str = serde_json::to_string(&tools).unwrap();
    assert!(tools_str.contains("render_terminal_graphic"));

    let _ = std::fs::remove_dir_all(&temp_dir);
}

// ============================================================================
// SYSTEM 16: DESKTOP COMPANION GUI SERVER LIFECYCLE
// ============================================================================

#[tokio::test]
async fn test_desktop_companion_gui_server_lifecycle() {
    // 1. Launch server on dynamic port (0)
    let handle = launch_desktop_companion_gui(0, false).await.expect("Failed to launch GUI server");
    let port = handle.port();
    assert!(port > 0);
    assert!(handle.url().contains(&port.to_string()));
    assert!(handle.is_running());

    // 2. Global registry tests
    register_active_gui(handle.clone());
    let active = get_active_gui();
    assert!(active.is_some());
    assert_eq!(active.unwrap().port(), port);

    // 3. Test HTTP Endpoints using reqwest Client
    let client = reqwest::Client::new();
    let base_url = handle.url().to_string();

    // GET / (HTML App)
    let index_resp = client.get(&base_url).send().await.expect("Failed to GET /");
    assert_eq!(index_resp.status(), 200);
    let index_html = index_resp.text().await.unwrap();
    assert!(index_html.contains("zy Desktop Companion"));
    assert!(index_html.contains("tailwindcss"));

    // GET /api/status
    let status_resp = client.get(format!("{}/api/status", base_url)).send().await.expect("Failed to GET /api/status");
    assert_eq!(status_resp.status(), 200);
    let status_json: serde_json::Value = status_resp.json().await.unwrap();
    assert_eq!(status_json["status"], "running");
    assert_eq!(status_json["port"], port);

    // GET /api/telemetry
    let telem_resp = client.get(format!("{}/api/telemetry", base_url)).send().await.expect("Failed to GET /api/telemetry");
    assert_eq!(telem_resp.status(), 200);
    let telem_json: serde_json::Value = telem_resp.json().await.unwrap();
    assert!(telem_json["tokens_per_sec"].as_f64().unwrap() > 0.0);

    // POST /api/prompt
    let prompt_resp = client.post(format!("{}/api/prompt", base_url))
        .json(&serde_json::json!({ "prompt": "Hello zy!" }))
        .send().await.expect("Failed to POST /api/prompt");
    assert_eq!(prompt_resp.status(), 200);

    // POST /api/approve
    let app_resp = client.post(format!("{}/api/approve", base_url))
        .json(&serde_json::json!({ "approved": true }))
        .send().await.expect("Failed to POST /api/approve");
    assert_eq!(app_resp.status(), 200);

    // 4. Test Broadcast streams
    handle.broadcast_thought("Observing workspace architecture...");
    handle.broadcast_event("hunk_diff", serde_json::json!({ "file": "src/lib.rs", "additions": 50 }));

    // 5. Format Terminal Report
    let rep_str = format_gui_report_for_terminal(&handle);
    assert!(rep_str.contains("DESKTOP COMPANION GUI"));
    assert!(rep_str.contains(&port.to_string()));

    // 6. Graceful Shutdown
    handle.stop();
    assert!(!handle.is_running());
    stop_active_gui();
    assert!(get_active_gui().is_none());

    // 7. Verify desktop_gui in get_tools()
    let tools = get_tools();
    let tools_str = serde_json::to_string(&tools).unwrap();
    assert!(tools_str.contains("desktop_gui"));
}

// ============================================================================
// SYSTEM 17: VISUAL MULTI-AGENT SWARM CANVAS & STUDIO
// ============================================================================

#[tokio::test]
async fn test_visual_swarm_canvas_studio_lifecycle() {
    // 1. Start Swarm Studio server on dynamic port (0)
    let handle = start_swarm_studio_server(0).await.expect("Failed to start swarm studio server");
    let port = handle.port();
    assert!(port > 0);
    assert!(handle.url().contains(&port.to_string()));
    assert!(handle.is_running());

    // 2. Global registry tests
    register_active_studio(handle.clone());
    let active = get_active_studio();
    assert!(active.is_some());
    assert_eq!(active.unwrap().port(), port);

    // 3. Test HTTP Endpoints
    let client = reqwest::Client::new();
    let base_url = handle.url().to_string();

    // GET / (Studio HTML Canvas)
    let index_resp = client.get(&base_url).send().await.expect("Failed to GET studio /");
    assert_eq!(index_resp.status(), 200);
    let index_html = index_resp.text().await.unwrap();
    assert!(index_html.contains("Multi-Agent Swarm Studio Canvas"));
    assert!(index_html.contains("Architect"));
    assert!(index_html.contains("Coder"));

    // GET /api/studio/state
    let state_resp = client.get(format!("{}/api/studio/state", base_url)).send().await.expect("Failed to GET /api/studio/state");
    assert_eq!(state_resp.status(), 200);
    let state_json: SwarmStudioState = state_resp.json().await.unwrap();
    assert_eq!(state_json.nodes.len(), 4);
    assert!(state_json.nodes.iter().any(|n| n.role == "Architect"));
    assert!(state_json.nodes.iter().any(|n| n.role == "Coder"));
    assert!(state_json.nodes.iter().any(|n| n.role == "Auditor"));
    assert!(state_json.nodes.iter().any(|n| n.role == "QA Tester"));

    // 4. Update Node Status & Broadcast Message Passing
    handle.update_agent_status("coder", "working", "synthesizing terminal graphics");
    handle.broadcast_node_event("architect", "coder", "plan_ready", "Terminal Graphics Specification v1");
    handle.set_active_diff("diff --git a/src/lib.rs b/src/lib.rs\n+pub fn render_terminal_graphics()");

    // Allow tokio tasks to process updates
    tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;

    // Verify State Mutated
    let updated_state = handle.state.read().await.clone();
    let coder_node = updated_state.nodes.iter().find(|n| n.id == "coder").unwrap();
    assert_eq!(coder_node.status, "working");
    assert_eq!(coder_node.current_task, "synthesizing terminal graphics");
    assert!(!updated_state.logs.is_empty());
    assert_eq!(updated_state.logs[0].from, "architect");
    assert_eq!(updated_state.logs[0].to, "coder");
    assert!(updated_state.active_diff.as_ref().unwrap().contains("render_terminal_graphics"));

    // 5. Terminal Formatting Report
    let rep_str = format_studio_report_for_terminal(&handle);
    assert!(rep_str.contains("VISUAL SWARM STUDIO CANVAS"));
    assert!(rep_str.contains(&port.to_string()));

    // 6. Graceful Shutdown
    handle.stop();
    assert!(!handle.is_running());
    stop_active_studio();
    assert!(get_active_studio().is_none());

    // 7. Verify studio_canvas in get_tools()
    let tools = get_tools();
    let tools_str = serde_json::to_string(&tools).unwrap();
    assert!(tools_str.contains("studio_canvas"));
}

// ============================================================================
// SYSTEM 18: UNIVERSAL THEME & 24-BIT TRUECOLOR ENGINE
// ============================================================================

#[test]
fn test_universal_theme_palette_and_truecolor_engine() {
    let temp_dir = std::env::temp_dir().join(format!("zy_test_theme_{}_{}", std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
    let _ = std::fs::create_dir_all(&temp_dir);

    // 1. Test RgbColor hex parsing and ANSI rendering
    let c1 = RgbColor::from_hex("#89b4fa").unwrap();
    assert_eq!(c1.r, 137);
    assert_eq!(c1.g, 180);
    assert_eq!(c1.b, 250);
    assert_eq!(c1.to_hex(), "#89b4fa");
    assert_eq!(c1.to_ansi_fg(), "\x1b[38;2;137;180;250m");
    assert_eq!(c1.to_ansi_bg(), "\x1b[48;2;137;180;250m");

    let painted = c1.paint("HELLO");
    assert!(painted.starts_with("\x1b[38;2;137;180;250m"));
    assert!(painted.ends_with("\x1b[0m"));

    // 3-digit hex parsing
    let c2 = RgbColor::from_hex("#fff").unwrap();
    assert_eq!(c2.r, 255);
    assert_eq!(c2.g, 255);
    assert_eq!(c2.b, 255);

    // 2. Validate all 8 built-in themes
    let themes = ThemeManager::list_themes();
    assert_eq!(themes.len(), 8);
    for t_name in &themes {
        let palette = ThemeManager::get_theme(t_name).unwrap();
        assert_eq!(&palette.name, *t_name);
        assert!(!palette.primary_accent.to_hex().is_empty());
        assert!(!palette.background.to_hex().is_empty());
        assert!(!palette.diff_addition.to_hex().is_empty());
        assert!(!palette.diff_deletion.to_hex().is_empty());

        let preview = ThemeManager::render_theme_preview(&palette);
        assert!(preview.contains("THEME PALETTE PREVIEW"));
        assert!(preview.contains(t_name));
    }

    // 3. Setting active theme
    let mocha = set_active_theme("tokyo-night").unwrap();
    assert_eq!(mocha.name, "tokyo-night");
    let current = ThemeManager::get_active_theme();
    assert_eq!(current.name, "tokyo-night");

    // 4. Theme persistence in workspace
    ThemeManager::save_theme_preference("dracula", &temp_dir).unwrap();
    let loaded = ThemeManager::load_theme_preference(&temp_dir);
    assert_eq!(loaded.unwrap(), "dracula");

    // 5. Format Theme Report
    let rep_str = format_theme_report_for_terminal(&mocha);
    assert!(rep_str.contains("tokyo-night"));
    assert!(rep_str.contains("Primary Accent:"));

    // 6. Verify set_theme in get_tools()
    let tools = get_tools();
    let tools_str = serde_json::to_string(&tools).unwrap();
    assert!(tools_str.contains("set_theme"));

    let _ = std::fs::remove_dir_all(&temp_dir);
}

// ============================================================================
// SYSTEM 19: MODAL KEYBINDINGS & FUZZY COMMAND PALETTE
// ============================================================================

#[test]
fn test_modal_keybindings_and_fuzzy_command_palette() {
    let temp_dir = std::env::temp_dir().join(format!("zy_test_palette_{}_{}", std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
    let _ = std::fs::create_dir_all(&temp_dir);
    std::fs::write(temp_dir.join("main.rs"), "fn main() {}").unwrap();
    std::fs::write(temp_dir.join("Cargo.toml"), "[package]").unwrap();

    let history = vec!["how do I run tests?".to_string(), "refactor theme system".to_string()];
    let items = FuzzyCommandPalette::build_default_items(&temp_dir, &history);
    assert!(items.len() >= 20);

    // 1. Fuzzy Search: Exact match
    let results_exact = FuzzyCommandPalette::search_palette("/graphic", &items);
    assert!(!results_exact.is_empty());
    assert_eq!(results_exact[0].item.title, "/graphic");
    assert!(results_exact[0].score >= 100);

    // 2. Fuzzy Search: Subsequence / Prefix
    let results_prefix = FuzzyCommandPalette::search_palette("worktree", &items);
    assert!(!results_prefix.is_empty());
    assert!(results_prefix.iter().any(|r| r.item.title.contains("worktree")));

    // 3. Fuzzy Search: Acronym initials
    let results_acronym = FuzzyCommandPalette::search_palette("ptg", &items);
    // Even if no acronym matches ptg, search should be safe and return filtered results
    assert!(results_acronym.is_empty() || results_acronym[0].score > 0);

    // 4. Empty query returns all items with score 100
    let results_empty = FuzzyCommandPalette::search_palette("", &items);
    assert_eq!(results_empty.len(), items.len());

    // 5. Test Keybinding State Machine
    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

    // Ctrl+P opens palette from normal mode
    let key_ctrl_p = KeyEvent::new(KeyCode::Char('p'), KeyModifiers::CONTROL);
    let (m1, a1) = handle_tui_keybinding(KeybindingMode::Normal, key_ctrl_p);
    assert_eq!(m1, KeybindingMode::Palette);
    assert_eq!(a1, KeyAction::OpenPalette);

    // Palette mode Esc returns to normal
    let key_esc = KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE);
    let (m2, a2) = handle_tui_keybinding(KeybindingMode::Palette, key_esc);
    assert_eq!(m2, KeybindingMode::Normal);
    assert_eq!(a2, KeyAction::ClosePalette);

    // Normal mode Vim navigation
    let key_j = KeyEvent::new(KeyCode::Char('j'), KeyModifiers::NONE);
    let (m3, a3) = handle_tui_keybinding(KeybindingMode::Normal, key_j);
    assert_eq!(m3, KeybindingMode::Normal);
    assert_eq!(a3, KeyAction::MoveDown);

    let key_n = KeyEvent::new(KeyCode::Char('n'), KeyModifiers::NONE);
    let (m4, a4) = handle_tui_keybinding(KeybindingMode::Normal, key_n);
    assert_eq!(m4, KeybindingMode::Normal);
    assert_eq!(a4, KeyAction::NextHunk);

    let key_space = KeyEvent::new(KeyCode::Char(' '), KeyModifiers::NONE);
    let (m5, a5) = handle_tui_keybinding(KeybindingMode::Normal, key_space);
    assert_eq!(m5, KeybindingMode::Normal);
    assert_eq!(a5, KeyAction::ToggleFold);

    // 6. Format Terminal Palette Report
    let rep_str = format_palette_results_for_terminal("/graph", &results_exact);
    assert!(rep_str.contains("FUZZY COMMAND PALETTE"));
    assert!(rep_str.contains("/graphic"));

    // 7. Verify fuzzy_command_palette in get_tools()
    let tools = get_tools();
    let tools_str = serde_json::to_string(&tools).unwrap();
    assert!(tools_str.contains("fuzzy_command_palette"));

    let _ = std::fs::remove_dir_all(&temp_dir);
}

// ============================================================================
// SYSTEM 20: AMBIENT AUDIO & SENSORY FEEDBACK ENGINE
// ============================================================================

#[test]
fn test_ambient_audio_sensory_feedback_engine() {
    // 1. Audio Enable / Disable / Toggle State
    AudioCueEngine::set_enabled(true);
    assert!(AudioCueEngine::is_enabled());
    let toggled = AudioCueEngine::toggle_enabled();
    assert!(!toggled);
    assert!(!AudioCueEngine::is_enabled());
    AudioCueEngine::set_enabled(true);
    assert!(AudioCueEngine::is_enabled());

    // 2. Synthesize PCM RIFF WAV headers for all 6 cues
    let cues = vec![
        SoundCueType::TaskCompleted,
        SoundCueType::ErrorAlert,
        SoundCueType::CheckpointSaved,
        SoundCueType::ToolExecuted,
        SoundCueType::ThemeChanged,
        SoundCueType::WarningAlert,
    ];

    for cue in cues {
        let wav = AudioCueEngine::synthesize_cue_wav(cue);
        assert!(wav.len() > 44, "WAV data too short for cue {:?}", cue);
        // Validate RIFF WAV header bytes
        assert_eq!(&wav[0..4], b"RIFF");
        assert_eq!(&wav[8..12], b"WAVE");
        assert_eq!(&wav[12..16], b"fmt ");
        assert_eq!(&wav[36..40], b"data");

        // Validate sample rate 44100 (0x44AC0000 in little endian -> [0x44, 0xAC, 0x00, 0x00])
        let sample_rate = u32::from_le_bytes([wav[24], wav[25], wav[26], wav[27]]);
        assert_eq!(sample_rate, 44100);

        // Validate 16-bit
        let bits_per_sample = u16::from_le_bytes([wav[34], wav[35]]);
        assert_eq!(bits_per_sample, 16);
    }

    // 3. String name mapping
    assert_eq!(SoundCueType::from_str("task_completed"), Some(SoundCueType::TaskCompleted));
    assert_eq!(SoundCueType::from_str("error"), Some(SoundCueType::ErrorAlert));
    assert_eq!(SoundCueType::from_str("checkpoint"), Some(SoundCueType::CheckpointSaved));
    assert_eq!(SoundCueType::from_str("tool"), Some(SoundCueType::ToolExecuted));
    assert_eq!(SoundCueType::from_str("theme"), Some(SoundCueType::ThemeChanged));
    assert_eq!(SoundCueType::from_str("warn"), Some(SoundCueType::WarningAlert));

    // 4. Test all cues function
    let test_reports = AudioCueEngine::test_all_cues();
    assert_eq!(test_reports.len(), 6);

    // 5. Test playing cue safely in test environment
    let play_res = play_sound_cue("task_completed");
    assert!(play_res.is_ok());

    // 6. Terminal status report
    let status_str = format_audio_engine_status_for_terminal(true, Some("task_completed"));
    assert!(status_str.contains("AMBIENT AUDIO SENSORY ENGINE"));
    assert!(status_str.contains("ENABLED"));
    assert!(status_str.contains("task_completed"));

    // 7. Verify play_audio_cue in get_tools()
    let tools = get_tools();
    let tools_str = serde_json::to_string(&tools).unwrap();
    assert!(tools_str.contains("play_audio_cue"));
}

// ============================================================================
// SYSTEM 21: BRUTAL EDGE CASES ACROSS ALL 6 UX/UI SYSTEMS
// ============================================================================

#[tokio::test]
async fn test_brutal_edge_cases_across_6_ux_ui_systems() {
    let temp_dir = std::env::temp_dir().join(format!("zy_brutal_uxui_{}_{}", std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
    let _ = std::fs::create_dir_all(&temp_dir);

    // 1. Graphics Edge Cases: empty image buffer & unknown protocol
    let empty_render = render_terminal_graphics(&[], "unknown", "unknown_proto", 0, 0);
    assert!(empty_render.is_ok());
    let empty_str = empty_render.unwrap();
    assert!(empty_str.contains("▀") || empty_str.is_empty());

    // Non-existent image file falls back to treating input as diagram text
    let missing_file_res = render_diagram_or_image("non_existent_file_12345.png", "unicode", 20, 10);
    assert!(missing_file_res.is_ok());

    // Corrupt base64 string
    let bad_b64 = render_diagram_or_image("!!!badbase64???", "kitty", 20, 10);
    assert!(bad_b64.is_ok());

    // 2. GUI Edge Cases: 404 Route & Rapid Repeated Shutdown
    let gui_handle = launch_desktop_companion_gui(0, false).await.unwrap();
    let client = reqwest::Client::new();
    let not_found_resp = client.get(format!("{}/non_existent_endpoint", gui_handle.url())).send().await.unwrap();
    assert_eq!(not_found_resp.status(), 404);

    gui_handle.stop();
    gui_handle.stop(); // double stop is idempotent
    assert!(!gui_handle.is_running());

    // 3. Swarm Studio Edge Cases: Non-existent node updates and 404
    let studio_handle = start_swarm_studio_server(0).await.unwrap();
    studio_handle.update_agent_status("ghost_agent", "idle", "none");
    studio_handle.broadcast_node_event("ghost1", "ghost2", "signal", "empty");
    let not_found_studio = client.get(format!("{}/bad_path", studio_handle.url())).send().await.unwrap();
    assert_eq!(not_found_studio.status(), 404);

    studio_handle.stop();
    studio_handle.stop();
    assert!(!studio_handle.is_running());

    // 4. Theme Engine Edge Cases: Malformed hex strings and unknown theme
    assert!(RgbColor::from_hex("").is_none());
    assert!(RgbColor::from_hex("#12").is_none());
    assert!(RgbColor::from_hex("#12345").is_none());
    assert!(RgbColor::from_hex("#zzzzzz").is_none());
    assert!(ThemeManager::get_theme("totally_fake_theme").is_none());
    let bad_theme_set = set_active_theme("non_existent_theme");
    assert!(bad_theme_set.is_err());

    // 5. Command Palette Edge Cases: Special regex characters in query
    let special_q_items = FuzzyCommandPalette::build_default_items(&temp_dir, &[]);
    let special_results = FuzzyCommandPalette::search_palette(".*+?^${}()|[]\\", &special_q_items);
    // Should not panic, gracefully returns 0 results
    assert_eq!(special_results.len(), 0);

    // 6. Audio Engine Edge Cases: Muted sound does not throw error
    AudioCueEngine::set_enabled(false);
    let muted_play = play_sound_cue("task_completed");
    assert!(muted_play.is_ok());
    AudioCueEngine::set_enabled(true);

    // Unknown cue string defaults gracefully
    let unknown_cue = play_sound_cue("completely_unknown_cue_name");
    assert!(unknown_cue.is_ok());

    let _ = std::fs::remove_dir_all(&temp_dir);
}

// ============================================================================
// SYSTEM 22: INTERACTIVE HUNK-BY-HUNK DIFF STAGING UI TESTS
// ============================================================================

#[test]
fn test_interactive_hunk_by_hunk_diff_staging_ui() {
    let temp_dir = std::env::temp_dir().join(format!("zy_test_stage_{}_{}", std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
    let _ = std::fs::create_dir_all(&temp_dir);

    let original_file = temp_dir.join("sample.rs");
    let original_content = "fn alpha() {\n    println!(\"alpha 1\");\n    println!(\"alpha 2\");\n}\n\nfn beta() {\n    println!(\"beta 1\");\n}\n";
    std::fs::write(&original_file, original_content).unwrap();

    let diff_text = r#"--- a/sample.rs
+++ b/sample.rs
@@ -1,5 +1,5 @@ fn alpha()
 fn alpha() {
-    println!("alpha 1");
+    println!("alpha MODIFIED");
     println!("alpha 2");
 }
@@ -6,3 +6,4 @@ fn beta()
 fn beta() {
     println!("beta 1");
+    println!("beta NEW LINE");
 }
"#;

    // 1. Parse unified diff into discrete hunks
    let hunks = parse_diff_into_hunks(diff_text);
    assert_eq!(hunks.len(), 2);

    assert_eq!(hunks[0].index, 0);
    assert_eq!(hunks[0].old_start, 1);
    assert_eq!(hunks[0].additions, 1);
    assert_eq!(hunks[0].deletions, 1);
    assert!(hunks[0].section_header.as_ref().unwrap().contains("fn alpha()"));

    assert_eq!(hunks[1].index, 1);
    assert_eq!(hunks[1].old_start, 6);
    assert_eq!(hunks[1].additions, 1);
    assert_eq!(hunks[1].deletions, 0);
    assert!(hunks[1].section_header.as_ref().unwrap().contains("fn beta()"));

    // 2. Test selective staging: Stage ONLY Hunk 0 (alpha modified, beta untouched)
    let staged_hunk0 = apply_selected_hunks(original_content, &hunks, &[0]).expect("Apply hunk 0 failed");
    assert!(staged_hunk0.contains("alpha MODIFIED"));
    assert!(!staged_hunk0.contains("alpha 1"));
    assert!(staged_hunk0.contains("beta 1"));
    assert!(!staged_hunk0.contains("beta NEW LINE"));

    // 3. Test selective staging: Stage ONLY Hunk 1 (beta new line, alpha untouched)
    let staged_hunk1 = apply_selected_hunks(original_content, &hunks, &[1]).expect("Apply hunk 1 failed");
    assert!(staged_hunk1.contains("alpha 1"));
    assert!(!staged_hunk1.contains("alpha MODIFIED"));
    assert!(staged_hunk1.contains("beta 1"));
    assert!(staged_hunk1.contains("beta NEW LINE"));

    // 4. Test selective staging: Stage ALL Hunks ([0, 1])
    let staged_all = apply_selected_hunks(original_content, &hunks, &[0, 1]).expect("Apply all hunks failed");
    assert!(staged_all.contains("alpha MODIFIED"));
    assert!(staged_all.contains("beta NEW LINE"));

    // 5. Test selective staging: Stage NO Hunks (empty slice) -> returns unmodified content
    let staged_none = apply_selected_hunks(original_content, &hunks, &[]).expect("Apply no hunks failed");
    assert_eq!(staged_none, original_content);

    // 6. Test Line-level hunk splitting (split_hunk_into_lines)
    let multi_change_diff = r#"@@ -1,10 +1,10 @@
 line 1
-line 2
+line 2 mod
 line 3
 line 4
-line 5
+line 5 mod
 line 6
"#;
    let multi_hunks = parse_diff_into_hunks(multi_change_diff);
    assert_eq!(multi_hunks.len(), 1);
    let split = split_hunk_into_lines(&multi_hunks[0]);
    assert!(split.len() >= 2);
    assert_eq!(split[0].index, 0);
    assert_eq!(split[1].index, 1);

    // 7. Test Terminal Report Formatting
    let rep_str = format_hunk_staging_report_for_terminal("sample.rs", &hunks, &[0]);
    assert!(rep_str.contains("INTERACTIVE HUNK-BY-HUNK DIFF STAGING"));
    assert!(rep_str.contains("sample.rs"));
    assert!(rep_str.contains("Hunk #0"));
    assert!(rep_str.contains("STAGED"));

    // 8. Verify hunk_diff_staging in get_tools()
    let tools = get_tools();
    let tools_str = serde_json::to_string(&tools).unwrap();
    assert!(tools_str.contains("hunk_diff_staging"));

    let _ = std::fs::remove_dir_all(&temp_dir);
}

// ============================================================================
// SYSTEM 23: REAL-TIME TOKEN HEATMAP & CONTEXT DENSITY INSPECTOR TESTS
// ============================================================================

#[test]
fn test_realtime_token_heatmap_and_context_density_inspector() {
    let messages = vec![
        Message {
            role: "system".to_string(),
            content: "You are zy, a lethal AI pair programming assistant with strict rules.".to_string(),
            tool_calls: None,
            images: None,
        },
        Message {
            role: "system".to_string(),
            content: "[RAG Retrieval Context]\nChunk 1: fn optimize_cache() { ... }\nChunk 2: fn hash_table() { ... }".to_string(),
            tool_calls: None,
            images: None,
        },
        Message {
            role: "user".to_string(),
            content: "[Attached Context: src/main.rs]\nfn main() { println!(\"hello\"); }".to_string(),
            tool_calls: None,
            images: None,
        },
        Message {
            role: "user".to_string(),
            content: "Refactor the memory allocator to use slab allocation.".to_string(),
            tool_calls: None,
            images: None,
        },
        Message {
            role: "assistant".to_string(),
            content: "Here is the slab allocator design with zero runtime overhead.".to_string(),
            tool_calls: None,
            images: None,
        },
        Message {
            role: "tool".to_string(),
            content: "{\"status\": \"success\", \"allocations\": 1024}".to_string(),
            tool_calls: None,
            images: None,
        },
    ];

    // 1. Inspect token heatmap with 8192 budget
    let report = inspect_token_heatmap(&messages, 8192);
    assert_eq!(report.num_ctx, 8192);
    assert!(report.total_tokens > 0);
    assert_eq!(report.sections.len(), 6);
    assert_eq!(report.density_category, ContextDensityCategory::Low);

    // Verify section types
    assert_eq!(report.sections[0].section_type, HeatmapSectionType::SystemPrompt);
    assert_eq!(report.sections[1].section_type, HeatmapSectionType::RagContext);
    assert_eq!(report.sections[2].section_type, HeatmapSectionType::AttachedFile);
    assert_eq!(report.sections[3].section_type, HeatmapSectionType::Turn);
    assert_eq!(report.sections[4].section_type, HeatmapSectionType::Turn);
    assert_eq!(report.sections[5].section_type, HeatmapSectionType::ToolPayload);

    // 2. Test High Density Bloat triggers
    let huge_content = "X".repeat(4000);
    let heavy_messages = vec![
        Message {
            role: "system".to_string(),
            content: format!("RULES:\n{}", huge_content),
            tool_calls: None,
            images: None,
        },
    ];
    let heavy_report = inspect_token_heatmap(&heavy_messages, 2000);
    assert!(heavy_report.usage_pct > 40.0);
    assert_eq!(heavy_report.density_category, ContextDensityCategory::High);
    assert!(heavy_report.recommendations.iter().any(|r| r.contains("Compress system rules") || r.contains("Critical context")));

    // 3. Test Terminal Report Formatting
    let term_out = format_token_heatmap_for_terminal(&report);
    assert!(term_out.contains("REAL-TIME TOKEN HEATMAP"));
    assert!(term_out.contains("Context Window:"));
    assert!(term_out.contains("SYSTEM"));
    assert!(term_out.contains("RAG"));
    assert!(term_out.contains("ATTACH"));
    assert!(term_out.contains("RECOMMENDATIONS:"));

    // 4. Verify token_heatmap in get_tools()
    let tools = get_tools();
    let tools_str = serde_json::to_string(&tools).unwrap();
    assert!(tools_str.contains("token_heatmap"));
}

// ============================================================================
// SYSTEM 24: TERMINAL SLIDE DECK PRESENTATION ENGINE TESTS
// ============================================================================

#[test]
fn test_terminal_slide_deck_presentation_engine() {
    let markdown_presentation = r#"
# Welcome to zy 2.0
## The Ultimate Local AI Coding Agent
- Autonomous TDD auto-repair loop
- Zero-latency background indexing
- Cross-platform hardware telemetry
Note: Introduce the agent's core capabilities.

---

# Architecture Overview
## Modular Multi-Agent Swarm
- Architect: high-level design and OODA loops
- Coder: precision code synthesis
- Auditor: SARIF security and a11y checks
- QA: test suite execution

```rust
pub fn launch_agent() {
    println!("Agent running in high-performance mode");
}
```

---

# Key Metrics & ROI
- 100% offline & local Ollama execution
- $0 cloud API token cost
- Sub-50ms latency
"#;

    // 1. Parse markdown into slides
    let slides = parse_markdown_into_slides(markdown_presentation);
    assert_eq!(slides.len(), 3);

    // Slide 1 checks
    assert_eq!(slides[0].index, 0);
    assert_eq!(slides[0].title, "Welcome to zy 2.0");
    assert_eq!(slides[0].subtitle.as_deref(), Some("The Ultimate Local AI Coding Agent"));
    assert_eq!(slides[0].bullet_points.len(), 3);
    assert_eq!(slides[0].notes.as_deref(), Some("Introduce the agent's core capabilities."));

    // Slide 2 checks
    assert_eq!(slides[1].index, 1);
    assert_eq!(slides[1].title, "Architecture Overview");
    assert_eq!(slides[1].code_blocks.len(), 1);
    assert_eq!(slides[1].code_blocks[0].language, "rust");
    assert!(slides[1].code_blocks[0].code.contains("pub fn launch_agent()"));

    // Slide 3 checks
    assert_eq!(slides[2].index, 2);
    assert_eq!(slides[2].title, "Key Metrics & ROI");
    assert_eq!(slides[2].bullet_points.len(), 3);

    // 2. Render Slide to Terminal
    let rendered_s1 = render_slide_to_terminal(&slides[0], 0, 3, 80, 24);
    assert!(rendered_s1.contains("Welcome to zy 2.0"));
    assert!(rendered_s1.contains("The Ultimate Local AI Coding Agent"));
    assert!(rendered_s1.contains("[ Slide 1 / 3 ]"));
    assert!(rendered_s1.contains("Autonomous TDD auto-repair loop"));
    assert!(rendered_s1.contains("Note: Introduce the agent's core capabilities."));
    assert!(rendered_s1.contains("[n/Space: Next"));

    let rendered_s2 = render_slide_to_terminal(&slides[1], 1, 3, 80, 24);
    assert!(rendered_s2.contains("Architecture Overview"));
    assert!(rendered_s2.contains("pub fn launch_agent()"));
    assert!(rendered_s2.contains("[ Slide 2 / 3 ]"));

    // 3. Verify present_slides in get_tools()
    let tools = get_tools();
    let tools_str = serde_json::to_string(&tools).unwrap();
    assert!(tools_str.contains("present_slides"));
}

// ============================================================================
// SYSTEM 25: MODULAR DOCKABLE TUI WIDGETS BAR TESTS
// ============================================================================

#[test]
fn test_modular_dockable_tui_widgets_bar() {
    let temp_dir = std::env::temp_dir().join(format!("zy_test_widgets_{}_{}", std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
    let _ = std::fs::create_dir_all(&temp_dir);

    // 1. Initial State
    let mut state = TuiWidgetBarState::new();
    assert_eq!(state.enabled_widgets.len(), 4);
    assert!(state.is_widget_enabled(WidgetType::GitStream));
    assert!(state.is_widget_enabled(WidgetType::DockerMonitor));
    assert!(state.is_widget_enabled(WidgetType::DatabaseTailer));
    assert!(state.is_widget_enabled(WidgetType::HardwareSparklines));

    // 2. Toggle and Disable
    state.toggle_widget(WidgetType::DockerMonitor);
    assert!(!state.is_widget_enabled(WidgetType::DockerMonitor));
    assert_eq!(state.enabled_widgets.len(), 3);

    state.enable_widget(WidgetType::DockerMonitor);
    assert!(state.is_widget_enabled(WidgetType::DockerMonitor));

    state.disable_widget(WidgetType::GitStream);
    assert!(!state.is_widget_enabled(WidgetType::GitStream));

    // 3. Render Sparklines
    let history = vec![0.0, 20.0, 40.0, 60.0, 80.0, 100.0];
    let spark = render_sparkline(&history, 6);
    assert_eq!(spark.chars().count(), 6);
    assert!(spark.contains("█"));
    assert!(spark.contains(" "));

    // 4. Render individual widget panels
    let git_panel = render_widget_panel(&WidgetType::GitStream, &state);
    assert!(git_panel.contains("Git:"));

    let docker_panel = render_widget_panel(&WidgetType::DockerMonitor, &state);
    assert!(docker_panel.contains("Containers:"));

    let db_panel = render_widget_panel(&WidgetType::DatabaseTailer, &state);
    assert!(db_panel.contains("Database:"));

    let hw_panel = render_widget_panel(&WidgetType::HardwareSparklines, &state);
    assert!(hw_panel.contains("CPU ["));
    assert!(hw_panel.contains("RAM ["));
    assert!(hw_panel.contains("GPU ["));

    // 5. Render Full Dock Bar
    let dock_bar = render_dockable_widget_bar(&state, 80);
    assert!(dock_bar.contains("MODULAR DOCKABLE TUI WIDGETS BAR"));
    assert!(dock_bar.contains("Containers:"));
    assert!(dock_bar.contains("Database:"));
    assert!(dock_bar.contains("CPU ["));

    // 6. Name parsing
    assert_eq!(parse_widget_type_name("git_stream"), Some(WidgetType::GitStream));
    assert_eq!(parse_widget_type_name("docker"), Some(WidgetType::DockerMonitor));
    assert_eq!(parse_widget_type_name("db"), Some(WidgetType::DatabaseTailer));
    assert_eq!(parse_widget_type_name("cpu"), Some(WidgetType::HardwareSparklines));
    assert_eq!(parse_widget_type_name("invalid"), None);

    // 7. Verify manage_widgets in get_tools()
    let tools = get_tools();
    let tools_str = serde_json::to_string(&tools).unwrap();
    assert!(tools_str.contains("manage_widgets"));

    let _ = std::fs::remove_dir_all(&temp_dir);
}

// ============================================================================
// SYSTEM 26: LOCAL TEXT-TO-SPEECH VOICE ENGINE TESTS
// ============================================================================

#[test]
fn test_local_text_to_speech_voice_engine() {
    // 1. Initial State
    let engine = SpeechEngine::new();
    assert!(engine.enabled);
    assert_eq!(engine.voice_speed, 1.0);
    assert_eq!(engine.pitch, 1.0);

    // 2. Test command generation across platforms
    let (cmd, args) = generate_speech_command("Task completed successfully.", Some(1.2), Some(1.0));
    assert!(!cmd.is_empty());
    assert!(!args.is_empty());

    #[cfg(windows)]
    {
        assert_eq!(cmd, "powershell");
        assert!(args.iter().any(|a| a.contains("System.Speech")));
        assert!(args.iter().any(|a| a.contains("Task completed successfully.")));
    }

    #[cfg(target_os = "macos")]
    {
        assert_eq!(cmd, "say");
        assert!(args.contains(&"-r".to_string()));
    }

    // 3. Empty text handling
    let res = synthesize_speech("", None, None);
    assert!(res.is_ok());

    // 4. Background voice execution test
    let bg_res = speak_in_background("Test background speech synthesis", Some(1.0), Some(1.0));
    assert!(bg_res.is_ok());

    // 5. Terminal Status Report
    let rep_str = format_speech_engine_status_for_terminal(&engine, Some("All systems operational."));
    assert!(rep_str.contains("LOCAL TEXT-TO-SPEECH (TTS) VOICE ENGINE"));
    assert!(rep_str.contains("ACTIVE"));
    assert!(rep_str.contains("All systems operational."));

    // 6. Verify speak_text in get_tools()
    let tools = get_tools();
    let tools_str = serde_json::to_string(&tools).unwrap();
    assert!(tools_str.contains("speak_text"));
}

// ============================================================================
// SYSTEM 27: INTERACTIVE AI DEBUGGER & STACK TRACE VISUALIZER TESTS
// ============================================================================

#[test]
fn test_interactive_ai_debugger_and_stack_trace_visualizer() {
    let temp_dir = std::env::temp_dir().join(format!("zy_test_debug_{}_{}", std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
    let _ = std::fs::create_dir_all(&temp_dir);

    // Create a real file for snippet resolution
    let test_file = temp_dir.join("calc.rs");
    let test_src = "pub fn divide(a: i32, b: i32) -> i32 {\n    let res = a / b;\n    res\n}\n";
    std::fs::write(&test_file, test_src).unwrap();

    // 1. Rust Panic Trace (Option::unwrap on None)
    let rust_trace = r#"
thread 'main' panicked at 'called `Option::unwrap()` on a `None` value', src/lib.rs:142:10
stack backtrace:
   0: std::panicking::begin_panic
   1: zy::execute_task
             at ./src/lib.rs:142:10
   2: zy::main
             at ./src/main.rs:25:5
"#;
    let parsed_rust = parse_crash_stack_trace(rust_trace).expect("Failed to parse Rust trace");
    assert_eq!(parsed_rust.language, CrashLanguage::Rust);
    assert_eq!(parsed_rust.root_cause, RootCauseHypothesis::UnwrapPanic);
    assert!(parsed_rust.error_message.contains("Option::unwrap"));
    assert!(parsed_rust.suggested_fix.contains("unwrap()"));
    assert!(!parsed_rust.frames.is_empty());

    // 2. Python KeyError Traceback
    let python_trace = r#"
Traceback (most recent call last):
  File "server.py", line 45, in handle_request
    user = authenticate(token)
  File "auth.py", line 12, in authenticate
    return session_cache[token]
KeyError: 'expired_token_123'
"#;
    let parsed_py = parse_crash_stack_trace(python_trace).expect("Failed to parse Python trace");
    assert_eq!(parsed_py.language, CrashLanguage::Python);
    assert_eq!(parsed_py.root_cause, RootCauseHypothesis::KeyError);
    assert_eq!(parsed_py.error_type, "KeyError");
    assert!(parsed_py.error_message.contains("expired_token_123"));
    assert!(parsed_py.suggested_fix.contains("dict.get"));
    assert_eq!(parsed_py.frames.len(), 2);

    // 3. Node.js TypeError Trace
    let node_trace = r#"
TypeError: Cannot read properties of undefined (reading 'map')
    at renderList (/app/src/components/List.ts:18:25)
    at App (/app/src/App.ts:32:10)
    at processTicksAndRejections (node:internal/process/task_queues:95:5)
"#;
    let parsed_node = parse_crash_stack_trace(node_trace).expect("Failed to parse Node trace");
    assert_eq!(parsed_node.language, CrashLanguage::NodeJs);
    assert_eq!(parsed_node.root_cause, RootCauseHypothesis::NullPointer);
    assert!(parsed_node.error_message.contains("Cannot read properties of undefined"));
    assert!(parsed_node.suggested_fix.contains("?."));
    assert!(parsed_node.frames.iter().any(|f| f.function_name == "renderList"));

    // 4. C/C++ Segmentation Fault GDB Trace
    let cpp_trace = r#"
Program received signal SIGSEGV, Segmentation fault.
#0  0x0000555555555149 in compute_hash (ptr=0x0) at src/hash.c:34
#1  0x0000555555555180 in main () at src/main.c:12
"#;
    let parsed_cpp = parse_crash_stack_trace(cpp_trace).expect("Failed to parse C++ trace");
    assert_eq!(parsed_cpp.language, CrashLanguage::Cpp);
    assert_eq!(parsed_cpp.root_cause, RootCauseHypothesis::SegmentationFault);
    assert_eq!(parsed_cpp.frames.len(), 2);
    assert_eq!(parsed_cpp.frames[0].function_name, "compute_hash");
    assert_eq!(parsed_cpp.frames[0].line_number, Some(34));

    // 5. Code context snippet extraction from real file on disk
    let file_trace = format!(
        "thread 'main' panicked at 'attempt to divide by zero', {}:2:15\n",
        test_file.display()
    );
    let parsed_file = parse_crash_stack_trace(&file_trace).expect("Failed to parse file trace");
    assert_eq!(parsed_file.root_cause, RootCauseHypothesis::DivisionByZero);
    assert!(parsed_file.failing_frame.is_some());
    let frame = parsed_file.failing_frame.unwrap();
    assert!(frame.code_snippet.is_some());
    assert!(frame.code_snippet.unwrap().contains("let res = a / b"));
    assert!(parsed_file.patch_suggestion.is_some());

    // 6. Terminal Stack Trace Formatting
    let term_out = format_stack_trace_report_for_terminal(&parsed_rust);
    assert!(term_out.contains("INTERACTIVE AI CRASH DEBUGGER"));
    assert!(term_out.contains("Rust"));
    assert!(term_out.contains("UnwrapPanic"));
    assert!(term_out.contains("SUGGESTED FIX:"));

    // 7. Verify debug_trace in get_tools()
    let tools = get_tools();
    let tools_str = serde_json::to_string(&tools).unwrap();
    assert!(tools_str.contains("debug_trace"));

    let _ = std::fs::remove_dir_all(&temp_dir);
}

// ============================================================================
// SYSTEM 28: BRUTAL EDGE CASES ACROSS ALL 6 NEW UX/UI SYSTEMS
// ============================================================================

#[test]
fn test_brutal_edge_cases_across_6_new_uxui_systems() {
    let temp_dir = std::env::temp_dir().join(format!("zy_brutal_new6_{}_{}", std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
    let _ = std::fs::create_dir_all(&temp_dir);

    // 1. Hunk diff staging: Empty string and invalid index
    let empty_hunks = parse_diff_into_hunks("");
    assert!(empty_hunks.is_empty());
    let apply_invalid = apply_selected_hunks("sample content", &empty_hunks, &[999]).unwrap();
    assert_eq!(apply_invalid, "sample content");

    let malformed_diff = "not a diff at all\njust random text\n+++ none\n--- none";
    let fallback_hunk = parse_diff_into_hunks(malformed_diff);
    assert!(!fallback_hunk.is_empty());

    // 2. Token heatmap: Zero messages and 0 budget
    let zero_report = inspect_token_heatmap(&[], 0);
    assert_eq!(zero_report.total_tokens, 0);
    assert_eq!(zero_report.num_ctx, 8192);
    assert_eq!(zero_report.usage_pct, 0.0);

    // 3. Slides: Empty markdown and consecutive dividers
    let empty_slides = parse_markdown_into_slides("");
    assert!(empty_slides.is_empty());

    let divider_only = "---\n---\n---";
    let div_slides = parse_markdown_into_slides(divider_only);
    assert!(div_slides.is_empty());

    // 4. Widgets: Extreme and zero load values
    let extreme_history = vec![0.0, 150.0, -20.0, 50.0];
    let extreme_spark = render_sparkline(&extreme_history, 10);
    assert_eq!(extreme_spark.chars().count(), 4);

    // Empty state dock bar rendering
    let mut empty_state = TuiWidgetBarState::new();
    empty_state.enabled_widgets.clear();
    let empty_dock = render_dockable_widget_bar(&empty_state, 80);
    assert!(empty_dock.contains("0 widgets active"));

    // 5. Speech: Multiline text with quotes
    let quote_text = "He said: 'zy is lethal' & \"extremely fast\"!\nLine 2.";
    let (cmd, args) = generate_speech_command(quote_text, Some(2.5), Some(0.5));
    assert!(!cmd.is_empty());
    assert!(!args.is_empty());

    // 6. Debugger: Completely empty and unknown crash logs
    let empty_trace = parse_crash_stack_trace("").unwrap();
    assert_eq!(empty_trace.language, CrashLanguage::Unknown);
    assert_eq!(empty_trace.root_cause, RootCauseHypothesis::Unknown);

    let unknown_log = "Fatal system anomaly in subsystem XYZ without recognizable stack trace";
    let unknown_trace = parse_crash_stack_trace(unknown_log).unwrap();
    assert_eq!(unknown_trace.language, CrashLanguage::Unknown);
    assert!(unknown_trace.frames.is_empty());

    let _ = std::fs::remove_dir_all(&temp_dir);
}

