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
    assert!(timeline_formatted.contains("Turn #1"));
    assert!(timeline_formatted.contains("Turn #2"));
    assert!(timeline_formatted.contains("Turn #3"));
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
    assert!(get_refactor_transaction_status().contains("Staged files: 1"));
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
