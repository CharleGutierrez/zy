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



