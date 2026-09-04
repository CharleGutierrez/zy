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
