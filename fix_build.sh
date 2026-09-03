#!/bin/bash
# Fix single_prompt signature
sed -i 's/force: bool,/force: bool,\n    executor: Option<String>,/g' src/main.rs

# Fix Message missing images field
sed -i 's/tool_calls: None,/tool_calls: None,\n                images: None,/g' src/main.rs
# But wait, sed on "tool_calls: None," might match multiple lines. Let's do it carefully.
