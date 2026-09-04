import re
import json

with open('src/lib.rs', 'r') as f:
    content = f.read()

funcs = [
    r'pub fn generate_openapi_sdk\(.*?\)\s*->\s*.*?\{.*?\n\}',
    r'pub fn find_dead_code_symbols\(.*?\)\s*->\s*.*?\{.*?\n\}',
    r'pub fn start_swarm_studio_server\(.*?\)\s*->\s*.*?\{.*?\n\}',
    r'pub fn query_spotlight\(.*?\)\s*->\s*.*?\{.*?\n\}',
    r'pub fn generate_analytics_report\(.*?\)\s*->\s*.*?\{.*?\n\}',
    r'pub fn calculate\(.*?\)\s*->\s*.*?\{.*?\n\}',
]

for func_regex in funcs:
    match = re.search(func_regex, content, flags=re.DOTALL)
    if match:
        print(f"FOUND: {match.group(0)[:100]}...")
    else:
        print(f"NOT FOUND: {func_regex}")

