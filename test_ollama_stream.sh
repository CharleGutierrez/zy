curl http://127.0.0.1:11434/api/chat -d '{
  "model": "qwen2.5-coder:1.5b",
  "stream": true,
  "messages": [{"role": "user", "content": "What is the weather in London?"}],
  "tools": [{"type": "function", "function": {"name": "get_weather", "description": "Get weather", "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}}}]
}'
