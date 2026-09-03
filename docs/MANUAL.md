# 📚 zy Agent: Command Reference Manual

## Command Line Flags
| Flag | Short | Description |
|------|-------|-------------|
| `--model` | `-m` | Specify the Ollama model to use (e.g., `llama3.2`). |
| `--system`| `-s` | Inject a custom system prompt/persona. |
| `--file`  | `-f` | Mount specific files (or images) directly into context. |
| `--agent` | `-a` | Enable autonomous tool execution (Bash/Files). |
| `--force` | `-F` | Disable safety confirmation prompts for dangerous tools. |
| `--rag`   | `-r` | Enable automatic vector-search against `.zy_rag_index.json`. |
| `--executor` | | Set the secondary model for Swarm Orchestration. |

## Interactive Slash Commands
Inside the `zy` REPL, you can hot-swap features mid-conversation:
- `/help` - Show available commands.
- `/clear` - Clear terminal & context window.
- `/save <name>` - Save session to `.zy_session_<name>.json`.
- `/model <name>` - Hot-swap the active brain.
- `/agent <on/off>` - Toggle Bash/File writing capabilities.
- `/rag <on/off>` - Toggle codebase search capability.
- `/executor <name>` - Engage Swarm Mode (Architect + Executor).
- `/strategist` - Force the AI to output an OODA Loop (`<STRATEGY>`) before acting.
- `/listen` - Hands-free Voice-to-Code via Whisper.
- `/evolve <req>` - Zy will read its own source code, rewrite it, and recompile.
- `/worker` - Automatically picks up local `.projectmem/issues/` and fixes them.
- `/sleep` - Deep memory compression (summarizes history into core memory).
- `/webhook <url>` - Set an endpoint for autonomous push notifications.
- `/undo` - Git reset the codebase to HEAD~1.
