# `zy` Program Study & Architecture Map

## Overview
`zy` is an extremely ambitious, high-performance local AI Agent and Multi-Modal CLI tool written in Rust. It serves as a unified intelligent developer environment, tightly integrated with `Ollama` for local LLM inference. Its primary goal is to provide developers with a single binary that encompasses intelligent code generation, multi-agent swarm orchestration, static analysis, and a highly immersive User Experience (UX).

---

## ✊ The "Broke Developer" Manifesto & Philosophy
As outlined in the project's documentation, `zy` explicitly targets developers who want Copilot-level AI without the $50+/mo subscriptions or internet dependency.
*   **AiTuner Engine**: Dynamically profiles system hardware. If a user has <12GB RAM, it forces the AI into "ECO MODE" (throttling context windows and CPU threads to prevent Out-Of-Memory crashes). If it detects high-end hardware, it unleashes "TURBO MODE".
*   **100% Offline**: It champions privacy and accessibility, pointing out that your code never leaves your machine. 

---

## 🏗️ Core Architecture & Module Map

The codebase is highly modularized, separating concerns between AI logic, developer tooling, and user interfaces.

* **`src/main.rs` & `src/lib.rs`**
  The entry points of the application. `main.rs` handles the asynchronous `tokio` runtime initialization and dispatches execution, while `lib.rs` defines the massive `clap` CLI parser encompassing over 30 advanced subcommands.

* **`src/agent.rs`**
  The core AI loop. It handles streaming responses, dual-model speculative routing (scout vs. main model), and the "Swarm Mode" where an **Architect** model plans a task and an **Executor** model physically runs tools.

* **`src/rag.rs`**
  The Retrieval-Augmented Generation engine. It features the "Vella" zero-latency OS watcher daemon, embedding generation, and local vector indexing, allowing the LLM to instantly query the local codebase context.

* **`src/tools.rs` & `src/commands.rs`**
  The execution environment. `commands.rs` handles interactive slash commands (e.g., `/train`, `/exit`), while `tools.rs` houses the implementations for file modifications, bash execution, and system tools that the agent invokes.

* **`src/tuner.rs` & `src/ollama_optimizer.rs`**
  Performance and inference optimization. These modules actively tune Ollama configurations (`num_ctx`, `temperature`, thread allocation, KV cache reuse) based on the user's hardware to achieve maximum Tokens Per Second (TPS).

* **`src/tier3_ux.rs` & `src/ux_stack.rs`**
  The presentation layer. This is where `zy` breaks out of the standard CLI mold, rendering interactive TUI dashboards (`ratatui`), Web UI servers, TrueColor theme palettes, and even processing ambient audio feedback and terminal graphics.

---

## ⚡ Key Capabilities & Subsystems

`zy` aims to replace dozens of disparate developer tools by embedding AI into every workflow:

### 1. Multi-Agent Swarm Orchestration
Unlike simple chatbots, `zy` can run a Swarm. A user can define a `--swarm` goal, prompting `zy` to spawn an Architect agent to draft a JSON execution plan, which is then passed to Executor agents that iterate, compile, and debug code autonomously.

### 2. Deep Static Analysis & Refactoring
Through tools like `AstGrep`, `Prune`, `A11y`, and `Review`, `zy` parses the Abstract Syntax Tree (AST) of the workspace. It doesn't just guess using LLM weights; it uses deterministic logic (`syn` for Rust) to find dead code, flag accessibility violations, or find structural patterns before asking the LLM to fix them.

### 3. Hyper-Immersive Interfaces (The "UX Stack")
`zy` implements multiple UI paradigms:
* **TUI Dashboard**: A full-screen `ratatui` interface overlaying chat, logs, and system metrics.
* **Spotlight HUD**: A background daemon that provides an OS-level fuzzy-search overlay.
* **Web UI**: An embedded local web server to visualize complex Git graphs or Agent Swarm topologies that don't fit in the terminal.

### 4. Codebase Economics & Profiling
The `Stats` and `Bench` commands aggregate tokens generated and cross-reference them against cloud API pricing (like OpenAI or Anthropic) to calculate real-world cost savings achieved by running models locally.

---

## 🛡️ Cybersecurity: White Hat vs Black Hat

The documentation explicitly acknowledges the dual-use nature of an air-gapped, autonomous agent:
*   **The Black Hat Loophole**: Users can plug in uncensored models. If an uncensored model is run with `zy --agent --force`, the AI can be weaponized to autonomously run scripts or attempt lateral network movement with zero human intervention.
*   **The Blue Team Defender**: `zy` is a superpower for cybersecurity researchers. Because it is 100% offline, researchers can feed it live malware inside an air-gapped sandbox, allowing the AI to reverse-engineer the malware without the payload ever dialing home.
*   **The Kill Switch**: To mitigate risks, `zy` implements a strict interactive Safety Prompt before executing any system command, acting as the human-in-the-loop kill switch unless explicitly bypassed.

---

## ⚙️ Technical Stack Highlights
* **Language**: Rust (Strict, memory-safe, fearless concurrency)
* **Async Runtime**: `tokio` (Powers the concurrent RAG indexing and multi-agent network requests)
* **CLI Parser**: `clap` (Handles the massive tree of flags and subcommands)
* **AI Backend**: Local `Ollama` REST API (`reqwest`)
* **Terminal UI**: `ratatui`, `crossterm`, `colored`, `inquire`
* **Data Processing**: `syn` (AST), `rusqlite` (Metrics DB), `serde` (JSON/YAML)
