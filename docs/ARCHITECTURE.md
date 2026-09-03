# 🏗️ zy Architecture & Periodicals

## The AiTuner Subsystem
`zy` is designed to run on everything from a Raspberry Pi to a Threadripper. On boot, `sysinfo` scans bare-metal OS parameters.
- **ECO MODE**: Triggered if RAM < 12GB. Context window is restricted to 2048, and threads are halved to prevent OS lockup.
- **TURBO MODE**: Triggered on high-end rigs. Context expands to 8192, maximizing thread saturation.

## The Vella Zero-Latency OS Watcher
The `zy watch` daemon utilizes the `notify` OS crate. Instead of manual re-indexing, the daemon intercepts OS-level file `Save` events and dynamically routes them to the local embedding model (`nomic-embed-text`). This allows the agent to maintain 100% real-time synchronized codebase memory.
