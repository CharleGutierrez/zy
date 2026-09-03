# 🚀 zy Agent: Getting Started Tutorial

Welcome to `zy`! This tutorial will take you from zero to autonomous AI orchestration in 5 minutes.

## Lesson 1: Your First Chat
Simply boot the agent by running:
```bash
cargo run
```
You will be greeted by the **Interactive Dashboard**. Use your arrow keys to select `🚀 Start Chatting`, pick your installed model (e.g., `llama3.2`), and hit Enter!

## Lesson 2: Letting the Agent Edit Files
Type this into the chat:
`zy ❯ Create a file named hello.py that prints "Hello World"`

Because `zy` is in **Agent Mode**, it will autonomously use the `write_file` tool to create `hello.py` on your hard drive! 
*(Tip: Use `/agent on` if you forgot to enable it in the menu).*

## Lesson 3: Codebase RAG (Searching your code)
Have a massive project? Let `zy` memorize it.
1. Exit the chat (`/exit`).
2. Run `cargo run -- index .` to embed your entire project into the Vector Database.
3. Start a chat, type `/rag on`, and ask: `"Where is my database connection logic located?"`

## Lesson 4: The Chaos Monkey (Brutal Resilience)
Want to test if your tests actually work?
Type `/chaos` in the chat. `zy` will randomly delete 5 lines of code from a file in your project. See if you (or `zy`) can fix it! If things go too wrong, type `/undo` to instantly Git-revert.
