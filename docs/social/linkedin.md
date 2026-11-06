# LinkedIn post

What does it take to run AI on your own hardware and make it actually do work? I spent the last few weeks finding out. 🧠

I built a local AI system from the ground up on my laptop (RTX 4070), and the interesting part is the layers underneath, more than the app on top:

🔹 **Local inference engine.** Open-weight models run through llama.cpp, one isolated process per model. The OS cleans up those processes if anything crashes.

🔹 **Hardware scheduling.** A model can run on one GPU, be split layer-by-layer across an NVIDIA and an AMD GPU at once (via Vulkan), or fall back to the CPU, and it adapts when a GPU is unplugged.

🔹 **Measuring instead of assuming.** The system reads the inference engine's own logs and the operating system's per-process GPU counters. It reports exactly where every layer landed and how much video memory each model uses. (This started because Task Manager suggested my model was on the wrong GPU. It wasn't.)

🔹 **Constrained decoding.** Small models are unreliable at structured output. Converting a JSON schema into a grammar, so the model can only produce valid actions, turned a 7B model that described its work into one that does it.

🔹 **Agent loop and context management.** A reason → act → observe cycle with tool calling. Long sessions are summarised automatically while the original goal and recent steps are kept, so the model never runs out of context.

🔹 **Grounding.** When the model needs facts, it searches the web, reads the pages, and must cite them. This stops it from inventing answers.

🔹 **Safety layers.** Every action is checked against a permission model: file access is sandboxed to one folder, destructive commands always need human approval, every step is audit-logged, and every change can be undone.

🔹 **Provider abstraction.** The same interface drives local models and cloud APIs (Claude, GPT, DeepSeek), with API keys encrypted by the operating system. Local-first, cloud when needed.

On top of these layers sit a chat assistant and a coding agent. Both are just applications of the same foundations.

Biggest lesson: the model is only one component. Reliability comes from the engineering around it: constraints, verification, memory and guardrails.

Built with Python, FastAPI, llama.cpp, React and TypeScript, with ~280 automated tests. Next, I'm testing the dual-GPU split on a real external AMD card.

Happy to go deeper on any of these layers. 👇

#AI #LLM #LocalAI #MachineLearning #SoftwareEngineering #SystemsDesign #Python
