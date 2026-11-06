<!-- prompt: tools_text | version: 1 -->
# Tools
You can call tools. To call a tool, write exactly one block in this format and then STOP writing (the result will be sent back to you):

<tool_call>
{"name": "TOOL_NAME", "arguments": {ARGUMENTS_AS_JSON}}
</tool_call>

Rules:
- The block must contain valid JSON with "name" and "arguments". Use double quotes; escape newlines inside strings as \n.
- Call one tool at a time and wait for its result, which arrives in a message starting with "Tool result".
- When you do not need a tool, answer normally without any <tool_call> block.

Available tools (JSON schemas of the arguments):
{{tool_catalog}}

Example — read a file:
<tool_call>
{"name": "read_file", "arguments": {"path": "src/main.py"}}
</tool_call>
