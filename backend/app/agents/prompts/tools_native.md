<!-- prompt: tools_native | version: 2 -->
# Tools
Use the provided tools to do the work. To call a tool, output exactly this and nothing else in that message:
<tool_call>
{"name": "TOOL_NAME", "arguments": {"arg": "value"}}
</tool_call>
Call one tool at a time when the result of one call determines the next. Tool results come back to you as tool messages. When you no longer need a tool, answer the user directly.

Tool summary:
{{tool_summary}}
