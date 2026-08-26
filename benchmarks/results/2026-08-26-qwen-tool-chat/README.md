# Qwen tool-call chat template

The C++ template accepts tool schemas, assistant tool calls, and ordered tool
responses without Python or Jinja. Invalid roles, orphan responses, and non-object
arguments fail explicitly. Basic system/user/assistant output remains unchanged.

![Tool branch](tool-chat.svg)
