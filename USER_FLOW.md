# User flow

Start a conversation by asking a Frappe or ERPNext question in the chat box.
LangGraph saves the recent conversation in its checkpoint for that chat, so
short follow-up messages continue the same topic.

For ordinary Frappe questions, the general assistant replies directly. For a
Sales request, the Sales specialist first checks the tools currently offered by
the connected MCP server. It uses those tools to look up records, prepare a
preview, or inspect live document metadata.

When you ask what is required to create a document, the assistant asks the MCP
server for its live required-field metadata. It gives a short answer based on
that result. Connect the Frappe metadata MCP if your Sales MCP does not offer
the required-field tools. The assistant tells you when no metadata tool is
available instead of guessing.

Creating a Customer, Item, Sales Order, or Quotation is a two-step action. The
assistant first shows a preview. Review it, then choose **Confirm and create**
to create the record, or **Cancel** to stop. A preview expires after its MCP
server time limit.

Messages can contain simple formatting. Bold text is shown normally without
asterisks. Text enclosed in three backticks is shown as a code block, preserving
line breaks and spacing. Code blocks are display-only; they are not executed.

The browser does not save chat messages, conversation IDs, or approval tokens.
LangGraph keeps the latest twelve messages and the active approval state in its
checkpoint. The browser receives only an HttpOnly session cookie that identifies
the checkpoint thread; approval state still expires at the MCP server.
