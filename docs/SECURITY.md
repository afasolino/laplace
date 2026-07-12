# Security model

The service binds to loopback, rejects unsupported extensions, normalizes upload names, caps uploads at 50 MiB, and never executes document or model text. Source hashes identify immutable copies. Exports must redact local paths when shared. No cloud inference or automatic acquisition is implemented.

The chat service accepts only the active project selected by `laplace --start`; browser requests cannot choose arbitrary project roots. Conversations and staged attachments are project-local. Markdown is escaped before the small client renderer applies safe formatting; model-produced HTML or JavaScript is not executed. Attachment filenames are reduced to basenames, signatures and extensions are checked, and source-opening routes enforce active-project/Library containment. Conversation deletion requires `confirm=true` and only marks the conversation deleted; Library documents and indexes remain untouched.
