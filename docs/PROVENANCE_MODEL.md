# Provenance model

Every local chunk carries document ID, SHA-256, filename, source class, page range, section, and chunk ID. Library records additionally preserve absolute source path, relative Library path, title/authors/year/DOI when extracted, parser, ingestion timestamp, collection, and availability. Online records carry provider, provider ID, title, authors, year, venue, abstract level, DOI, canonical/PDF URLs, access state, retrieval timestamp, raw-record hash, query, rank, and access level.

Availability labels are `METADATA_ONLY`, `ABSTRACT_ONLY`, `PUBLIC_WEB_TEXT`, `COMPLETE_LOCAL_PDF`, and `COMPLETE_DOWNLOADED_PDF`; access blockers use `DOWNLOAD_REQUIRES_LOGIN`, `ACCESS_DENIED`, and `API_KEY_REQUIRED`. Page-grounded citation validation accepts only filename/page/chunk tuples present in the evidence packet. Unsupported claims receive `[SOURCE REQUIRED]`.

Chat citation objects add a stable citation number, title, quoted evidence snippet, source class, retrieval score, availability, and optional DOI/source path. The normal chat response never embeds the full evidence packet. Clicking a citation requests the message evidence endpoint; controlled source opening accepts only files under the active project or configured Library root. Model citation IDs that do not resolve exactly are rejected, audited, and replaced by a grounded extractive fallback.

The model-facing packet uses compact evidence IDs (`E1`, `E2`, …) mapped to the immutable filename/page/section/chunk tuple. Markdown markers and legacy numeric IDs are both accepted, but only tuples present in the retrieved packet validate. A rejected candidate remains available in the audit and conversation history; its fallback is a new message with its own citations and a `fallback_of_message_id` link.
