# Provenance model

Every local chunk carries document ID, SHA-256, filename, source class, page range, section, and chunk ID. Library records additionally preserve absolute source path, relative Library path, title/authors/year/DOI when extracted, parser, ingestion timestamp, collection, and availability. Online records carry provider, provider ID, title, authors, year, venue, abstract level, DOI, canonical/PDF URLs, access state, retrieval timestamp, raw-record hash, query, rank, and access level.

Availability labels are `METADATA_ONLY`, `ABSTRACT_ONLY`, `PUBLIC_WEB_TEXT`, `COMPLETE_LOCAL_PDF`, and `COMPLETE_DOWNLOADED_PDF`; access blockers use `DOWNLOAD_REQUIRES_LOGIN`, `ACCESS_DENIED`, and `API_KEY_REQUIRED`. Page-grounded citation validation accepts only filename/page/chunk tuples present in the evidence packet. Unsupported claims receive `[SOURCE REQUIRED]`.

