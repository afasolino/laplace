# Online research security

Only configured scholarly APIs and public HTTP(S) pages are queried. Crossref, OpenAlex, and arXiv require no secret; IEEE reads only `IEEE_XPLORE_API_KEY` from the environment. Keys are never printed, persisted, or passed to the model. The safe fetcher rejects `file:`, localhost, private, loopback, link-local, and reserved addresses, bounds response size, blocks redirects, and validates content type.

Open-access acquisition requires an explicit provider `open_access: true`, HTTPS, public DNS, PDF MIME/magic bytes, a size limit, atomic temporary storage, and SHA-256. Subscribed IEEE downloads are never automatic and require a visible headed browser, manual login, explicit per-item approval, and immediate stop on CAPTCHA, 401/403/429, access warnings, expired sessions, or ambiguous controls.

