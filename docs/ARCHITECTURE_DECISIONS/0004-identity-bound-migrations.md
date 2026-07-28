# ADR 0004: identity-bound fixture-first migrations

Status: accepted

Migration commands never infer a state root. They require both an explicit root and
the manifest's state identity, then perform permission, integrity and hash preflight.
An automatic backup, exclusive lock and restart journal precede changes. Production
migration is an operationally approved action distinct from fixture certification.
