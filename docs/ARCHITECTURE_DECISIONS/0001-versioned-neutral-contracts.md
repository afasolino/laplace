# ADR 0001: versioned neutral contracts

Status: accepted

Laplace has two legitimate operating modes and several mature stores. A destructive
rewrite would couple UI migrations to state migrations. We therefore place strict,
versioned provider-neutral records and protocols between mode-specific adapters and
existing services. Compatibility adapters remain until fixture migrations and
release policy permit removal.
