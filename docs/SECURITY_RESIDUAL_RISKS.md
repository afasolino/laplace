# Security residual risks

The v7 CPU/fixture certification does not prove the security of a live model runtime,
GPU driver, provider implementation, reverse proxy, SSH/HTTPS sync backend or
production backup encryption provider. Those require deployment-specific review and
live certification. GPU/live-model work is
`BLOCKED_BY_USER_CONSTRAINT_GPU_UNAVAILABLE`.

Other residual risks:

- malicious files may exploit third-party parsers after admission checks;
- a compromised local account or administrator can read its accessible plaintext;
- denial of service below configured limits remains possible;
- Unicode confusables that do not normalize identically still need human review;
- Markdown safety ultimately depends on the frontend renderer retaining its sanitizer;
- filesystem race defense also relies on OS permissions and careful final-open flags;
- offline dependency inventory cannot identify newly disclosed vulnerabilities;
- license metadata may be missing and is marked `UNKNOWN_REVIEW_REQUIRED`;
- TLS, SSH host keys, certificate rotation and external backup keys remain operator
  responsibilities;
- a fixture provider cannot reveal malformed behavior unique to a live provider.

Operators should run current dependency/vulnerability tooling in an approved networked
release environment, keep state permissions private, terminate remote access at a
reviewed proxy, test encrypted restore regularly and repeat live-provider security
tests before enabling a new route.

