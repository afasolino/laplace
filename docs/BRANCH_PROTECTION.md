# Branch protection

Protect the production branch with:

- pull requests required;
- at least one independent approving review;
- dismissal of stale approvals after new commits;
- conversation resolution;
- linear history;
- signed commits/tags where organizational signing is configured;
- no force pushes or deletion;
- administrators included unless emergency policy explicitly says otherwise.

Required status checks:

```text
lint-and-types
unit-and-integration-tests
browser-fixture-tests
package-build
migration-tests
security
documentation
```

Require the branch to be current before merge. The manual `release-candidate` gate is
required for a version tag or deployment but need not block every development pull
request.

Repository Actions permissions should remain read-only by default. Do not expose
deployment, model, SSH, signing or AI-agent credentials to pull-request workflows.
Any release-token expansion requires separate approval and a protected environment.

