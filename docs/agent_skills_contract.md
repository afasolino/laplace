# Agent skills contract

Frozen skills live under `codex_a6000/skills`. Each directory contains
`SKILL.md` and `skill.json`. The seven required skills are requirements
normalization, RTL implementation, systematic debugging, verification before
completion, reviewer grounding, bounded correction, and run finalization.

`FrozenSkillRegistry` validates names, front matter, role declarations, and
content hashes. It writes a deterministically ordered `skills.lock.json`.
Measured runs use the locked bytes; skills cannot self-modify and runtime
research cannot update them.

To change a skill:

1. edit the repository source;
2. run the skill validation and reproducibility tests;
3. review the changed hash;
4. commit it before starting a measured run.

An old run continues to name its old skill-lock hash. A new lock never mutates
historical run evidence.

