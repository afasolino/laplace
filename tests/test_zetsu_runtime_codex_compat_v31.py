from __future__ import annotations

from pathlib import Path

from research_workspace import zetsu_cli
from research_workspace.zetsu_config import DEFAULT_ENDPOINT, DEFAULT_TOKEN_ENV, ZetsuStatus


def _status(repo: Path) -> ZetsuStatus:
    return ZetsuStatus(configured=False, repository=str(repo), config_path=str(repo/'codex.toml'), skill_path=str(repo/'.agents/skills/zetsu/SKILL.md'), endpoint=None, token_env_var=None, token_available=False, skill_installed=True, codex_available=True, config_version=None, expected_config_version=4, skill_version='1.5.0', expected_skill_version='1.5.0', schema_version='1.5', compatible=False)

def _recognition(repo: Path, state_root: Path, bridge: Path) -> dict[str, object]:
    return {'recognized':True,'configuration':{'name':'zetsu','enabled':True,'transport':{'type':'stdio','command':str(bridge),'args':['--repo',str(repo),'--state-root',str(state_root),'--endpoint',DEFAULT_ENDPOINT],'env':{'CODEX_MCP_PROTOCOL_VERSION':'2026-07-28'}}}}

def test_recognizes_exact_codex_owned_stdio_bridge(tmp_path: Path, monkeypatch) -> None:
    repo=tmp_path/'repo'; repo.mkdir(); state_root=repo/'.runtime/v2-live-cert'; bridge=repo/'.venv/bin/laplace-zetsu-mcp'; bridge.parent.mkdir(parents=True); bridge.write_text('',encoding='utf-8')
    monkeypatch.setattr(zetsu_cli.shutil,'which',lambda name: str(bridge) if name=='laplace-zetsu-mcp' else None)
    assert zetsu_cli._recognized_stdio_bridge_settings(repo,state_root,_recognition(repo,state_root,bridge)) == {'endpoint':DEFAULT_ENDPOINT,'token_env_var':DEFAULT_TOKEN_ENV,'mode':'codex_stdio_bridge'}

def test_stdio_bridge_rejects_wrong_repository(tmp_path: Path, monkeypatch) -> None:
    repo=tmp_path/'repo'; repo.mkdir(); other=tmp_path/'other'; other.mkdir(); state_root=repo/'.runtime/v2-live-cert'; bridge=repo/'.venv/bin/laplace-zetsu-mcp'; bridge.parent.mkdir(parents=True); bridge.write_text('',encoding='utf-8')
    monkeypatch.setattr(zetsu_cli.shutil,'which',lambda name: str(bridge) if name=='laplace-zetsu-mcp' else None)
    assert zetsu_cli._recognized_stdio_bridge_settings(other,state_root,_recognition(repo,state_root,bridge)) is None

def test_diagnostics_accept_stdio_without_legacy_managed_block(tmp_path: Path, monkeypatch) -> None:
    repo=tmp_path/'repo'; repo.mkdir(); state_root=repo/'.runtime/v2-live-cert'; bridge=repo/'.venv/bin/laplace-zetsu-mcp'; bridge.parent.mkdir(parents=True); bridge.write_text('',encoding='utf-8'); recognition=_recognition(repo,state_root,bridge)
    monkeypatch.setattr(zetsu_cli,'zetsu_status',lambda _repo:_status(repo)); monkeypatch.setattr(zetsu_cli,'_codex_recognition',lambda _repo:recognition); monkeypatch.setattr(zetsu_cli.shutil,'which',lambda name: str(bridge) if name=='laplace-zetsu-mcp' else None); monkeypatch.setattr(zetsu_cli,'_load_command_token',lambda endpoint,token_env,root: True); monkeypatch.setattr(zetsu_cli,'_online_probe',lambda endpoint,token_env,timeout,*,retrieval:{'reachable':True,'retrieval_check':retrieval}); monkeypatch.setattr(zetsu_cli,'_laplace_status',lambda endpoint,token_env,timeout:{'status':'READY'}); monkeypatch.setattr(zetsu_cli,'_laplace_readiness',lambda endpoint,timeout:{'status':'READY','reasons':[]}); monkeypatch.setattr(zetsu_cli,'_repository_readiness',lambda repository,root:{'agent_task_ready':True,'state':'ready'})
    payload=zetsu_cli._diagnostic_payload(repo,command='test',offline=False,timeout=1.0,state_root=state_root)
    assert payload['ok'] is True and payload['configuration_mode']=='codex_stdio_bridge' and payload['endpoint']==DEFAULT_ENDPOINT and payload['token_env_var']==DEFAULT_TOKEN_ENV and payload['configured'] is True and payload['compatible'] is True

def test_start_does_not_reclaim_codex_owned_stdio_config(tmp_path: Path, monkeypatch) -> None:
    repo=tmp_path/'repo'; repo.mkdir(); state_root=repo/'state'; bridge=repo/'.venv/bin/laplace-zetsu-mcp'; bridge.parent.mkdir(parents=True); bridge.write_text('',encoding='utf-8'); recognition=_recognition(repo,state_root,bridge)
    monkeypatch.setattr(zetsu_cli,'zetsu_status',lambda _repo:_status(repo)); monkeypatch.setattr(zetsu_cli,'_codex_recognition',lambda _repo:recognition); monkeypatch.setattr(zetsu_cli.shutil,'which',lambda name: str(bridge) if name=='laplace-zetsu-mcp' else None)
    def forbidden(*args,**kwargs): raise AssertionError('configure_zetsu must not reclaim stdio config')
    monkeypatch.setattr(zetsu_cli,'configure_zetsu',forbidden); monkeypatch.setattr(zetsu_cli,'start_local_runtime',lambda *args,**kwargs:{'status':'READY'}); monkeypatch.setattr(zetsu_cli,'_diagnostic_payload',lambda *args,**kwargs:{'ok':True})
    assert zetsu_cli.main(['start','--repo',str(repo),'--state-root',str(state_root),'--nocodev']) == 0
