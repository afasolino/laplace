from research_workspace.chat_capability_views import corpus_overview, runtime_metrics_view


class FakeCorpusClient:
    def personal_corpora(self, *, include_archived: bool = False):
        assert include_archived is False
        return {"corpora": [{"corpus_id": "pc_" + "a" * 32, "name": "papers", "revision": 3, "state": "READY"}]}

    def personal_corpus(self, corpus_id: str):
        assert corpus_id == "pc_" + "a" * 32
        return {
            "sources": [
                {"logical_path": "papers/hbm4.pdf", "media_type": "application/pdf"},
                {"logical_path": "notes.txt", "media_type": "text/plain"},
            ]
        }


def test_runtime_metrics_never_invents_tokens_per_second() -> None:
    view = runtime_metrics_view({"model_lanes": ["quality", "standard"]})
    assert view["evidence_kind"] == "runtime_metrics"
    assert view["completion_tokens_per_second"] is None
    assert view["throughput_status"] == "UNAVAILABLE"


def test_corpus_overview_is_bounded_metadata_not_topic_hallucination() -> None:
    view = corpus_overview(FakeCorpusClient())
    assert view["evidence_kind"] == "corpus_overview"
    assert view["corpus_count"] == 1
    assert view["source_count"] == 2
    assert view["media_type_counts"] == {"application/pdf": 1, "text/plain": 1}
    assert view["sample_source_names"] == ["papers/hbm4.pdf", "notes.txt"]
    assert "topic" not in view
