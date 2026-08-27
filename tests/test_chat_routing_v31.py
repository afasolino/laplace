from research_workspace.chat_routing import route_message


def test_general_questions_do_not_route_to_repository_agent() -> None:
    assert route_message("What is 2 + 2?").route == "chat"
    assert route_message("Why is the sky blue?").route == "chat"
    assert route_message("What has keys but cannot open locks?").route == "chat"


def test_runtime_queries_route_only_to_runtime() -> None:
    assert route_message("What token/s are you running at?").route == "runtime"
    assert route_message("What model are you using?").route == "runtime"
    assert route_message("Show GPU memory usage").route == "runtime"


def test_corpus_overview_and_retrieval_are_distinct() -> None:
    assert route_message("What is my corpus about?").route == "corpus"
    assert route_message("List my corpus").route == "corpus"
    assert route_message("Search my corpus for HBM4").route == "retrieval"
    assert route_message("According to my notes, what did I write about PIM?").route == "retrieval"


def test_repository_engineering_routes_to_agent() -> None:
    assert route_message("Inspect src/research_workspace/chat_cli.py").route == "agent"
    assert route_message("Fix the failing test in the repository").route == "agent"
    assert route_message("Where is derive_task_label defined?").route == "agent"


def test_explicit_override_always_wins() -> None:
    assert route_message("What is 2 + 2?", "agent").route == "agent"
    assert route_message("Fix src/a.py", "chat").route == "chat"
