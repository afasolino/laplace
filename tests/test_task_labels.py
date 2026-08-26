from research_workspace.task_labels import derive_task_label


def test_task_label_is_deterministic_bounded_and_informative() -> None:
    instruction = (
        "Inspect the current repository and trace how laplace chat works end-to-end "
        "from the CLI entrypoint"
    )
    assert derive_task_label(instruction) == "Repository Laplace Chat End-to-end"
    assert derive_task_label(instruction) == derive_task_label(instruction)


def test_task_label_has_two_to_four_words_and_safe_fallback() -> None:
    assert derive_task_label("scheduler") == "Scheduler Task"
    assert derive_task_label("the and to") == "Repository Task"
    label = derive_task_label("X" * 100 + " second third fourth fifth")
    assert 2 <= len(label.split()) <= 4
    assert len(label) <= 64
