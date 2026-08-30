from tools.runtime_control import RuntimeControl


def test_queue_starts_unpaused(tmp_path):
    control = RuntimeControl(tmp_path / "test.db")

    assert control.is_queue_paused() is False


def test_pause_queue_persists(tmp_path):
    db = tmp_path / "test.db"

    control = RuntimeControl(db)

    assert control.pause_queue() is True
    assert control.is_queue_paused() is True

    reloaded = RuntimeControl(db)

    assert reloaded.is_queue_paused() is True


def test_resume_queue_persists(tmp_path):
    db = tmp_path / "test.db"

    control = RuntimeControl(db)

    control.pause_queue()
    assert control.resume_queue() is True

    reloaded = RuntimeControl(db)

    assert reloaded.is_queue_paused() is False


def test_snapshot(tmp_path):
    control = RuntimeControl(tmp_path / "test.db")

    assert control.snapshot() == {
        "paused": False,
    }

    control.pause_queue()

    assert control.snapshot() == {
        "paused": True,
    }
