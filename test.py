from loop import run_once


def test_run_once():
    result = run_once()
    assert isinstance(result, str)
    assert len(result) > 0


if __name__ == "__main__":
    test_run_once()
    print("OK")
