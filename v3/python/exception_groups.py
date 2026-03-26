import sys

if sys.version_info >= (3, 11):
    try:
        raise ExceptionGroup("Group", [ValueError("bad"), TypeError("wrong")])
    except* ValueError as e:
        print(f"Caught ValueError: {e}")
    except* TypeError as e:
        print(f"Caught TypeError: {e}")
else:
    print("Exception groups require Python 3.11+")
