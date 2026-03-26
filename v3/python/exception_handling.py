def safe_divide(a, b):
    try:
        result = a / b
    except ZeroDivisionError:
        return "Error: Division by zero"
    except TypeError:
        return "Error: Invalid operand type"
    else:
        return f"Result: {result}"
    finally:
        print("Division attempted")

if __name__ == "__main__":
    print(safe_divide(10, 2))
    print(safe_divide(10, 0))
    print(safe_divide(10, "2"))
