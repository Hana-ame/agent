import contextlib
import io

def noisy_function():
    print("This is a noisy output")
    print("Another line")

f = io.StringIO()
with contextlib.redirect_stdout(f):
    noisy_function()

output = f.getvalue()
print("Captured output:")
print(output)
