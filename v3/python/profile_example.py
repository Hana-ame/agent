import cProfile
import pstats
import io

def slow_function():
    total = 0
    for i in range(1000000):
        total += i
    return total

pr = cProfile.Profile()
pr.enable()
result = slow_function()
pr.disable()

s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
ps.print_stats(10)
print("Result:", result)
print("Profile output (first 500 chars):", s.getvalue()[:500])
