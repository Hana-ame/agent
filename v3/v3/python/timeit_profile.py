import timeit
import cProfile
import pstats
import io

def slow_sum(n):
    total = 0
    for i in range(n):
        total += i
    return total

def fast_sum(n):
    return sum(range(n))

t_slow = timeit.timeit("slow_sum(10000)", globals=globals(), number=100)
t_fast = timeit.timeit("fast_sum(10000)", globals=globals(), number=100)
print(f"slow_sum: {t_slow:.4f}s, fast_sum: {t_fast:.4f}s")

pr = cProfile.Profile()
pr.enable()
slow_sum(1000000)
pr.disable()

s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
ps.print_stats(5)
print("Profile output (first 500 chars):")
print(s.getvalue()[:500])
