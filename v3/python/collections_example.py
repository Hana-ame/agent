from collections import Counter, defaultdict, deque

# Counter
words = ["apple", "banana", "apple", "orange", "banana", "apple"]
counter = Counter(words)
print("Counter:", counter)
print("Most common:", counter.most_common(2))

# defaultdict
dd = defaultdict(list)
dd['fruits'].append('apple')
dd['fruits'].append('banana')
dd['vegetables'].append('carrot')
print("DefaultDict:", dict(dd))

# deque
dq = deque([1, 2, 3])
dq.appendleft(0)
dq.append(4)
print("Deque:", list(dq))
