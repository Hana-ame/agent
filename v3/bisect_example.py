import bisect

a = [1, 3, 5, 7, 9]

print("Insert 4 in position:", bisect.bisect_left(a, 4))
bisect.insort(a, 4)
print("After insort 4:", a)

print("Insert 5 in position (right):", bisect.bisect_right(a, 5))
bisect.insort_right(a, 5)
print("After insort_right 5:", a)
