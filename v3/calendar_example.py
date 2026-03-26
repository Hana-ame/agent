import calendar

print("Calendar 2024:\n", calendar.calendar(2024, w=2, l=1, c=6))

print("Month 2024-12:\n", calendar.month(2024, 12))

print("Weekday of 2024-12-25:", calendar.weekday(2024, 12, 25))
print("Is 2024 leap year?", calendar.isleap(2024))
