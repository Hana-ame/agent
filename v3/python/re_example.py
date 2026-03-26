import re

text = "The quick brown fox jumps over the lazy dog."

# 查找所有单词
words = re.findall(r'\b\w+\b', text)
print("Words:", words)

# 替换
replaced = re.sub(r'fox', 'cat', text)
print("Replaced:", replaced)

# 匹配邮箱
email_text = "Contact us at support@example.com or sales@example.org"
emails = re.findall(r'[\w\.-]+@[\w\.-]+', email_text)
print("Emails:", emails)
