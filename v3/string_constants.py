import string

print("ASCII letters:", string.ascii_letters)
print("Digits:", string.digits)
print("Punctuation:", string.punctuation)
print("Whitespace:", repr(string.whitespace))

# 格式化
template = string.Template("Hello $name, your score is $score")
print(template.substitute(name="Alice", score=95))
