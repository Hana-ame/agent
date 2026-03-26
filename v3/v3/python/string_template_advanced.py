import string

class MyTemplate(string.Template):
    delimiter = '@'
    idpattern = r'[a-z]+_[a-z]+'

t = MyTemplate('@name_username says: @message')
print(t.substitute(name_username='Alice', message='Hello'))

# 使用 safe_substitute 避免 KeyError
print(t.safe_substitute(name_username='Bob'))
