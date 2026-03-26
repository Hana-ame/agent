import reprlib

long_list = list(range(1000))
print("Default repr:", reprlib.repr(long_list))

# 自定义最大长度
repr_instance = reprlib.Repr()
repr_instance.maxlist = 10
print("Custom repr:", repr_instance.repr(long_list))
