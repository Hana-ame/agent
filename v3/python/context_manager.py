class ManagedFile:
    def __init__(self, filename, mode):
        self.filename = filename
        self.mode = mode
        self.file = None

    def __enter__(self):
        self.file = open(self.filename, self.mode)
        return self.file

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()

if __name__ == "__main__":
    with ManagedFile("test.txt", "w") as f:
        f.write("Hello, context manager!")
    
    with open("test.txt", "r") as f:
        content = f.read()
        print(content)
    
    # 清理测试文件
    import os
    os.remove("test.txt")
