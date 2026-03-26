"""协程示例（非async）"""
def coroutine():
    print("Coroutine started")
    while True:
        x = yield
        print(f"Received: {x}")

if __name__ == "__main__":
    c = coroutine()
    next(c)  # 启动
    c.send(10)
    c.send(20)
    c.close()
