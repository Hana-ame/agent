import concurrent.futures
import time

def square(n):
    time.sleep(0.1)
    return n * n

def main():
    # 线程池（不会触发 Windows 多进程启动问题）
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        results = list(executor.map(square, range(10)))
    print("ThreadPool results:", results)

    # 进程池（必须放在 __main__ 保护下）
    with concurrent.futures.ProcessPoolExecutor(max_workers=3) as executor:
        results = list(executor.map(square, range(10)))
    print("ProcessPool results:", results)

if __name__ == "__main__":
    main()
