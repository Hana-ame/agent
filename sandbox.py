import docker
import tarfile
import io
import os

class DockerSandbox:
    def __init__(self, image="python:3.11-slim", timeout=10, memory_limit="128m"):
        """
        初始化沙箱配置
        :param image: Docker镜像，建议用 slim 版本
        :param timeout: 代码执行超时时间(秒)，防止死循环
        :param memory_limit: 内存限制，防止内存溢出攻击
        """
        self.client = docker.from_env()
        self.image = image
        self.timeout = timeout
        self.memory_limit = memory_limit
        
        # 预先拉取镜像，防止运行时等待
        try:
            self.client.images.get(self.image)
        except:
            print(f"Pulling image {self.image}...")
            self.client.images.pull(self.image)

    def execute(self, code: str):
        """
        在容器中执行 Python 代码
        """
        container = None
        try:
            # 1. 启动容器
            # network_disabled=True: 禁止联网，防止代码向外发送数据或挖矿
            # remove=True: (在run方法中有效，detach模式需手动remove)
            container = self.client.containers.run(
                self.image,
                command="sleep infinity", # 让容器保持运行，等待我们exec
                detach=True,
                mem_limit=self.memory_limit,
                network_disabled=True 
            )

            # 2. 准备代码文件 (将字符串包装成 tar 流上传到容器)
            # 我们不直接用 python -c "code" 是因为复杂代码里有很多转义字符很难处理
            # 写入文件运行最稳妥
            code_file_content = code.encode('utf-8')
            tar_stream = io.BytesIO()
            with tarfile.open(fileobj=tar_stream, mode='w') as tar:
                tarinfo = tarfile.TarInfo(name='script.py')
                tarinfo.size = len(code_file_content)
                tar.addfile(tarinfo, io.BytesIO(code_file_content))
            tar_stream.seek(0)
            
            # 将代码文件上传到容器的 /app 目录（或根目录）
            container.put_archive("/", tar_stream)

            # 3. 执行代码
            # 使用 exec_run 执行 python script.py
            exec_result = container.exec_run(
                cmd="python script.py",
                workdir="/",
                demux=True  # 分离 stdout 和 stderr
            )

            stdout, stderr = exec_result.output
            
            # 4. 处理结果
            output = ""
            if stdout:
                output += f"Output:\n{stdout.decode('utf-8')}\n"
            if stderr:
                output += f"Error:\n{stderr.decode('utf-8')}\n"
            
            if not output:
                output = "Code executed successfully (no output)."

            return output

        except Exception as e:
            return f"System Error: {str(e)}"
        
        finally:
            # 5. 清理战场：强制销毁容器
            if container:
                try:
                    container.kill() # 强制停止
                    container.remove() # 删除容器
                except:
                    pass

# --- 测试集成 ---

# 实例化沙箱
sandbox = DockerSandbox()

# 测试 1: 正常的代码
code_normal = """
print('Hello from Docker!')
a = 10 + 20
print(f'Calculation result: {a}')
"""
print("=== Test Normal ===")
print(sandbox.execute(code_normal))

# 测试 2: 危险代码 (模拟 rm -rf /)
# 注意：在容器里这也是危险的，会删掉容器内的 python 环境导致后续报错，
# 但绝不会影响你的宿主机 Mac/Windows/Linux。
code_dangerous = """
import os
print('Trying to delete root...')
# 实际上我们用 Python 模拟删除动作，或者直接 os.system('rm -rf /bin')
# 为了演示效果，我们删除一个关键目录看看
os.system('rm -rf /bin') 
print('Deleted /bin inside container. Host is safe.')
"""
print("\n=== Test Dangerous ===")
print(sandbox.execute(code_dangerous))

# 测试 3: 再次运行正常代码
# 因为每次 execute 都是全新的容器，所以之前的删除不会影响这一次
print("\n=== Test Normal Again ===")
print(sandbox.execute(code_normal))