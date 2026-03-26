import os
import platform

print("OS name:", os.name)
print("Current directory:", os.getcwd())
print("Environment variable (PATH):", os.environ.get('PATH', '')[:100])
print("Platform system:", platform.system())
print("Platform release:", platform.release())
print("Processor:", platform.processor())
