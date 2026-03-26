import sysconfig

print("Python version:", sysconfig.get_python_version())
print("Platform:", sysconfig.get_platform())
print("Install paths:", sysconfig.get_path_names())
print("Config variable (prefix):", sysconfig.get_config_var('prefix'))
