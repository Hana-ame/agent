class RegistryMeta(type):
    registry = {}

    def __new__(cls, name, bases, dct):
        new_class = super().__new__(cls, name, bases, dct)
        if name != "BasePlugin":
            cls.registry[name] = new_class
        return new_class

class BasePlugin(metaclass=RegistryMeta):
    pass

class PluginA(BasePlugin):
    def run(self):
        return "PluginA running"

class PluginB(BasePlugin):
    def run(self):
        return "PluginB running"

if __name__ == "__main__":
    print("Registered plugins:", list(RegistryMeta.registry.keys()))
    for name, plugin_cls in RegistryMeta.registry.items():
        print(f"{name}: {plugin_cls().run()}")
