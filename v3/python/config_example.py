import configparser
import os

config = configparser.ConfigParser()
config['DEFAULT'] = {'ServerAliveInterval': '45', 'Compression': 'yes'}
config['bitbucket.org'] = {'User': 'hg'}
config['topsecret.server.com'] = {'Host Port': '50022', 'ForwardX11': 'no'}

with open('example.ini', 'w') as f:
    config.write(f)

# 读取配置
config.read('example.ini')
print("Sections:", config.sections())
print("bitbucket.org user:", config['bitbucket.org']['User'])
print("Default compression:", config['DEFAULT']['Compression'])

os.remove('example.ini')
