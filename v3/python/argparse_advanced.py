import argparse

parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(dest='command', required=True)

# add 子命令
add_parser = subparsers.add_parser('add')
add_parser.add_argument('x', type=int)
add_parser.add_argument('y', type=int)

# multiply 子命令
mul_parser = subparsers.add_parser('multiply')
mul_parser.add_argument('x', type=int)
mul_parser.add_argument('y', type=int)

args = parser.parse_args(['add', '3', '5'])  # 模拟命令行
if args.command == 'add':
    print(f"{args.x} + {args.y} = {args.x + args.y}")
elif args.command == 'multiply':
    print(f"{args.x} * {args.y} = {args.x * args.y}")
