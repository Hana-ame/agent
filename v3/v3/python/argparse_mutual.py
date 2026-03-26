import argparse

parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group()
group.add_argument('--verbose', action='store_true')
group.add_argument('--quiet', action='store_true')

args = parser.parse_args(['--verbose'])
print("Verbose:", args.verbose)
print("Quiet:", args.quiet)

# 互斥会报错
try:
    parser.parse_args(['--verbose', '--quiet'])
except SystemExit:
    print("Mutually exclusive arguments error")
