import argparse

parser = argparse.ArgumentParser(description="A simple example of argparse")
parser.add_argument("--name", default="World", help="Name to greet")
parser.add_argument("--times", type=int, default=1, help="Number of times to greet")
parser.add_argument("-v", "--verbose", action="store_true", help="Increase output verbosity")

if __name__ == "__main__":
    args = parser.parse_args()
    if args.verbose:
        print(f"Greeting {args.name} {args.times} time(s)")
    for _ in range(args.times):
        print(f"Hello, {args.name}!")
