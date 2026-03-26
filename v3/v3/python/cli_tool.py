import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Simple CLI tool")
    parser.add_argument('--name', default='World')
    parser.add_argument('--count', type=int, default=1)
    args = parser.parse_args()
    for _ in range(args.count):
        print(f"Hello, {args.name}!")

if __name__ == "__main__":
    main()
