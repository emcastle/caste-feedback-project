# src/Caste_Project/cli.py
import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="sanity")
    args = ap.parse_args()
    print("Running mode:", args.mode)

if __name__ == "__main__":
    main()
