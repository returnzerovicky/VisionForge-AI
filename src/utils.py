import json


def save_json(data, filename="output.json"):

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def print_header(title):
    print("=" * 60)
    print(title)
    print("=" * 60)