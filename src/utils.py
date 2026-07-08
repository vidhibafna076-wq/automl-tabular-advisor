def print_section(title: str):
    print("\n" + " ")
    print(title)
    print(" ")


def print_list(items):
    if not items:
        print("None")
    else:
        for item in items:
            print(f"- {item}")