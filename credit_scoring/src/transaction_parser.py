import re

def parse_transactions(text):
    lines = text.split("\n")

    transactions = []

    for line in lines:
        line = line.strip()

        # keep only lines that look like transactions
        if any(char.isdigit() for char in line):
            transactions.append(line)

    return transactions