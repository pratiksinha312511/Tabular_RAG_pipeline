"""Reconstruct transaction data from the PDF printout into a proper CSV."""

import re
import csv
import os
import PyPDF2

PDF_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "assessment_transaction_data.xlsx - Google Sheets.pdf",
)
OUTPUT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "transactions.csv",
)


def main():
    reader = PyPDF2.PdfReader(PDF_PATH)

    left_rows = []   # user_id, user_name, date, amount
    right_rows = []  # category, merchant

    for page in reader.pages:
        text = page.extract_text()
        if not text:
            continue

        for line in text.strip().split("\n"):
            line = line.strip()
            if not line:
                continue

            # Skip header lines
            if "user_id" in line.lower() or "transaction_category" in line.lower():
                continue

            # Left-side rows: usr_ID  Name  Date  Amount
            m = re.match(
                r"(usr_\w+)\s+(.+?)\s+(\d{4}-\d{2}-\d{2})\s+(-?\d+)", line
            )
            if m:
                left_rows.append({
                    "user_id": m.group(1),
                    "user_name": m.group(2).strip(),
                    "transaction_date": m.group(3),
                    "transaction_amount": int(m.group(4)),
                })
                continue

            # Right-side rows: CATEGORY  Merchant Name
            m = re.match(r"([A-Z]+_[A-Z]+)\s+(.+)", line)
            if m:
                right_rows.append({
                    "transaction_category_detail": m.group(1),
                    "merchant_name": m.group(2).strip(),
                })
                continue

    print(f"Left rows (amounts):     {len(left_rows)}")
    print(f"Right rows (categories): {len(right_rows)}")

    if len(left_rows) != len(right_rows):
        print(f"WARNING: Row mismatch — trimming to minimum")
        n = min(len(left_rows), len(right_rows))
        left_rows, right_rows = left_rows[:n], right_rows[:n]

    # Merge
    rows = [{**l, **r} for l, r in zip(left_rows, right_rows)]

    # Save
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    fields = [
        "user_id", "user_name", "transaction_date",
        "transaction_amount", "transaction_category_detail", "merchant_name",
    ]
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved {len(rows)} rows to {OUTPUT_PATH}")

    # Verify first 5 rows
    print("\nFirst 5 rows:")
    for r in rows[:5]:
        print(f"  {r['user_id']} | {r['user_name']} | {r['transaction_date']} | "
              f"{r['transaction_amount']:>6} | {r['transaction_category_detail']} | {r['merchant_name']}")

    # Per-user stats
    users = sorted(set(r["user_id"] for r in rows))
    print(f"\n{len(users)} users:")
    for uid in users:
        subset = [r for r in rows if r["user_id"] == uid]
        print(f"  {uid} ({subset[0]['user_name']}): {len(subset)} transactions")


if __name__ == "__main__":
    main()
