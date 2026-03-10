import csv

# ----- CONFIGURE -----
INPUT_CSV = "amazon_train.csv"
OUTPUT_TXT = "amazon_reviews.txt"
LIMIT = 8000   # set number of reviews you want (None = no limit)
# ----------------------


def extract_third_column(input_path, output_path, limit=None):
    count = 0

    with open(input_path, "r", encoding="utf-8", newline="") as csvfile, \
         open(output_path, "w", encoding="utf-8") as outfile:

        reader = csv.reader(csvfile)

        for row in reader:

            # Skip rows with fewer than 3 cols
            if len(row) < 3:
                continue

            review = row[2].strip()  # third column

            # Write review to file
            outfile.write(review + "\n")

            count += 1

            # Stop if limit reached
            if limit is not None and count >= limit:
                break

    print(f"Done! Extracted {count} reviews into {output_path}")


if __name__ == "__main__":
    extract_third_column(INPUT_CSV, OUTPUT_TXT, LIMIT)
