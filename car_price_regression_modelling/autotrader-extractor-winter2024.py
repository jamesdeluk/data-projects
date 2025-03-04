import csv

# Load data from raw_car_data.txt
def load_data(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

# Define helper functions to extract relevant data
def parse_data(data):
    lines = data.splitlines()
    rows = []
    current_row = {}

    for i, line in enumerate(lines):
        line = line.strip()
        if line.startswith('\u00a3') and i + 1 < len(lines) and "Ford Fiesta" in lines[i + 1]:
            if current_row:  # Save the previous record before starting a new one
                for key, value in current_row.items():
                    if value is None:
                        print(f"Warning: Missing {key} for row: {current_row}")
                rows.append(current_row)
            current_row = {
                "Price": line.replace('\u00a3', '').replace(',', ''),
                "Year": None,
                "Generation": None,
                "Trim": None,
                "Mileage": None,
                "Distance": None,
                "Seller": None,
                "Owners": None
            }

        elif "Ford Fiesta" in line:
            continue

        elif line.startswith("1.6T") or line.startswith("1.5T"):
            current_row["Generation"] = "7" if "1.6T" in line or "1.6L" in line else "8"
            current_row["Trim"] = line.split(" ")[2]

        elif "reg)" in line:
            current_row["Year"] = line.split()[0]

        elif "miles)" in line:
            current_row["Distance"] = line.split("(")[1].replace(")", "").split(' ')[0]

        elif "miles" in line:
            current_row["Mileage"] = line.replace(" miles", "").replace(",", "")

        elif "owners" in line:
            current_row["Owners"] = line.split()[0]

        elif "See all" in line or "reviews" in line or "dealer logo" in line:
            current_row["Seller"] = "Trade"

        elif current_row and not current_row.get("Seller"):
            current_row["Seller"] = "Private"

    if current_row:  # Append the last record
        for key, value in current_row.items():
            if value is None:
                print(f"Warning: Missing {key} for row: {current_row}")
        rows.append(current_row)

    return rows

# Convert parsed data to CSV format
def write_to_csv(rows, filename):
    fieldnames = ["Price", "Year", "Generation", "Trim", "Mileage", "Distance", "Seller", "Owners"]
    with open(filename, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

# Main execution
if __name__ == "__main__":
    raw_data = load_data("raw_car_data.txt")
    parsed_data = parse_data(raw_data)
    write_to_csv(parsed_data, "ford_fiesta_data.csv")
