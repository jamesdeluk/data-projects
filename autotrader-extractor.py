import re
import csv

# Function to extract the required fields from a block of text
def extract_data(text):
    # Regex patterns for each field
    price_pattern = r"£(\d{1,3},\d{3})"
    year_pattern = r"(\d{4}) \(\d{2} reg\)"
    mileage_pattern = r"(\d{1,3},\d{3}) miles"
    distance_pattern = r"\((\d+) miles\)"

    # Search for patterns
    price_match = re.search(price_pattern, text)
    year_match = re.search(year_pattern, text)
    mileage_match = re.search(mileage_pattern, text)
    distance_match = re.search(distance_pattern, text)

    # Extract and clean the data
    price = price_match.group(1).replace(",", "") if price_match else None
    year = year_match.group(1) if year_match else None
    mileage = mileage_match.group(1).replace(",", "") if mileage_match else None
    distance = distance_match.group(1) if distance_match else None

    return [price, year, mileage, distance]

# Read the text file
input_file_path = 'data.txt'
with open(input_file_path, 'r', encoding='utf-8') as file:
    text = file.read()

# Split the text into blocks for each car entry
car_entries = text.split('£')[1:]  # Split by the £ symbol (price always comes first)

# Open a CSV file to write the extracted data
output_file_path = 'car_data.csv'
with open(output_file_path, 'w', newline='', encoding='utf-8') as csvfile:
    csvwriter = csv.writer(csvfile)
    # Write the header
    csvwriter.writerow(['Price', 'Year', 'Mileage', 'Distance'])

    # Extract data from each car entry and write it to the CSV
    for entry in car_entries:
        entry = '£' + entry  # Re-add the '£' symbol to make sure price is part of the entry
        data = extract_data(entry)
        if all(data):  # Ensure all fields are extracted
            csvwriter.writerow(data)

print(f'Data extraction complete. CSV saved as {output_file_path}.')