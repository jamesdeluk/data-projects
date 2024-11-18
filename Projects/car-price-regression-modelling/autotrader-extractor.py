import re
import csv

# Function to clean text by removing lines with invalid £ symbols
def clean_text(text):
    # Define a valid price pattern
    valid_price_pattern = r'^£\d{1,3}(?:,\d{3})*$'  # Matches only full price lines like £9,999
    
    cleaned_lines = []
    for line in text.splitlines():
        # If the line has a £ symbol, keep it only if it matches the valid price pattern
        if '£' in line:
            if re.match(valid_price_pattern, line.strip()):
                cleaned_lines.append(line)
        else:
            # Keep lines without £ symbols
            cleaned_lines.append(line)
    
    return "\n".join(cleaned_lines)

# Function to extract data
def extract_data(text):
    # Define regex patterns
    price_pattern = r'£(\d{1,3}(?:,\d{3})*)'  # Matches price like £9,999
    year_pattern = r'(\d{4})(?: \(\d+ reg\))?'  # Matches year with or without (reg)
    mileage_pattern = r'(\d{1,3}(?:,\d{3})*) miles'  # Matches mileage like 38,032 miles
    engine_pattern = r'(1\.\d)'  # Matches engine size like 1.5L or 1.6L
    distance_pattern = r'\((\d+) miles\)'  # Matches distance like (1 miles)
    seller_pattern = r'reviews|See all \d+ cars'  # Matches reviews and "See all cars" for seller type
    trim_pattern = r'ST-\d'  # Matches ST trim level like ST-2, ST-3

    # Define generation based on engine size
    generation_mapping = {'1.6': 7, '1.5': 8}

    data = []
    
    # Split the text into chunks starting with a price (£)
    entries = re.split(r'(£\d{1,3}(?:,\d{3})*)', text)  # Splitting with prices included
    for i in range(1, len(entries), 2):  # Process every second entry (price and car details)
        price = entries[i].replace('£', '').replace(',', '')  # Clean price
        chunk = entries[i+1]
        
        # Extract other fields
        year_match = re.search(year_pattern, chunk)
        mileage_match = re.search(mileage_pattern, chunk)
        engine_match = re.search(engine_pattern, chunk)
        distance_match = re.search(distance_pattern, chunk)
        seller_match = re.search(seller_pattern, chunk)
        trim_match = re.search(trim_pattern, chunk)  # Extract trim level
        
        # Extract fields
        year = year_match.group(1) if year_match else ''
        mileage = mileage_match.group(1).replace(',', '') if mileage_match else ''
        engine_size = engine_match.group(1) if engine_match else ''
        generation = generation_mapping.get(engine_size, '')
        distance = distance_match.group(1) if distance_match else ''
        seller = "Trade" if seller_match else "Private"  # "trade" for dealer, "private" for private seller
        trim_level = trim_match.group(0) if trim_match else ''  # Extract trim level (ST-2, ST-3)

        # Check for empty fields
        if not price or not year or not mileage or not generation or not distance:
            print(f"Alert: Incomplete entry found: Price: {price}, Year: {year}, Generation: {generation}, Trim: {trim_level}, Mileage: {mileage}, Distance: {distance}, Seller: {seller}")

        # Append the extracted data as a row
        data.append([price, year, generation, trim_level, mileage, distance, seller])
    
    return data

# Read the text file
input_file = 'raw_car_data.txt'  # Changed input file name
with open(input_file, 'r', encoding='utf-8') as file:  # Added encoding='utf-8'
    raw_text = file.read()

# Clean the text to remove invalid £ lines
cleaned_text = clean_text(raw_text)

# Extract data
car_data = extract_data(cleaned_text)

# Write to a CSV file
output_file = 'car_data.csv'
with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:  # Ensure writing in UTF-8 too
    writer = csv.writer(csvfile)
    # Write header
    writer.writerow(['Price', 'Year', 'Generation', 'Trim', 'Mileage', 'Distance', 'Seller'])
    # Write data
    writer.writerows(car_data)

print(f'Data extraction complete. CSV file saved as {output_file}')
