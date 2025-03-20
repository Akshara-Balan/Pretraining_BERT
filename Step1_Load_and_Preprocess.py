import pandas as pd

# Load the CSV file
df = pd.read_csv("file.txt")

# Inspect the data
print(df.head())

# Ensure the 'text' column exists
if 'Malayalam' not in df.columns:
    raise ValueError("CSV file must contain a 'text' column.")

# Drop rows with missing values
df = df.dropna(subset=['Malayalam'])

# Save the cleaned text to a temporary file for tokenizer training
with open("malayalam_text.txt", "w", encoding="utf-8") as f:
    for line in df['Malayalam']:
        f.write(line + "\n")
