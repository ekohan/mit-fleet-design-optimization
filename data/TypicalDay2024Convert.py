import pandas as pd

# Read the CSV file
df = pd.read_csv('data/TypicalDay2024.csv', sep=';')

# Create the mapping dictionary
product_type_mapping = {
    'Seco': 'Dry',
    'Refrigerado': 'Chilled',
    'Congelado': 'Frozen'
}

# Apply the mapping to the ProductType column
df['ProductType'] = df['ProductType'].map(product_type_mapping)

# Save the modified file
df.to_csv('data/TypicalDay2024_converted.csv', sep=';', index=False)