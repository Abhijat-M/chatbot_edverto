import json

# File paths
existing_file = "intents.json"  
new_file = "new.json" 
output_file = "intents.json"  # Output file to store merged intents

# Load the existing JSON file
with open(existing_file, 'r') as f:
    existing_data = json.load(f)

# Load the new JSON file
with open(new_file, 'r') as f:
    new_data = json.load(f)

# Merge the two JSON lists
merged_data = existing_data + new_data

# Save the merged data to a new JSON file
with open(output_file, 'w') as f:
    json.dump(merged_data, f, indent=4)

print(f"Merged JSON file saved as {output_file}")
