import os
import json

# Directory containing the .tex files
tex_directory = "./output-tex"

# Function to read all .tex files and save them into a JSON file
def save_tex_files_to_json(tex_directory, output_json_file):
    tex_files = [f for f in os.listdir(tex_directory) if f.endswith(".tex")]
    tex_data = {}

    for tex_file in tex_files:
        tex_file_path = os.path.join(tex_directory, tex_file)
        
        # Read the .tex file content
        with open(tex_file_path, "r", encoding="utf-8") as file:
            content = file.read()
        
        # Remove all whitespace characters (spaces, tabs, newlines, etc.)
        content_no_spaces = content.replace(" ", "").replace("\n", "").replace("\t", "")

        # Store the modified content in the dictionary with filename as key
        tex_data[tex_file] = content_no_spaces
    
    # Save the dictionary to a JSON file
    with open(output_json_file, "w", encoding="utf-8") as json_file:
        json.dump(tex_data, json_file, ensure_ascii=False, indent=4)

# Output JSON file path
output_json_file = "./tex_files.json"

# Call the function to save the .tex files to the JSON file
save_tex_files_to_json(tex_directory, output_json_file)

print(f"All .tex files have been saved to {output_json_file}")
