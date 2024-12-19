#!/bin/bash



source_dir="backup_cvrp_instances"
dest_dir="cvrp_instances"

# Check if directories exist
if [ ! -d "$source_dir" ]; then
    echo "Error: Source directory $source_dir does not exist"
    exit 1
fi

if [ ! -d "$dest_dir" ]; then
    echo "Error: Destination directory $dest_dir does not exist"
    exit 1
fi

# Get all .sol and .vrp files
files=("$source_dir"/*.{sol,vrp})
total_files=${#files[@]}

if [ $total_files -eq 0 ]; then
    echo "No .sol or .vrp files found in $source_dir"
    exit 1
fi

# Calculate number of files to move (20%)
move_count=$((total_files / 5))

# Randomly select and move files
selected=($(printf "%s\n" "${files[@]}" | sort -R | head -n $move_count))

for file in "${selected[@]}"; do
    if [ -f "$file" ]; then
        echo "Moving $(basename "$file")"
        mv -n "$file" "$dest_dir/"
    fi
done

echo "Moved $move_count files to $dest_dir"