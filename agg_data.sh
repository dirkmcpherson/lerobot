#!/bin/bash

# --- Configuration ---
# The prefix expected before the number (e.g., "episode_")
readonly PREFIX="episode_"
# The number of digits expected in the numeric part (e.g., 6 for ######)
readonly NUM_DIGITS=6

# --- Functions ---
usage() {
  echo "Usage: $0 -s <source_directory> -d <destination_directory> -a <number_to_add>"
  echo
  echo "Description:"
  echo "  Scans <source_directory> for files matching the pattern '${PREFIX}######.*',"
  echo "  where '######' is a ${NUM_DIGITS}-digit number (e.g., ${PREFIX}000123.mp4)."
  echo "  Adds the integer <number_to_add> to the numeric part of each matching file."
  echo "  Copies the original file to <destination_directory> with the new calculated name,"
  echo "  preserving the ${NUM_DIGITS}-digit format with leading zeros."
  echo
  echo "Options:"
  echo "  -s DIR    Specify the source directory containing the files."
  echo "  -d DIR    Specify the destination directory for the copied files."
  echo "  -a NUM    Specify the integer value to add to the episode numbers."
  echo "  -h        Show this help message and exit."
  exit 1
}

# --- Argument Parsing ---
SOURCE_DIR=""
DEST_DIR=""
ADD_VALUE=""

while getopts "s:d:a:h" opt; do
  case $opt in
    s) SOURCE_DIR="$OPTARG" ;;
    d) DEST_DIR="$OPTARG" ;;
    a) ADD_VALUE="$OPTARG" ;;
    h) usage ;;
    \?) echo "Invalid option: -$OPTARG" >&2; usage ;;
    :) echo "Option -$OPTARG requires an argument." >&2; usage ;;
  esac
done

# --- Input Validation ---
if [[ -z "$SOURCE_DIR" || -z "$DEST_DIR" || -z "$ADD_VALUE" ]]; then
  echo "Error: Source directory (-s), destination directory (-d), and number to add (-a) are required." >&2
  usage
fi

if [[ ! -d "$SOURCE_DIR" ]]; then
  echo "Error: Source directory '$SOURCE_DIR' not found or is not a directory." >&2
  exit 1
fi

# Validate ADD_VALUE is an integer (positive, negative, or zero)
if ! [[ "$ADD_VALUE" =~ ^-?[0-9]+$ ]]; then
    echo "Error: Value to add ('$ADD_VALUE') must be an integer." >&2
    exit 1
fi

# --- Ensure Destination Directory Exists ---
# Use mkdir -p to create parent directories if needed and not error if it exists
mkdir -p "$DEST_DIR"
if [[ $? -ne 0 ]]; then
    echo "Error: Could not create destination directory '$DEST_DIR'." >&2
    exit 1
fi
# Get absolute path for destination for clarity in output
DEST_DIR_ABS="$(cd "$DEST_DIR" && pwd)" || exit 1


echo "Source Directory:      $SOURCE_DIR"
echo "Destination Directory: $DEST_DIR_ABS"
echo "Value to Add:          $ADD_VALUE"
echo "Filename Pattern:      ${PREFIX}<${NUM_DIGITS}_digits>.<extension>"
echo "---"

# --- Main Processing Loop ---

# Use nullglob to prevent loop execution if no files match the pattern
shopt -s nullglob

processed_count=0
skipped_count=0
error_count=0

# Regex to capture number and extension
# Matches the literal PREFIX, exactly NUM_DIGITS digits (group 1),
# a literal dot, and the rest of the filename (group 2, the extension)
regex="${PREFIX}([0-9]{${NUM_DIGITS}})\.(.+)$"

# Loop through all files in the source directory
for file in "$SOURCE_DIR"/*; do
    # Skip if not a regular file
    [[ -f "$file" ]] || continue

    filename=$(basename "$file")

    # Check if the filename matches the required pattern
    if [[ "$filename" =~ $regex ]]; then
        current_num_str="${BASH_REMATCH[1]}"
        extension="${BASH_REMATCH[2]}"

        # Convert current number string to integer (explicitly base 10 for safety)
        # Bash's ((...)) usually handles leading zeros correctly as decimal
        current_num=$((10#$current_num_str))

        # Calculate the new number
        new_num=$((current_num + ADD_VALUE))

        # --- Validation for the new number ---
        # Check if the new number is negative
        if (( new_num < 0 )); then
            echo "Skipping '$filename': Calculated new number ($new_num) is negative."
            ((skipped_count++))
            continue
        fi

        # Optional: Check if new number exceeds the intended digit count significantly.
        # This prevents `episode_999999.txt + 1` becoming `episode_1000000.txt` if you
        # strictly want to maintain the original number of digits space.
        # However, printf will handle larger numbers correctly, just maybe not what you expect visually.
        # max_val_exclusive=$((10**NUM_DIGITS))
        # if (( new_num >= max_val_exclusive )); then
        #     echo "Warning: New number for '$filename' ($new_num) exceeds ${NUM_DIGITS} digits."
        #     # Decide whether to skip or proceed (here we proceed but warn)
        # fi

        # Format the new number with leading zeros to match NUM_DIGITS width
        # E.g., printf "%06d" 123 -> 000123
        # E.g., printf "%06d" 1000000 -> 1000000 (will exceed width if needed)
        new_num_formatted=$(printf "%0${NUM_DIGITS}d" "$new_num")

        # Construct the new filename and the full destination path
        new_filename="${PREFIX}${new_num_formatted}.${extension}"
        new_path="$DEST_DIR_ABS/$new_filename"

        # Check for potential overwrite (optional, uncomment if needed)
        # if [[ -e "$new_path" ]]; then
        #   echo "Skipping '$filename': Target file '$new_path' already exists."
        #   ((skipped_count++))
        #   continue
        # fi

        # Copy the file
        echo "Processing: '$filename' -> '$new_filename'"
        cp -p "$file" "$new_path" # Use -p to preserve mode, ownership, timestamps
        if [[ $? -eq 0 ]]; then
             ((processed_count++))
        else
            echo "Error: Failed to copy '$file' to '$new_path'." >&2
            ((error_count++))
        fi
    # else
        # Optional: uncomment to see which files *don't* match the pattern
        # echo "Debug: Skipping '$filename' (does not match pattern '$regex')"
    fi
done

# Restore default glob behavior
shopt -u nullglob

echo "---"
echo "Processing complete."
echo "Files successfully copied: $processed_count"
echo "Files skipped (validation): $skipped_count"
echo "Copy errors: $error_count"

# Exit with success code if no errors occurred during copy
if (( error_count == 0 )); then
  exit 0
else
  exit 1 # Exit with error code if copy errors happened
fi