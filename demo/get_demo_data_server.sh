#!/bin/bash

# -----------------------------------
# Download + unzip TBL_data.zip
# Safe for:  source script.sh
# -----------------------------------

# Colors
red=$(tput setaf 1)
yellow=$(tput setaf 3)
green=$(tput setaf 2)
reset=$(tput sgr0)

# -----------------------------------
# Configuration
# -----------------------------------
URL="https://www.datadepot.rcac.purdue.edu/bouman/data/TBL_data.zip"
ZIPFILE="TBL_data.zip"
DEST_FOLDER="./demo/data"     # ← this must already exist

echo "${green}Attempting secure download from:${reset}"
echo "${green}   $URL${reset}"

# -----------------------------------
# Download (secure first, fallback to insecure)
# -----------------------------------
curl -L --fail "$URL" -o "$ZIPFILE" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "${yellow}SSL validation failed (expected for DataDepot).${reset}"
    echo "${yellow}Retrying with --insecure ...${reset}"

    curl -L -k "$URL" -o "$ZIPFILE"
    if [ $? -ne 0 ]; then
        echo "${red}❌ Download failed even in insecure mode.${reset}"
        return 1
    fi
fi

echo "${green}✔ Download successful:${reset} $ZIPFILE"

# -----------------------------------
# Ensure DEST_FOLDER exists
# -----------------------------------
if [ ! -d "$DEST_FOLDER" ]; then
    echo "${red}❌ DEST_FOLDER does not exist: $DEST_FOLDER${reset}"
    return 1
fi

# -----------------------------------
# Unzip to temporary directory
# -----------------------------------
TMPDIR=$(mktemp -d)
echo "${green}Unzipping into temp directory:${reset} $TMPDIR"

unzip -q "$ZIPFILE" -d "$TMPDIR"
if [ $? -ne 0 ]; then
    echo "${red}❌ Unzip failed${reset}"
    return 1
fi

# Find top-level folder (expected: TBL_data)
TOP_LEVEL_DIR=$(find "$TMPDIR" -mindepth 1 -maxdepth 1 -type d | head -n 1)
BASENAME=$(basename "$TOP_LEVEL_DIR")

if [ -z "$TOP_LEVEL_DIR" ]; then
    echo "${red}❌ No folder found inside ZIP${reset}"
    return 1
fi

echo "${green}Found extracted folder:${reset} $BASENAME"

# -----------------------------------
# Move folder *inside* existing DEST_FOLDER
# -----------------------------------
TARGET_PATH="$DEST_FOLDER/$BASENAME"

echo "${green}Moving ${BASENAME} → ${DEST_FOLDER}${reset}"
mv "$TOP_LEVEL_DIR" "$TARGET_PATH"

# -----------------------------------
# Cleanup
# -----------------------------------
rm "$ZIPFILE"
rm -rf "$TMPDIR"

echo "${green}✔ Complete!${reset}"
echo "${green}Extracted folder now located at:${reset} $TARGET_PATH"
``
