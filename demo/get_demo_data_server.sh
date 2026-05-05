#!/bin/bash

# -----------------------------------
# Download + unzip TBL_data & pre_trained_models
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
data_URL="https://www.datadepot.rcac.purdue.edu/bouman/data/TBL_data.zip"
data_ZIPFILE="TBL_data.zip"
data_DEST_FOLDER="./demo/data"

models_URL="https://www.datadepot.rcac.purdue.edu/bouman/data/pre_trained_models.zip"
models_ZIPFILE="pre_trained_models.zip"
models_DEST_FOLDER="./demo/pre_trained_models"

# -----------------------------------
# Helper Function: Download & Extract
# -----------------------------------
fetch_and_unzip() {
    local URL="$1"
    local ZIPFILE="$2"
    local DEST_FOLDER="$3"
    local STRATEGY="$4"  # Accepts "NESTED" or "CONTENTS"

    echo "${green}----------------------------------------${reset}"
    echo "${green}Attempting secure download from:${reset}"
    echo "${green}   $URL${reset}"

    # Download (secure first, fallback to insecure)
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

    # Ensure DEST_FOLDER exists
    if [ ! -d "$DEST_FOLDER" ]; then
        echo "${red}❌ Destination folder does not exist: $DEST_FOLDER${reset}"
        return 1
    fi

    # Unzip to temporary directory
    local TMPDIR=$(mktemp -d)
    echo "${green}Unzipping into temp directory:${reset} $TMPDIR"

    unzip -q "$ZIPFILE" -d "$TMPDIR"
    if [ $? -ne 0 ]; then
        echo "${red}❌ Unzip failed${reset}"
        return 1
    fi

    # Determine if the ZIP is wrapped in a single top-level folder
    local NUM_ITEMS=$(find "$TMPDIR" -mindepth 1 -maxdepth 1 | wc -l)
    local TOP_LEVEL_DIR=$(find "$TMPDIR" -mindepth 1 -maxdepth 1 -type d | head -n 1)

    local SOURCE_DIR
    if [ "$NUM_ITEMS" -eq 1 ] && [ -n "$TOP_LEVEL_DIR" ]; then
        SOURCE_DIR="$TOP_LEVEL_DIR"
    else
        SOURCE_DIR="$TMPDIR"
    fi

    # Execute move based on requested strategy
    if [ "$STRATEGY" = "NESTED" ]; then
        local BASENAME=$(basename "$SOURCE_DIR")
        local TARGET_PATH="$DEST_FOLDER/$BASENAME"

        echo "${green}Moving ${BASENAME} → ${DEST_FOLDER}${reset}"
        mv "$SOURCE_DIR" "$DEST_FOLDER/"
        echo "${green}Extracted folder now located at:${reset} $TARGET_PATH"

    elif [ "$STRATEGY" = "CONTENTS" ]; then
        echo "${green}Moving contents directly into → ${DEST_FOLDER}${reset}"

        # shopt -s dotglob ensures `*` matches hidden files (e.g., .gitignore)
        shopt -s dotglob
        mv "$SOURCE_DIR"/* "$DEST_FOLDER/"
        shopt -u dotglob

        echo "${green}Files extracted directly to:${reset} $DEST_FOLDER"
    else
        echo "${red}❌ Invalid strategy specified: $STRATEGY${reset}"
        return 1
    fi

    # Cleanup
    rm "$ZIPFILE"
    rm -rf "$TMPDIR"

    echo "${green}✔ Complete!${reset}"
    echo ""
}

# -----------------------------------
# Execution
# -----------------------------------

# Process TBL_data (Creates a nested folder inside demo/data)
fetch_and_unzip "$data_URL" "$data_ZIPFILE" "$data_DEST_FOLDER" "NESTED" || return 1

# Process pre_trained_models (Pours contents directly into demo/pre_trained_models)
fetch_and_unzip "$models_URL" "$models_ZIPFILE" "$models_DEST_FOLDER" "CONTENTS" || return 1

echo "${green}All downloads and extractions finished successfully.${reset}"