#!/bin/bash

# Remove all directories ./gan/i/ where i is 10, 20, 30...
for i in {1..1000}; do
    dir="./vae/$i/"
    if [ -d "$dir" ]; then
        echo "Removing directory: $dir"
        rm -rf "$dir"
    else
        echo "Directory does not exist: $dir"
    fi
done