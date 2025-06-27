#!/bin/bash

TARGET_DIR_PETS="./dataset/amazon/raw/pets"
TARGET_DIR_OFFICE="./dataset/amazon/raw/office"

if [ ! -d "$TARGET_DIR_PETS" ]; then
  echo "Directory $TARGET_DIR_PETS does not exist. Creating..."
  mkdir -p "$TARGET_DIR_PETS"
else
  echo "Directory $TARGET_DIR_PETS already exists."
fi

if [ ! -d "$TARGET_DIR_OFFICE" ]; then
  echo "Directory $TARGET_DIR_OFFICE does not exist. Creating..."
  mkdir -p "$TARGET_DIR_OFFICE"
else
  echo "Directory $TARGET_DIR_OFFICE already exists."
fi

cd "$TARGET_DIR_PETS" || exit 1

# Pets dataset
wget -O meta.json.gz "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Pet_Supplies.json.gz"
wget -O reviews.json.gz "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Pet_Supplies.json.gz"

gunzip -k meta.json.gz
gunzip -k reviews.json.gz

cd ../office || exit 1

# Office dataset
wget -O meta.json.gz "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Office_Products.json.gz"
wget -O reviews.json.gz "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Office_Products.json.gz"

gunzip -k meta.json.gz
gunzip -k reviews.json.gz