#!/usr/bin/env python
"""
Inspect Source Data.xlsx to understand its structure
"""

import pandas as pd
from pathlib import Path

print("="*70)
print("INSPECTING SOURCE DATA.XLSX")
print("="*70)

data_file = Path('data/raw/Source Data.xlsx')

if not data_file.exists():
    print(f"✗ File not found: {data_file}")
    exit(1)

# Load all sheets
print("\nLoading Excel file...")
excel_file = pd.ExcelFile(data_file)

print(f"✓ Found {len(excel_file.sheet_names)} sheet(s):")
print()

# Inspect each sheet
for sheet_name in excel_file.sheet_names:
    print("="*70)
    print(f"SHEET: '{sheet_name}'")
    print("="*70)
    
    df = pd.read_excel(data_file, sheet_name=sheet_name)
    
    print(f"Shape: {df.shape} (rows × columns)")
    print(f"\nColumns ({len(df.columns)}):")
    for i, col in enumerate(df.columns, 1):
        # Show non-null count
        non_null = df[col].notna().sum()
        dtype = df[col].dtype
        print(f"  {i:2d}. {col:<40} | Non-null: {non_null:4d}/{len(df):4d} | Type: {dtype}")
    
    print(f"\nFirst 3 rows:")
    print(df.head(3).to_string(max_colwidth=50))
    print()
