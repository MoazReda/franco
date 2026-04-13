"""
Prepare 500 best Franco sentences for manual annotation.
Exports to Excel file for easy translation.
"""

import pandas as pd
import numpy as np

# Load processed dataset
df = pd.read_csv('data/processed/franco_dataset_v1.csv')

print(f"Total dataset: {len(df):,} messages")

# Select best 500 sentences for annotation
# Strategy: high franco_ratio + diverse sources + ideal length
df_candidates = df[
    (df['franco_ratio'] >= 0.7) &      # High Franco quality
    (df['word_count'] >= 4) &           # Not too short
    (df['word_count'] <= 15)            # Not too long - easier to translate
].copy()

print(f"Candidates after filter: {len(df_candidates):,}")

# Get diverse sample - mix of sources
whatsapp_sample = df_candidates[
    df_candidates['source_type'] == 'whatsapp'
].sample(400, random_state=42)

youtube_sample = df_candidates[
    df_candidates['source_type'] == 'youtube'
].sample(min(80, len(df_candidates[df_candidates['source_type']=='youtube'])), 
         random_state=42)

reddit_sample = df_candidates[
    df_candidates['source_type'] == 'reddit'
].sample(min(20, len(df_candidates[df_candidates['source_type']=='reddit'])), 
         random_state=42)

# Combine
annotation_df = pd.concat([whatsapp_sample, youtube_sample, reddit_sample])
annotation_df = annotation_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Prepare annotation sheet
annotation_sheet = pd.DataFrame({
    'id': range(1, len(annotation_df) + 1),
    'franco': annotation_df['franco'].values,
    'arabic': '',        
    'english': '',       
    'source': annotation_df['source_type'].values,
    'franco_ratio': annotation_df['franco_ratio'].round(2).values,
    'word_count': annotation_df['word_count'].values,
    'notes': ''          
})

# Save to Excel
output_path = 'data/annotated/annotation_sheet.xlsx'

with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    annotation_sheet.to_excel(writer, index=False, sheet_name='Annotation')
    
    # Format the sheet
    workbook = writer.book
    worksheet = writer.sheets['Annotation']
    
    # Set column widths
    worksheet.column_dimensions['A'].width = 5   # id
    worksheet.column_dimensions['B'].width = 50  # franco
    worksheet.column_dimensions['C'].width = 50  # arabic
    worksheet.column_dimensions['D'].width = 50  # english
    worksheet.column_dimensions['E'].width = 12  # source
    worksheet.column_dimensions['F'].width = 12  # franco_ratio
    worksheet.column_dimensions['G'].width = 10  # word_count
    worksheet.column_dimensions['H'].width = 20  # notes

print(f"\n✓ Annotation sheet saved to: {output_path}")
print(f"\n=== ANNOTATION SHEET SUMMARY ===")
print(f"Total sentences: {len(annotation_sheet)}")
print(f"\nSource breakdown:")
print(annotation_sheet['source'].value_counts())
print(f"\nAvg word count: {annotation_sheet['word_count'].mean():.1f}")
print(f"Avg franco ratio: {annotation_sheet['franco_ratio'].mean():.2%}")
print(f"\nSample of sentences to translate:")
for _, row in annotation_sheet.head(10).iterrows():
    print(f"  {row['id']:3}. {row['franco'][:60]}")