import pandas as pd

timestamp = '20250410-1415'

file1 = f'/home/vkaryakina/OB/unet-olfactory-bulb-segmentation/experiments.ave_4n7_140_20_140/{timestamp}/metrics/{timestamp}__boundary_mistakes.csv'  # путь к первому CSV файлу
file2 = f'/home/vkaryakina/OB/unet-olfactory-bulb-segmentation/experiments.ave_4n7_140_20_140/{timestamp}/metrics/{timestamp}__boundary_mistakes_weighted.csv'  # путь ко второму CSV файлу

df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

boundary_columns = ['Boundary Error (0-1)', 'Boundary Error (1-2)', 'Boundary Error (2-3)', 'Boundary Error (3-4)', 'Boundary Error (4-5)']
diff_columns = ['Diff > 1 (0)', 'Diff > 1 (1)', 'Diff > 1 (2)', 'Diff > 1 (3)', 'Diff > 1 (4)', 'Diff > 1 (5)']

df1['common mistake'] = df1[boundary_columns].sum(axis=1) + df1[diff_columns].sum(axis=1)

df1['common mistake weighted'] = df2[boundary_columns].sum(axis=1) + df2[diff_columns].sum(axis=1)

df1['mistake ratio'] = df1['common mistake'] / df1['common mistake weighted']

# df1 = df1[['Image', 'common mistake', 'common mistake weighted', 'mistake ratio']].sort_values(by='mistake ratio', ascending=False)

output_file = f'/home/vkaryakina/OB/unet-olfactory-bulb-segmentation/experiments.ave_4n7_140_20_140/{timestamp}/metrics/{timestamp}_ratio.csv'
df1[['Image', 'common mistake', 'common mistake weighted', 'mistake ratio']].to_csv(output_file, index=False)

print(df1[['Image', 'common mistake', 'common mistake weighted', 'mistake ratio']])