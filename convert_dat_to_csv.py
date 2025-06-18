import pandas as pd

# Read the .dat file
dat_file = 'EoS_DD2MEV_p15_CSS1_1.4_ncrit_0.335561_ecrit_334.285_pcrit_42.1651.dat'
df = pd.read_csv(dat_file, sep='\t', header=None, names=['epsilon', 'p', 'n', 'mu'])

# Save as CSV
csv_file = dat_file.replace('.dat', '.csv')
df.to_csv(csv_file, index=False)

print(f"Converted {dat_file} to {csv_file}")