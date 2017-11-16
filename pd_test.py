import pandas as pd

caller = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3', 'K4', 'K5'],
                       'A': ['A0', 'A1', 'A2', 'A3', 'A4', 'A5']})

other1 = pd.DataFrame({'key': ['K0', 'K1', 'K2'],
                       'B': ['B0', 'B1', 'B2'],
                       'A': ['A6', 'A7', 'A8']})


other = pd.DataFrame({'key': ['K0', 'K1', 'K2'],
                       'B': ['B0', 'B1', 'B2'],
                       'A': ['A6', 'A7', 'A8']})


# Join DataFrames using their indexes.
print other, "A" in other, "ok" in other

print caller.join(other.set_index("key"), lsuffix='_caller', rsuffix='_other', how='inner', on="key")
print pd.merge(caller, other, on="key")

