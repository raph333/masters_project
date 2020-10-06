import os
import pandas as pd

pd.options.display.max_columns = 15


df_old = pd.concat([
        pd.read_csv('/scratch1/rpeer/tmp/competition_data/raw/dev/dev_target.csv'),
        pd.read_csv('/scratch1/rpeer/tmp/competition_data/raw/valid/valid_target.csv')
    ], ignore_index=True)
df_old = df_old.set_index('gdb_idx')
print(df_old.shape)
ranges_old = df_old.apply(lambda col: col.max() - col.min())
sdev_old = df_old.apply(lambda col: col.std())
print(ranges_old.mean())
print(sdev_old.mean())

common_ids = set(df_old.index)

df_new = pd.read_csv('/scratch1/rpeer/tmp/full_data/raw/ground_truth.csv').drop('atom number', axis=1)
df_new = df_new.set_index('gdb_idx')
# df_new = df_new[df_new.index.isin(common_ids)]
print(df_new.shape)

df_new_norm = (df_new - df_new.mean()) / df_new.std()

ranges_new = df_new_norm.apply(lambda col: col.max() - col.min())
sdev_new = df_new_norm.apply(lambda col: col.std())
print(ranges_new.mean())
print(sdev_new.mean())


# dev_ids = pd.read_csv('/scratch1/rpeer/tmp/old_data/raw/dev/dev_target.csv').gdb_idx.tolist()
# val_ids = pd.read_csv('/scratch1/rpeer/tmp/old_data/raw/valid/valid_target.csv').gdb_idx.tolist()
# non_test = set(dev_ids + val_ids)
# tst_ids = [i for i in df_new.index if i not in non_test]
# assert set(dev_ids).union(val_ids).union(tst_ids).symmetric_difference(set(df_new.index)) == set()
#
# id_df = pd.concat([
#     pd.DataFrame({'set_': ['test'] * len(dev_ids), 'gdb_idx': dev_ids}),
#     pd.DataFrame({'set_': ['valid'] * len(val_ids), 'gdb_idx': val_ids}),
#     pd.DataFrame({'set_': ['dev']*len(tst_ids), 'gdb_idx': tst_ids})
# ])
# id_df.to_csv('../../../old_ds_split.csv', index=False)