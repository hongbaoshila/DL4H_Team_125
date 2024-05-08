import os
import re
import pandas as pd
from sklearn.model_selection import KFold


def preprocess1(x):
    y = re.sub('\\[(.*?)\\]', '', x)  # remove de-identified brackets
    y = re.sub('[0-9]+\.', '', y)  # remove 1.2. since the segmenter segments based on this
    y = re.sub('dr\.', 'doctor', y)
    y = re.sub('m\.d\.', 'md', y)
    y = re.sub('admission date:', '', y)
    y = re.sub('discharge date:', '', y)
    y = re.sub('--|__|==', '', y)
    return y


def preprocessing(df_less_n, tokenizer):
    df_less_n['TEXT'] = df_less_n['TEXT'].fillna(' ')
    df_less_n['TEXT'] = df_less_n['TEXT'].str.replace('\n', ' ')
    df_less_n['TEXT'] = df_less_n['TEXT'].str.replace('\r', ' ')
    df_less_n['TEXT'] = df_less_n['TEXT'].apply(str.strip)
    df_less_n['TEXT'] = df_less_n['TEXT'].str.lower()

    df_less_n['TEXT'] = df_less_n['TEXT'].apply(lambda x: preprocess1(x))

    sen = df_less_n['TEXT'].values
    tokenized_texts = [tokenizer.tokenize(x) for x in sen]
    # print("First sentence tokenized")
    # print(tokenized_texts[0])
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
    df_less_n['Input_ID'] = input_ids
    return df_less_n[['Adm_ID', 'Note_ID', 'TEXT', 'Input_ID', 'Label', 'chartdate', 'charttime']]


def main():
    # df = pd.read_csv("MICROBIOLOGYEVENTS.csv").fillna("")
    # df_ecoli = df.loc[df['ORG_ITEMID'] == 80002]
    # df_ecoli.to_csv("ecoli.csv")
    # df_nonecoli = df.loc[df['ORG_ITEMID'] != 80002]
    # df_nonecoli.to_csv("nonecoli.csv")
    #
    # df_ecoli = pd.read_csv("ecoli.csv")
    # ecoli_id = df_ecoli['HADM_ID'].unique()
    # df_nonecoli = pd.read_csv("nonecoli.csv")
    # nonecoli_id = df_nonecoli['HADM_ID'].unique()
    #
    original_df = pd.read_csv('NOTEEVENTS.csv')
    # keep_col = ["ROW_ID","HADM_ID","CHARTDATE","CHARTTIME","TEXT"]
    # new_f = original_df[keep_col]
    # for i in range(int(np.ceil(len(new_f) / 10000))):
    #     label = []
    #     df_chunk = new_f.iloc[i * 10000:(i + 1) * 10000].copy()
    #     for index, row in df_chunk.iterrows():
    #         if row['HADM_ID'] in ecoli_id:
    #             label.append(1)
    #         elif row['HADM_ID'] in nonecoli_id:
    #             label.append(0)
    #         else:
    #             df_chunk.drop(index, inplace=True)
    #     df_chunk['LABEL'] = label
    #     temp_file_dir = './tmp/Processed_{}.csv'.format(i)
    #     df_chunk.to_csv(temp_file_dir, index=False)

    # tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
    # for i in range(int(np.ceil(len(original_df) / 10000))):
    #     temp_file_dir = './tmp/Processed_{}.csv'.format(i)
    #     df_chunk = pd.read_csv(temp_file_dir, header=0)
    #     df_processed_chunk = df_chunk[['HADM_ID', 'ROW_ID', 'CHARTDATE', 'CHARTTIME', 'TEXT','LABEL']]
    #     df_processed_chunk.rename(columns={'HADM_ID': "Adm_ID",
    #                                 'ROW_ID': "Note_ID",
    #                                 'CHARTDATE': "chartdate",
    #                                 'CHARTTIME': "charttime",
    #                                 'TEXT': "TEXT",
    #                                 'LABEL': "Label"}, inplace=True)
    #     df_processed_chunk = df_processed_chunk.astype({'Adm_ID': 'int64', 'Note_ID': 'int64', 'Label': 'int64'})
    #     df_processed_chunk = preprocessing(df_processed_chunk, tokenizer)
    #     temp_file_dir = './tmp1/Processed_{}.csv'.format(i)
    #     df_processed_chunk.to_csv(temp_file_dir, index=False)

    # df = pd.DataFrame({'Adm_ID': [], 'Note_ID': [], 'TEXT': [], 'Input_ID': [],
    #                    'Label': [], 'chartdate': [], 'charttime': []})
    # for i in range(40):
    #     temp_file_dir = './tmp1/Processed_{}.csv'.format(i)
    #     df_chunk = pd.read_csv(temp_file_dir, header=0)
    #     df = df.append(df_chunk, ignore_index=True)
    # df.to_csv('./Ecoli/data.csv', index=False)
    #
    # result = df.Label.value_counts()
    # print("In the full dataset Positive Patients' Notes: {}, Negative Patients' Notes: {}".format(result[1], result[0]))
    # ecoli_ID = pd.Series(df[df.Label == 1].Adm_ID.unique())
    # not_ecoli_ID = pd.Series(df[df.Label == 0].Adm_ID.unique())
    # print("Total Positive Patients' ids: {}, Total Negative Patients' ids: {}".format(len(ecoli_ID), len(not_ecoli_ID)))

    # df = pd.read_csv('./Ecoli/data.csv', header=0)
    # ecoli_ID = pd.Series(df[df.Label == 1].Adm_ID.unique())
    # not_ecoli_ID = pd.Series(df[df.Label == 0].Adm_ID.unique())
    #
    # RANDOM_SEED = 1
    # not_ecoli_ID_use = not_ecoli_ID.sample(n=500, random_state=RANDOM_SEED)
    # ecoli_ID_use = ecoli_ID.sample(n=500, random_state=RANDOM_SEED)
    #
    # Kfold = None
    # if Kfold is None:
    #     id_val_test_t = ecoli_ID_use.sample(frac=0.2, random_state=RANDOM_SEED)
    #     id_val_test_f = not_ecoli_ID_use.sample(frac=0.2, random_state=RANDOM_SEED)
    #
    #     id_train_t = ecoli_ID_use.drop(id_val_test_t.index)
    #     id_train_f = not_ecoli_ID_use.drop(id_val_test_f.index)
    #
    #     id_val_t = id_val_test_t.sample(frac=0.5, random_state=RANDOM_SEED)
    #     id_test_t = id_val_test_t.drop(id_val_t.index)
    #     id_val_f = id_val_test_f.sample(frac=0.5, random_state=RANDOM_SEED)
    #     id_test_f = id_val_test_f.drop(id_val_f.index)
    #
    #     id_test = pd.concat([id_test_t, id_test_f])
    #     test_id_label = pd.DataFrame(data=list(zip(id_test, [1] * len(id_test_t) + [0] * len(id_test_f))),
    #                                  columns=['id', 'label'])
    #
    #     id_val = pd.concat([id_val_t, id_val_f])
    #     val_id_label = pd.DataFrame(data=list(zip(id_val, [1] * len(id_val_t) + [0] * len(id_val_f))),
    #                                 columns=['id', 'label'])
    #
    #     id_train = pd.concat([id_train_t, id_train_f])
    #     train_id_label = pd.DataFrame(data=list(zip(id_train, [1] * len(id_train_t) + [0] * len(id_train_f))),
    #                                   columns=['id', 'label'])
    #
    #     mortality_train = df[df.Adm_ID.isin(train_id_label.id)]
    #     mortality_val = df[df.Adm_ID.isin(val_id_label.id)]
    #     mortality_test = df[df.Adm_ID.isin(test_id_label.id)]
    #     mortality_not_use = df[
    #         (~df.Adm_ID.isin(train_id_label.id)) & (
    #                     ~df.Adm_ID.isin(val_id_label.id) & (~df.Adm_ID.isin(test_id_label.id)))]
    #
    #     train_result = mortality_train.Label.value_counts()
    #
    #     val_result = mortality_val.Label.value_counts()
    #
    #     test_result = mortality_test.Label.value_counts()
    #
    #     no_result = mortality_not_use.Label.value_counts()
    #
    #     mortality_train.to_csv('./Ecoli/train.csv', index=False)
    #     mortality_val.to_csv('./Ecoli/val.csv', index=False)
    #     mortality_test.to_csv('./Ecoli/test.csv', index=False)
    #     mortality_not_use.to_csv('./Ecoli/not_use.csv', index=False)
    #     df.to_csv('./Ecoli/full.csv', index=False)
    #
    #     if len(no_result) == 2:
    #         print(("In the train dataset Positive Patients' Notes: {}, Negative Patients' Notes: {}\n"
    #                "In the validation dataset Positive Patients' Notes: {}, Negative Patients' Notes: {}\n"
    #                "In the test dataset Positive Patients' Notes: {}, Negative Patients' Notes: {}\n"
    #                "In the not use dataset Positive Patients' Notes: {}, Negative Patients' Notes: {}").format(
    #             train_result[1],
    #             train_result[0],
    #             val_result[1],
    #             val_result[0],
    #             test_result[1],
    #             test_result[0],
    #             no_result[1],
    #             no_result[0]))
    #     else:
    #         try:
    #             print(("In the train dataset Positive Patients' Notes: {}, Negative  Patients' Notes: {}\n"
    #                    "In the validation dataset Positive Patients' Notes: {}, Negative Patients' Notes: {}\n"
    #                    "In the test dataset Positive Patients' Notes: {}, Negative Patients' Notes: {}\n"
    #                    "In the not use dataset Negative Patients' Notes: {}").format(train_result[1],
    #                                                                                  train_result[0],
    #                                                                                  val_result[1],
    #                                                                                  val_result[0],
    #                                                                                  test_result[1],
    #                                                                                  test_result[0],
    #                                                                                  no_result[0])
    #                   )
    #         except KeyError:
    #             print(("In the train dataset Positive Patients' Notes: {}, Negative  Patients' Notes: {}\n"
    #                    "In the validation dataset Positive Patients' Notes: {}, Negative Patients' Notes: {}\n"
    #                    "In the test dataset Positive Patients' Notes: {}, Negative Patients' Notes: {}\n"
    #                    "In the not use dataset Positive Patients' Notes: {}").format(train_result[1],
    #                                                                                  train_result[0],
    #                                                                                  val_result[1],
    #                                                                                  val_result[0],
    #                                                                                  test_result[1],
    #                                                                                  test_result[0],
    #                                                                                  no_result[1])
    #                   )
    #
    # else:
    #     folds_t = KFold(Kfold, False, RANDOM_SEED)
    #     folds_f = KFold(Kfold, False, RANDOM_SEED)
    #     ecoli_ID_use.reset_index(inplace=True, drop=True)
    #     not_ecoli_ID_use.reset_index(inplace=True, drop=True)
    #     for num, ((train_t, test_t), (train_f, test_f)) in enumerate(zip(folds_t.split(ecoli_ID_use),
    #                                                                      folds_f.split(not_ecoli_ID_use))):
    #         id_train_t = ecoli_ID_use[train_t]
    #         id_val_test_t = ecoli_ID_use[test_t]
    #         id_train_f = not_ecoli_ID_use[train_f]
    #         id_val_test_f = not_ecoli_ID_use[test_f]
    #         id_val_t = id_val_test_t.sample(frac=0.5, random_state=RANDOM_SEED)
    #         id_test_t = id_val_test_t.drop(id_val_t.index)
    #         id_val_f = id_val_test_f.sample(frac=0.5, random_state=RANDOM_SEED)
    #         id_test_f = id_val_test_f.drop(id_val_f.index)
    #
    #         id_test = pd.concat([id_test_t, id_test_f])
    #         test_id_label = pd.DataFrame(data=list(zip(id_test, [1] * len(id_test_t) + [0] * len(id_test_f))),
    #                                      columns=['id', 'label'])
    #
    #         id_val = pd.concat([id_val_t, id_val_f])
    #         val_id_label = pd.DataFrame(data=list(zip(id_val, [1] * len(id_val_t) + [0] * len(id_val_f))),
    #                                     columns=['id', 'label'])
    #
    #         id_train = pd.concat([id_train_t, id_train_f])
    #         train_id_label = pd.DataFrame(data=list(zip(id_train, [1] * len(id_train_t) + [0] * len(id_train_f))),
    #                                       columns=['id', 'label'])
    #
    #         mortality_train = df[df.Adm_ID.isin(train_id_label.id)]
    #         mortality_val = df[df.Adm_ID.isin(val_id_label.id)]
    #         mortality_test = df[df.Adm_ID.isin(test_id_label.id)]
    #         mortality_not_use = df[
    #             (~df.Adm_ID.isin(train_id_label.id)) & (
    #                     ~df.Adm_ID.isin(val_id_label.id) & (~df.Adm_ID.isin(test_id_label.id)))]
    #
    #         train_result = mortality_train.Label.value_counts()
    #
    #         val_result = mortality_val.Label.value_counts()
    #
    #         test_result = mortality_test.Label.value_counts()
    #
    #         no_result = mortality_not_use.Label.value_counts()
    #
    #         mortality_train.to_csv('./Ecoli/train.csv', index=False)
    #         mortality_val.to_csv('./Ecoli/val.csv', index=False)
    #         mortality_test.to_csv('./Ecoli/test.csv', index=False)
    #         mortality_not_use.to_csv('./Ecoli/not_use.csv', index=False)
    #         df.to_csv(os.path.join('./Ecoli/', str(num), 'full.csv'), index=False)
    #
    #         if len(no_result) == 2:
    #             print(("In the {}th split of {} folds\n"
    #                    "In the train dataset Positive Patients' Notes: {}, Negative Patients' Notes: {}\n"
    #                    "In the validation dataset Positive Patients' Notes: {}, Negative Patients' Notes: {}\n"
    #                    "In the test dataset Positive Patients' Notes: {}, Negative Patients' Notes: {}\n"
    #                    "In the not use dataset Positive Patients' Notes: {}, Negative Patients' Notes: {}").format(
    #                 num,
    #                 Kfold,
    #                 train_result[1],
    #                 train_result[0],
    #                 val_result[1],
    #                 val_result[0],
    #                 test_result[1],
    #                 test_result[0],
    #                 no_result[1],
    #                 no_result[0])
    #             )
    #         else:
    #             try:
    #                 print(("In the {}th split of {} folds\n"
    #                        "In the train dataset Positive Patients' Notes: {}, Negative  Patients' Notes: {}\n"
    #                        "In the validation dataset Positive Patients' Notes: {}, Negative Patients' Notes: {}\n"
    #                        "In the test dataset Positive Patients' Notes: {}, Negative Patients' Notes: {}\n"
    #                        "In the not use dataset Negative Patients' Notes: {}").format(num,
    #                                                                                      Kfold,
    #                                                                                      train_result[1],
    #                                                                                      train_result[0],
    #                                                                                      val_result[1],
    #                                                                                      val_result[0],
    #                                                                                      test_result[1],
    #                                                                                      test_result[0],
    #                                                                                      no_result[0])
    #                       )
    #             except KeyError:
    #                 print(("In the {}th split of {} folds\n"
    #                        "In the train dataset Positive Patients' Notes: {}, Negative  Patients' Notes: {}\n"
    #                        "In the validation dataset Positive Patients' Notes: {}, Negative Patients' Notes: {}\n"
    #                        "In the test dataset Positive Patients' Notes: {}, Negative Patients' Notes: {}\n"
    #                        "In the not use dataset Positive Patients' Notes: {}").format(num,
    #                                                                                      Kfold,
    #                                                                                      train_result[1],
    #                                                                                      train_result[0],
    #                                                                                      val_result[1],
    #                                                                                      val_result[0],
    #                                                                                      test_result[1],
    #                                                                                      test_result[0],
    #                                                                                      no_result[1])
    #                       )

    # df = pd.read_csv("./Ecoli/train1.csv")
    # df = df.drop('TEXT', axis=1)
    # df.to_csv('./Ecoli/train.csv', index=False)
    #
    # df1 = pd.read_csv("./Ecoli/test1.csv")
    # df1 = df1.drop('TEXT', axis=1)
    # df1.to_csv('./Ecoli/test.csv', index=False)

#     df2 = pd.read_csv("./Ecoli/val1.csv")
#     df2 = df2.drop('TEXT', axis=1)
#     df2.to_csv('./Ecoli/val.csv', index=False)
#
#
# if __name__ == "__main__":
#     main()
