import pandas as pd
import numpy as np


#converts data to excel to find the frequency of numbers
def data_to_excel():
    train_data = pd.read_csv('emnist-letters-train.csv')
    test_data = pd.read_csv('emnist-letters-test.csv')
    train_alphabet_data = train_data.ix[:, 0]
    test_alphabet_data = test_data.ix[:, 0]
    train_alphabets, train_frequency = np.unique(train_alphabet_data, return_counts=True)
    test_alphabets, test_frequency = np.unique(test_alphabet_data, return_counts=True)
    train_frequency_table = np.asarray((train_alphabets, train_frequency))
    test_frequency_table = np.asarray((test_alphabets, test_frequency))
    train_df = pd.DataFrame(train_frequency_table.T)
    test_df = pd.DataFrame(test_frequency_table.T)
    train_df.to_excel("train_frequency.xlsx", index=False)
    test_df.to_excel("test_frequency.xlsx", index=False)

#gets required data for training
def extract_data(input_file, output_file):
    data = pd.read_csv(input_file).as_matrix()
    list = []
    m = data.shape[0]
    for i in range(0, m):
        if data[i][0] in chosen_digits_set:
            list.append(data[i])
    list = np.array(list)
    df = pd.DataFrame(list)
    df.to_csv(output_file, encoding='utf-8', index=False)

chosen_digits = {1: 0, 2: 1, 8: 2, 9:3, 10: 4, 11: 5, 12: 6, 13: 7, 18: 8}
chosen_digits_set = set(chosen_digits.keys())

extract_data('emnist-letters-train.csv',"final_training_data.csv")
extract_data('emnist-letters-test.csv', "final_test_data.csv")

def myOneHotEncoder(output_array):
    ndigits = len(chosen_digits)
    list = []
    for label in output_array:
        encoded_label = [0 for x in range(len(chosen_digits))]
        encoded_label[chosen_digits[label[0]]] = 1
        list.append(encoded_label)
    return np.array(list)

# Test case for myOneHotEncoder
#ar = np.array([1,2,8,9,10,11,12,18,1,2,8,9])
#a = myOneHotEncoder(ar.reshape(12, 1))
#print a