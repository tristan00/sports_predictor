import pandas as pd



class DataManager():
    def __init__(self, file_path):
        with open(file_path):
            self.raw_data = pd.read_csv(file_path, sep = '|')

        for i in self.raw_data.columns:
            a = self.raw_data[i]
            a


if __name__ == '__main__':
    manager = DataManager(r'C:\Users\trist\OneDrive\Desktop\mma_data/raw_1561253613.csv')
    a = 1