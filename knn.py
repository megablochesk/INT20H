import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split


def PrepareData(data):
    columns = data.columns.copy().to_numpy()
    columns = np.delete(columns, 0)
    columns = np.delete(columns, 0)

    # print(data.head())

    def mergeFunc(df):
        values = df.iloc[0].to_numpy()[:-1]
        values = np.concatenate((values, df.iloc[1].to_numpy()[:-1]))
        values = np.concatenate((values, df.iloc[2].to_numpy()[:-1]))
        values = np.concatenate((values, df.iloc[3].to_numpy()))
        columnsWT = np.delete(columns, -1)

        resColumns = columnsWT + '_0'
        resColumns = np.concatenate((resColumns, columnsWT + '_1'))
        resColumns = np.concatenate((resColumns, columnsWT + '_2'))
        resColumns = np.concatenate((resColumns, columnsWT + '_3'))
        resColumns = np.concatenate((resColumns, ['target']))
        return pd.Series(values, index=resColumns)

    data = data.groupby('Id', as_index=False)[columns].apply(mergeFunc)

    return data


def main():
    trainData = pd.read_csv("train.csv")

    trainData = PrepareData(trainData)
    trainData = trainData.dropna()

    X_train = trainData.drop("target", axis=1)
    y_train = trainData["target"]

    testData = pd.read_csv("test.csv")
    testData = PrepareData(testData)

    X_test = testData.drop("Id", axis=1)
    ids = testData["Id"]

    resData = pd.DataFrame([], columns=['Id', 'Predicted'])

    k = 43

    for ind in X_test.index:
        data_point = X_test.loc[ind]
        distances = np.linalg.norm(X_train - data_point, axis=1)
        nearest_neighbor_ids = distances.argsort()[:k]
        nearest_neighbor_values = y_train.iloc[nearest_neighbor_ids]

        # getting class using mode
        modes = nearest_neighbor_values.mode()

        prediction = modes[0]
        resId = ids[ind]
        predRes = {'Id': str(resId), 'Predicted': prediction}
        resData = resData.append(predRes, ignore_index=True)

    resData.to_csv("result.csv", index=False, float_format='%.1f')


if __name__ == '__main__':
    main()
