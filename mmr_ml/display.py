import pandas as pd
from prettytable import PrettyTable


def display_results(results, save=True, output_file='summary.csv'):
    data = []
    table = PrettyTable()
    table.field_names = [
        "Train Set",
        "Train R",
        "Train NR",
        "Test Set",
        "Test R",
        "Test NR",
        "Meth. & Params",
        "Features",
        "TP",
        "FP",
        "TN",
        "FN",
        "Acc.",
        "Prec.",
        "Rec.",
        "F1",
    ]
    for result in results:
        table.add_row(result)
        data.append(result)
    # save data list to csv file summary.csv if save is True. If summary.csv exists, append data to it.
    if save:
        pd.DataFrame(data=data, columns=table.field_names).to_csv(output_file, mode='a', index=False, header=True)
    print(table)
