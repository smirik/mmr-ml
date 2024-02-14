from prettytable import PrettyTable


def pretty_print(all_results, N=20):
    base_field_names = ["Rank", "Features", "Accuracy", "Precision", "Recall", "F1 score"]
    param_keys = set()
    for _, _, _, _, params, _ in all_results:
        param_keys.update(params.keys())
    param_keys_sorted = sorted(param_keys)
    display_param_keys = [key.split('__')[-1].capitalize() for key in param_keys_sorted]

    full_headers = base_field_names + display_param_keys

    rows = []
    for rank, (accuracy, precision, recall_score, f1_score, params, subset) in enumerate(all_results, start=1):
        row = [rank, ", ".join(str(s) for s in subset), f"{accuracy:.3f}", f"{precision:.3f}", f"{recall_score:.3f}", f"{f1_score:.3f}"]
        for key in param_keys_sorted:
            row.append(params.get(key, 'N/A'))
        rows.append(row)

    table = PrettyTable()
    table.field_names = full_headers
    for field in full_headers:
        table.align[field] = "r"
    for field in ['Features']:
        table.align[field] = "l"

    for row in rows[:N]:
        table.add_row(row)

    return (rows, table)
