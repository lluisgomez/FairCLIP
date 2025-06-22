import json
import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Build LaTeX table from race/gender bias evaluation.")
    parser.add_argument("--json_files", nargs="+", required=True, help="List of JSON files")
    return parser.parse_args()

def extract_results(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
        race = data.get("race_results", {})
        gender = data.get("gender_results", {})
        return gender, race

def collect_all_categories(files):
    categories = set()
    for file in files:
        gender, race = extract_results(file)
        categories.update(gender.keys())
        categories.update(race.keys())
    return sorted(categories)

def build_latex_table(file_data, categories):
    header_top = "Model & " + " & ".join([f"\\multicolumn{{2}}{{c}}{{{cat}}}" for cat in categories]) + " \\\\"
    header_mid = " & " + " & ".join(["G & R" for _ in categories]) + " \\\\ \\hline"
    
    rows = []
    for model_name, (gender, race) in file_data.items():
        row = [model_name]
        for cat in categories:
            g_val = gender.get(cat, "N/A")
            r_val = race.get(cat, "N/A")
            g_str = f"{g_val:.2f}" if isinstance(g_val, float) else g_val
            r_str = f"{r_val:.2f}" if isinstance(r_val, float) else r_val
            row += [g_str, r_str]
        rows.append(" & ".join(row) + " \\\\")
    
    col_spec = "l" + "cc" * len(categories)
    return "\\begin{tabular}{" + col_spec + "}\n\\hline\n" + header_top + "\n" + header_mid + "\n" + "\n".join(rows) + "\n\\hline\n\\end{tabular}"

def main():
    args = parse_args()
    file_data = {}
    for file in args.json_files:
        model_name = Path(file).parent.name
        gender, race = extract_results(file)
        file_data[model_name] = (gender, race)
    
    all_categories = collect_all_categories(args.json_files)
    latex = build_latex_table(file_data, all_categories)
    print(latex)

if __name__ == "__main__":
    main()

