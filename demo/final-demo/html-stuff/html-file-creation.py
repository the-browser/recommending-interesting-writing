import argparse
from pathlib import Path
import os

# get arguments for script and parse
def expand_path(string):
    return Path(os.path.expandvars(string))


parser = argparse.ArgumentParser(
    description="Train model on article data and test evaluation"
)

parser.add_argument(
    "--header", default=Path("header.html"), help="Path to header of html file."
)
parser.add_argument(
    "--footer", default=Path("footer.html"), help="Path to footer of html file."
)
parser.add_argument(
    "--browser_head",
    default=Path("browser/browser-head.html"),
    help="Path to head of brower section.",
)
parser.add_argument(
    "--corona_head",
    default=Path("corona/corona-head.html"),
    help="Path to head of corona section.",
)
parser.add_argument(
    "--section_tail",
    default=Path("section-tail.html"),
    help="Path to sectiontail HTML.",
)
parser.add_argument(
    "--browser_table",
    default=Path("browser/browser-table.html"),
    help="Path to populated HTML browser table.",
)
parser.add_argument(
    "--corona_table",
    default=Path("corona/corona-table.html"),
    help="Path to populated HTML corona table.",
)
parser.add_argument(
    "--output_file",
    default=Path("/users/rohan/browser/docs/index.html"),
    help="Path to output file from resulting generation",
)

args = parser.parse_args()

with open(args.header, "r", encoding="utf-8") as file:
    header = file.read()

with open(args.footer, "r", encoding="utf-8") as file:
    footer = file.read()

with open(args.browser_head, "r", encoding="utf-8") as file:
    browser_head = file.read()

with open(args.corona_head, "r", encoding="utf-8") as file:
    corona_head = file.read()

with open(args.section_tail, "r", encoding="utf-8") as file:
    section_tail = file.read()

with open(args.browser_table, "r", encoding="utf-8") as file:
    browser_table = file.read()

with open(args.corona_table, "r", encoding="utf-8") as file:
    corona_table = file.read()

final_output = ""
final_output += header
final_output += browser_head
final_output += browser_table
final_output += section_tail
final_output += corona_head
final_output += corona_table
final_output += section_tail
final_output += footer

with open(args.output_file, "w", encoding="utf-8") as file:
    file.write(final_output)

print("Final HTML created and saved to output file!")
