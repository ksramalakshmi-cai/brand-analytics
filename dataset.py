from datasets import load_dataset

dataset = load_dataset("axonstan/LogoDet-3K")

classes = dataset["train"].features["company_name"].names

targets = [
"google",
"coca cola",
"budweiser",
"toyota",
"hyundai",
"emirates",
"aramco",
"dp world",
"sobha",
"flipkart",
"royal stag",
"apollo",
"livpure",
"toyota",
]

for t in targets:
    matches = [c for c in classes if t.lower() in c.lower()]
    print(t, matches)