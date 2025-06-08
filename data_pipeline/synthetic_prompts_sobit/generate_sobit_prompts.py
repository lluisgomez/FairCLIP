import json
import csv
from itertools import product

# Load input files
with open("../../evals/So-B-IT_taxonomy.json", "r") as f:
    taxonomy = json.load(f)

with open("../../evals/So-B-IT_categories.json", "r") as f:
    pos_tags = json.load(f)

with open("sobit_prompt_templates.json", "r") as f:
    templates = json.load(f)

with open("sobit_scene_contexts.json", "r") as f:
    scene_contexts = json.load(f)

with open("photo_style_prefixes.json", "r") as f:
    photo_prefixes = json.load(f)

time_periods = [
  "from the 1950s", "from the 1960s", "from the 1970s", "from the 1980s",
  "from the 1990s", "from the early 2000s", "from the 2010s", "from the 2020s"
]


# Define demographics
genders = ["male", "female", "nonbinary"]
ethnicities = ["white", "black", "east asian", "south asian", "latino", "middle eastern", "indian"]
ages = ["young adult", "middle-aged", "elderly"]
income_levels = ["low-income", "mid-income", "high-income"]
demographics = list(product(genders, ethnicities, ages, income_levels))

# Utility to safely handle template keys
def safe_format(template, **kwargs):
    for key in list(kwargs.keys()):
        template = template.replace("{" + key.capitalize() + "}", "{" + key + "}")
        template = template.replace("{" + key.upper() + "}", "{" + key + "}")
    return template.format(**kwargs)

# Generate prompts
prompt_rows = []

for category, words in taxonomy.items():
    for word in words:
        pos = pos_tags.get(word)
        if pos not in ["NOUN", "ADJECTIVE"]:
            continue

        cat_templates = templates.get(category, {}).get(pos, [])
        cat_scenes = scene_contexts.get(category, [])

        for gender, ethnicity, age, income_level in demographics:
            demographic = f"{income_level} {age} {ethnicity} {gender}"
            for scene in cat_scenes:
                for template in cat_templates:
                    for prefix in photo_prefixes:
                        for time_period in time_periods:
                            try:
                                prompt = safe_format(template, word=word, demographic=demographic,
                                                     scene=scene, prefix=prefix, time=time_period)
                                training_caption = f"A person who is {word}" if pos == "ADJECTIVE" else f"A person who is a {word}"
                                prompt_rows.append({
                                    "prompt": prompt,
                                    "training_caption": training_caption
                                })
                            except KeyError:
                                continue  # Skip malformed templates

# Write to CSV
output_path = "synthetic_prompts_sobit.csv"
with open(output_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=prompt_rows[0].keys())
    writer.writeheader()
    writer.writerows(prompt_rows)

print(f"✅ Generated {len(prompt_rows)} prompts and saved to {output_path}")

