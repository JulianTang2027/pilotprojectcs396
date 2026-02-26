import nbformat
from nbstripout import strip_output

# Load your notebook
with open("Fine_tuning_Leads_to_Forgetting.ipynb", "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

# Strip outputs
strip_output(nb)

# Save back
with open("Fine_tuning_Leads_to_Forgetting_clean.ipynb", "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print("Notebook outputs cleared!")