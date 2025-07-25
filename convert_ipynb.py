import json
import sys
import os

def convert_ipynb_to_py(ipynb_path, py_path):
    # Charger le notebook
    with open(ipynb_path, "r", encoding="utf-8") as f:
        notebook = json.load(f)

    lines = []
    for cell in notebook.get("cells", []):
        cell_type = cell.get("cell_type", "")
        source = cell.get("source", [])

        if cell_type == "markdown":
            # Convertir Markdown en commentaires
            lines.append("# " + "=" * 60)
            for line in source:
                # On retire les retours à la ligne éventuels
                line_clean = line.rstrip("\n")
                if line_clean.strip():
                    lines.append("# " + line_clean)
                else:
                    lines.append("#")
            lines.append("# " + "=" * 60)
            lines.append("")  # Ligne vide après un bloc markdown
        elif cell_type == "code":
            lines.extend([l.rstrip("\n") for l in source])
            lines.append("")  # Ligne vide après un bloc code

    # Sauvegarde du fichier .py
    with open(py_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Conversion terminée : {py_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Utilisation : python convert_ipynb_to_py.py <notebook.ipynb> <output.py>")
    else:
        ipynb_path = sys.argv[1]
        py_path = sys.argv[2]

        if not os.path.isfile(ipynb_path):
            print(f"Erreur : le fichier {ipynb_path} n'existe pas.")
            sys.exit(1)

        convert_ipynb_to_py(ipynb_path, py_path)
