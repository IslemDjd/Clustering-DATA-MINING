import csv

def parse_fasta(filepath):
    """Lit un fichier FASTA et retourne une liste de dictionnaires."""
    data = []
    entry = entry_name = organism = protein = ""
    sequence = ""

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                # Sauvegarder la précédente séquence si elle existe
                if entry:
                    data.append({
                        "ID": entry,
                        "name": entry_name,
                        "protein": protein,
                        "species": organism,
                        "sequence": sequence
                    })
                    sequence = ""

                # Exemple de header :
                # >sp|P12345|PROT_HUMAN Myosin heavy chain OS=Homo sapiens GN=MYH7 PE=1 SV=2
                parts = line.split()
                id_parts = parts[0].split("|")

                entry = id_parts[1] if len(id_parts) > 1 else ""
                entry_name = id_parts[2] if len(id_parts) > 2 else ""

                # Récupérer la description (le nom du protéine)
                desc_parts = line.split("OS=")[0].split(maxsplit=1)
                protein = desc_parts[1] if len(desc_parts) > 1 else ""

                # Récupérer l’organisme
                organism = ""
                for part in parts:
                    if part.startswith("OS="):
                        organism = part[3:]
                        break
            else:
                sequence += line

    # Ajouter la dernière entrée
    if entry:
        data.append({
            "ID": entry,
            "name": entry_name,
            "protein": protein,
            "species": organism,
            "sequence": sequence
        })

    return data


def merge_fasta_files(file_list):
    """Fusionne plusieurs fichiers FASTA en un seul DataFrame (liste de dictionnaires)."""
    merged_data = []
    for file in file_list:
        print(f"Lecture de {file} ...")
        merged_data.extend(parse_fasta(file))
    return merged_data


def save_as_fasta(df, output_path):
    """Sauvegarde la liste de dictionnaires dans un fichier FASTA."""
    with open(output_path, "w", encoding="utf-8") as f:
        for row in df:
            entry = row.get("ID", "")
            entry_name = row.get("name", "")
            organism = row.get("species", "")
            sequence = row.get("sequence", "")

            header = f">sp|{entry}|{entry_name} OS={organism}\n"
            f.write(header)

            for i in range(0, len(sequence), 60):
                f.write(sequence[i:i+60] + "\n")


def save_as_csv(df, output_path):
    """Sauvegarde la liste de dictionnaires dans un fichier CSV."""
    if not df:
        print("Aucune donnée à sauvegarder.")
        return

    fieldnames = ["ID", "name", "protein", "species", "sequence"]

    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in df:
            writer.writerow(row)


# Liste des fichiers FASTA à fusionner
files = [
    "./Datasets/bovine.fasta",
    "./Datasets/human.fasta",
    "./Datasets/mouse.fasta",
    "./Datasets/rat.fasta",
    "./Datasets/zebrafish.fasta"
]

# Fusion des fichiers
df_merged = merge_fasta_files(files)

# Nombre total d’entrées
print(f"\nNombre total de séquences fusionnées : {len(df_merged)}")

# Sauvegarde dans un fichier FASTA
fasta_output = "./Results/merged_files.fasta"
save_as_fasta(df_merged, fasta_output)
print(f"✅ Fichier FASTA fusionné sauvegardé : {fasta_output}")

# Sauvegarde dans un fichier CSV
csv_output = "./Results/merged_files.csv"
save_as_csv(df_merged, csv_output)
print(f"✅ Fichier CSV structuré sauvegardé : {csv_output}")
