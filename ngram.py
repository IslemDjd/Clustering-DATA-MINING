import pandas as pd
import os

# Les 20 acides aminés standards
STANDARD_AMINO_ACIDS = list('ACDEFGHIKLMNPQRSTVWY')

def generer_features_exactes():
    """Génère EXACTEMENT 8,420 features (1-, 2-, et 3-grammes) avec ID complet."""
    print("⚡ Génération des 8,420 features...")

    # Charger les données fusionnées
    df = pd.read_csv("./Results/merged_files.csv")

    # Vérification des colonnes disponibles
    print(f"🧬 Colonnes trouvées dans merged_files.csv : {list(df.columns)}")

    # Génération des combinaisons théoriques
    features = []

    # n=1
    for a in STANDARD_AMINO_ACIDS:
        features.append(a)

    # n=2
    for a in STANDARD_AMINO_ACIDS:
        for b in STANDARD_AMINO_ACIDS:
            features.append(a + b)

    # n=3
    for a in STANDARD_AMINO_ACIDS:
        for b in STANDARD_AMINO_ACIDS:
            for c in STANDARD_AMINO_ACIDS:
                features.append(a + b + c)

    print(f"✅ Features théoriques : {len(features)}")

    # Créer le DataFrame final
    data_final = []

    for _, row in df.iterrows():
        # Certains fichiers ont ces colonnes : "entry", "entry_name", "organism", "sequence"
        # On va créer un ID complet plus descriptif
        full_id = f"{row.get('ID', '')}".strip("|")

        # Retirer les lettres non standard avant le comptage
        sequence = ''.join([aa for aa in str(row["sequence"]) if aa in STANDARD_AMINO_ACIDS])

        ligne = {
            "Full_ID": full_id,
        }

        # Comptage des occurrences de chaque n-gramme
        for feature in features:
            ligne[feature] = sequence.count(feature)

        data_final.append(ligne)

    features_df = pd.DataFrame(data_final)

    # Sauvegarde finale
    output_path = "./Results/FINAL_8420_features.csv"
    features_df.to_csv(output_path, index=False)

    print(f"🎉 TERMINÉ !")
    print(f"📁 Fichier : {output_path}")
    print(f"📊 Dimensions : {features_df.shape[0]} séquences × {features_df.shape[1]} colonnes")
    print(f"🔢 Features : {features_df.shape[1] - 2} n-grams + 2 IDs")

    return features_df


# EXÉCUTION
if __name__ == "__main__":
    generer_features_exactes()
