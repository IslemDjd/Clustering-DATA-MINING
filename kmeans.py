# =====================================================
#  K-MEANS MANUEL + SELECTION FEATURES + SILHOUETTE (LIBRARY)
# =====================================================

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score  # ✅ silhouette only

# -----------------------------
#   1️⃣ NETTOYAGE & SELECTION
# -----------------------------
def nettoyer_et_selectionner(df, id_col="ID", var_threshold=0.01, corr_threshold=0.9):
    print("🧹 Nettoyage des données...")

    # Garder seulement les colonnes numériques
    df_num = df.select_dtypes(include=[np.number]).copy()

    # Supprimer NaN
    df_num = df_num.dropna(axis=1, how='all').dropna()
    print(f"   - Dimensions après nettoyage : {df_num.shape}")

    # Filtrer par variance
    variances = df_num.var()
    keep = variances[variances > var_threshold].index
    df_num = df_num[keep]
    print(f"   - Colonnes après filtre variance : {len(keep)}")

    # Filtrer par corrélation
    corr = df_num.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > corr_threshold)]
    df_num = df_num.drop(columns=to_drop)
    print(f"   - Colonnes après filtre corrélation : {df_num.shape[1]}")

    # Normalisation (Z-score)
    df_norm = (df_num - df_num.mean()) / df_num.std()

    return df_norm.values, df[id_col].values if id_col in df.columns else None, df_num.columns.tolist()


# -----------------------------
#   2️⃣ IMPLEMENTATION MANUELLE K-MEANS
# -----------------------------
class KMeansManuel:
    def __init__(self, k=3, max_iter=100):
        self.k = k
        self.max_iter = max_iter

    def distance(self, a, b):
        return np.sqrt(np.sum((a - b) ** 2))

    def initialiser_centroides(self, data):
        indices = np.random.choice(data.shape[0], self.k, replace=False)
        return data[indices]

    def assigner_clusters(self, data, centroides):
        clusters = []
        for x in data:
            distances = [self.distance(x, c) for c in centroides]
            clusters.append(np.argmin(distances))
        return np.array(clusters)

    def calculer_centroides(self, data, clusters):
        centroides = []
        for i in range(self.k):
            points = data[clusters == i]
            if len(points) > 0:
                centroides.append(points.mean(axis=0))
            else:
                # Cluster vide → centroïde aléatoire
                centroides.append(data[np.random.randint(0, data.shape[0])])
        return np.array(centroides)

    def fit(self, data):
        self.centroides = self.initialiser_centroides(data)
        for iteration in range(self.max_iter):
            clusters = self.assigner_clusters(data, self.centroides)
            nouveaux_centroides = self.calculer_centroides(data, clusters)

            if np.allclose(self.centroides, nouveaux_centroides):
                print(f"✅ Convergence après {iteration + 1} itérations")
                break
            self.centroides = nouveaux_centroides

        self.clusters = clusters
        return self

    def inertie(self, data):
        inertie = 0
        for i, x in enumerate(data):
            c = self.clusters[i]
            inertie += np.sum((x - self.centroides[c]) ** 2)
        return inertie


# -----------------------------
#   3️⃣ PIPELINE PRINCIPAL
# -----------------------------
def main():
    print("=" * 60)
    print("CLUSTERING MANUEL K-MEANS + SILHOUETTE (LIBRARY)")
    print("=" * 60)

    # === 1. Charger données ===
    df = pd.read_csv("./Results/FINAL_8420_features.csv")
    print(f"Dimensions initiales : {df.shape}")

    # === 2. Nettoyage + sélection ===
    X, ids, features = nettoyer_et_selectionner(df, id_col="ID")

    # === 3. Méthode du coude + silhouette ===
    print("\n🔍 Calcul du coude et silhouette pour différents k...")
    K = range(2, 8)
    inerties = []
    silhouettes = []

    for k in K:
        model = KMeansManuel(k=k).fit(X)
        inertie = model.inertie(X)

        # ✅ Silhouette score avec sklearn
        try:
            score_sil = silhouette_score(X, model.clusters)
        except Exception:
            score_sil = np.nan

        inerties.append(inertie)
        silhouettes.append(score_sil)

        print(f"  k={k} → inertie={inertie:.2f}, silhouette={score_sil:.4f}")

    # === 4. Choisir meilleur k (max silhouette) ===
    meilleur_k = K[np.nanargmax(silhouettes)]
    print(f"\n🎯 Meilleur k selon silhouette : {meilleur_k}")

    # === 5. Clustering final ===
    final_model = KMeansManuel(k=meilleur_k).fit(X)
    resultats = pd.DataFrame({
        "ID": ids if ids is not None else np.arange(len(X)),
        "Cluster": final_model.clusters
    })
    resultats.to_csv("./Results/resultats_kmeans_silhouette.csv", index=False)

    print("✅ Résultats sauvegardés : ./Results/resultats_kmeans_silhouette.csv")

    # === 6. Résumé final ===
    print("\n📊 RÉSUMÉ FINAL :")
    print(f"  - Features utilisées : {len(features)}")
    print(f"  - Nombre de séquences : {X.shape[0]}")
    print(f"  - Clusters formés : {meilleur_k}")
    for i in range(meilleur_k):
        print(f"    Cluster {i} → {np.sum(final_model.clusters == i)} séquences")

    print(f"  - Score silhouette global : {np.nanmax(silhouettes):.4f}")

if __name__ == "__main__":
    main()
