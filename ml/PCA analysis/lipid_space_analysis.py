from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import sqlite3
import joblib

def main():

    BASE_DIR = Path(__file__).resolve().parent.parent.parent

    LIPID_DB = (
        BASE_DIR
        / "db"
        / "work"
        / "lipid_properties.db"
    )


    # Load lipid data
    with sqlite3.connect(LIPID_DB) as conn:
        lipids = pd.read_sql(
            "SELECT * FROM lipid_properties",
            conn
        )


    #print(lipids.head())

    # -----------------------
    # Select descriptors
    # -----------------------

    descriptor_cols = [
        c for c in lipids.columns
        if c != "lipid_name"
    ]

    X = lipids[descriptor_cols]

    # -----------------------
    # Scale
    # -----------------------

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    # -----------------------
    # PCA
    # -----------------------

    pca = PCA()

    X_pca = pca.fit_transform(X_scaled)

    # Variance explained

    variance = pca.explained_variance_ratio_

    for i, v in enumerate(variance):
        print(f"PC{i+1}: {v:.2%}")

    # -----------------------
    # Loadings
    # -----------------------

    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f"PC{i+1}" for i in range(pca.components_.shape[0])],
        index=descriptor_cols
    )

    print("\nPCA loadings:")
    print(loadings.head(20))

    # -----------------------
    # Save PCA results
    # -----------------------

    output_file = BASE_DIR / "lipid_pca_results.xlsx"

    # Variance explained
    variance_df = pd.DataFrame({
        "Principal Component": [f"PC{i+1}" for i in range(len(variance))],
        "Explained Variance": variance,
        "Explained Variance (%)": variance * 100,
        "Cumulative Variance (%)": variance.cumsum() * 100,
    })

    # PCA scores (lipid coordinates)
    scores_df = pd.DataFrame(
        X_pca,
        columns=[f"PC{i+1}" for i in range(X_pca.shape[1])]
    )
    scores_df.insert(0, "lipid_name", lipids["lipid_name"])

    with pd.ExcelWriter(output_file) as writer:
        variance_df.to_excel(
            writer,
            sheet_name="Variance Explained",
            index=False,
        )

        loadings.to_excel(
            writer,
            sheet_name="Loadings",
        )

        scores_df.to_excel(
            writer,
            sheet_name="Scores",
            index=False,
        )

    print(f"\nPCA results saved to:\n{output_file}")

    # -----------------------
    # Plot
    # -----------------------

    plt.figure(figsize=(8,6))

    plt.scatter(
        X_pca[:,0],
        X_pca[:,1]
    )

    for i, lipid in enumerate(lipids["lipid_name"]):
        plt.text(
            X_pca[i,0],
            X_pca[i,1],
            lipid
        )

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(
        "Lipid chemical space"
    )
    
    plot_file = BASE_DIR / "lipid_pca_plot.png"

    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    print(f"PCA plot saved to:\n{plot_file}")

    plt.show()

    # -----------------------
    # Save PCA model
    # -----------------------
    MODEL_PATH = BASE_DIR /"models" / "lipid_pca_model.joblib"

    joblib.dump(
        {
            "scaler": scaler,
            "pca": pca,
            "lipid_names": lipids["lipid_name"].values,
            "scores": X_pca
        },
        MODEL_PATH
    )

    

if __name__ == "__main__":
    main()