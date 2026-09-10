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

    descriptor_cols = lipids.select_dtypes(
        include="number"
    ).columns.tolist()

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
    # Plot: PC1 vs PC2
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

    plt.xlabel(
        f"PC1 ({variance[0]:.1%})"
    )

    plt.ylabel(
        f"PC2 ({variance[1]:.1%})"
    )

    plt.title(
        "Lipid chemical space (PC1 vs PC2)"
    )
    
    plot_file = BASE_DIR / "lipid_pca_plot.png"

    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    print(f"PCA plot saved to:\n{plot_file}")

    plt.show()


    # -----------------------
    # Plot: PC1 vs PC2 vs PC3
    # -----------------------

    fig = plt.figure(figsize=(10, 8))

    ax = fig.add_subplot(
        111,
        projection="3d"
    )

    ax.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        X_pca[:, 2]
    )

    for i, lipid in enumerate(lipids["lipid_name"]):

        ax.text(
            X_pca[i, 0],
            X_pca[i, 1],
            X_pca[i, 2],
            lipid,
            fontsize=7
        )

    ax.set_xlabel(
        f"PC1 ({variance[0]:.1%})"
    )

    ax.set_ylabel(
        f"PC2 ({variance[1]:.1%})"
    )

    ax.set_zlabel(
        f"PC3 ({variance[2]:.1%})"
    )

    ax.set_title(
        "Lipid chemical space (PC1–PC3)"
    )

    plot_3d_file = BASE_DIR / "lipid_pca_plot_3d.png"

    plt.savefig(
        plot_3d_file,
        dpi=300,
        bbox_inches="tight"
    )

    print(
        f"3D PCA plot saved to:\n{plot_3d_file}"
    )

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