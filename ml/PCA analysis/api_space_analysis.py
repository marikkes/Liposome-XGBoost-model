from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import sqlite3
import joblib


def main():

    BASE_DIR = Path(__file__).resolve().parent.parent.parent

    API_DB = (
        BASE_DIR
        / "db"
        / "work"
        / "api_properties.db"
    )


    # -----------------------
    # Load API data
    # -----------------------

    with sqlite3.connect(API_DB) as conn:
        apis = pd.read_sql(
            "SELECT * FROM api_properties",
            conn
        )


    print(apis.head())


    # -----------------------
    # Select descriptors
    # -----------------------

    descriptor_cols = [
        "molecular_weight",
        "logp",
        "rotatable_bond_count",
        "hbond_acceptor_count",
        "hbond_donor_count",
        "heavy_atom_count",
        "tpsa",
        "complexity",
        "defined_atom_stereocenter_count",
        "defined_bond_stereocenter_count",
        "number_of_rings_calc"
    ]


    # Remove APIs without calculated ring count
    pca_apis = apis.dropna(
        subset=descriptor_cols
    )


    print(
        "\nAPIs excluded from PCA:"
    )

    print(
        apis.loc[
            ~apis["api"].isin(pca_apis["api"]),
            "api"
        ].to_string(index=False)
    )


    X = pca_apis[descriptor_cols]


    print("\nMissing values:")
    print(X.isna().sum())


    # -----------------------
    # Scale descriptors
    # -----------------------

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)


    # -----------------------
    # PCA
    # -----------------------

    pca = PCA()

    X_pca = pca.fit_transform(X_scaled)


    # -----------------------
    # Explained variance
    # -----------------------

    variance = pca.explained_variance_ratio_

    print("\nExplained variance:")

    for i, v in enumerate(variance):
        print(f"PC{i+1}: {v:.2%}")


    # -----------------------
    # Loadings
    # -----------------------

    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[
            f"PC{i+1}"
            for i in range(pca.components_.shape[0])
        ],
        index=descriptor_cols
    )


    print("\nPCA loadings:")
    print(loadings)


    # -----------------------
    # Save results
    # -----------------------

    output_file = BASE_DIR / "api_pca_results.xlsx"


    variance_df = pd.DataFrame({
        "Principal Component":
            [f"PC{i+1}" for i in range(len(variance))],

        "Explained Variance":
            variance,

        "Explained Variance (%)":
            variance * 100,

        "Cumulative Variance (%)":
            variance.cumsum() * 100,
    })


    scores_df = pd.DataFrame(
        X_pca,
        columns=[
            f"PC{i+1}"
            for i in range(X_pca.shape[1])
        ]
    )

    scores_df.insert(
        0,
        "api",
        pca_apis["api"]
    )

    excluded_df = apis.loc[
        ~apis["api"].isin(pca_apis["api"])
    ]


    with pd.ExcelWriter(output_file) as writer:

        variance_df.to_excel(
            writer,
            sheet_name="Variance Explained",
            index=False
        )

        loadings.to_excel(
            writer,
            sheet_name="Loadings"
        )

        scores_df.to_excel(
            writer,
            sheet_name="Scores",
            index=False
        )

        excluded_df.to_excel(
            writer,
            sheet_name="Excluded APIs",
            index=False
        )


    print(
        f"\nPCA results saved to:\n{output_file}"
    )


    # -----------------------
    # Plot
    # -----------------------

    plt.figure(figsize=(8,6))

    plt.scatter(
        X_pca[:,0],
        X_pca[:,1]
    )


    for i, api in enumerate(pca_apis["api"]):

        plt.text(
            X_pca[i,0],
            X_pca[i,1],
            api,
            fontsize=8
        )


    plt.xlabel(
        "PC1"
    )

    plt.ylabel(
        "PC2"
    )

    plt.title(
        "API chemical space"
    )


    plot_file = BASE_DIR / "api_pca_plot.png"

    plt.savefig(
        plot_file,
        dpi=300,
        bbox_inches="tight"
    )


    print(
        f"PCA plot saved to:\n{plot_file}"
    )


    plt.show()


    # -----------------------
    # Save PCA model
    # -----------------------

    MODEL_PATH = (
        BASE_DIR
        / "models"
        / "api_pca_model.joblib"
    )


    joblib.dump(
        {
            "scaler": scaler,
            "pca": pca,
            "api_names": pca_apis["api"].values,
            "scores": X_pca
        },
        MODEL_PATH
    )


    print(
        f"PCA model saved to:\n{MODEL_PATH}"
    )


if __name__ == "__main__":
    main()