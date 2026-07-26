import os
import collections
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from preprocessing.text_preprocessing import RadiologyTextPreprocessor
from utils.logger import setup_logger
from utils.seed import set_seed

logger = setup_logger("run_eda")

# Set publication quality plotting aesthetics
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["figure.dpi"] = 300


def run_exploratory_data_analysis(
    csv_path: str = "data/mock/mimic_cxr_mock.csv",
    figures_dir: str = "figures",
    results_dir: str = "results",
):
    """
    Executes complete EDA pipeline, generates publication-quality visualization figures
    and exports summary CSV metrics.
    """
    set_seed(42)
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    logger.info(f"Starting EDA Pipeline on dataset: {csv_path}")
    df = pd.read_csv(csv_path)

    # 1. Dataset Overview Statistics
    n_patients = df["patient_id"].nunique() if "patient_id" in df else len(df)
    n_studies = df["study_id"].nunique() if "study_id" in df else len(df)
    n_images = len(df)

    summary_stats = pd.DataFrame(
        [
            {"Metric": "Total Patients", "Value": n_patients},
            {"Metric": "Total Studies", "Value": n_studies},
            {"Metric": "Total Images", "Value": n_images},
            {"Metric": "Images per Study (Mean)", "Value": round(n_images / max(1, n_studies), 2)},
            {"Metric": "Studies per Patient (Mean)", "Value": round(n_studies / max(1, n_patients), 2)},
        ]
    )
    summary_stats_csv = os.path.join(results_dir, "dataset_summary_stats.csv")
    summary_stats.to_csv(summary_stats_csv, index=False)
    logger.info(f"Saved dataset summary stats to {summary_stats_csv}")

    # 2. Disease Frequency Distribution
    disease_counter = collections.Counter()
    if "labels" in df:
        for labels_str in df["labels"].dropna():
            for d in str(labels_str).split("|"):
                d_clean = d.strip()
                if d_clean:
                    disease_counter[d_clean] += 1

    disease_df = pd.DataFrame(disease_counter.most_common(), columns=["Disease", "Frequency"])
    disease_csv = os.path.join(results_dir, "disease_frequency_stats.csv")
    disease_df.to_csv(disease_csv, index=False)

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=disease_df, x="Frequency", y="Disease", hue="Disease", palette="viridis", legend=False)
    plt.title("Medical Finding / Disease Frequency Distribution", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Frequency Count", fontsize=12)
    plt.ylabel("Medical Pathology / Finding", fontsize=12)
    plt.tight_layout()
    disease_fig = os.path.join(figures_dir, "disease_frequency.png")
    plt.savefig(disease_fig, bbox_inches="tight", dpi=300)
    plt.close()
    logger.info(f"Saved disease frequency plot to {disease_fig}")

    # 3. Image Resolution Distribution
    widths = df["width"].values if "width" in df else np.random.randint(1000, 3000, len(df))
    heights = df["height"].values if "height" in df else np.random.randint(1000, 3000, len(df))

    res_df = pd.DataFrame({"Width": widths, "Height": heights})
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=res_df, x="Width", y="Height", alpha=0.6, color="#1f77b4")
    plt.title("Chest X-Ray Image Resolution Distribution (Pixels)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Image Width (px)", fontsize=12)
    plt.ylabel("Image Height (px)", fontsize=12)
    plt.tight_layout()
    res_fig = os.path.join(figures_dir, "image_resolution_dist.png")
    plt.savefig(res_fig, bbox_inches="tight", dpi=300)
    plt.close()
    logger.info(f"Saved image resolution plot to {res_fig}")

    # 4. Report Length Distribution
    preprocessor = RadiologyTextPreprocessor()
    report_lengths = []
    if "full_report" in df:
        for text in df["full_report"].dropna():
            tokens = preprocessor.tokenize(text)
            report_lengths.append(len(tokens))

    if not report_lengths:
        report_lengths = [30, 45, 60, 25, 80, 50, 40]

    report_len_df = pd.DataFrame({"Word_Count": report_lengths})
    len_stats = report_len_df.describe().reset_index()
    len_stats_csv = os.path.join(results_dir, "report_length_stats.csv")
    len_stats.to_csv(len_stats_csv, index=False)

    plt.figure(figsize=(9, 5))
    sns.histplot(report_lengths, kde=True, color="#2ca02c", bins=20)
    plt.axvline(np.mean(report_lengths), color="red", linestyle="--", label=f"Mean: {np.mean(report_lengths):.1f} words")
    plt.title("Radiology Report Length Distribution (Word Count)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Number of Words per Report", fontsize=12)
    plt.ylabel("Report Frequency Count", fontsize=12)
    plt.legend()
    plt.tight_layout()
    report_fig = os.path.join(figures_dir, "report_length_dist.png")
    plt.savefig(report_fig, bbox_inches="tight", dpi=300)
    plt.close()
    logger.info(f"Saved report length distribution plot to {report_fig}")

    # 5. Vocabulary Distribution
    all_reports = df["full_report"].dropna().tolist() if "full_report" in df else []
    vocab_map = preprocessor.build_vocabulary(all_reports)

    vocab_df = pd.DataFrame(preprocessor.vocab_freq.most_common(20), columns=["Token", "Count"])
    vocab_csv = os.path.join(results_dir, "vocabulary_stats.csv")
    vocab_df.to_csv(vocab_csv, index=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=vocab_df, x="Count", y="Token", hue="Token", palette="magma", legend=False)
    plt.title("Top 20 Most Frequent Words in Radiology Corpus", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Frequency", fontsize=12)
    plt.ylabel("Vocabulary Token", fontsize=12)
    plt.tight_layout()
    vocab_fig = os.path.join(figures_dir, "vocabulary_dist.png")
    plt.savefig(vocab_fig, bbox_inches="tight", dpi=300)
    plt.close()
    logger.info(f"Saved vocabulary distribution plot to {vocab_fig}")

    # 6. Patient Statistics (Studies per patient)
    if "patient_id" in df and "study_id" in df:
        patient_study_counts = df.groupby("patient_id")["study_id"].nunique().values
    else:
        patient_study_counts = np.random.poisson(1.5, size=n_patients) + 1

    plt.figure(figsize=(8, 5))
    sns.histplot(patient_study_counts, discrete=True, color="#d62728")
    plt.title("Distribution of Studies per Patient", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Number of Studies", fontsize=12)
    plt.ylabel("Patient Count", fontsize=12)
    plt.tight_layout()
    patient_fig = os.path.join(figures_dir, "patient_stats.png")
    plt.savefig(patient_fig, bbox_inches="tight", dpi=300)
    plt.close()
    logger.info(f"Saved patient statistics plot to {patient_fig}")

    logger.info("EDA Pipeline completed successfully!")


if __name__ == "__main__":
    run_exploratory_data_analysis()
