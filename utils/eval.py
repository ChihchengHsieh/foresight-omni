import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns


markers = ["o", "s", "D", "^", "P", "*", "X", "v", "H"]


def get_dash_styles(n_modalities: int):
    """
    Generate a list of dash styles for the given number of modalities.

    Each dash style is a (on_length, off_length) tuple used in matplotlib line styling.
    """
    base_styles = [(2, 2), (4, 2), (6, 2), (1, 1), (5, 3)]

    # If number of modalities is less than or equal to base styles, return the subset
    if n_modalities <= len(base_styles):
        return base_styles[:n_modalities]

    # Extend styles by increasing the on/off lengths systematically
    extended_styles = base_styles.copy()
    i = 0
    while len(extended_styles) < n_modalities:
        # Use modulo to rotate through base styles and slightly alter them
        base_on, base_off = base_styles[i % len(base_styles)]
        new_style = (
            base_on + (i // len(base_styles)) * 2,
            base_off + ((i // len(base_styles)) % 3),
        )
        if new_style not in extended_styles:
            extended_styles.append(new_style)
        i += 1

    return extended_styles


def get_style_dict(unique_modalities):

    dash_styles = get_dash_styles(len(unique_modalities))

    return {
        "dashes": {
            mod: dash_styles[i % len(dash_styles)]
            for i, mod in enumerate(unique_modalities)
        },
        "markers": {
            mod: markers[i % len(markers)] for i, mod in enumerate(unique_modalities)
        },
    }


def MAP_evaluator_to_clean_dict(evaluator):
    perf_dict = evaluator.compute()
    if "classes" in perf_dict:
        del perf_dict["classes"]
    return {k: v.item() for k, v in perf_dict.items()}

def save_all_disease_dashboards_to_pdf_without_multiclasses(
    df,
    disease_list,
    metrics,
    saving_path,
    style_dict,
    pdf_filename="all_disease_dashboards.pdf",
    multiclass_metrics=[],
    png_dir=None,  # Optional directory to save PNGs
):
    os.makedirs(saving_path, exist_ok=True)
    pdf_path = os.path.join(saving_path, pdf_filename)
    with PdfPages(pdf_path) as pdf:
        for disease in disease_list:
            fig = plt.figure(constrained_layout=True, figsize=(24, 36))
            subfigs = fig.subfigures(3, 1, height_ratios=[1, 1, 1])

            # Section 1: Time-based dashboard
            axes1 = subfigs[0].subplots(2, 3).flatten()
            for i, metric in enumerate(metrics):
                pattern = re.compile(f"has_{disease}_in_(\\d+)_years_{metric}")
                data = []
                for col in df.columns:
                    match = pattern.match(col)
                    if match:
                        year = int(match.group(1))
                        for _, row in df.iterrows():
                            data.append(
                                {
                                    "year": year,
                                    "value": row[col],
                                    "modality": row["modalities"],
                                }
                            )
                if data:
                    plot_df = pd.DataFrame(data)
                    sns.lineplot(
                        data=plot_df,
                        x="year",
                        y="value",
                        hue="modality",
                        markers=True,
                        style="modality",
                        dashes=style_dict["dashes"],
                        markersize=15,
                        ax=axes1[i],
                    )
                    axes1[i].set_title(f"{metric.replace('_', ' ').title()}")
                    axes1[i].grid(True)

            pos_pat = re.compile(f"has_{disease}_in_(\\d+)_years_positive_samples")
            neg_pat = re.compile(f"has_{disease}_in_(\\d+)_years_negative_samples")
            sample_data = []
            for col in df.columns:
                for _, row in df.iterrows():
                    if pos_pat.match(col):
                        year = int(pos_pat.match(col).group(1))
                        sample_data.append(
                            {"year": year, "count": row[col], "label": "Positive"}
                        )
                    elif neg_pat.match(col):
                        year = int(neg_pat.match(col).group(1))
                        sample_data.append(
                            {"year": year, "count": row[col], "label": "Negative"}
                        )
            if sample_data:
                sample_df = pd.DataFrame(sample_data)
                ax = axes1[-1]
                sns.barplot(
                    data=sample_df, x="year", y="count", hue="label", ci=None, ax=ax
                )
                ax.set_title("Sample Counts")
                ax.grid(True)
                for container in ax.containers:
                    ax.bar_label(container, fmt="%.0f", fontsize=8, padding=2)

            # Section 2: Range-based dashboard
            axes2 = subfigs[1].subplots(2, 3).flatten()
            for i, metric in enumerate(metrics):
                pattern = re.compile(f"{disease}_(\\d+)_(\\d+)_" + metric)
                data = []
                for col in df.columns:
                    match = pattern.match(col)
                    if match:
                        year_range = f"{match.group(1)}–{match.group(2)}"
                        for _, row in df.iterrows():
                            data.append(
                                {
                                    "year_range": year_range,
                                    "value": row[col],
                                    "modality": row["modalities"],
                                }
                            )
                if data:
                    plot_df = pd.DataFrame(data)
                    sns.lineplot(
                        data=plot_df,
                        x="year_range",
                        y="value",
                        hue="modality",
                        markers=True,
                        style="modality",
                        dashes=style_dict["dashes"],
                        markersize=15,
                        ax=axes2[i],
                    )
                    axes2[i].set_title(f"Range {metric.replace('_', ' ').title()}")
                    axes2[i].grid(True)

            range_sample_data = []
            pos_pat = re.compile(f"{disease}_(\\d+)_(\\d+)_positive_samples")
            neg_pat = re.compile(f"{disease}_(\\d+)_(\\d+)_negative_samples")
            for col in df.columns:
                for _, row in df.iterrows():
                    if pos_pat.match(col):
                        range_ = f"{pos_pat.match(col).group(1)}–{pos_pat.match(col).group(2)}"
                        range_sample_data.append(
                            {
                                "year_range": range_,
                                "count": row[col],
                                "label": "Positive",
                            }
                        )
                    elif neg_pat.match(col):
                        range_ = f"{neg_pat.match(col).group(1)}–{neg_pat.match(col).group(2)}"
                        range_sample_data.append(
                            {
                                "year_range": range_,
                                "count": row[col],
                                "label": "Negative",
                            }
                        )
            if range_sample_data:
                range_df = pd.DataFrame(range_sample_data)
                ax = axes2[-1]
                sns.barplot(
                    data=range_df,
                    x="year_range",
                    y="count",
                    hue="label",
                    ci=None,
                    ax=ax,
                )
                ax.set_title("Sample Counts by Year Range")
                ax.grid(True)
                for container in ax.containers:
                    ax.bar_label(container, fmt="%.0f", fontsize=8, padding=2)

            # Section 3: Multiclass dashboard
            # axes3 = subfigs[2].subplots(3, 3).flatten()

            # Calculate how many subplots are needed in section 3
            n_barplots = len(multiclass_metrics)
            n_confmats = min(3, len(df))
            n_special_charts = 2  # AUROC per Class + Class Counts
            total_axes_needed = n_barplots + n_confmats + n_special_charts

            # Determine appropriate grid size
            n_cols = 3
            n_rows = (total_axes_needed + n_cols - 1) // n_cols  # ceil division

            axes3 = subfigs[2].subplots(n_rows, n_cols).flatten()

            # for i, (_, row) in enumerate(df.iterrows()):
            #     mat = np.zeros((5, 5), dtype=int)
            #     for p_idx, p_cls in enumerate(multiclass_classes):
            #         for t_idx, t_cls in enumerate(multiclass_classes):
            #             col = f"{disease}_multiclass_confmat_pred_{p_cls}_true_{t_cls}_counts"
            #             if col in df.columns:
            #                 mat[t_idx, p_idx] = row[col]
            #     sns.heatmap(
            #         mat,
            #         annot=True,
            #         fmt="d",
            #         cmap="Blues",
            #         xticklabels=multiclass_classes,
            #         yticklabels=multiclass_classes,
            #         ax=axes3[i],
            #     )
            #     axes3[i].set_title(f"Confusion Matrix\n({row['modalities']})")
            #     axes3[i].set_ylabel("Predicted")
            #     axes3[i].set_xlabel("True")

            fig.suptitle(f"{disease.upper()} - Full Dashboard", fontsize=24)
            pdf.savefig(fig)

            if png_dir is not None:
                os.makedirs(png_dir, exist_ok=True)
                fig.savefig(
                    os.path.join(png_dir, f"{disease}_dashboard_full.png"),
                    bbox_inches="tight",
                    dpi=300,
                )

                # Save each individual subplot (Axes)
                for i, ax in enumerate(fig.get_axes()):
                    temp_fig, temp_ax = plt.subplots(figsize=(8, 6))

                    # Copy content from original Axes to new Axes
                    for line in ax.get_lines():
                        temp_ax.plot(
                            line.get_xdata(),
                            line.get_ydata(),
                            label=line.get_label(),
                            linestyle=line.get_linestyle(),
                            marker=line.get_marker(),
                            color=line.get_color(),
                        )

                    for bar_container in ax.containers:
                        temp_ax.bar(
                            [bar.get_x() for bar in bar_container],
                            [bar.get_height() for bar in bar_container],
                            width=[bar.get_width() for bar in bar_container],
                            label=bar_container.get_label(),
                        )

                    temp_ax.set_title(ax.get_title())
                    temp_ax.set_xlabel(ax.get_xlabel())
                    temp_ax.set_ylabel(ax.get_ylabel())
                    temp_ax.legend(loc="best")
                    temp_ax.grid(True)

                    temp_fig.tight_layout()
                    temp_fig.savefig(
                        os.path.join(png_dir, f"{disease}_subplot_{i+1}.png"),
                        bbox_inches="tight",
                        dpi=300,
                    )
                    plt.close(temp_fig)

            plt.close(fig)



def save_all_disease_dashboards_to_pdf(
    df,
    disease_list,
    metrics,
    multiclass_classes,
    saving_path,
    style_dict,
    pdf_filename="all_disease_dashboards.pdf",
    multiclass_metrics=[],
    png_dir=None,  # Optional directory to save PNGs
):
    os.makedirs(saving_path, exist_ok=True)
    pdf_path = os.path.join(saving_path, pdf_filename)
    with PdfPages(pdf_path) as pdf:
        for disease in disease_list:
            fig = plt.figure(constrained_layout=True, figsize=(24, 36))
            subfigs = fig.subfigures(3, 1, height_ratios=[1, 1, 1])

            # Section 1: Time-based dashboard
            axes1 = subfigs[0].subplots(2, 3).flatten()
            for i, metric in enumerate(metrics):
                pattern = re.compile(f"has_{disease}_in_(\\d+)_years_{metric}")
                data = []
                for col in df.columns:
                    match = pattern.match(col)
                    if match:
                        year = int(match.group(1))
                        for _, row in df.iterrows():
                            data.append(
                                {
                                    "year": year,
                                    "value": row[col],
                                    "modality": row["modalities"],
                                }
                            )
                if data:
                    plot_df = pd.DataFrame(data)
                    sns.lineplot(
                        data=plot_df,
                        x="year",
                        y="value",
                        hue="modality",
                        markers=True,
                        style="modality",
                        dashes=style_dict["dashes"],
                        markersize=15,
                        ax=axes1[i],
                    )
                    axes1[i].set_title(f"{metric.replace('_', ' ').title()}")
                    axes1[i].grid(True)

            pos_pat = re.compile(f"has_{disease}_in_(\\d+)_years_positive_samples")
            neg_pat = re.compile(f"has_{disease}_in_(\\d+)_years_negative_samples")
            sample_data = []
            for col in df.columns:
                for _, row in df.iterrows():
                    if pos_pat.match(col):
                        year = int(pos_pat.match(col).group(1))
                        sample_data.append(
                            {"year": year, "count": row[col], "label": "Positive"}
                        )
                    elif neg_pat.match(col):
                        year = int(neg_pat.match(col).group(1))
                        sample_data.append(
                            {"year": year, "count": row[col], "label": "Negative"}
                        )
            if sample_data:
                sample_df = pd.DataFrame(sample_data)
                ax = axes1[-1]
                sns.barplot(
                    data=sample_df, x="year", y="count", hue="label", ci=None, ax=ax
                )
                ax.set_title("Sample Counts")
                ax.grid(True)
                for container in ax.containers:
                    ax.bar_label(container, fmt="%.0f", fontsize=8, padding=2)

            # Section 2: Range-based dashboard
            axes2 = subfigs[1].subplots(2, 3).flatten()
            for i, metric in enumerate(metrics):
                pattern = re.compile(f"{disease}_(\\d+)_(\\d+)_" + metric)
                data = []
                for col in df.columns:
                    match = pattern.match(col)
                    if match:
                        year_range = f"{match.group(1)}–{match.group(2)}"
                        for _, row in df.iterrows():
                            data.append(
                                {
                                    "year_range": year_range,
                                    "value": row[col],
                                    "modality": row["modalities"],
                                }
                            )
                if data:
                    plot_df = pd.DataFrame(data)
                    sns.lineplot(
                        data=plot_df,
                        x="year_range",
                        y="value",
                        hue="modality",
                        markers=True,
                        style="modality",
                        dashes=style_dict["dashes"],
                        markersize=15,
                        ax=axes2[i],
                    )
                    axes2[i].set_title(f"Range {metric.replace('_', ' ').title()}")
                    axes2[i].grid(True)

            range_sample_data = []
            pos_pat = re.compile(f"{disease}_(\\d+)_(\\d+)_positive_samples")
            neg_pat = re.compile(f"{disease}_(\\d+)_(\\d+)_negative_samples")
            for col in df.columns:
                for _, row in df.iterrows():
                    if pos_pat.match(col):
                        range_ = f"{pos_pat.match(col).group(1)}–{pos_pat.match(col).group(2)}"
                        range_sample_data.append(
                            {
                                "year_range": range_,
                                "count": row[col],
                                "label": "Positive",
                            }
                        )
                    elif neg_pat.match(col):
                        range_ = f"{neg_pat.match(col).group(1)}–{neg_pat.match(col).group(2)}"
                        range_sample_data.append(
                            {
                                "year_range": range_,
                                "count": row[col],
                                "label": "Negative",
                            }
                        )
            if range_sample_data:
                range_df = pd.DataFrame(range_sample_data)
                ax = axes2[-1]
                sns.barplot(
                    data=range_df,
                    x="year_range",
                    y="count",
                    hue="label",
                    ci=None,
                    ax=ax,
                )
                ax.set_title("Sample Counts by Year Range")
                ax.grid(True)
                for container in ax.containers:
                    ax.bar_label(container, fmt="%.0f", fontsize=8, padding=2)

            # Section 3: Multiclass dashboard
            # axes3 = subfigs[2].subplots(3, 3).flatten()

            # Calculate how many subplots are needed in section 3
            n_barplots = len(multiclass_metrics)
            n_confmats = min(3, len(df))
            n_special_charts = 2  # AUROC per Class + Class Counts
            total_axes_needed = n_barplots + n_confmats + n_special_charts

            # Determine appropriate grid size
            n_cols = 3
            n_rows = (total_axes_needed + n_cols - 1) // n_cols  # ceil division

            axes3 = subfigs[2].subplots(n_rows, n_cols).flatten()

            for i, (_, row) in enumerate(df.iterrows()):
                mat = np.zeros((5, 5), dtype=int)
                for p_idx, p_cls in enumerate(multiclass_classes):
                    for t_idx, t_cls in enumerate(multiclass_classes):
                        col = f"{disease}_multiclass_confmat_pred_{p_cls}_true_{t_cls}_counts"
                        if col in df.columns:
                            mat[t_idx, p_idx] = row[col]
                sns.heatmap(
                    mat,
                    annot=True,
                    fmt="d",
                    cmap="Blues",
                    xticklabels=multiclass_classes,
                    yticklabels=multiclass_classes,
                    ax=axes3[i],
                )
                axes3[i].set_title(f"Confusion Matrix\n({row['modalities']})")
                axes3[i].set_ylabel("Predicted")
                axes3[i].set_xlabel("True")

            fig.suptitle(f"{disease.upper()} - Full Dashboard", fontsize=24)
            pdf.savefig(fig)

            if png_dir is not None:
                os.makedirs(png_dir, exist_ok=True)
                fig.savefig(
                    os.path.join(png_dir, f"{disease}_dashboard_full.png"),
                    bbox_inches="tight",
                    dpi=300,
                )

                # Save each individual subplot (Axes)
                for i, ax in enumerate(fig.get_axes()):
                    temp_fig, temp_ax = plt.subplots(figsize=(8, 6))

                    # Copy content from original Axes to new Axes
                    for line in ax.get_lines():
                        temp_ax.plot(
                            line.get_xdata(),
                            line.get_ydata(),
                            label=line.get_label(),
                            linestyle=line.get_linestyle(),
                            marker=line.get_marker(),
                            color=line.get_color(),
                        )

                    for bar_container in ax.containers:
                        temp_ax.bar(
                            [bar.get_x() for bar in bar_container],
                            [bar.get_height() for bar in bar_container],
                            width=[bar.get_width() for bar in bar_container],
                            label=bar_container.get_label(),
                        )

                    temp_ax.set_title(ax.get_title())
                    temp_ax.set_xlabel(ax.get_xlabel())
                    temp_ax.set_ylabel(ax.get_ylabel())
                    temp_ax.legend(loc="best")
                    temp_ax.grid(True)

                    temp_fig.tight_layout()
                    temp_fig.savefig(
                        os.path.join(png_dir, f"{disease}_subplot_{i+1}.png"),
                        bbox_inches="tight",
                        dpi=300,
                    )
                    plt.close(temp_fig)

            plt.close(fig)




def save_all_disease_dashboards_to_pdf_backup(
    df,
    disease_list,
    metrics,
    multiclass_classes,
    saving_path,
    style_dict,
    pdf_filename="all_disease_dashboards.pdf",
    multiclass_metrics=[],
    png_dir=None,  # Optional directory to save PNGs
):
    os.makedirs(saving_path, exist_ok=True)
    pdf_path = os.path.join(saving_path, pdf_filename)
    with PdfPages(pdf_path) as pdf:
        for disease in disease_list:
            fig = plt.figure(constrained_layout=True, figsize=(24, 36))
            subfigs = fig.subfigures(3, 1, height_ratios=[1, 1, 1])

            # Section 1: Time-based dashboard
            axes1 = subfigs[0].subplots(2, 3).flatten()
            for i, metric in enumerate(metrics):
                pattern = re.compile(f"has_{disease}_in_(\\d+)_years_{metric}")
                data = []
                for col in df.columns:
                    match = pattern.match(col)
                    if match:
                        year = int(match.group(1))
                        for _, row in df.iterrows():
                            data.append(
                                {
                                    "year": year,
                                    "value": row[col],
                                    "modality": row["modalities"],
                                }
                            )
                if data:
                    plot_df = pd.DataFrame(data)
                    sns.lineplot(
                        data=plot_df,
                        x="year",
                        y="value",
                        hue="modality",
                        markers=True,
                        style="modality",
                        dashes=style_dict["dashes"],
                        markersize=15,
                        ax=axes1[i],
                    )
                    axes1[i].set_title(f"{metric.replace('_', ' ').title()}")
                    axes1[i].grid(True)

            pos_pat = re.compile(f"has_{disease}_in_(\\d+)_years_positive_samples")
            neg_pat = re.compile(f"has_{disease}_in_(\\d+)_years_negative_samples")
            sample_data = []
            for col in df.columns:
                for _, row in df.iterrows():
                    if pos_pat.match(col):
                        year = int(pos_pat.match(col).group(1))
                        sample_data.append(
                            {"year": year, "count": row[col], "label": "Positive"}
                        )
                    elif neg_pat.match(col):
                        year = int(neg_pat.match(col).group(1))
                        sample_data.append(
                            {"year": year, "count": row[col], "label": "Negative"}
                        )
            if sample_data:
                sample_df = pd.DataFrame(sample_data)
                ax = axes1[-1]
                sns.barplot(
                    data=sample_df, x="year", y="count", hue="label", ci=None, ax=ax
                )
                ax.set_title("Sample Counts")
                ax.grid(True)
                for container in ax.containers:
                    ax.bar_label(container, fmt="%.0f", fontsize=8, padding=2)

            # Section 2: Range-based dashboard
            axes2 = subfigs[1].subplots(2, 3).flatten()
            for i, metric in enumerate(metrics):
                pattern = re.compile(f"{disease}_(\\d+)_(\\d+)_" + metric)
                data = []
                for col in df.columns:
                    match = pattern.match(col)
                    if match:
                        year_range = f"{match.group(1)}–{match.group(2)}"
                        for _, row in df.iterrows():
                            data.append(
                                {
                                    "year_range": year_range,
                                    "value": row[col],
                                    "modality": row["modalities"],
                                }
                            )
                if data:
                    plot_df = pd.DataFrame(data)
                    sns.lineplot(
                        data=plot_df,
                        x="year_range",
                        y="value",
                        hue="modality",
                        markers=True,
                        style="modality",
                        dashes=style_dict["dashes"],
                        markersize=15,
                        ax=axes2[i],
                    )
                    axes2[i].set_title(f"Range {metric.replace('_', ' ').title()}")
                    axes2[i].grid(True)

            range_sample_data = []
            pos_pat = re.compile(f"{disease}_(\\d+)_(\\d+)_positive_samples")
            neg_pat = re.compile(f"{disease}_(\\d+)_(\\d+)_negative_samples")
            for col in df.columns:
                for _, row in df.iterrows():
                    if pos_pat.match(col):
                        range_ = f"{pos_pat.match(col).group(1)}–{pos_pat.match(col).group(2)}"
                        range_sample_data.append(
                            {
                                "year_range": range_,
                                "count": row[col],
                                "label": "Positive",
                            }
                        )
                    elif neg_pat.match(col):
                        range_ = f"{neg_pat.match(col).group(1)}–{neg_pat.match(col).group(2)}"
                        range_sample_data.append(
                            {
                                "year_range": range_,
                                "count": row[col],
                                "label": "Negative",
                            }
                        )
            if range_sample_data:
                range_df = pd.DataFrame(range_sample_data)
                ax = axes2[-1]
                sns.barplot(
                    data=range_df,
                    x="year_range",
                    y="count",
                    hue="label",
                    ci=None,
                    ax=ax,
                )
                ax.set_title("Sample Counts by Year Range")
                ax.grid(True)
                for container in ax.containers:
                    ax.bar_label(container, fmt="%.0f", fontsize=8, padding=2)

            # Section 3: Multiclass dashboard
            # axes3 = subfigs[2].subplots(3, 3).flatten()

            # Calculate how many subplots are needed in section 3
            n_barplots = len(multiclass_metrics)
            n_confmats = min(3, len(df))
            n_special_charts = 2  # AUROC per Class + Class Counts
            total_axes_needed = n_barplots + n_confmats + n_special_charts

            # Determine appropriate grid size
            n_cols = 3
            n_rows = (total_axes_needed + n_cols - 1) // n_cols  # ceil division

            axes3 = subfigs[2].subplots(n_rows, n_cols).flatten()

            for i, metric in enumerate(multiclass_metrics):
                col = f"{disease}_multiclass_{metric}"
                if col in df.columns:
                    ax = axes3[i]
                    sns.barplot(data=df, x="modalities", y=col, ax=ax)
                    ax.set_title(f"Multiclass {metric.title()}")
                    ax.tick_params(axis="x", rotation=45)
                    ax.grid(True)
                    for container in ax.containers:
                        ax.bar_label(container, fmt="%.2f", padding=3, fontsize=8)

            auroc_data = []
            for cls in multiclass_classes:
                col = f"{disease}_multiclass_class_{cls}_auroc"
                if col in df.columns:
                    for _, row in df.iterrows():
                        auroc_data.append(
                            {
                                "modality": row["modalities"],
                                "class": cls,
                                "auroc": row[col],
                            }
                        )
            if auroc_data:
                auroc_df = pd.DataFrame(auroc_data)
                ax = axes3[4]
                sns.barplot(
                    data=auroc_df,
                    x="class",
                    y="auroc",
                    hue="modality",
                    ax=ax,
                    dodge=True,
                )
                ax.set_title("AUROC per Class")
                ax.grid(True)
                for container in ax.containers:
                    for bar in container:
                        height = bar.get_height()
                        if height > 0:
                            ax.annotate(
                                f"{height:.4f}",
                                xy=(bar.get_x() + bar.get_width() / 2, height),
                                xytext=(0, 3),
                                textcoords="offset points",
                                ha="center",
                                va="bottom",
                                fontsize=8,
                                rotation=90,
                            )

            count_data = []
            for cls in multiclass_classes:
                col = f"{disease}_multiclass_class_{cls}_counts"
                if col in df.columns:
                    for _, row in df.iterrows():
                        count_data.append(
                            {
                                "modality": row["modalities"],
                                "class": cls,
                                "count": row[col],
                            }
                        )
            if count_data:
                count_df = pd.DataFrame(count_data)
                ax = axes3[5]
                sns.barplot(
                    data=count_df,
                    x="class",
                    y="count",
                    hue="modality",
                    ax=ax,
                    dodge=True,
                )
                ax.set_title("Class Counts")
                ax.grid(True)
                for container in ax.containers:
                    for bar in container:
                        height = bar.get_height()
                        if height > 0:
                            ax.annotate(
                                f"{int(height)}",
                                xy=(bar.get_x() + bar.get_width() / 2, height),
                                xytext=(0, 3),
                                textcoords="offset points",
                                ha="center",
                                va="bottom",
                                fontsize=8,
                                rotation=90,
                            )

            for i, (_, row) in enumerate(df.iterrows()):
                if i >= 3:
                    break
                mat = np.zeros((5, 5), dtype=int)
                for p_idx, p_cls in enumerate(multiclass_classes):
                    for t_idx, t_cls in enumerate(multiclass_classes):
                        col = f"{disease}_multiclass_confmat_pred_{p_cls}_true_{t_cls}_counts"
                        if col in df.columns:
                            mat[t_idx, p_idx] = row[col]
                sns.heatmap(
                    mat,
                    annot=True,
                    fmt="d",
                    cmap="Blues",
                    xticklabels=multiclass_classes,
                    yticklabels=multiclass_classes,
                    ax=axes3[6 + i],
                )
                axes3[6 + i].set_title(f"Confusion Matrix\n({row['modalities']})")
                axes3[6 + i].set_xlabel("Predicted")
                axes3[6 + i].set_ylabel("True")

            fig.suptitle(f"{disease.upper()} - Full Dashboard", fontsize=24)
            pdf.savefig(fig)

            if png_dir is not None:
                os.makedirs(png_dir, exist_ok=True)
                fig.savefig(
                    os.path.join(png_dir, f"{disease}_dashboard_full.png"),
                    bbox_inches="tight",
                    dpi=300,
                )

                # Save each individual subplot (Axes)
                for i, ax in enumerate(fig.get_axes()):
                    temp_fig, temp_ax = plt.subplots(figsize=(8, 6))

                    # Copy content from original Axes to new Axes
                    for line in ax.get_lines():
                        temp_ax.plot(
                            line.get_xdata(),
                            line.get_ydata(),
                            label=line.get_label(),
                            linestyle=line.get_linestyle(),
                            marker=line.get_marker(),
                            color=line.get_color(),
                        )

                    for bar_container in ax.containers:
                        temp_ax.bar(
                            [bar.get_x() for bar in bar_container],
                            [bar.get_height() for bar in bar_container],
                            width=[bar.get_width() for bar in bar_container],
                            label=bar_container.get_label(),
                        )

                    temp_ax.set_title(ax.get_title())
                    temp_ax.set_xlabel(ax.get_xlabel())
                    temp_ax.set_ylabel(ax.get_ylabel())
                    temp_ax.legend(loc="best")
                    temp_ax.grid(True)

                    temp_fig.tight_layout()
                    temp_fig.savefig(
                        os.path.join(png_dir, f"{disease}_subplot_{i+1}.png"),
                        bbox_inches="tight",
                        dpi=300,
                    )
                    plt.close(temp_fig)

            plt.close(fig)
