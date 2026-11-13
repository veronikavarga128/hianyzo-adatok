"""Ábrák az elemzéshez. Minden függvény elmenti az ábrát, és visszaadja a figure-t."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")

MODEL_NAMES = {"dummy": "Átlagjósló", "baseline": "Baseline (Ridge)", "advanced": "Haladó (Gradient boosting)"}
MODEL_COLORS = {"dummy": "#b8b8b8", "baseline": "#4c78a8", "advanced": "#e4793c"}


def _save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def plot_reduction_experiment(df, title, path):
    """A feature engineering és a dimenziócsökkentés hatása (RMSE)."""
    d = df.copy()
    d["model"] = d["model"].map(MODEL_NAMES)
    d["feature_engineering"] = d["feature_engineering"].map({False: "nincs új jellemző", True: "új jellemzőkkel"})
    d["reduction"] = d["reduction"].map({"none": "nincs csökkentés", "pca": "PCA (95%)", "kbest": "20 legjobb jellemző"})
    g = sns.catplot(
        data=d, x="reduction", y="rmse", hue="feature_engineering", col="model",
        kind="bar", height=3.6, aspect=1.15, palette=["#9ecae1", "#3182bd"],
    )
    g.set_axis_labels("", "RMSE (jegypont)")
    g.set_titles("{col_name}")
    for ax in g.axes.flat:
        plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
        lo = d["rmse"].min()
        ax.set_ylim(lo * 0.9, d["rmse"].max() * 1.04)
    g.legend.set_title("")
    g.figure.suptitle(title, y=1.04)
    return _save(g.figure, path)


def plot_pca_scree(ratio, title, path):
    cum = np.cumsum(ratio)
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.bar(range(1, len(ratio) + 1), ratio, color="#9ecae1", label="komponens varianciahányada")
    ax.plot(range(1, len(ratio) + 1), cum, color="#e4793c", marker="o", ms=3, label="kumulált")
    ax.axhline(0.95, color="grey", ls="--", lw=1)
    n95 = int(np.argmax(cum >= 0.95)) + 1
    ax.axvline(n95, color="grey", ls="--", lw=1)
    ax.text(n95 + 0.5, 0.5, f"{n95} komponens\nadja a variancia 95%-át", fontsize=9)
    ax.set_xlabel("Főkomponens sorszáma")
    ax.set_ylabel("Magyarázott variancia aránya")
    ax.set_title(title)
    ax.legend(loc="center right")
    return _save(fig, path)


def plot_cv_boxplot(scores, title, path):
    """A keresztvalidációs körök RMSE-je modellenként."""
    d = scores.copy()
    order = [m for m in ["dummy", "baseline", "advanced"] if m in d["model"].unique()]
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(data=d, x="model", y="rmse", order=order, palette=MODEL_COLORS, width=0.5,
                showmeans=True, meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": "black"}, ax=ax)
    sns.stripplot(data=d, x="model", y="rmse", order=order, color="black", alpha=0.35, size=3, ax=ax)
    ax.set_xticks(range(len(order)), [MODEL_NAMES[m] for m in order])
    ax.set_xlabel("")
    ax.set_ylabel("RMSE (jegypont) a CV körökben")
    ax.set_title(title)
    return _save(fig, path)


def plot_pred_vs_actual(oof, title, path):
    """Jósolt és tényleges végső jegy a tanításhoz nem használt (out-of-fold) adatokon."""
    models = [m for m in ["baseline", "advanced"] if m in oof["model"].unique()]
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4.6), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, models):
        d = oof[oof["model"] == m]
        ax.scatter(d["y_true"], d["y_pred"], s=14, alpha=0.4, color=MODEL_COLORS[m])
        ax.plot([0, 20], [0, 20], color="black", lw=1, ls="--")
        ax.set_title(MODEL_NAMES[m])
        ax.set_xlabel("Tényleges G3")
    axes[0].set_ylabel("Jósolt G3")
    fig.suptitle(title, y=1.02)
    return _save(fig, path)


def plot_permutation_importance(imp, title, path, top=12):
    """Csoportos permutációs fontosság: az RMSE növekedése a csoport megkeverése után."""
    models = [m for m in ["baseline", "advanced"] if m in imp["model"].unique()]
    fig, axes = plt.subplots(1, len(models), figsize=(6.2 * len(models), 5), sharex=False)
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, models):
        d = imp[imp["model"] == m].nlargest(top, "atlag").iloc[::-1]
        ax.barh(d["csoport"], d["atlag"], xerr=d["szoras"], color=MODEL_COLORS[m], ecolor="black", capsize=2)
        ax.set_title(MODEL_NAMES[m])
        ax.set_xlabel("RMSE növekedése megkeverés után")
    fig.suptitle(title, y=1.01)
    fig.tight_layout()
    return _save(fig, path)


def plot_shap(shap_values, Z, names, title, path_beeswarm, path_bar, top=15):
    """SHAP összefoglaló ábrák (méhraj és átlagos abszolút hatás)."""
    import shap

    Zdf = pd.DataFrame(Z, columns=names)
    plt.figure()
    shap.summary_plot(shap_values, Zdf, max_display=top, show=False)
    fig = plt.gcf()
    fig.suptitle(title, y=1.0)
    _save(fig, path_beeswarm)
    plt.close(fig)

    plt.figure()
    shap.summary_plot(shap_values, Zdf, plot_type="bar", max_display=top, show=False)
    fig2 = plt.gcf()
    fig2.suptitle(title, y=1.0)
    _save(fig2, path_bar)
    plt.close(fig2)