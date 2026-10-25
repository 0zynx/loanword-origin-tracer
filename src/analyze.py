"""UMAP coloured by origin + borrowing graph."""
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from config import PROCESSED, FIGS, SEED

p = argparse.ArgumentParser()
p.add_argument("--umap", action="store_true")
p.add_argument("--graph", action="store_true")
p.add_argument("--view", default="translit")
args = p.parse_args()

df = pd.read_csv(PROCESSED / "splits.csv")
col = "word_translit" if args.view == "translit" else "word_native"

if args.umap:
    import umap
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2)
    X = vec.fit_transform(df[col])
    X = TruncatedSVD(128, random_state=SEED).fit_transform(X)
    E = umap.UMAP(n_neighbors=25, min_dist=0.1, random_state=SEED).fit_transform(X)
    plt.figure(figsize=(9, 7))
    for lab in df.label.unique():
        m = (df.label == lab).values
        plt.scatter(E[m, 0], E[m, 1], s=6, alpha=0.7, label=lab)
    plt.legend(markerscale=3, fontsize=8); plt.title("UMAP of word shapes by origin")
    plt.tight_layout(); plt.savefig(FIGS / f"umap_{args.view}.png", dpi=220); plt.close()
    print("wrote", FIGS / f"umap_{args.view}.png")

if args.graph:
    e = df[df.label != "native"].groupby(["lang_name", "label"]).size().reset_index(name="w")
    G = nx.DiGraph()
    for _, r in e.iterrows():
        G.add_edge(r.lang_name, r.label, weight=int(r.w))
    pos = nx.spring_layout(G, seed=SEED, k=0.9)
    plt.figure(figsize=(9, 8))
    nx.draw_networkx_nodes(G, pos, node_size=[600 + 30 * G.degree(n) for n in G],
                           node_color="#8ecae6", edgecolors="k")
    nx.draw_networkx_labels(G, pos, font_size=9)
    nx.draw_networkx_edges(G, pos, arrowsize=14,
                           width=[0.4 + 0.4 * G[u][v]["weight"] for u, v in G.edges],
                           connectionstyle="arc3,rad=0.12", edge_color="#555")
    plt.title("Loanword flow: borrower -> source")
    plt.axis("off"); plt.tight_layout()
    plt.savefig(FIGS / "borrowing_graph.png", dpi=220); plt.close()
    e.sort_values("w", ascending=False).to_csv(PROCESSED / "borrowing_edges.csv", index=False)
    print("wrote", FIGS / "borrowing_graph.png")
