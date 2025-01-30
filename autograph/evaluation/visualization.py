import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx


def component_laplacian(nx_graph, k, normalization="unit"):
    if normalization not in ("unit", "sqrt"):
        raise ValueError("Unknown normalization")

    eigenvector_encoding = np.zeros((nx_graph.number_of_nodes(), k))
    eigenvalue_encoding = np.zeros((nx_graph.number_of_nodes(), k))
    component_index = np.zeros(nx_graph.number_of_nodes())
    num_components = 0

    node_list = sorted(list(nx_graph.nodes))
    whole_laplacian = nx.normalized_laplacian_matrix(
        nx_graph, nodelist=node_list
    ).toarray()

    for idx, node_set in enumerate(nx.connected_components(nx_graph)):
        node_set = sorted(list(node_set))
        component_index[node_set] = idx
        # induced = nx.induced_subgraph(nx_graph, node_set)
        # laplacian = nx.normalized_laplacian_matrix(induced).toarray()
        laplacian = whole_laplacian[node_set, :][:, node_set]
        eig_vals, eig_vecs = np.linalg.eigh(laplacian)
        # we randomly flip the eigenvectors
        eig_vecs = eig_vecs * (
            2 * np.random.randint(0, 2, size=(1, eig_vecs.shape[1])) - 1
        )
        if normalization == "sqrt":
            eig_vecs *= np.sqrt(len(node_set))
        if eig_vecs.shape[1] < k + 1:
            eig_vecs = np.pad(
                eig_vecs, [(0, 0), (0, k + 1 - eig_vecs.shape[1])], mode="constant"
            )
            eig_vals = np.pad(
                eig_vals, [(0, k + 1 - eig_vals.shape[0])], mode="constant"
            )
        eigenvector_encoding[node_set] = eig_vecs[:, 1 : k + 1]
        eigenvalue_encoding[node_set] = eig_vals[1 : k + 1]
        num_components += 1

    return eigenvector_encoding, eigenvalue_encoding, component_index, num_components


def draw_nx_graph(graph, layout, ax, node_color=None):
    if node_color is None:
        fiedler, _, _, _ = component_laplacian(graph, 1, "sqrt")
        node_color = fiedler[:, 0]
    nx.draw(graph, pos=layout, font_size=5, node_size=100, node_color=node_color, ax=ax)

def draw_nx_graph(graph, layout, ax):
    # Set node colors based on the eigenvectors
    w, U = np.linalg.eigh(nx.normalized_laplacian_matrix(graph).toarray())
    vmin, vmax = np.min(U[:, 1]), np.max(U[:, 1])
    m = max(np.abs(vmin), vmax)
    vmin, vmax = -m, m

    nx.draw(graph, pos=layout, font_size=5, node_size=25, node_color=U[:, 1],
            cmap=plt.cm.coolwarm, vmin=vmin, vmax=vmax, edge_color='grey')

def plot_gridspec_graphs(graphs, num_columns=3):
    num_graphs = len(graphs)
    num_rows = (num_graphs - 1) // num_columns + 1  # Number of rows in the grid

    fig_samples = plt.figure(figsize=(5.5 * num_columns, 5.5 * num_rows))
    gs_samples = fig_samples.add_gridspec(num_rows, num_columns)

    for i, graph in enumerate(graphs):
        if isinstance(graph, Data):
            graph = to_networkx(graph, to_undirected=True)
        ax = fig_samples.add_subplot(gs_samples[i])
        layout = nx.spring_layout(graph, iterations=100, seed=42)
        draw_nx_graph(graph, layout, ax)
        ax.set_aspect("equal")

    fig_samples.tight_layout()

    return fig_samples

def plot_smiles(outdir, smiles_list, save_svg=False, save_pdf=False):
    from rdkit import Chem
    from rdkit.Chem import Draw
    from rdkit.Chem.Draw import rdMolDraw2D
    if save_pdf:
        save_svg = True

    for i, mol in enumerate(smiles_list):
        if mol is None:
            continue
        m = Chem.MolFromSmiles(mol)
        Draw.MolToFile(m, f'{outdir}/mol{i}.png', size=(600, 600))
        if save_svg:
            img = rdMolDraw2D.MolDraw2DSVG(100, 100)
            img.drawOptions().minFontSize = 12
            img.DrawMolecule(m)
            img.FinishDrawing()
            outpath = f'{outdir}/mol{i}.svg'
            with open(outpath, 'w') as f:
                f.write(img.GetDrawingText())
            if save_pdf:
                from svglib.svglib import svg2rlg
                from reportlab.graphics import renderPDF
                drawing = svg2rlg(outpath)
                renderPDF.drawToFile(drawing, f'{outdir}/mol{i}.pdf')
