import matplotlib.pyplot as plt
import networkx as nx

# Create a directed graph to visualize ResNet-50 flow
G = nx.DiGraph()

# Add nodes for each processing step
G.add_node("Input Image", color="lightblue")
G.add_node("Conv + Pooling (Basic Feature Extraction)", color="blue")
G.add_node("Residual Block 1 (Basic Shapes & Textures)", color="green")
G.add_node("Residual Block 2 (Mid-Level Features)", color="green")
G.add_node("Residual Block 3 (Complex Features & Object Parts)", color="green")
G.add_node("Residual Block 4 (Full Object Representation)", color="green")
G.add_node("Global Average Pooling (Summarizing Features)", color="orange")
G.add_node("Fully Connected Layer (Classification)", color="red")
G.add_node("Output: Object Label (Dog, Plane, etc.)", color="yellow")

# Connect nodes to show data flow
edges = [
    ("Input Image", "Conv + Pooling (Basic Feature Extraction)"),
    ("Conv + Pooling (Basic Feature Extraction)", "Residual Block 1 (Basic Shapes & Textures)"),
    ("Residual Block 1 (Basic Shapes & Textures)", "Residual Block 2 (Mid-Level Features)"),
    ("Residual Block 2 (Mid-Level Features)", "Residual Block 3 (Complex Features & Object Parts)"),
    ("Residual Block 3 (Complex Features & Object Parts)", "Residual Block 4 (Full Object Representation)"),
    ("Residual Block 4 (Full Object Representation)", "Global Average Pooling (Summarizing Features)"),
    ("Global Average Pooling (Summarizing Features)", "Fully Connected Layer (Classification)"),
    ("Fully Connected Layer (Classification)", "Output: Object Label (Dog, Plane, etc.)")
]

G.add_edges_from(edges)

# Draw the model architecture
plt.figure(figsize=(10, 6))
nx.draw(G, with_labels=True, node_color="lightblue", node_size=3000, edge_color="gray", font_size=8, font_weight="bold")
plt.title("📌 Visualization of ResNet-50 Image Processing Flow")
plt.show()
