import torch.nn as nn
from graphviz import Digraph


class RNNVisualizer:

    def __init__(self, model, output_layer=None):
        if not isinstance(model, (nn.RNN, nn.GRU, nn.LSTM)):
            raise ValueError("Model must be RNN, GRU, or LSTM")

        self.model = model
        self.output_layer = output_layer

        self.input_size = model.input_size
        self.hidden_size = model.hidden_size
        self.num_layers = model.num_layers
        self.bidirectional = model.bidirectional
        self.cell_type = model.__class__.__name__

    # -------------------------------------------------------
    # Folded representation
    # -------------------------------------------------------

    def draw_folded(self, filename="folded_rnn"):

        g = Digraph("FoldedRNN")
        g.attr(rankdir="LR")

        g.node("input", f"x_t\n({self.input_size})")

        prev = "input"

        for l in range(self.num_layers):  # noqa: E741

            layer_name = f"{self.cell_type}_layer{l}"

            label = f"{self.cell_type} layer {l}\nhidden={self.hidden_size}"

            if self.bidirectional:
                label += "\n(bidirectional)"

            g.node(layer_name, label)

            g.edge(prev, layer_name)
            g.edge(layer_name, layer_name, label="h_(t-1)")

            prev = layer_name

        if self.output_layer:

            g.node("output", "Output Layer")

            g.edge(prev, "output")

        g.render(filename, format="png", cleanup=True)

        print(f"Saved folded diagram → {filename}.png")

    # -------------------------------------------------------
    # Unfolded representation
    # -------------------------------------------------------

    def draw_unfolded(self, seq_len=5, filename="unfolded_rnn"):

        g = Digraph("UnfoldedRNN")
        g.attr(rankdir="LR")

        # create nodes

        for t in range(seq_len):

            g.node(f"x{t}", f"x{t}")

            for l in range(self.num_layers):  # noqa: E741

                node_name = f"h{t}_l{l}"

                g.node(node_name,
                       f"{self.cell_type}\nL{l}")

        # connect inputs

        for t in range(seq_len):

            g.edge(f"x{t}", f"h{t}_l0")

        # recurrent connections

        for t in range(1, seq_len):

            for l in range(self.num_layers):  # noqa: E741

                g.edge(f"h{t-1}_l{l}",
                       f"h{t}_l{l}",
                       label="h")

        # vertical layer connections

        for t in range(seq_len):

            for l in range(1, self.num_layers):  # noqa: E741

                g.edge(f"h{t}_l{l-1}",
                       f"h{t}_l{l}")

        # output layer

        if self.output_layer:

            for t in range(seq_len):

                g.node(f"y{t}", "y")

                g.edge(f"h{t}_l{self.num_layers-1}",
                       f"y{t}")

        g.render(filename, format="png", cleanup=True)

        print(f"Saved unfolded diagram → {filename}.png")

    # -------------------------------------------------------
    # Show weight sharing
    # -------------------------------------------------------

    def draw_weight_sharing(self, filename="rnn_weights"):

        g = Digraph("Weights")
        g.attr(rankdir="LR")

        g.node("Wx", "Input weights\nWx")
        g.node("Wh", "Recurrent weights\nWh")

        g.node("cell", f"{self.cell_type} Cell")

        g.edge("Wx", "cell")
        g.edge("Wh", "cell")

        g.render(filename, format="png", cleanup=True)

        print(f"Saved weight diagram → {filename}.png")


# -------------------------------------------------------
# Example usage
# -------------------------------------------------------

if __name__ == "__main__":

    # Example model
    rnn = nn.LSTM(
        input_size=16,
        hidden_size=32,
        num_layers=2,
        bidirectional=True
    )

    output_layer = nn.Linear(32 * 2, 10)

    viz = RNNVisualizer(rnn, output_layer)

    viz.draw_folded()

    viz.draw_unfolded(seq_len=6)

    viz.draw_weight_sharing()