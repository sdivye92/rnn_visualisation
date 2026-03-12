import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(0)

###############################################
# Sentence
###############################################

sentence = "the cat sat on the warm mat"
tokens = sentence.split()
vocab = {w:i for i,w in enumerate(set(tokens))}
inv_vocab = {i:w for w,i in vocab.items()}

seq = torch.tensor([vocab[w] for w in tokens])
seq_len = len(seq)

###############################################
# Hyperparameters
###############################################

embedding_dim = 8
hidden_dim = 16
vocab_size = len(vocab)

###############################################
# Model Definitions
###############################################

class RNNModel(nn.Module):
    def __init__(self, rnn_type):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        if rnn_type == "RNN":
            self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)

        elif rnn_type == "LSTM":
            self.rnn = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

        elif rnn_type == "GRU":
            self.rnn = nn.GRU(embedding_dim, hidden_dim, batch_first=True)

        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        emb = self.embedding(x)
        out, hidden = self.rnn(emb)
        out = self.fc(out)
        return out, emb, hidden


###############################################
# Training Function
###############################################

def train_model(model):

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    inputs = seq[:-1].unsqueeze(0)
    targets = seq[1:].unsqueeze(0)

    for _ in range(200):

        optimizer.zero_grad()
        outputs, _, _ = model(inputs)

        loss = criterion(
            outputs.reshape(-1, vocab_size),
            targets.reshape(-1)
        )

        loss.backward()
        optimizer.step()

    return model


###############################################
# Gradient Influence
###############################################

def gradient_influence(model):

    inputs = seq[:-1].unsqueeze(0)

    outputs, embeddings, _ = model(inputs)

    final_state = outputs[0,-1]

    grads = []

    for i in range(seq_len-1):

        g = torch.autograd.grad(
            final_state.sum(),
            embeddings,
            retain_graph=True
        )[0][0,i]

        grads.append(g.norm().item())

    return grads


###############################################
# Occlusion Influence
###############################################

def occlusion_influence(model):

    inputs = seq[:-1].unsqueeze(0)

    base_out,_ ,_ = model(inputs)
    base_vec = base_out[0,-1].detach()

    scores = []

    for i in range(seq_len-1):

        masked = inputs.clone()
        masked[0,i] = 0

        out,_,_ = model(masked)

        diff = torch.norm(base_vec - out[0,-1]).item()

        scores.append(diff)

    return scores


###############################################
# Plot Function
###############################################

def plot_scores(scores, title):

    plt.figure()
    plt.bar(range(len(scores)), scores)
    plt.xticks(range(len(scores)), tokens[:-1], rotation=45)
    plt.title(title)
    plt.ylabel("Influence")
    plt.show()

def plot_scores2(scores, title):

    plt.bar(range(len(scores)), scores)
    plt.xticks(range(len(scores)), tokens[:-1], rotation=45)
    plt.title(title)
    plt.ylabel("Influence")


###############################################
# Information Flow Diagram
###############################################

def flow_diagram(scores):

    scores = np.array(scores)

    # normalize to [0,1]
    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

    plt.figure(figsize=(8,3))

    x = np.arange(len(scores))

    for i,s in enumerate(scores):

        plt.arrow(
            i, 1,
            0, -0.6,
            head_width=0.1,
            head_length=0.1,
            length_includes_head=True,
            alpha=0.2 + 0.8*s,   # ensure visibility
            linewidth=2
        )

    plt.xticks(x, tokens[:-1], rotation=45)
    plt.yticks([])
    plt.title("Information Flow Strength")
    plt.ylim(0,1.2)

    plt.show()


def flow_heatmap(scores):

    scores = np.array(scores).reshape(1,-1)

    plt.figure(figsize=(8,1.5))
    plt.imshow(scores, cmap="viridis", aspect="auto")

    plt.xticks(range(len(tokens)-1), tokens[:-1], rotation=45)
    plt.yticks([])

    plt.colorbar(label="Influence")
    plt.title("Token Influence Heatmap")

    plt.show()

###############################################
# Train Models
###############################################

rnn_model = train_model(RNNModel("RNN"))
lstm_model = train_model(RNNModel("LSTM"))
gru_model = train_model(RNNModel("GRU"))

###############################################
# Compute Influences
###############################################

rnn_grad = gradient_influence(rnn_model)
lstm_grad = gradient_influence(lstm_model)
gru_grad = gradient_influence(gru_model)

rnn_occ = occlusion_influence(rnn_model)

###############################################
# Visualizations
###############################################

plt.figure(figsize=(12,4))
plt.subplot(1, 3, 1)
plot_scores2(rnn_grad, "Gradient Influence (RNN)")
plt.subplot(1, 3, 2)
plot_scores2(lstm_grad, "Gradient Influence (LSTM)")
plt.subplot(1, 3, 3)
plot_scores2(gru_grad, "Gradient Influence (GRU)")
plt.show()

# plot_scores(rnn_occ, "Occlusion Influence (RNN)")

###############################################
# BPTT Decay Curve
###############################################

plt.figure()

plt.plot(rnn_grad, label="RNN")
plt.plot(lstm_grad, label="LSTM")
plt.plot(gru_grad, label="GRU")

plt.xticks(range(len(tokens)-1), tokens[:-1], rotation=45)
plt.title("Memory Retention Curve")
plt.ylabel("Gradient Influence")
plt.legend()

plt.show()

###############################################
# Information Flow
###############################################

flow_diagram(rnn_grad)
flow_heatmap(rnn_grad)