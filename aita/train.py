import wandb
import pickle
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import EditedNearestNeighbours

class AITA_Net(nn.Module):
    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 output_size,
                 hidden_dim,
                 n_layers,
                 embedding_data=None,
                 freeze_embedding=False,
                 drop_prob=0.5):
        super(AITA_Net, self).__init__()
        
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        if embedding_data is None:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
        else:
            embedding_weights = torch.FloatTensor(embedding_data)
            self.embedding = nn.Embedding.from_pretrained(embedding_weights,
                                                          freeze=freeze_embedding)
            embedding_dim = embedding_data.shape[1]

        self.dropout = nn.Dropout(drop_prob)
        if n_layers == 1: drop_prob = 0
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers,
                            dropout=drop_prob, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size = x.size(0)
        x = x.long()
        embedding = self.embedding(x)
        lstm_out, self.hidden = self.lstm(embedding, self.hidden)
        # lstm_out = lstm_out[:, -1, :]
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)

        out = self.dropout(lstm_out)
        out = self.fc(out)
        out = self.sigmoid(out)

        out = out.view(batch_size, -1)
        out = out[:, -1]
        return out

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data

        hidden = (weight.new(self.n_layers, batch_size,
                             self.hidden_dim).zero_(),
                      weight.new(self.n_layers, batch_size,
                                 self.hidden_dim).zero_())
        return hidden

def unpack_data(file_name):
    with open(file_name, 'rb') as of:
        save_dict = pickle.load(of)
    titles = save_dict["titles"]
    labels = save_dict["labels"]
    vocab = save_dict["vocabulary"]
    word2idx = save_dict["word2idx"]
    idx2word = save_dict["idx2word"]
    return titles, labels, vocab, word2idx, idx2word

def build_model(word2idx, embeddings):
    vocab_size = len(word2idx) + 1
    output_size = 1
    embedding_dim = 512
    hidden_dim = 256
    n_layers = 1
    model = AITA_Net(vocab_size,
                     embedding_dim,
                     output_size,
                     hidden_dim,
                     n_layers,
                     embedding_data=embeddings)
    lr = 0.005
    penalty = 0.0001
    bce = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=penalty)
    return model, bce, optimizer

def calc_ratio(labels):
    t = sum(labels)
    f = len(labels) - t
    ratio = t / f
    print(f"True Examples: {t} |  False Examples: {f}")
    print(f"Ratio: {ratio}")

def build_loaders(titles, labels, batch_size,
                  under_sample=False, over_sample=False):
    train_titles, test_titles, train_labels, test_labels = \
        train_test_split(titles, labels, test_size=0.1)
    val_titles, test_titles, val_labels, test_labels = \
        train_test_split(test_titles, test_labels, test_size=0.01)

    steps = []
    if under_sample:
        steps.append(("Under", EditedNearestNeighbours(n_neighbors=2)))
    if over_sample:
        steps.append(("Over", SMOTE(sampling_strategy=1)))
    if under_sample or over_sample:
        pipeline = Pipeline(steps=steps)
        train_titles, train_labels = pipeline.fit_resample(train_titles,
                                                           train_labels)
    print("Train:")
    calc_ratio(train_labels)
    print("Validation:")
    calc_ratio(val_labels)
    print("Test:")
    calc_ratio(test_labels)

    train = TensorDataset(torch.from_numpy(train_titles),
                          torch.from_numpy(train_labels))
    val = TensorDataset(torch.from_numpy(val_titles),
                        torch.from_numpy(val_labels))
    test = TensorDataset(torch.from_numpy(test_titles),
                         torch.from_numpy(test_labels))

    train_loader = DataLoader(train, shuffle=True, batch_size=batch_size,
                              drop_last=True)
    test_loader = DataLoader(test, shuffle=True, batch_size=batch_size,
                             drop_last=True)
    val_loader = DataLoader(val, shuffle=True, batch_size=batch_size,
                            drop_last=True)

    return train_loader, test_loader, val_loader

def main():
    data = unpack_data('processed_dataset.pickle')
    titles, labels, vocab, word2idx, idx2word = data
    with open("embedding_matrix.pickle", 'rb') as of:
        embeddings = pickle.load(of)

    batch_size = 512
    train_loader, test_loader, val_loader = build_loaders(titles,
                                                          labels,
                                                          batch_size)

    model, bce, optimizer = build_model(word2idx, embeddings)

    wandb.init(project="aita_classifier")
    wandb.watch(model)

    epochs = 100
    counter = 0
    log_period = 100
    clip = 5
    val_loss_min = np.Inf

    model.train()
    for e in range(epochs):
        train_num_correct = 0
        train_num_pred = 0
        train_auc = []
        train_losses = []
        for titles, labels in train_loader:
            counter += 1
            # Prepare model
            model.zero_grad()
            model.hidden = model.init_hidden(batch_size)
           
            # Get output and loss
            output = model(titles)
            loss = bce(output, labels.float())

            # Calculate, clip and apply gradient
            loss.backward()
            # nn.utils.clip_grad_norm(model.parameters(), clip)
            optimizer.step()

            with torch.no_grad():
                # Save values for loggin
                prediction = torch.round(output)
                labels = labels.float().view_as(prediction)
                train_auc.append(roc_auc_score(labels, output.detach()))
                correct = prediction.eq(labels)
                correct = np.squeeze(correct.numpy())
                train_num_correct += np.sum(correct)
                train_num_pred += output.size(0)
                train_losses.append(loss.item())

            if counter % log_period == 0:
                validation_losses = []
                val_num_correct = 0
                val_num_pred = 0
                val_auc = []
                model.eval()
                print("Validating...")
                for v_titles, v_labels in tqdm(val_loader):
                    with torch.no_grad():
                        model.hidden = model.init_hidden(batch_size)
                        output = model(v_titles)
                        loss = bce(output, v_labels.float())
                        validation_losses.append(loss.item())

                        prediction = torch.round(output)
                        labels = labels.float().view_as(prediction)
                        val_auc.append(roc_auc_score(labels, output.detach()))
                        correct = prediction.eq(labels)
                        correct = np.squeeze(correct.numpy())
                        val_num_correct += np.sum(correct)
                        val_num_pred += output.size(0)
                model.train()

                avg_val_loss = np.mean(validation_losses)
                val_accuracy = val_num_correct / val_num_pred
                avg_val_auc = np.mean(val_auc)
                avg_train_loss = np.mean(train_losses)
                train_accuracy = train_num_correct / train_num_pred
                avg_train_auc = np.mean(train_auc)
                print(f"Epoch: {e} | Step: {counter}")
                print(f"Train Accuracy: {train_accuracy:.6f} | Validation Accuracy: {val_accuracy:.6f}")
                print(f"Train Loss: {avg_train_loss:.6f} |  Validation Loss: {avg_val_loss:.6f}")
                print(f"Train AUCROC: {avg_train_auc:.6f} | Validation AUCROC: {avg_val_auc:.6f}")

                wandb.log({"Train Accuracy": train_accuracy,
                           "Train Loss": avg_train_loss,
                           "Train AUC": avg_train_auc,
                           "Validation Accuracy": val_accuracy,
                           "Validation Loss": avg_val_loss,
                           "Validation AUC": avg_val_auc,
                           "Step": counter})

                if avg_val_loss < val_loss_min:
                    print("Validation loss decreased! Saving model...")
                    print(f"Loss decreased from {val_loss_min:.6f} to {avg_val_loss:.6f}")
                    torch.save(model.state_dict(),
                               f'model_data/best_state_dict_{e}_{counter}.pt')
                    val_loss_min = avg_val_loss
                print("")

                # Reset average loggin metrics
                train_num_correct = 0
                train_num_pred = 0
                train_loss = []
                train_auc = []

if __name__ == "__main__":
    main()
