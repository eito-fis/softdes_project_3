import wandb
import pickle
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

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
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers,
                            dropout=drop_prob, batch_first=True)
        self.dropout = nn.Dropout(drop_prob)
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
    hidden_dim = 512
    n_layers = 1
    model = AITA_Net(vocab_size,
                     embedding_dim,
                     output_size,
                     hidden_dim,
                     n_layers,
                     embedding_data=embeddings)
    lr = 0.005
    bce = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    return model, bce, optimizer

def build_loaders(titles, labels, batch_size):
    dataset = TensorDataset(torch.from_numpy(titles), torch.from_numpy(labels))

    dataset_size = len(titles)
    train_proportion = 0.8
    test_proportion = 0.1

    train_size = int(dataset_size * train_proportion)
    test_size = int(dataset_size * test_proportion)
    val_size = int(dataset_size - train_size - test_size)
    split = [train_size, test_size, val_size]

    train, test, val = torch.utils.data.random_split(dataset, split)
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

    epochs = 20
    counter = 0
    log_period = 25
    clip = 5
    val_loss_min = np.Inf

    model.train()
    for e in range(epochs):
        train_num_correct = 0
        train_num_pred = 0
        train_losses = []
        for titles, labels in train_loader:
            counter += 1
            # Prepare model
            model.zero_grad()
            model.hidden = model.init_hidden(batch_size)
           
            # Get output and loss
            output = model(titles)
            loss = bce(output, labels.float())

            # Save values for loggin
            prediction = torch.round(output)
            correct = prediction.eq(labels.float().view_as(prediction))
            correct = np.squeeze(correct.numpy())
            train_num_correct += np.sum(correct)
            train_num_pred += output.size(0)
            train_losses.append(loss.item())

            # Calculate, clip and apply gradient
            loss.backward()
            nn.utils.clip_grad_norm(model.parameters(), clip)
            optimizer.step()

            if counter % log_period == 0:
                validation_losses = []
                val_num_correct = 0
                val_num_pred = 0
                model.eval()
                for v_titles, v_labels in val_loader:
                    model.hidden = model.init_hidden(batch_size)
                    output = model(v_titles)
                    loss = bce(output, v_labels.float())
                    validation_losses.append(loss.item())

                    prediction = torch.round(output)
                    correct = prediction.eq(labels.float().view_as(prediction))
                    correct = np.squeeze(correct.numpy())
                    val_num_correct += np.sum(correct)
                    val_num_pred += output.size(0)
                model.train()

                avg_val_loss = np.mean(validation_losses)
                val_accuracy = val_num_correct / val_num_pred
                avg_train_loss = np.mean(train_losses)
                train_accuracy = train_num_correct / train_num_pred
                print("-" * 30)
                print(f"Epoch: {e} | Step: {counter}")
                print(f"Train Accuracy: {train_accuracy}")
                print(f"Train Loss: {avg_train_loss}")
                print(f"Validation Accuracy: {val_accuracy}")
                print(f"Validation Loss: {avg_val_loss:.6f}")

                wandb.log({"Train Accuracy": train_accuracy,
                           "Train Loss": avg_train_loss,
                           "Validation Accuracy": val_accuracy,
                           "Validation Loss": avg_val_loss,
                           "Step": counter})

                if avg_val_loss < val_loss_min:
                    torch.save(model.state_dict(),
                               f'model_data/state_dict_{e}_{counter}.pt')
                    torch.save(optimizer.state_dict(),
                               f'model_data/optimizer_{e}_{counter}.pth')
                    print(f"""Validation loss decreased:
                          ({val_loss_min:.6f} --> {avg_val_loss:.6f})""")
                    print("Saved new model")
                    val_loss_min = avg_val_loss

                # Reset average loggin metrics
                train_num_correct = 0
                train_num_pred = 0
                train_loss = []

if __name__ == "__main__":
    main()
