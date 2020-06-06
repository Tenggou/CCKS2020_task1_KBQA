import torch
import torch.nn as nn


def mask_(seq, seq_lens):
    mask = torch.zeros_like(seq)
    for i, l in enumerate(seq_lens):
        mask[i, :l].fill_(1)
    return mask


def mask_fill(seq, mask, fill_value):
    return seq.masked_fill(mask == 0, fill_value)


class BiLSTM(nn.Module):
    """
    Bidirectional LSTM
    """
    def __init__(self, emb_dim=300, hidden_dim=300, num_layers=1, device=torch.device('cpu'), size_vocabulary=0):
        super(BiLSTM, self).__init__()
        self.device = device

        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(num_embeddings=size_vocabulary, embedding_dim=emb_dim, padding_idx=0)
        self.embedding.weight.requires_grad = True

        self.rnn = nn.LSTM(input_size=self.emb_dim,
                           hidden_size=self.hidden_dim,
                           num_layers=self.num_layers,
                           bidirectional=True,  #
                           batch_first=True).to(device=self.device)  #

    def forward(self, x):
        """
        x is a list of tensors
        :param x:
        :return:
        """
        h = (torch.ones(2, len(x), self.hidden_dim).to(device=self.device),
             torch.ones(2, len(x), self.hidden_dim).to(device=self.device))  # hidden state / cell state

        x = [torch.Tensor(each).to(device=self.device) for each in x]
        lengths = [len(sent) for sent in x]
        padded_x = nn.utils.rnn.pad_sequence(x, batch_first=True).long()
        padded_x = self.embedding(padded_x)

        packed_x = nn.utils.rnn.pack_padded_sequence(padded_x, lengths=lengths, batch_first=True, enforce_sorted=False).to(device=self.device)
        _, h_out = self.rnn(packed_x, h)
        return h_out[0].transpose(1, 0).contiguous().view(-1, self.hidden_dim * 2)
