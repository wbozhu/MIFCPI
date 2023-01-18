# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from torch.autograd import Variable
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
class GATLayer(nn.Module):

    def __init__(self, in_features, out_features, dropout=0.5, alpha=0.2, concat=True):
        super(GATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, h, adj):
        Wh = torch.matmul(h, self.W)
        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = F.leaky_relu(torch.matmul(a_input, self.a).squeeze(3), self.alpha)

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=2)
        h_prime = torch.bmm(attention, Wh)

        return F.elu(h_prime) if self.concat else h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        b = Wh.size()[0]
        N = Wh.size()[1]

        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=1)
        Wh_repeated_alternating = Wh.repeat_interleave(N, dim=0).view(b, N*N, self.out_features)
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=2)

        return all_combinations_matrix.view(b, N, N, 2 * self.out_features)
class CPIMIF(nn.Module):
    def __init__(self,hp,n_atom, n_amino,
                 protein_MAX_LENGH = 1000,
                 drug_MAX_LENGH = 100):
        super(CPIMIF, self).__init__()
        self.dim = hp.char_dim
        self.conv = hp.conv
        self.drug_MAX_LENGH = drug_MAX_LENGH
        self.drug_kernel = hp.drug_kernel
        self.protein_MAX_LENGH = protein_MAX_LENGH
        self.protein_kernel = hp.protein_kernel
        self.conv_layers = nn.ModuleList([nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2 * 5 + 1,
                                                    stride=1, padding=5) for _ in range(4)])
        self.protein_embed = nn.Embedding(26, self.dim,padding_idx=0)
        self.drug_embed = nn.Embedding(65, self.dim,padding_idx=0)
        self.Drug_CNNs = nn.Sequential(
            nn.Conv1d(in_channels=self.dim, out_channels= self.conv,  kernel_size=self.drug_kernel[0]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv, out_channels= self.conv*2,  kernel_size=self.drug_kernel[1]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv*2, out_channels= self.conv*4,  kernel_size=self.drug_kernel[2]),
            nn.ReLU(),
        )
        self.Drug_max_pool = nn.MaxPool1d(self.drug_MAX_LENGH-self.drug_kernel[0]-self.drug_kernel[1]-self.drug_kernel[2]+3)
        self.Protein_CNNs = nn.Sequential(
            nn.Conv1d(in_channels=self.dim, out_channels=self.conv, kernel_size=self.protein_kernel[0]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv, out_channels=self.conv * 2, kernel_size=self.protein_kernel[1]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv * 2, out_channels=self.conv * 4, kernel_size=self.protein_kernel[2]),
            nn.ReLU(),
        )
        self.Protein_max_pool = nn.MaxPool1d(self.protein_MAX_LENGH - self.protein_kernel[0] - self.protein_kernel[1] - self.protein_kernel[2] + 3)
        self.attention_layer = nn.Linear(self.conv*4, self.conv*4)
        self.protein_attention_layer = nn.Linear(self.conv * 4, self.conv * 4)
        self.drug_attention_layer = nn.Linear(self.conv * 4, self.conv * 4)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.dropout3 = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.leaky_relu = nn.LeakyReLU()
        self.fc1 = nn.Linear(96, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, 2)
        global n_word
        n_word = n_amino
        # print(n_amino)
        comp_dim = 32
        prot_dim = 32
        latent_dim = 32
        dropout = 0.1
        alpha = 0.1
        gat_dim = 32
        num_head = 3
        window = 5
        layer_cnn = 4

        self.embedding_layer_atom = nn.Embedding(n_atom + 1, comp_dim)
        self.embedding_layer_amino = nn.Embedding(n_amino + 1, prot_dim)
        # GAT
        self.gat_layers = [GATLayer(comp_dim, gat_dim, dropout=dropout, alpha=alpha, concat=True)
                           for _ in range(num_head)]
        for i, layer in enumerate(self.gat_layers):
            self.add_module('gat_layer_{}'.format(i), layer)
        self.gat_out = GATLayer(gat_dim * num_head, comp_dim, dropout=dropout, alpha=alpha, concat=False)
        # GAT end
        self.W_comp = nn.Linear(comp_dim, latent_dim)
        self.conv_layers = nn.ModuleList([nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2 * window + 1,
                                                    stride=1, padding=window) for _ in range(layer_cnn)])
        self.W_prot = nn.Linear(prot_dim, latent_dim)
        self.alpha = 0.1
        self.fp0 = nn.Parameter(torch.empty(size=(1024, latent_dim)))
        nn.init.xavier_uniform_(self.fp0, gain=1.414)
        self.fp1 = nn.Parameter(torch.empty(size=(latent_dim, latent_dim)))
        nn.init.xavier_uniform_(self.fp1, gain=1.414)
        self.bidat_num = 4
        self.U = nn.ParameterList(
            [nn.Parameter(torch.empty(size=(latent_dim, latent_dim))) for _ in range(self.bidat_num)])
        for i in range(self.bidat_num):
            nn.init.xavier_uniform_(self.U[i], gain=1.414)

        self.transform_c2p = nn.ModuleList([nn.Linear(latent_dim, latent_dim) for _ in range(self.bidat_num)])
        self.transform_p2c = nn.ModuleList([nn.Linear(latent_dim, latent_dim) for _ in range(self.bidat_num)])

        self.bihidden_c = nn.ModuleList([nn.Linear(latent_dim, latent_dim) for _ in range(self.bidat_num)])
        self.bihidden_p = nn.ModuleList([nn.Linear(latent_dim, latent_dim) for _ in range(self.bidat_num)])
        self.biatt_c = nn.ModuleList([nn.Linear(latent_dim * 2, 1) for _ in range(self.bidat_num)])
        self.biatt_p = nn.ModuleList([nn.Linear(latent_dim * 2, 1) for _ in range(self.bidat_num)])

        self.comb_c = nn.Linear(latent_dim * self.bidat_num, latent_dim)
        self.comb_p = nn.Linear(latent_dim * self.bidat_num, latent_dim)
        self.output = nn.Linear(latent_dim * latent_dim * 2, 2)
        hid_dim = 32
        kernel_size = 7
        n_layers = 3
        n_heads = 8
        pf_dim = 256
        self.fc = nn.Linear(prot_dim, hid_dim)
        self.convs = nn.ModuleList(
            [nn.Conv1d(hid_dim, 2 * hid_dim, kernel_size, padding=(kernel_size - 1) // 2) for _ in range(3)])  # convolutional layers

        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to('cuda:0')
        self.ln = nn.LayerNorm(hid_dim)
        self.layers = nn.ModuleList(
            [DecoderLayer(hid_dim, n_heads, pf_dim, SelfAttention, PositionwiseFeedforward, dropout, 'cuda:0')
             for _ in range(n_layers)])

        self.fc_1 = nn.Linear(hid_dim, 256)
        self.fc_2 = nn.Linear(256, 2)
        self.bilstm = nn.LSTM(prot_dim, 80, 2, dropout=0.2, batch_first=True, bidirectional=True)

        # transformer
        n_encoder=3
        n_decoder=1
        dim=32
        d_ff=160
        heads=2
        dropout0 = 0
        self.positional_encoder = positional_encoder(prot_dim, dropout0)
        self.encoder = encoder(n_encoder, dim, d_ff, dropout, heads=heads)
        self.decoder = decoder(n_decoder, dim, d_ff, dropout, heads=heads)
        self.W_cnn = nn.ModuleList([nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2 * 11 + 1,
                                              stride=1, padding=11) for _ in range(3)])
        self.pro_bert = PRO_BERT()
        self.Autoencoder = Autoencoder(32, 32)
    def comp_gat(self, atoms, atoms_mask, adj):
        atoms_vector = self.embedding_layer_atom(atoms)
        atoms_multi_head = torch.cat([gat(atoms_vector, adj) for gat in self.gat_layers], dim=2)
        atoms_vector = F.elu(self.gat_out(atoms_multi_head, adj))
        atoms_vector = F.leaky_relu(self.W_comp(atoms_vector), self.alpha)
        return atoms_vector
    def transformer(self, compound, protein):
        protein = self.positional_encoder(protein)
        protein = self.encoder(protein)
        return protein
    def attention_cnn(self, x, xs, layer):
        xs = torch.unsqueeze(torch.unsqueeze(xs, 0), 0)
        for i in range(layer):
            xs = torch.relu(self.W_cnn[i](xs))
        xs = torch.squeeze(torch.squeeze(xs, 0), 0)
        return xs
    def prot_bilstm(self,amino):
        amino_vector = self.embedding_layer_amino(amino)
        bilstms, _ = self.bilstm(amino_vector)
        atoms_vector = F.leaky_relu(self.W_comp(bilstms), self.alpha)
        return atoms_vector
    def prot_cnn(self, amino, amino_mask):
        amino_vector = self.embedding_layer_amino(amino)
        amino_vector = torch.unsqueeze(amino_vector, 1)
        for i in range(4):
            amino_vector = F.leaky_relu(self.conv_layers[i](amino_vector), self.alpha)
        amino_vector = torch.squeeze(amino_vector, 1)
        amino_vector = F.leaky_relu(self.W_prot(amino_vector), self.alpha)
        return amino_vector
    def mask_softmax(self, a, mask, dim=-1):
        a_max = torch.max(a, dim, keepdim=True)[0]
        a_exp = torch.exp(a - a_max)
        a_exp = a_exp * mask
        a_softmax = a_exp / (torch.sum(a_exp, dim, keepdim=True) + 1e-6)
        return a_softmax
    def bidirectional_attention_prediction(self, atoms_vector, atoms_mask, fps, amino_vector, amino_mask):
        b = atoms_vector.shape[0]
        for i in range(self.bidat_num):
            A = torch.tanh(torch.matmul(torch.matmul(atoms_vector, self.U[i]), amino_vector.transpose(1, 2)))
            A = A * torch.matmul(atoms_mask.view(b, -1, 1), amino_mask.view(b, 1, -1))

            atoms_trans = torch.matmul(A, torch.tanh(self.transform_p2c[i](amino_vector)))
            amino_trans = torch.matmul(A.transpose(1, 2), torch.tanh(self.transform_c2p[i](atoms_vector)))

            atoms_tmp = torch.cat([torch.tanh(self.bihidden_c[i](atoms_vector)), atoms_trans], dim=2)
            amino_tmp = torch.cat([torch.tanh(self.bihidden_p[i](amino_vector)), amino_trans], dim=2)

            atoms_att = self.mask_softmax(self.biatt_c[i](atoms_tmp).view(b, -1), atoms_mask.view(b, -1))
            amino_att = self.mask_softmax(self.biatt_p[i](amino_tmp).view(b, -1), amino_mask.view(b, -1))

            cf = torch.sum(atoms_vector * atoms_att.view(b, -1, 1), dim=1)
            pf = torch.sum(amino_vector * amino_att.view(b, -1, 1), dim=1)

            if i == 0:
                cat_cf = cf
                cat_pf = pf
            else:
                cat_cf = torch.cat([cat_cf.view(b, -1), cf.view(b, -1)], dim=1)
                cat_pf = torch.cat([cat_pf.view(b, -1), pf.view(b, -1)], dim=1)

        cf_final = torch.cat([self.comb_c(cat_cf).view(b, -1), fps.view(b, -1)], dim=1)
        pf_final = self.comb_p(cat_pf)
        cf_pf = F.leaky_relu(torch.matmul(cf_final.view(b, -1, 1), pf_final.view(b, 1, -1)).view(b, -1), 0.1)
        return self.output(cf_pf)
    # CPI-MIF
    def forward(self, atoms, atoms_mask, adjacency, amino, amino_mask, fps):
        """Compound vector with GAT."""
        atoms_vector = self.comp_gat(atoms, atoms_mask, adjacency)
        compound_vector = torch.mean(atoms_vector, 1)
        """update AE2"""
        fusion_vector, aeInput, aeOutput = self.Autoencoder([compound_vector])
        HTrainOptimizor = optim.Adam([self.Autoencoder.H], lr=0.1)
        loss = F.mse_loss(aeOutput, aeInput)
        HTrainOptimizor.zero_grad()
        loss = loss.requires_grad_()
        loss.backward()
        HTrainOptimizor.step()
        """Protein vector with PRO-BERT."""
        amino_vector = self.pro_bert(amino, amino_mask)
        protein_vector = torch.mean(amino_vector, 1)
        """update AE2"""
        fusion_vector, aeInput, aeOutput = self.Autoencoder([protein_vector])
        HTrainOptimizor = optim.Adam([self.Autoencoder.H], lr=0.1)
        loss = F.mse_loss(aeOutput, aeInput)
        HTrainOptimizor.zero_grad()
        loss = loss.requires_grad_()
        loss.backward()
        HTrainOptimizor.step()
        """attention layer"""
        drug_att = self.drug_attention_layer(atoms_vector)
        protein_att = self.protein_attention_layer(amino_vector)
        d_att_layers = torch.unsqueeze(drug_att, 2).repeat(1, 1, amino_vector.shape[-2],1)
        p_att_layers = torch.unsqueeze(protein_att, 1).repeat(1, atoms_vector.shape[-2], 1, 1)
        Atten_matrix = self.attention_layer(self.relu(d_att_layers + p_att_layers))
        Compound_atte = torch.mean(Atten_matrix, 2)
        Protein_atte = torch.mean(Atten_matrix, 1)

        Compound_atte = self.sigmoid(Compound_atte.permute(0, 2, 1))
        Protein_atte = self.sigmoid(Protein_atte.permute(0, 2, 1))

        drugConv =atoms_vector.permute(0, 2, 1)
        proteinConv =amino_vector.permute(0, 2, 1)
        drugConv = drugConv * 0.5 + drugConv * Compound_atte
        proteinConv = proteinConv * 0.5 + proteinConv * Protein_atte

        Drug_max_pool = nn.MaxPool1d(drugConv.shape[-1])
        Protein_max_pool = nn.MaxPool1d(proteinConv.shape[-1])
        drugConv = Drug_max_pool(drugConv).squeeze(2)
        proteinConv = Protein_max_pool(proteinConv).squeeze(2)

        fusion_vector = self.Autoencoder.getH()
        pair = torch.cat([drugConv, proteinConv, fusion_vector], dim=1)
        pair = self.dropout1(pair)
        x1 = self.fc1(pair)
        fully1 = self.leaky_relu(x1)
        fully1 = self.dropout2(fully1)
        fully2 = self.leaky_relu(self.fc2(fully1))
        fully2 = self.dropout3(fully2)
        fully3 = self.leaky_relu(self.fc3(fully2))
        predict = self.out(fully3)
        return predict

# transformer
def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
class positional_encoder(nn.Module):
    def __init__(self, d_model, dropout, max_len=13100):
        super(positional_encoder, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, d_model, 2)).type(torch.FloatTensor) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position.type(torch.FloatTensor) * div_term.type(torch.FloatTensor))
        pe[:, 1::2] = torch.cos(position.type(torch.FloatTensor) * div_term.type(torch.FloatTensor))
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)
    def forward(self, x):
        x = x + Variable(self.pe[0:x.size(0), :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)
# transformer encoder & decoder
class encoder(nn.Module):
    def __init__(self, n, dim, d_ff, dropout, heads):
        super(encoder, self).__init__()
        self.layers = clones(encoder_layer(dim, heads,
                                           self_attn(heads, dim, dropout).to('cuda:0'),
                                           PositionwiseFeedForward(dim, d_ff), dropout), n)

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x
class decoder(nn.Module):
    def __init__(self, n, dim, d_ff, dropout, heads):
        super(decoder, self).__init__()
        self.layers = clones(decoder_layer(dim, heads,
                                           tgt_attn(heads, dim, dropout).to('cuda:0'),
                                           self_attn(heads, dim, dropout).to('cuda:0'),
                                           PositionwiseFeedForward(dim, d_ff), dropout), n)
        self.tgt_out = tgt_out(heads, dim, dropout)
        self.final_norm = LayerNorm(dim)

    def forward(self, x, tgt):
        for layer in self.layers:
            x = layer(x, tgt)
        x = self.tgt_out(tgt, x, x)
        x = self.final_norm(x)
        return x

# encoder & decoder layers
class encoder_layer(nn.Module):
    def __init__(self, dim, heads, self_attn, feedforward, dropout):
        super(encoder_layer, self).__init__()
        self.res_layer = [residual_layer(dim, dropout, self_attn),
                          residual_layer(dim, dropout, feedforward)]
        self.dim = dim

    def forward(self, x, mask=None):
        x = self.res_layer[0](x, x, x)
        return self.res_layer[1](x)
class decoder_layer(nn.Module):
    def __init__(self, dim, heads, tgt_attn, self_attn, feedforward, dropout):
        super(decoder_layer, self).__init__()
        self.res_layer = [residual_layer(dim, dropout, tgt_attn),
                          residual_layer(dim, dropout, self_attn),
                          residual_layer(dim, dropout, feedforward)]

    def forward(self, x, tgt):
        x = self.res_layer[0](x, tgt, x)  # res_layer: v, q, k
        x = self.res_layer[1](x, x, x)
        return self.res_layer[2](x)

## end of encoder & decoder layers
## attentions:
class self_attn(nn.Module):
    def __init__(self, h, dim, dropout=0):
        super(self_attn, self).__init__()
        assert dim % h == 0
        # We assume d_v always equals d_k
        self.d_k = dim // h
        self.h = h
        self.linears = clones(nn.Linear(dim, dim), 3)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nwords = key.size(0)

        query, key, value = \
            [l(x).view(nwords, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        # qkv.size() = length,heads,1,dk

        query = query.squeeze(2).transpose(0, 1)  # heads, length, dk
        key = key.squeeze(2).transpose(0, 1).transpose(1, 2)  # heads, dk, length
        value = value.squeeze(2).transpose(0, 1)  # heads, length, dk

        scores = torch.matmul(query, key)  # heads, length, length
        p_attn = F.softmax(scores, dim=2)  # heads, length, length
        if self.dropout is not None:
            p_attn = self.dropout(p_attn)

        x = torch.matmul(p_attn, value)  # heads, length, dk
        x = x.transpose(0, 1).contiguous().view([nwords, self.h * self.d_k])
        # x=x.transpose(0,1).view([nwords,self.h * self.d_k])
        self.attn = p_attn

        return self.linears[-1](x).unsqueeze(1)
class tgt_out(nn.Module):
    def __init__(self, h, dim, dropout=0):
        super(tgt_out, self).__init__()
        assert dim % h == 0
        # We assume d_v always equals d_k
        self.d_k = dim // h
        self.h = h
        self.tgt_linear = nn.Linear(10, dim)
        self.linears = clones(nn.Linear(dim, dim), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):  # q=tgt, k=self, v=self
        nwords = key.size(0)
        query = self.tgt_linear(query)  # from gnn_dim to dim
        query = self.linears[0](query).view(-1, self.h, self.d_k).transpose(0, 1)  # heads, 1, dk
        key = self.linears[1](key).view(nwords, -1, self.h, self.d_k).transpose(1, 2)  # length, heads, 1, dk
        value = self.linears[2](value).view(nwords, -1, self.h, self.d_k).transpose(1, 2)  # length, heads, 1, dk

        key = key.squeeze(2).transpose(0, 1).transpose(1, 2)  # heads, dk, length
        scores = torch.matmul(query, key)
        p_attn = F.softmax(scores, dim=2)  # heads,1,length

        if self.dropout is not None:
            p_attn = self.dropout(p_attn)
        value = value.squeeze(2).transpose(0, 1)  # heads,length,dk

        x = torch.matmul(p_attn, value)
        x = x.transpose(0, 1).contiguous().view([1, self.h * self.d_k])
        self.attn = p_attn

        return self.linears[-1](x)
class tgt_attn(nn.Module):
    def __init__(self, h, dim, dropout=0):
        super(tgt_attn, self).__init__()
        assert dim % h == 0
        # We assume d_v always equals d_k
        self.d_k = dim // h
        self.h = h
        self.tgt_linear = nn.Linear(160, dim)
        self.linears = clones(nn.Linear(dim, dim), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):  # q=tgt, k=self, v=self
        nwords = key.size(0)
        query = self.tgt_linear(query)  # from gnn_dim to dim
        query = self.linears[0](query).view(-1, self.h, self.d_k).transpose(0, 1)  # heads, 1, dk
        key = self.linears[1](key).view(nwords, -1, self.h, self.d_k).transpose(1, 2)  # length, heads, 1, dk
        value = self.linears[2](value).view(nwords, -1, self.h, self.d_k).transpose(1, 2)  # length, heads, 1, dk

        key = key.squeeze(2).transpose(0, 1).transpose(1, 2)  # heads, dk, length
        scores = torch.matmul(query, key)
        p_attn = F.softmax(scores, dim=2).transpose(1, 2)  # heads,length,1

        if self.dropout is not None:
            p_attn = self.dropout(p_attn)

        value = value.squeeze(2).transpose(0, 1)  # heads,length,dk

        x = p_attn * value  # heads,length,dk
        x = x.transpose(0, 1).contiguous().view([nwords, self.h * self.d_k])
        self.attn = p_attn  # length, dim

        return self.linears[-1](x)
    ## end of attentions
class residual_layer(nn.Module):
    def __init__(self, size, dropout, sublayer):
        super(residual_layer, self).__init__()
        self.norm = LayerNorm(size).to('cuda:0')
        self.dropout = nn.Dropout(dropout)
        self.sublayer = sublayer

    def forward(self, x, q=None, k=None):  # q and k are None if sublayer is ff, x is v
        if (q != None and k != None):
            return self.norm(x + self.dropout(self.sublayer(q, k, x).squeeze(1)))
        else:
            return self.norm(x + self.dropout(self.sublayer(x).squeeze(1)))
class LayerNorm(nn.Module):
    def __init__(self, size, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(size))
        self.b_2 = nn.Parameter(torch.zeros(size))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        # return norm+bias
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff).to('cuda:0')
        self.w_2 = nn.Linear(d_ff, d_model).to('cuda:0')
        self.dropout = nn.Dropout(dropout).to('cuda:0')

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
# end
class DecoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, self_attention, positionwise_feedforward, dropout, device):
        super().__init__()

        self.ln = nn.LayerNorm(hid_dim)
        self.sa = self_attention(hid_dim, n_heads, dropout, device)
        self.ea = self_attention(hid_dim, n_heads, dropout, device)
        self.pf = positionwise_feedforward(hid_dim, pf_dim, dropout)
        self.do = nn.Dropout(dropout)

    def forward(self, trg, src, trg_mask=None, src_mask=None):
        # trg = [batch_size, compound len, atom_dim]
        # src = [batch_size, protein len, hid_dim] # encoder output
        # trg_mask = [batch size, compound sent len]
        # src_mask = [batch size, protein len]

        trg = self.ln(trg + self.do(self.sa(trg, trg, trg, trg_mask)))

        trg = self.ln(trg + self.do(self.ea(trg, src, src, src_mask)))

        trg = self.ln(trg + self.do(self.pf(trg)))

        return trg
class SelfAttention(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_heads = n_heads

        assert hid_dim % n_heads == 0

        self.w_q = nn.Linear(hid_dim, hid_dim)
        self.w_k = nn.Linear(hid_dim, hid_dim)
        self.w_v = nn.Linear(hid_dim, hid_dim)

        self.fc = nn.Linear(hid_dim, hid_dim)

        self.do = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads])).to(device)

    def forward(self, query, key, value, mask=None):
        bsz = query.shape[0]

        # query = key = value [batch size, sent len, hid dim]

        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)

        # Q, K, V = [batch size, sent len, hid dim]

        Q = Q.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        K = K.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        V = V.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)

        # K, V = [batch size, n heads, sent len_K, hid dim // n heads]
        # Q = [batch size, n heads, sent len_q, hid dim // n heads]
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        # energy = [batch size, n heads, sent len_Q, sent len_K]
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = self.do(F.softmax(energy, dim=-1))

        # attention = [batch size, n heads, sent len_Q, sent len_K]

        x = torch.matmul(attention, V)

        # x = [batch size, n heads, sent len_Q, hid dim // n heads]

        x = x.permute(0, 2, 1, 3).contiguous()

        # x = [batch size, sent len_Q, n heads, hid dim // n heads]

        x = x.view(bsz, -1, self.n_heads * (self.hid_dim // self.n_heads))

        # x = [batch size, src sent len_Q, hid dim]

        x = self.fc(x)

        # x = [batch size, sent len_Q, hid dim]

        return x
class PositionwiseFeedforward(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.pf_dim = pf_dim

        self.fc_1 = nn.Conv1d(hid_dim, pf_dim, 1)  # convolution neural units
        self.fc_2 = nn.Conv1d(pf_dim, hid_dim, 1)  # convolution neural units

        self.do = nn.Dropout(dropout)

    def forward(self, x):
        # x = [batch size, sent len, hid dim]

        x = x.permute(0, 2, 1)

        # x = [batch size, hid dim, sent len]

        x = self.do(F.relu(self.fc_1(x)))

        # x = [batch size, pf dim, sent len]

        x = self.fc_2(x)

        # x = [batch size, hid dim, sent len]

        x = x.permute(0, 2, 1)

        # x = [batch size, sent len, hid dim]

        return x

# PRO_BERT
class PRO_BERT(nn.Module):
    def __init__(self):
        super(PRO_BERT, self).__init__()

        global max_len, n_layers, n_head, d_model, d_ff, d_k, d_v, vocab_size, device, c_in
        max_len = 16000
        n_layers = 3
        n_head = 8
        d_model = 32
        d_ff = 32
        d_k = 4
        d_v = 4
        c_in = 32
        vocab_size = n_word
        device = 'cuda:0'
        self.embedding = Embedding()
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])
        self.fc_task = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(d_model // 2, 2),
        )
        self.classifier = nn.Linear(2, 2)
        self.downConv = ConvLayer(c_in)

    def forward(self, input_ids,enc_self_attn_mask):
        # input_ids[batch_size, seq_len] like[8,1975]
        output = self.embedding(input_ids)  # [batch_size, seq_len, d_model]
        enc_self_attn_mask = get_attn_pad_mask(input_ids)  # [batch_size, maxlen, maxlen]
        hidden_layers=[]
        n = 0
        for layer in self.layers:
            output = layer(output, enc_self_attn_mask)
            n=n+1
            if n<3:
                output = self.downConv(output)
        #     hidden_layers.append(output)
        # fusion_layers=torch.cat([hidden_layers[0],hidden_layers[1],hidden_layers[2]],dim=1)
        # return fusion_layers
        return output

def get_attn_pad_mask(seq):
    batch_size, seq_len = seq.size()
    pad_attn_mask = seq.data.eq(0).unsqueeze(1)  # [batch_size, 1, seq_len]
    pad_attn_mask_expand = pad_attn_mask.expand(batch_size, seq_len, seq_len)  # [batch_size, seq_len, seq_len]
    return pad_attn_mask_expand
class Embedding(nn.Module):
    def __init__(self):
        super(Embedding, self).__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)  # token embedding (look-up table)
        self.pos_embed = nn.Embedding(max_len, d_model)  # position embedding
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        seq_len = x.size(1)  # x: [batch_size, seq_len]
        pos = torch.arange(seq_len, device='cuda:0', dtype=torch.long)  # [seq_len]
        pos = pos.unsqueeze(0).expand_as(x)  # [seq_len] -> [batch_size, seq_len]
        embedding = self.pos_embed(pos)
        embedding = embedding + self.tok_embed(x)
        embedding = self.norm(embedding)
        return embedding
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)  # scores : [batch_size, n_head, seq_len, seq_len]
        scores.masked_fill_(attn_mask, -1e9)  # Fills elements of self tensor with value where mask is one.
        attn = nn.Softmax(dim=-1)(scores)  # [batch_size, n_head, seq_len, seq_len]
        context = torch.matmul(attn, V)  # [batch_size, n_head, seq_len, d_v]
        return context, attn
class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_head)
        self.W_K = nn.Linear(d_model, d_k * n_head)
        self.W_V = nn.Linear(d_model, d_v * n_head)

        self.linear = nn.Linear(n_head * d_v, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, Q, K, V, attn_mask):
        residual, batch_size = Q, Q.size(0)
        q_s = self.W_Q(Q).view(batch_size, -1, n_head, d_k).transpose(1, 2)  # q_s: [batch_size, n_head, seq_len, d_k]
        k_s = self.W_K(K).view(batch_size, -1, n_head, d_k).transpose(1, 2)  # k_s: [batch_size, n_head, seq_len, d_k]
        v_s = self.W_V(V).view(batch_size, -1, n_head, d_v).transpose(1, 2)  # v_s: [batch_size, n_head, seq_len, d_v]
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_head, 1, 1)
        context, attention_map = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1,
                                                            n_head * d_v)  # context: [batch_size, seq_len, n_head * d_v]
        output = self.linear(context)
        output = self.norm(output + residual)
        return output, attention_map

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_ff) -> (batch_size, seq_len, d_model)
        return self.fc2(self.relu(self.fc1(x)))
class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()
        self.attention_map = None

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attention_map = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs,
                                                        enc_self_attn_mask)  # enc_inputs to same Q,K,V
        self.attention_map = attention_map
        enc_outputs = self.pos_ffn(enc_outputs)  # enc_outputs: [batch_size, seq_len, d_model]
        return enc_outputs

"""AE2"""
class AENet(nn.Module):
    def __init__(self, inputDim, hiddenDim, prelr, totlr):
        super(AENet, self).__init__()

        self.enfc = nn.Linear(inputDim, hiddenDim)
        self.defc = nn.Linear(hiddenDim, inputDim)

    def encoder(self, x):
        return torch.sigmoid(self.enfc(x))

    def decoder(self, zHalf):
        return torch.sigmoid(self.defc(zHalf))

    def totolTrainOnce(self, trainDataList, g, lamda):
        g = torch.autograd.Variable(g, requires_grad=False)
        trainLoader = DataLoader(
            dataset=TensorDataset(trainDataList, g),
            batch_size=1,
            shuffle=True
        )
        for x, g in trainLoader:
            x = x.float()
            zHalf = self.encoder(x)
            z = self.decoder(zHalf)
        return z
class DGNet(nn.Module):
    def __init__(self, targetDim, hiddenDim, lr=0.001):
        super(DGNet, self).__init__()

        self.dgfc = nn.Linear(targetDim, hiddenDim)

    def degradation(self, h):
        return torch.sigmoid(self.dgfc(h))

    def totalTrainDgOnce(self, hList, zHalfList, lamda):
        hList = torch.autograd.Variable(hList, requires_grad=False)
        zHalfList = torch.autograd.Variable(zHalfList, requires_grad=False)
        trainLoader = DataLoader(
            dataset=TensorDataset(hList, zHalfList),
            batch_size=1,
            shuffle=True
        )
        for h, zHalf in trainLoader:
            g = self.degradation(h)
        return g
class Autoencoder(nn.Module):
    def __init__(self, dimList, targetDim, hiddenDim=100, preTrainLr=0.001,
                 aeTotleTrainLr=0.001, dgTotleTrainLr=0.001, lamda=1.0, HTrainLr=0.1):
        super(Autoencoder, self).__init__()
        dimList=[32]
        self.viewNum = 0
        self.nSample = 1
        self.lamda = lamda
        self.HTrainLr = HTrainLr
        self.aeNetList = [AENet(d, hiddenDim, preTrainLr, aeTotleTrainLr).cuda() for d in dimList]
        self.dgNetList = [DGNet(targetDim, hiddenDim, dgTotleTrainLr).cuda() for d in dimList]
        self.H = nn.Parameter(torch.FloatTensor(np.random.uniform(0, 1, [self.nSample, targetDim])))

        self.input = []
        self.output = []

    def forward(self, trainDataList, nSample=1):
        # totleTrain
        self.nSample = nSample
        self.viewNum = len(trainDataList)  # 1
        # 1.Update aenets
        g = [dgnet.degradation(self.H) for dgnet in self.dgNetList]
        for v in range(self.viewNum):
            self.aeNetList[v].totolTrainOnce(trainDataList[v], g[v], self.lamda)

        # 2.Update dgnets&AE2
        for v in range(self.viewNum):
            zHalfList = self.aeNetList[v].encoder(trainDataList[v].float())
            # 2.1 Update denets
            self.dgNetList[v].totalTrainDgOnce(self.H, zHalfList, self.lamda)

            # 2.2 Update AE2
            tmpZHalfList = torch.autograd.Variable(zHalfList, requires_grad=False)
            trainLoader = DataLoader(
                dataset=TensorDataset(self.H, tmpZHalfList),
                batch_size=100,
                shuffle=True
            )
            for h, zHalf in trainLoader:
                self.input = zHalf
                self.output = self.dgNetList[v].degradation(h)

        return self.H, self.input, self.output

    def getH(self):
        return self.H
class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=2,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1,2)
        return x
