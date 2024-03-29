# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from anomaly_detector.multivariate.contract import MultiADConfig


class GAT(nn.Module):
    """
    Graph Attention Network
    """

    def __init__(self, num_feats: int):
        """
        :param num_feats: number of nodes in the graph
        """
        super(GAT, self).__init__()
        self.num_feats = num_feats
        self.weight_key = nn.Parameter(torch.zeros(size=(self.num_feats, 1)))
        self.weight_query = nn.Parameter(torch.zeros(size=(self.num_feats, 1)))
        nn.init.xavier_uniform_(self.weight_key, gain=1.414)
        nn.init.xavier_uniform_(self.weight_query, gain=1.414)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        :param x: input tensor shape (batch_size, num_nodes, num_feats)
        :return attn_feat: output tensor shape (batch_size, num_nodes, num_feats)
        :return attn_map: attention map shape (batch_size, num_nodes, num_nodes)
        """
        batch_size = x.size(0)
        num_nodes = x.size(1)
        num_feats = x.size(2)
        key = torch.matmul(x, self.weight_key)
        query = torch.matmul(x, self.weight_query)
        attn_input = key.repeat(1, 1, num_nodes).view(
            batch_size, num_nodes * num_nodes, 1
        ) + query.repeat(1, num_nodes, 1)
        attn_output = attn_input.squeeze(2).view(batch_size, num_nodes, num_nodes)
        attn_output = F.leaky_relu(attn_output, negative_slope=0.2)
        attn_map = F.softmax(attn_output, dim=2)
        attention = self.dropout(attn_map)
        attn_feat = torch.matmul(attention, x).permute(0, 2, 1)
        return attn_feat, attn_map


class VAE(nn.Module):
    """
    Variational Autoencoder
    """

    def __init__(
        self, input_dim: int, z_dim: int, enc_dim: int, dec_dim: int, output_dim: int
    ):
        """
        :param input_dim: input dimension
        :param z_dim: latent dimension
        :param enc_dim: encoder dimension
        :param dec_dim: decoder dimension
        :param output_dim: output dimension
        """
        super(VAE, self).__init__()
        self.Encoder = nn.Sequential(
            nn.Linear(input_dim, enc_dim),
            nn.BatchNorm1d(enc_dim),
            nn.ReLU(True),
            nn.Linear(enc_dim, enc_dim),
            nn.BatchNorm1d(enc_dim),
            nn.ReLU(True),
        )

        self.Decoder = nn.Sequential(
            nn.Linear(z_dim, dec_dim),
            nn.BatchNorm1d(dec_dim),
            nn.ReLU(True),
            nn.Linear(dec_dim, dec_dim),
            nn.BatchNorm1d(dec_dim),
            nn.ReLU(True),
        )

        self.linear_mu = nn.Linear(enc_dim, z_dim)
        self.linear_logv = nn.Linear(enc_dim, z_dim)
        self.linear_recon = nn.Linear(dec_dim, output_dim)
        self.register_buffer("eps", torch.randn(z_dim))  # for reproducibility

    def encode(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        :param x: input tensor, shape (batch_size, input_dim)
        :return mu: mean of the latent distribution, shape (batch_size, z_dim)
        :rtype mu: torch.Tensor
        :return logv: log variance of the latent distribution, shape (batch_size, z_dim)
        :rtype logv: torch.Tensor
        """
        hidden = self.Encoder(x)
        mu, logv = self.linear_mu(hidden), self.linear_logv(hidden)
        return mu, logv

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        :param z: latent variable, shape (batch_size, z_dim)
        :return: reconstructed input, shape (batch_size, input_dim)
        """
        hidden = self.Decoder(z)
        recon = self.linear_recon(hidden)
        return torch.sigmoid(recon)

    def reparameterize(self, mu: torch.Tensor, logv: torch.Tensor) -> torch.Tensor:
        """
        :param mu: mean of the latent distribution, shape (batch_size, z_dim)
        :param logv: log variance of the latent distribution, shape (batch_size, z_dim)
        :return: sampled latent variable, shape (batch_size, z_dim)
        """
        std = torch.exp(torch.mul(logv, 0.5))
        # for reproducibility, a fixed eps is used during inference
        eps = torch.randn_like(std) if self.training else self.eps
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        """
        :param x: input tensor, shape (batch_size, input_dim)
        :return recon_x: reconstructed input, shape (batch_size, input_dim)
        :return latent_mu: mean of the latent distribution, shape (batch_size, z_dim)
        :return latent_logv: log variance of the latent distribution, shape (batch_size, z_dim)
        :rtype latent_logv: torch.Tensor
        """
        latent_mu, latent_logv = self.encode(x)
        z = self.reparameterize(latent_mu, latent_logv)
        recon_x = self.decode(z)
        return recon_x, latent_mu, latent_logv


class MultivariateGraphAttnDetector(nn.Module):
    """
    Multivariate Graph Attention Network for Anomaly Detection
    """

    def __init__(self, config: MultiADConfig):
        super(MultivariateGraphAttnDetector, self).__init__()
        self.cnn_kernel_size = config.cnn_kernel_size
        self.data_dim = config.data_dim
        self.input_size = config.input_size - self.cnn_kernel_size + 1
        self.gat_window_size = config.gat_window_size
        self.hidden_size = config.hidden_size
        self.z_dim = config.z_dim
        self.enc_dim = config.enc_dim
        self.dec_dim = config.dec_dim
        self.feat_gat = GAT(self.gat_window_size)
        self.time_gat = GAT(self.data_dim)
        self.dropout = nn.Dropout(p=0.5)
        self.register_buffer(
            "moving_avg_weight",
            torch.div(torch.ones((1, 1, self.cnn_kernel_size)), self.cnn_kernel_size),
        )
        self.gru = nn.GRU(self.data_dim * 3, self.hidden_size, batch_first=False)
        self.spatial_pool = nn.AdaptiveMaxPool1d(self.gat_window_size)
        self.fc_layers = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(self.hidden_size),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.data_dim),
        )
        self.vae = VAE(
            self.hidden_size, self.z_dim, self.enc_dim, self.dec_dim, self.data_dim
        )
        self.device = config.device
        self.to(self.device)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(x.size(0) * x.size(1), 1, x.size(2))
        x = F.conv1d(x, weight=self.moving_avg_weight)
        x = x.squeeze(1).view(batch_size, -1, x.size(2)).permute(0, 2, 1).contiguous()
        # bz x Ts x dim -> bz x dim x Ts -> sampling -> bz x dim x 100 -> bz x 100 x dim
        x = self.spatial_pool(x.permute(0, 2, 1)).permute(0, 2, 1)
        x_time = x  # bz x Tw x Dim
        x_feat = x.permute(0, 2, 1)
        h_time, _ = self.time_gat(x_time)
        h_time = h_time.permute(0, 2, 1)
        h_feat, attn_feat = self.feat_gat(x_feat)
        x = torch.cat([x, h_feat, h_time], dim=2).permute(1, 0, 2)
        _, x = self.gru(x)
        x = x.permute(1, 0, 2).squeeze(1)
        x_pred = torch.sigmoid(self.fc_layers(x))
        x_recon, latent_mu, latent_logv = self.vae(x)
        return x_pred, x_recon, latent_mu, latent_logv, attn_feat


class AnomalyDetectionCriterion(nn.Module):
    @staticmethod
    def forward(pred, recon, target, mu, logv, beta=1.0):
        # prediction loss
        pred_loss = F.mse_loss(pred, target, reduction="sum")

        # recon loss
        recon_loss = F.mse_loss(recon, target, reduction="sum")

        # KL divergence
        kl_loss = -0.5 * torch.sum(1.0 + logv - mu.pow(2) - logv.exp())
        vae_loss = recon_loss + 1.0 * kl_loss
        loss = beta * vae_loss + pred_loss
        return loss, pred_loss, vae_loss


class DetectionResults:
    # compatability
    def __init__(self, x_pred, x_recon, latent_mu, latent_logv, attn_map=None) -> None:
        self.preds = x_pred
        self.recons = x_recon
        self.latent_mu = latent_mu
        self.latent_logv = latent_logv
        self.attn_map = attn_map

    def compute_detection_results(
        self, targets, pct_weight, hard_th_lower, hard_th_upper, threshold_window
    ):
        inference_length = len(targets) - threshold_window + 1
        contributor_rmses = (
            torch.exp(
                torch.min(
                    2.0 * torch.abs(self.preds - targets), torch.ones_like(self.preds)
                )
            )
            - 1.0
        )
        rmses = torch.mean(contributor_rmses * torch.Tensor(pct_weight), dim=1)
        stacked_rmses = rmses.unfold(0, threshold_window, 1)
        sorted_rmses, _ = stacked_rmses.sort(dim=1)
        pos = int(stacked_rmses.size(1) * 0.95)
        thresholds = sorted_rmses[:, pos][:inference_length]

        inference_score = rmses[-inference_length:]

        is_anomalies = torch.logical_or(
            torch.logical_and(
                torch.ge(inference_score, thresholds),
                torch.ge(inference_score, hard_th_lower),
            ),
            torch.gt(inference_score, hard_th_upper),
        )

        severity = inference_score / (np.e - 1)

        if self.attn_map is not None:
            previous_attn = self.attn_map.unfold(0, threshold_window, 1).mean(-1)
            attn_map = (
                self.attn_map[-inference_length:] - previous_attn[-inference_length:]
            )
        else:
            attn_map = None
        contributor_scores = contributor_rmses[-inference_length:]
        contributor_scores = contributor_scores / contributor_scores.sum(-1).unsqueeze(
            -1
        )
        forecast = self.preds[-inference_length:]

        # all tensors
        return (
            is_anomalies,
            inference_score,
            severity,
            forecast,
            contributor_scores,
            attn_map,
        )

    def get_loss_items(self):
        return self.preds, self.recons, self.latent_mu, self.latent_logv

    def get_attn_map(self):
        return self.attn_map
