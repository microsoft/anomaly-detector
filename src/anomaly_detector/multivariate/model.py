# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import os
import time
from dataclasses import fields
from typing import Any, Dict, List, Optional

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.utils.data as torch_data
from anomaly_detector.base import BaseAnomalyDetector
from anomaly_detector.common.constants import FillNAMethod
from anomaly_detector.common.data_processor import MultiADDataProcessor
from anomaly_detector.common.exception import DataFormatError
from anomaly_detector.multivariate.contract import (
    MultiADConfig,
    MultiADConstants,
    NormalizeBase,
)
from anomaly_detector.multivariate.dataset import MultiADDataset
from anomaly_detector.multivariate.module import (
    AnomalyDetectionCriterion,
    DetectionResults,
    MultivariateGraphAttnDetector,
)
from anomaly_detector.multivariate.util import (
    AverageMeter,
    append_result,
    compute_severity,
    get_multiple_variables_pct_weight_score,
    get_threshold,
)
from torch import nn
from tqdm import tqdm


class MultivariateAnomalyDetector(BaseAnomalyDetector):
    def __init__(self):
        super(MultivariateAnomalyDetector, self).__init__()
        self.config: Optional[MultiADConfig] = None
        self.model: Optional[MultivariateGraphAttnDetector] = None
        self.pct_weight: Optional[List] = None
        self.threshold: Optional[float] = None
        self.train_score_max: Optional[float] = None
        self.train_score_min: Optional[float] = None
        self.model_path: Optional[str] = None
        self.variables: Optional[List[str]] = None

    def fit(self, data: pd.DataFrame, params: Dict[str, Any] = None) -> None:
        variables, values, config = self._process_data_and_params(data, params)
        self.config = config
        self.variables = variables
        _window = self.config.threshold_window + self.config.input_size
        self.pct_weight = get_multiple_variables_pct_weight_score(data, _window)
        self.config.data_dim = values.shape[1]
        self.model_path = os.path.join(self.config.save, "best_multi_ad_model.pt")
        train_length = int(len(data) * self.config.train_ratio)
        valid_length = int(len(data) * self.config.val_ratio)
        train_data = values[:train_length]
        val_data = values[train_length: valid_length + train_length]
        max_values = np.max(train_data, axis=0)
        min_values = np.min(train_data, axis=0)
        self.config.normalize_base = NormalizeBase(
            max_values=max_values.tolist(), min_values=min_values.tolist()
        )

        torch.manual_seed(self.config.seed)
        train_set = MultiADDataset(
            train_data,
            window_size=self.config.input_size,
            interval=self.config.interval,
            horizon=self.config.horizon,
            max_values=max_values,
            min_values=min_values,
            clip_min=MultiADConstants.TRAIN_CLIP_MIN,
            clip_max=MultiADConstants.TRAIN_CLIP_MAX,
        )
        val_set = MultiADDataset(
            val_data,
            window_size=self.config.input_size,
            interval=1,
            horizon=1,
            max_values=max_values,
            min_values=min_values,
            clip_min=MultiADConstants.INFERENCE_CLIP_MIN,
            clip_max=MultiADConstants.INFERENCE_CLIP_MAX,
        )
        train_loader = torch_data.DataLoader(
            train_set,
            batch_size=self.config.batch_size,
            drop_last=True,
            shuffle=True,
            num_workers=0,
        )
        val_loader = torch_data.DataLoader(
            val_set, batch_size=self.config.batch_size, shuffle=False, num_workers=0
        )
        os.makedirs(self.config.save, exist_ok=True)
        model = MultivariateGraphAttnDetector(self.config)
        # print("init", list(model.parameters())[0])
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.lr)
        criterion = AnomalyDetectionCriterion().to(self.config.device)

        best_val = np.Inf
        for i in range(self.config.epochs):
            start = time.time()
            train_loss, train_vae_loss, train_pred_loss = self.train_epoch(
                model, optimizer, train_loader, criterion
            )
            valid_loss, valid_vae_loss, valid_pred_loss = self.evaluate_epoch(
                model, criterion, val_loader
            )
            end = time.time()
            if mlflow.active_run() is not None:
                mlflow.log_metric("train_loss", train_loss, step=i + 1)
                mlflow.log_metric("train_vae_loss", train_vae_loss, step=i + 1)
                mlflow.log_metric("train_pred_loss", train_pred_loss, step=i + 1)
                mlflow.log_metric("valid_loss", valid_loss, step=i + 1)
                mlflow.log_metric("valid_vae_loss", valid_vae_loss, step=i + 1)
                mlflow.log_metric("valid_pred_loss", valid_pred_loss, step=i + 1)
                mlflow.log_metric("epoch_time", end - start, step=i + 1)

            if valid_loss < best_val:
                best_val = valid_loss
                self.model = model
                self.save_checkpoint()
                # print("best_ckpt", list(model.parameters())[0])

        self.load_checkpoint(self.model_path)
        (
            self.threshold,
            self.train_score_max,
            self.train_score_min,
        ) = self.compute_thresholds(train_data)

    def save_checkpoint(self):
        self.model.cpu()
        ckpt = {
            "state_dict": self.model.state_dict(),
            "config": self.config,
            "pct_weight": self.pct_weight,
            "threshold": self.threshold,
            "train_score_max": self.train_score_max,
            "train_score_min": self.train_score_min,
        }
        torch.save(ckpt, self.model_path)
        self.model.to(self.config.device)

    def load_checkpoint(self, model_path):
        ckpt = torch.load(model_path)
        self.config = ckpt["config"]
        self.model = MultivariateGraphAttnDetector(self.config)
        self.model.load_state_dict(ckpt["state_dict"])
        self.pct_weight = ckpt["pct_weight"]
        self.threshold = ckpt["threshold"]
        self.train_score_max = ckpt["train_score_max"]
        self.train_score_min = ckpt["train_score_min"]

    def train_epoch(self, model, optimizer, dataloader, criterion):
        config = self.config
        loss_meter = AverageMeter()
        pred_loss_meter = AverageMeter()
        vae_loss_meter = AverageMeter()
        model.train()
        for _, (inputs, target) in enumerate(dataloader):
            inputs = inputs.to(self.config.device)
            target = target.to(self.config.device)
            inputs = inputs + torch.normal(
                mean=torch.zeros_like(inputs), std=torch.ones_like(inputs) * 0.1
            )
            x_pred, x_recon, latent_mu, latent_logv = DetectionResults(
                *model(inputs)
            ).get_loss_items()
            loss, pred_loss, vae_loss = criterion(
                x_pred, x_recon, target, latent_mu, latent_logv, config.beta
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_meter.update(loss.detach().item(), len(inputs), is_sum=True)
            pred_loss_meter.update(pred_loss.detach().item(), len(inputs), is_sum=True)
            vae_loss_meter.update(vae_loss.detach().item(), len(inputs), is_sum=True)
        return loss_meter.avg, vae_loss_meter.avg, pred_loss_meter.avg

    def evaluate_epoch(self, model, criterion, dataloader):
        config = self.config
        loss_meter = AverageMeter()
        pred_loss_meter = AverageMeter()
        vae_loss_meter = AverageMeter()
        model.eval()

        with torch.no_grad():
            for _, (inputs, target) in enumerate(dataloader):
                inputs = inputs.to(config.device)
                target = target.to(config.device)
                x_pred, x_recon, latent_mu, latent_logv = DetectionResults(
                    *model(inputs)
                ).get_loss_items()
                loss, pred_loss, vae_loss = criterion(
                    x_pred, x_recon, target, latent_mu, latent_logv, config.beta
                )
                loss_meter.update(loss.detach().item(), len(inputs), is_sum=True)
                pred_loss_meter.update(
                    pred_loss.detach().item(), len(inputs), is_sum=True
                )
                vae_loss_meter.update(
                    vae_loss.detach().item(), len(inputs), is_sum=True
                )
        return loss_meter.avg, vae_loss_meter.avg, pred_loss_meter.avg

    def predict(
            self, context, data: pd.DataFrame, params: Optional[Dict[str, Any]] = None
    ):
        variables, values, _ = self._process_data_and_params(data, params)

        if self.model is None:
            try:
                self.load_checkpoint(self.model_path)
            except Exception as ex:
                raise ValueError(f"Cannot load model. Please train model. {repr(ex)}")
        if len(values) < self.config.threshold_window + self.config.input_size:
            raise ValueError(
                f"Not enough data. Minimum size is {self.config.threshold_window + self.config.input_size}"
            )
        hard_th_upper = max(MultiADConstants.ANOMALY_UPPER_THRESHOLD, self.threshold)
        hard_th_lower = min(MultiADConstants.ANOMALY_LOWER_THRESHOLD, self.threshold)
        torch.manual_seed(self.config.seed)
        self.model.to(self.config.device)
        (
            contributor_rmses,
            total_rmses,
            _,
            _,
            _,
            attn_feats,
        ) = self.inference(values, True)
        contributor_scores = np.array(contributor_rmses)
        contributor_scores = contributor_scores / (
            contributor_scores.sum(-1)[:, np.newaxis]
        )
        inference_scores = total_rmses
        result_n = len(inference_scores)
        thresholds = [
            get_threshold(
                inference_scores[
                    max(0, result_n - self.config.threshold_window - i): len(inference_scores) - i
                ]
            )
            for i in range(result_n - 1, self.config.threshold_window - 2, -1)
        ]
        inference_scores = inference_scores[self.config.threshold_window - 1:]
        contributor_scores = contributor_scores[self.config.threshold_window - 1:]
        is_anomalies = np.array(
            [
                (s >= t and s >= hard_th_lower) or s > hard_th_upper
                for s, t in zip(inference_scores, thresholds)
            ]
        )
        severity = compute_severity(inference_scores)
        severity[~is_anomalies] = 0.0
        if attn_feats is not None:
            inference_length = len(is_anomalies)
            attn_feats = torch.from_numpy(attn_feats)
            previous_attn = attn_feats.unfold(0, self.config.threshold_window, 1).mean(
                -1
            )
            attn_feats = (
                    attn_feats[-inference_length:] - previous_attn[-inference_length:]
            )
        attn_feats = attn_feats.numpy()
        index_values = data.index.to_list()
        return self._pack_response(
            index_values,
            variables,
            is_anomalies,
            inference_scores,
            severity,
            contributor_scores,
            attn_feats,
        )

    def inference(self, values, compute_attn=True):
        max_values = self.config.normalize_base.max_values
        min_values = self.config.normalize_base.min_values
        self.model.eval()
        test_set = MultiADDataset(
            values,
            window_size=self.config.input_size,
            interval=1,
            horizon=1,
            max_values=max_values,
            min_values=min_values,
            clip_min=MultiADConstants.INFERENCE_CLIP_MIN,
            clip_max=MultiADConstants.INFERENCE_CLIP_MAX,
        )
        test_loader = torch_data.DataLoader(
            test_set, batch_size=self.config.batch_size, shuffle=False, num_workers=0
        )

        contributor_probs = None
        contributor_rmses = None
        total_rmses = None
        total_probs = None
        forecasts = None
        attn_feats = None
        pct_weight = np.array(self.pct_weight)
        with torch.no_grad():
            for _, (inputs, target) in enumerate(tqdm(test_loader)):
                inputs = inputs.to(self.config.device)
                target = target.to(self.config.device)
                detection_results = DetectionResults(*self.model(inputs))
                (
                    x_pred,
                    x_recon,
                    _,
                    _,
                ) = detection_results.get_loss_items()
                batch_contributor_rmses = (
                        torch.exp(
                            torch.min(
                                2 * torch.abs(x_pred - target), torch.ones_like(x_pred)
                            )
                        )
                        - 1
                )

                batch_contributor_probs = (
                        torch.exp(
                            torch.min(
                                2 * torch.abs(x_recon - target), torch.ones_like(x_pred)
                            )
                        )
                        - 1
                )
                batch_probs = torch.mean(batch_contributor_probs, dim=1)

                batch_contributor_rmses_weight = (
                        batch_contributor_rmses.cpu().numpy() * pct_weight
                )
                contributor_rmses = append_result(
                    contributor_rmses, batch_contributor_rmses_weight
                )
                total_rmses = append_result(
                    total_rmses, np.mean(batch_contributor_rmses_weight, axis=1)
                )
                contributor_probs = append_result(
                    contributor_probs, batch_contributor_probs.cpu().numpy()
                )
                total_probs = append_result(total_probs, batch_probs.cpu().numpy())
                if compute_attn:
                    if detection_results.get_attn_map() is not None:
                        attn_feats = append_result(
                            attn_feats, detection_results.get_attn_map().cpu().numpy()
                        )
                    forecasts = append_result(forecasts, x_pred.cpu().numpy())
        return (
            contributor_rmses,
            total_rmses,
            contributor_probs,
            total_probs,
            forecasts,
            attn_feats,
        )

    @staticmethod
    def compute_window_sizes(sliding_window: int) -> (int, int):
        if sliding_window:
            try:
                sliding_window = int(sliding_window)
            except Exception as e:
                raise e
            input_size = sliding_window // 2
            threshold_window = min(200, sliding_window - input_size)
        else:
            input_size = 200
            threshold_window = 200
        return input_size, threshold_window

    def compute_thresholds(self, data) -> (float, float, float):
        _, total_rmses, _, _, _, _ = self.inference(data, compute_attn=False)
        train_scores = total_rmses
        sorted_scores = np.sort(train_scores)
        threshold = sorted_scores[int(len(sorted_scores) * self.config.level)]
        train_score_min = np.min(train_scores)
        train_score_max = np.max(train_scores)
        return threshold, train_score_max, train_score_min

    def _process_data_and_params(self, data: pd.DataFrame, params: Dict[str, Any] = None):
        if params is None:
            params = {}

        # check data
        if not isinstance(data, pd.DataFrame):
            raise DataFormatError(f"data must be pandas.DataFrame not {type(data)}.")

        # check params
        fill_na_method = params.get("fill_na_method", FillNAMethod.Linear.name)
        fill_na_value = params.get("fill_na_value", 0.0)
        params["input_size"], params["threshold_window"] = self.compute_window_sizes(
            params.get("sliding_window", None)
        )

        valid_fields = [f.name for f in fields(MultiADConfig)]
        for key in list(params.keys()):
            if key not in valid_fields:
                params.pop(key)

        data_processor = MultiADDataProcessor(
            fill_na_method=fill_na_method,
            fill_na_value=fill_na_value,
        )
        data = data_processor.process(data)
        variables = data.columns.to_list()
        params["data_dim"] = len(variables)
        values = data.values
        config = MultiADConfig(**params)
        return variables, values, config

    @staticmethod
    def _pack_response(
            index_values,
            variables,
            is_anomalies,
            inference_scores,
            severity_scores,
            contributor_scores,
            attn_feats,
    ):
        contributor_scores = torch.from_numpy(contributor_scores)
        top_k_contributors_idx = torch.argsort(
            contributor_scores, dim=-1, descending=True
        )
        top_k_contributors = torch.gather(
            contributor_scores, dim=-1, index=top_k_contributors_idx
        )
        top_k_contributors = top_k_contributors / top_k_contributors.sum(
            dim=-1, keepdim=True
        )
        attn_feats = torch.from_numpy(attn_feats)
        top_k_attn_scores = torch.gather(
            attn_feats,
            dim=1,
            index=top_k_contributors_idx.unsqueeze(-1).repeat(
                1, 1, attn_feats.size(-1)
            ),
        )
        top_attn_scores_idx = torch.argsort(
            torch.abs(top_k_attn_scores), dim=-1, descending=True
        )[..., : MultiADConstants.TOP_ATTENTION_COUNT]
        top_attn_scores = torch.gather(
            top_k_attn_scores, dim=-1, index=top_attn_scores_idx
        )
        contributor_scores = contributor_scores.cpu().numpy()
        top_k_contributors = top_k_contributors.cpu().numpy()
        top_k_contributors_idx = top_k_contributors_idx.cpu().numpy()
        top_attn_scores = top_attn_scores.cpu().numpy()
        top_attn_scores_idx = top_attn_scores_idx.cpu().numpy()

        num_series_names = len(variables)
        num_index_values = len(index_values)
        num_results = len(is_anomalies)
        num_contributors = top_k_contributors.shape[1]
        num_attentions = top_attn_scores.shape[2]
        diff = num_index_values - num_results
        assert diff >= 0, "invalid length"
        results = []
        for i in range(diff, num_index_values):
            idx = i - diff
            is_anomaly = bool(is_anomalies[idx])
            score = float(inference_scores[idx])
            severity = float(severity_scores[idx])
            interpretation = []
            for j in range(num_contributors):
                changed_values = []
                changed_variables = []
                for k in range(num_attentions):
                    if abs(top_attn_scores[idx, j, k]) > min(
                            0.001, 1.0 / (1.25 * num_series_names)
                    ):
                        changed_values.append(float(top_attn_scores[idx, j, k]))
                        var_idx = int(top_attn_scores_idx[idx, j, k])
                        changed_variables.append(variables[var_idx])
                var_idx = top_k_contributors_idx[idx, j]
                interpretation.append(
                    {
                        "variable_name": str(variables[var_idx]),
                        "contribution_score": float(contributor_scores[idx, var_idx]),
                        "correlation_changes": {
                            "changed_variables": changed_variables,
                            "changed_values": changed_values,
                        },
                    }
                )
            results.append(
                {
                    "index": index_values[i],
                    "is_anomaly": is_anomaly,
                    "score": score,
                    "severity": severity,
                    "interpretation": interpretation
                }
            )
        return results
