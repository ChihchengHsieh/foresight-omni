import torch
import torch.nn.functional as F


class DeepHitCIndexEvaluator:
    """
    Evaluator for DeepHit-style survival outputs.

    Collects:
      - logits  : [B, T]
      - t_bin   : [B] (0..T-1)
      - event   : [B] (0/1)
      - time    : [B] float (obs_time_years if provided else t_bin)

    Computes:
      - c_index (Harrell-like; comparable pairs i had event and time_i < time_j)
      - AUROC per time bin k for "event by bin k" using score = CDF[:, k]
      - Accuracy per time bin k (threshold on CDF[:, k])
      - Counts per time_bin, and per (time_bin, event)
    """

    def __init__(
        self,
        *,
        include_baseline_event: bool = False,  # your current behavior: incident-only eval
        acc_threshold: float = 0.5,
        eps: float = 1e-8,
    ):
        self.include_baseline_event = bool(include_baseline_event)
        self.acc_threshold = float(acc_threshold)
        self.eps = float(eps)

        self.logits = []  # list of [B, T]
        self.t_bins = []  # list of [B]
        self.events = []  # list of [B]
        self.times = []   # list of [B] float (years or bins)

    def reset(self):
        self.logits.clear()
        self.t_bins.clear()
        self.events.clear()
        self.times.clear()

    @torch.no_grad()
    def update(
        self,
        logits: torch.Tensor,
        t_bin: torch.Tensor,
        event: torch.Tensor,
        *,
        obs_time_years: torch.Tensor | None = None,
    ):
        """
        Args:
            logits: [B, T]
            t_bin:  [B] or [B,1] (int)
            event:  [B] or [B,1] (0/1)
            obs_time_years: optional [B] or [B,1] continuous time since baseline (years)
        """
        if logits.dim() != 2:
            raise RuntimeError(f"logits must be [B,T], got {tuple(logits.shape)}")

        if t_bin.dim() > 1:
            t_bin = t_bin.view(-1)
        if event.dim() > 1:
            event = event.view(-1)

        t_bin = t_bin.long()
        event = event.float()

        B, T = logits.shape
        if t_bin.numel() != B or event.numel() != B:
            raise RuntimeError(
                f"Batch mismatch: logits B={B}, t_bin={t_bin.numel()}, event={event.numel()}"
            )

        # Valid: finite, in-range
        valid = torch.isfinite(t_bin.float()) & torch.isfinite(event)
        valid = valid & (t_bin >= 0) & (t_bin < T)

        # Incident-only evaluation: drop baseline prevalence events (event=1 at bin 0)
        if not self.include_baseline_event:
            valid = valid & ~((event > 0.5) & (t_bin == 0))

        if valid.sum() == 0:
            return

        logits_v = logits[valid]
        t_bin_v = t_bin[valid]
        event_v = event[valid]

        # time used for C-index comparisons
        if obs_time_years is not None:
            if obs_time_years.dim() > 1:
                obs_time_years = obs_time_years.view(-1)
            obs_time_years = obs_time_years.float()
            if obs_time_years.numel() != B:
                raise RuntimeError(
                    f"obs_time_years mismatch: logits B={B}, obs_time_years={obs_time_years.numel()}"
                )
            time_v = obs_time_years[valid]
        else:
            time_v = t_bin_v.float()

        self.logits.append(logits_v.detach().cpu())
        self.t_bins.append(t_bin_v.detach().cpu())
        self.events.append(event_v.detach().cpu())
        self.times.append(time_v.detach().cpu())

    @staticmethod
    def _auroc_binary(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        AUROC for binary labels {0,1} without sklearn.
        Returns NaN if undefined (all labels same).
        """
        scores = scores.float()
        labels = labels.float()

        pos = (labels > 0.5)
        neg = ~pos
        n_pos = int(pos.sum().item())
        n_neg = int(neg.sum().item())
        if n_pos == 0 or n_neg == 0:
            return scores.new_tensor(float("nan"))

        # argsort for ranks
        order = torch.argsort(scores)
        ranks = torch.empty_like(order, dtype=torch.float)
        ranks[order] = torch.arange(1, scores.numel() + 1, device=scores.device).float()

        # tie handling: average ranks for equal scores
        sorted_scores = scores[order]
        diffs = torch.diff(sorted_scores)
        tie_breaks = torch.nonzero(diffs != 0, as_tuple=False).view(-1)

        bounds = torch.cat(
            [
                torch.tensor([0], device=scores.device, dtype=torch.long),
                (tie_breaks + 1).to(torch.long),
                torch.tensor([scores.numel()], device=scores.device, dtype=torch.long),
            ],
            dim=0,
        )

        for i in range(bounds.numel() - 1):
            a = int(bounds[i].item())
            b = int(bounds[i + 1].item())
            if b - a <= 1:
                continue
            seg = order[a:b]
            avg = ranks[seg].mean()
            ranks[seg] = avg

        sum_ranks_pos = ranks[pos].sum()
        auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
        return auc

    @torch.no_grad()
    def compute(self):
        if not self.logits:
            return {
                "c_index": torch.tensor(float("nan")),
                "auroc_per_bin": {},
                "accuracy_per_bin": {},
                "time_bin_counts": {},
                "time_bin_event_counts": {},
            }

        logits = torch.cat(self.logits, dim=0)   # [N, T]
        t_bin = torch.cat(self.t_bins, dim=0)    # [N]
        event = torch.cat(self.events, dim=0)    # [N]
        time = torch.cat(self.times, dim=0)      # [N]

        N, T = logits.shape
        if N < 2:
            return {
                "c_index": torch.tensor(float("nan")),
                "auroc_per_bin": {},
                "accuracy_per_bin": {},
                "time_bin_counts": {0: int(N)},
                "time_bin_event_counts": {},
            }

        # ---- probabilities / cdf ----
        p = F.softmax(logits, dim=-1)     # [N, T]
        cdf = p.cumsum(dim=-1)            # [N, T]
        risk_final = cdf[:, -1]           # [N]

        # ---- C-index ----
        time_i = time.unsqueeze(1)        # [N,1]
        time_j = time.unsqueeze(0)        # [1,N]
        event_i = event.unsqueeze(1)      # [N,1]

        cmp_mask = (event_i > 0.5) & (time_i < time_j)
        if cmp_mask.any():
            diff = risk_final.unsqueeze(1) - risk_final.unsqueeze(0)
            concordant = (diff > 0).float()
            ties = (diff == 0).float()
            concordant_pairs = (concordant * cmp_mask).sum()
            tied_pairs = (ties * cmp_mask).sum()
            total_pairs = cmp_mask.sum()
            c_index = (concordant_pairs + 0.5 * tied_pairs) / (total_pairs + self.eps)
        else:
            c_index = torch.tensor(float("nan"))

        # ---- counts per time bin ----
        max_bin = int(t_bin.max().item()) if t_bin.numel() > 0 else -1
        bin_counts = torch.bincount(t_bin, minlength=max_bin + 1).cpu().tolist()
        time_bin_counts = {i: int(c) for i, c in enumerate(bin_counts)}

        time_bin_event_counts = {}
        for b in range(max_bin + 1):
            m = (t_bin == b)
            if not m.any():
                continue
            e1 = int((event[m] > 0.5).sum().item())
            e0 = int(m.sum().item()) - e1
            time_bin_event_counts[int(b)] = {0: e0, 1: e1}

        # ---- AUROC / Accuracy per bin (event by bin k) ----
        auroc_per_bin = {}
        accuracy_per_bin = {}

        for k in range(T):
            # y_true_k = 1 if event occurred by bin k
            y_true_k = ((event > 0.5) & (t_bin <= k)).float()
            y_score_k = cdf[:, k]

            auroc_k = self._auroc_binary(y_score_k, y_true_k)

            y_pred_k = (y_score_k >= self.acc_threshold).float()
            acc_k = (y_pred_k == y_true_k).float().mean()

            auroc_per_bin[int(k)] = auroc_k
            accuracy_per_bin[int(k)] = acc_k

        flat = {}

        # ---- global metric ----
        flat["c_index"] = c_index

        # ---- per-bin AUROC / Accuracy ----
        for k, v in auroc_per_bin.items():
            flat[f"{k}_bin_auroc"] = v

        for k, v in accuracy_per_bin.items():
            flat[f"{k}_bin_acc"] = v

        # ---- counts per bin ----
        for k, v in time_bin_counts.items():
            flat[f"{k}_bin_counts"] = torch.tensor(v)

        # ---- counts per bin × event ----
        for k, d in time_bin_event_counts.items():
            flat[f"{k}_bin_censored_counts"] = torch.tensor(d.get(0, 0))
            flat[f"{k}_bin_event_counts"] = torch.tensor(d.get(1, 0))

        return flat