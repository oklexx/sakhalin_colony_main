from __future__ import annotations

import math
import os
import torch
import torch._dynamo  # noqa: F401  (must be module-level: a function-local
                      # `import torch._dynamo` would make `torch` local to
                      # __init__ and break every earlier `torch.*` reference)
import torch.nn as nn
from torch import distributions as D
from typing import Optional, Dict

from rl._nn_common import load_policy_state
from rl.actor_critic import ActorCritic
from rl.rollout_buffer import RolloutBuffer


class PPO:
    """PPO with GPU-resident buffer, AMP, optional torch.compile."""

    def __init__(
        self,
        model: ActorCritic,
        buffer: RolloutBuffer,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        n_epochs: int = 4,
        batch_size: int = 64,
        use_amp: bool = True,
        amp_dtype: str = "bfloat16",
        torch_compile: bool = False,
        device: Optional[torch.device] = None,
        lr_warmup_steps: int = 0,
        lr_decay: bool = True,
        total_training_steps: int = 0,
        target_kl: float = 0.0,
        obs_version: int = 1,
    ):
        self.model = model
        self.obs_version = obs_version
        self.buffer = buffer
        self.lr = lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.use_amp = use_amp
        self.device = device or model.device

        self.amp_dtype = torch.bfloat16 if amp_dtype == "bfloat16" else torch.float16
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=(amp_dtype == "float16"))

        if use_amp:
            if self.device.type == "cuda":
                if amp_dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
                    print(f"[PPO WARNING] bfloat16 requested but GPU "
                          f"({torch.cuda.get_device_name()}) does not support it. "
                          f"AMP autocast may fall back to float32.")
            else:
                print(f"[PPO WARNING] AMP enabled but device={self.device.type}. "
                      f"AMP only accelerates on CUDA GPUs.")

        from rl.actor_critic_hybrid import ActorCriticHybrid
        self.is_hybrid = isinstance(model, ActorCriticHybrid)

        self.optimizer = torch.optim.Adam(
            model.params, lr=lr, eps=1e-5, weight_decay=0.0
        )

        # Keep the un-compiled module for save/load and compile fallback.
        self._raw_model = model
        self._compiled = False
        if torch_compile:
            if self.device.type == "cuda" and torch.cuda.is_available():
                try:
                    # "reduce-overhead" (CUDA graphs) is counter-productive here:
                    # batch shapes vary (last partial mini-batch) and the module
                    # flips eval()/train() every rollout, which forces graph
                    # recaptures — usually SLOWER than eager on this tiny MLP.
                    # "default" is the safe choice.
                    # If the backend fails at call time (e.g. no triton on
                    # Windows), fall back to eager per-graph instead of crashing
                    # the whole training run.
                    torch._dynamo.config.suppress_errors = True
                    self.model = torch.compile(model, mode="default")
                    self._compiled = True
                    print("[PPO] torch.compile enabled (mode=default). "
                          "Note: first update is slow (compilation); for a "
                          "256x256 MLP the steady-state gain is small — the "
                          "bottleneck is env stepping, not GPU math.")
                except Exception as e:
                    self.model = model
                    print(f"[PPO] torch.compile failed: {e}. Falling back to eager.")
            else:
                print("[PPO] torch.compile requires CUDA. Disabling. "
                      "(On Windows torch.compile is not supported at all — "
                      "no triton backend; it is silently skipped.)")

        self.lr_warmup_steps = lr_warmup_steps
        self.lr_decay = lr_decay
        self._current_step = 0
        self._total_training_steps = total_training_steps
        self.target_kl = target_kl

        if lr_warmup_steps > 0 or lr_decay:
            from torch.optim.lr_scheduler import LambdaLR
            def lr_lambda(step):
                if step < lr_warmup_steps:
                    return float(step) / max(1, lr_warmup_steps)
                if lr_decay and total_training_steps > 0:
                    progress = float(step - lr_warmup_steps) / max(
                        1, total_training_steps - lr_warmup_steps)
                    progress = min(progress, 1.0)
                    return 0.1 + 0.5 * (1.0 + math.cos(math.pi * progress))
                return 1.0
            self.scheduler = LambdaLR(self.optimizer, lr_lambda)
        else:
            self.scheduler = None

    def _forward(self, *args):
        """Forward through the (possibly compiled) model with eager fallback.

        torch.compile on Windows has no triton backend: the wrapper is created
        fine but the FIRST forward raises BackendCompilerFailed. Without this
        guard the whole run crashes ~n_steps into the first rollout.
        """
        try:
            return self.model(*args)
        except Exception as e:
            if self._compiled:
                print(f"[PPO] compiled forward failed ({type(e).__name__}: {e}). "
                      f"Falling back to eager permanently.")
                self.model = self._raw_model
                self._compiled = False
                return self.model(*args)
            raise

    def collect_step(
        self, flat, minimap=None, action_masks=None
    ) -> Dict[str, torch.Tensor]:
        """Get action, log_prob, value for current obs (no grad).

        In hybrid mode call as collect_step(flat, minimap); otherwise pass a
        single obs tensor as `flat`.
        `action_masks`: [n_envs, n_actions] float tensor, 1.0=available, 0.0=blocked.

        NOTE: deliberately runs in float32 (no autocast). Rollout inference is
        one [n_envs, 209] forward per step — autocast cast kernels cost more
        than they save at this size, and bf16 log_probs/values would inject
        quantization noise into GAE and the PPO importance ratio. AMP is kept
        for update(), where big mini-batches actually benefit.
        """
        self.model.eval()
        with torch.no_grad():
            if self.is_hybrid:
                logits, values = self._forward(flat, minimap, action_masks)
            else:
                logits, values = self._forward(flat, action_masks)
            if action_masks is not None:
                # Mask unavailable actions: set logits to a large negative value
                # (-1e9 instead of -inf to avoid NaN when all actions are blocked)
                logits = logits.float().masked_fill(action_masks == 0, -1e9)
            else:
                logits = logits.float()
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            values = values.float().squeeze(-1)
        # Clamp actions to valid range (prevent CUDA assert from NaN logits)
        n_actions = self._raw_model.n_actions
        action = action.clamp(0, n_actions - 1)
        return {
            "action": action,
            "log_prob": log_prob,
            "value": values,
        }

    def update(
        self, last_value: torch.Tensor, last_done: torch.Tensor
    ) -> Dict[str, float]:
        """Run PPO update after buffer is full.

        last_value: [n_envs] value of terminal state
        last_done:  [n_envs] whether terminal
        """
        self.model.train()
        self.buffer.compute_gae(last_value, last_done)

        # Accumulate stats on-GPU; calling .item() per batch forces a
        # device sync (~3 per batch here) and measurably slows the update.
        sum_policy = torch.zeros((), device=self.device)
        sum_value = torch.zeros((), device=self.device)
        sum_entropy = torch.zeros((), device=self.device)
        sum_kl = torch.zeros((), device=self.device)
        n_batches = 0
        early_stop = False

        for _ in range(self.n_epochs):
            if early_stop:
                break
            for batch in self.buffer.get_batches(self.batch_size):
                obs = batch["obs"]
                actions = batch["actions"]
                old_log_probs = batch["old_log_probs"]
                advantages = batch["advantages"]
                returns = batch["returns"]
                old_values = batch["values"]
                flat = batch.get("flat", None) if self.is_hybrid else None
                action_masks = batch.get("action_masks", None)

                self.optimizer.zero_grad()

                if self.use_amp and self.device.type == "cuda":
                    with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype):
                        policy_loss, value_loss, entropy, approx_kl = self._compute_loss_components(
                            obs, actions, old_log_probs, advantages, returns, old_values,
                            flat=flat, action_masks=action_masks
                        )
                        loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy
                else:
                    policy_loss, value_loss, entropy, approx_kl = self._compute_loss_components(
                        obs, actions, old_log_probs, advantages, returns, old_values,
                        flat=flat, action_masks=action_masks
                    )
                    loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy

                if not torch.isfinite(loss):
                    self.optimizer.zero_grad()
                    continue

                if self.use_amp and self.amp_dtype == torch.float16 and self.device.type == "cuda":
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    if self.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self._raw_model.params, self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    if self.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self._raw_model.params, self.max_grad_norm)
                    self.optimizer.step()

                if self.scheduler is not None:
                    self.scheduler.step()
                    self._current_step += 1

                with torch.no_grad():
                    sum_policy += policy_loss.detach()
                    sum_value += value_loss.detach()
                    sum_entropy += entropy.detach()
                    sum_kl += approx_kl.detach()
                n_batches += 1

                # Early stopping: skip remaining epochs if KL diverges too much.
                # This is the one per-batch sync we keep (it must react now).
                if self.target_kl > 0 and float(approx_kl) > 1.5 * self.target_kl:
                    early_stop = True
                    break

        self.buffer.reset()

        nb = max(n_batches, 1)
        return {
            "policy_loss": float(sum_policy) / nb,
            "value_loss": float(sum_value) / nb,
            "entropy": float(sum_entropy) / nb,
            "approx_kl": float(sum_kl) / nb,
            "learning_rate": self.optimizer.param_groups[0]["lr"],
        }

    def _compute_loss_components(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        old_values: torch.Tensor,
        flat: Optional[torch.Tensor] = None,
        action_masks: Optional[torch.Tensor] = None,
    ):
        if self.is_hybrid:
            logits, values = self._forward(flat, obs, action_masks)
        else:
            logits, values = self._forward(obs, action_masks)
        # Apply action masks: set blocked actions to a large negative value.
        # Always upcast to float32: log_probs/entropy/ratio must not be
        # computed in bf16 (quantization noise directly biases the ratio).
        if action_masks is not None:
            logits = logits.float().masked_fill(action_masks == 0, -1e9)
        else:
            logits = logits.float()
        dist = D.Categorical(logits=logits)
        new_log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        values = values.squeeze(-1)
        # Huber loss (Smooth L1): защищает градиенты политики (актора) от подавления
        # при clip_grad_norm_, когда критик совершает крупные ошибки на размахах
        # отдачи (-25 000 .. +5 000). При ошибках > 10 лосс становится линейным.
        value_loss = nn.functional.smooth_l1_loss(values, returns, beta=10.0)

        with torch.no_grad():
            approx_kl = (old_log_probs - new_log_probs).mean().abs()

        return policy_loss, value_loss, entropy, approx_kl

    def save(self, path: str):
        # Always save from the un-compiled module: the OptimizedModule's
        # state_dict prefixes keys with "_orig_mod.", which older checkpoints
        # and the evaluator's architecture inference do not expect.
        #
        # Write beside the destination and replace it only after torch.save has
        # completed.  The GUI can start watch while training is still running;
        # a direct torch.save(path) exposes a partially-written ZIP checkpoint
        # and the watcher then exits immediately after "Loading policy ...".
        # os.replace is atomic on the same filesystem (including Windows).
        model = self._raw_model
        clean_state = {}
        for k, v in model.state_dict().items():
            ck = k.replace("_orig_mod.", "") if k.startswith("_orig_mod.") else k
            clean_state[ck] = v
        if hasattr(model, "hidden_sizes"):
            hidden_sizes = list(model.hidden_sizes)
        elif hasattr(model, "flat_trunk") and hasattr(model, "joint"):
            hidden_sizes = (
                [m.out_features for m in model.flat_trunk if isinstance(m, nn.Linear)] +
                [m.out_features for m in model.joint if isinstance(m, nn.Linear)]
            )
        elif hasattr(model, "trunk"):
            hidden_sizes = [m.out_features for m in model.trunk if isinstance(m, nn.Linear)]
        elif hasattr(model, "joint"):
            hidden_sizes = [m.out_features for m in model.joint if isinstance(m, nn.Linear)]
        elif hasattr(model, "flat_trunk"):
            hidden_sizes = [m.out_features for m in model.flat_trunk if isinstance(m, nn.Linear)]
        else:
            hidden_sizes = []
        extra = {}
        if self.is_hybrid:
            extra["n_channels"] = model.n_channels
            extra["grid_size"] = model.grid_size
            extra["obs_size"] = model.obs_size
        elif hasattr(model, "n_channels"):
            extra["n_channels"] = model.n_channels
            extra["grid_size"] = model.grid_size
        elif hasattr(model, "obs_size"):
            extra["obs_size"] = model.obs_size
        payload = {
            "model_state": clean_state,
            "optimizer_state": self.optimizer.state_dict(),
            "buffer_pos": self.buffer.pos,
            "n_actions": model.n_actions,
            "hidden_sizes": hidden_sizes,
            "obs_version": int(getattr(self, "obs_version", 1)),
            **extra,
        }
        destination = os.fspath(path)
        temporary = f"{destination}.tmp-{os.getpid()}"
        try:
            torch.save(payload, temporary)
            os.replace(temporary, destination)
        finally:
            # If serialization failed, do not leave a file that the model
            # scanner could mistake for a usable checkpoint.
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        # Newer models have a zero-initialized value-only mask projection.
        # Missing keys are expected for legacy checkpoints; output heads and
        # observation/action dimensions remain unchanged.
        load_policy_state(self._raw_model, ckpt["model_state"], ckpt_path=str(path))
        try:
            self.optimizer.load_state_dict(ckpt["optimizer_state"])
        except (KeyError, ValueError, RuntimeError):
            # Legacy checkpoints predate critic_mask_proj; weights remain valid
            # and the optimizer is intentionally left freshly initialized.
            pass
