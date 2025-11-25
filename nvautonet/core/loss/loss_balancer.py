"""
Adaptive Loss Balancing Algorithm
Based on Section 5 of the paper
"""

import torch
import numpy as np


class AdaptiveLossBalancer:
    """
    Adaptive loss balancing for multi-task learning.

    From paper Section 5 (Equation 14):
    At the start of each epoch:
    1. Compute total loss for each task: L_t = ∑_s L_{t,s}
    2. Compute weight: ŵ_t = c_t / L_t
    3. Normalize: w_t = ŵ_t / ∑_t ŵ_t

    where c_t is a configurable task prior/multiplier.

    Two-stage approach:
    1. First round: c_t = 1 for all tasks
    2. Second round: Tune c_t based on task importance and difficulty
    """

    def __init__(self, task_names, task_priors=None, update_frequency='epoch'):
        """
        Args:
            task_names: List of task names (e.g., ['detection', 'segmentation'])
            task_priors: Dict of task_name -> prior weight c_t
                        Default: 1.0 for all tasks
            update_frequency: 'epoch' or 'batch' - how often to update weights
        """
        self.task_names = task_names
        self.num_tasks = len(task_names)
        self.update_frequency = update_frequency

        # Task priors (c_t in paper)
        if task_priors is None:
            self.task_priors = {name: 1.0 for name in task_names}
        else:
            self.task_priors = task_priors

        # Initialize weights uniformly
        self.task_weights = {name: 1.0 / self.num_tasks for name in task_names}

        # Track losses for weight computation
        self.epoch_losses = {name: 0.0 for name in task_names}
        self.num_samples = 0

        # History for logging
        self.weight_history = []

    def reset_epoch_losses(self):
        """Reset loss accumulators at the start of each epoch"""
        self.epoch_losses = {name: 0.0 for name in self.task_names}
        self.num_samples = 0

    def accumulate_losses(self, task_losses):
        """
        Accumulate losses during training.

        Args:
            task_losses: Dict of task_name -> loss value (scalar)
        """
        for task_name, loss_value in task_losses.items():
            if task_name in self.epoch_losses:
                if isinstance(loss_value, torch.Tensor):
                    loss_value = loss_value.item()
                self.epoch_losses[task_name] += loss_value

        self.num_samples += 1

    def update_weights(self):
        """
        Update task weights based on accumulated losses.
        Call this at the end of each epoch.
        """
        # Compute weights: w_t = c_t / L_t
        raw_weights = {}

        for task_name in self.task_names:
            total_loss = self.epoch_losses[task_name]
            prior = self.task_priors[task_name]

            if total_loss > 0:
                raw_weights[task_name] = prior / total_loss
            else:
                # If no loss (shouldn't happen), keep previous weight
                raw_weights[task_name] = self.task_weights[task_name]

        # Normalize weights
        total_weight = sum(raw_weights.values())

        if total_weight > 0:
            self.task_weights = {
                name: w / total_weight
                for name, w in raw_weights.items()
            }
        else:
            # Fallback to uniform weights
            self.task_weights = {name: 1.0 / self.num_tasks for name in self.task_names}

        # Log weights
        self.weight_history.append(self.task_weights.copy())

        print(f"\n[Loss Balancer] Updated task weights:")
        for task_name, weight in self.task_weights.items():
            epoch_loss = self.epoch_losses[task_name]
            print(f"  {task_name}: weight={weight:.4f}, epoch_loss={epoch_loss:.4f}")

    def get_weighted_loss(self, task_losses):
        """
        Apply current weights to task losses.

        Args:
            task_losses: Dict of task_name -> loss tensor

        Returns:
            weighted_loss: Combined weighted loss (scalar tensor)
            loss_dict: Dict with individual weighted losses for logging
        """
        weighted_loss = 0.0
        loss_dict = {}

        for task_name, loss_value in task_losses.items():
            if task_name in self.task_weights:
                weight = self.task_weights[task_name]
                weighted = weight * loss_value

                weighted_loss += weighted
                loss_dict[f'{task_name}_weighted'] = weighted
                loss_dict[f'{task_name}_raw'] = loss_value
            else:
                # Unknown task, add without weighting
                weighted_loss += loss_value
                loss_dict[f'{task_name}_raw'] = loss_value

        loss_dict['total_weighted'] = weighted_loss

        return weighted_loss, loss_dict

    def state_dict(self):
        """Save state for checkpointing"""
        return {
            'task_weights': self.task_weights,
            'task_priors': self.task_priors,
            'epoch_losses': self.epoch_losses,
            'num_samples': self.num_samples,
            'weight_history': self.weight_history
        }

    def load_state_dict(self, state_dict):
        """Load state from checkpoint"""
        self.task_weights = state_dict['task_weights']
        self.task_priors = state_dict.get('task_priors', self.task_priors)
        self.epoch_losses = state_dict.get('epoch_losses', self.epoch_losses)
        self.num_samples = state_dict.get('num_samples', 0)
        self.weight_history = state_dict.get('weight_history', [])


class FixedLossBalancer:
    """
    Simple fixed-weight loss balancer for baseline comparison.
    """

    def __init__(self, task_names, task_weights=None):
        """
        Args:
            task_names: List of task names
            task_weights: Dict of task_name -> fixed weight
        """
        self.task_names = task_names

        if task_weights is None:
            # Uniform weights
            num_tasks = len(task_names)
            self.task_weights = {name: 1.0 / num_tasks for name in task_names}
        else:
            self.task_weights = task_weights

    def reset_epoch_losses(self):
        """No-op for fixed weights"""
        pass

    def accumulate_losses(self, task_losses):
        """No-op for fixed weights"""
        pass

    def update_weights(self):
        """No-op for fixed weights"""
        pass

    def get_weighted_loss(self, task_losses):
        """Apply fixed weights"""
        weighted_loss = 0.0
        loss_dict = {}

        for task_name, loss_value in task_losses.items():
            if task_name in self.task_weights:
                weight = self.task_weights[task_name]
                weighted = weight * loss_value

                weighted_loss += weighted
                loss_dict[f'{task_name}_weighted'] = weighted
                loss_dict[f'{task_name}_raw'] = loss_value

        loss_dict['total_weighted'] = weighted_loss

        return weighted_loss, loss_dict

    def state_dict(self):
        return {'task_weights': self.task_weights}

    def load_state_dict(self, state_dict):
        self.task_weights = state_dict['task_weights']
