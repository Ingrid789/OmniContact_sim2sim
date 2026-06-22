import numpy as np

class OmniContactMetricsMixin:
    def _reset_episode_metrics(self):
        self._action_rate_reported = False
        self._prev_policy_action = None
        self._action_delta_abs_sum = 0.0
        self._action_delta_sq_sum = 0.0
        self._action_delta_max = 0.0
        self._action_delta_samples = 0
        self._torso_tracking_error_sum = 0.0
        self._torso_tracking_error_samples = 0
        self._wrist_tracking_error_sum = 0.0
        self._wrist_tracking_error_samples = 0
        self._ankle_tracking_error_sum = 0.0
        self._ankle_tracking_error_samples = 0
        self.last_episode_metrics = {}

    def _record_torso_tracking_metrics(self):
        torso_goal = getattr(self.policy_output, "torso_goal", None)
        if torso_goal is None or self.torso_body_id < 0:
            return
        torso_goal = np.asarray(torso_goal, dtype=np.float32).reshape(-1)
        torso_pos = np.asarray(self.d.xpos[self.torso_body_id], dtype=np.float32).reshape(-1)
        if torso_goal.shape[0] < 3 or torso_pos.shape[0] < 3:
            return
        torso_err = float(np.linalg.norm(torso_pos[:3] - torso_goal[:3]))
        self._torso_tracking_error_sum += torso_err
        self._torso_tracking_error_samples += 1

    def _record_limb_tracking_metrics(self):
        wrist_goal = np.asarray(getattr(self.policy_output, "wrist_goal", np.zeros(0, dtype=np.float32)), dtype=np.float32).reshape(-1)
        if wrist_goal.shape[0] >= 14 and self.left_palm_body_id >= 0 and self.right_palm_body_id >= 0:
            left_wrist_goal = wrist_goal[:3]
            right_wrist_goal = wrist_goal[7:10]
            left_wrist_pos = np.asarray(self.d.xpos[self.left_palm_body_id], dtype=np.float32).reshape(-1)
            right_wrist_pos = np.asarray(self.d.xpos[self.right_palm_body_id], dtype=np.float32).reshape(-1)
            wrist_err = 0.5 * (
                float(np.linalg.norm(left_wrist_pos[:3] - left_wrist_goal))
                + float(np.linalg.norm(right_wrist_pos[:3] - right_wrist_goal))
            )
            self._wrist_tracking_error_sum += wrist_err
            self._wrist_tracking_error_samples += 1

        l_ankle_goal = np.asarray(getattr(self.policy_output, "l_ankle_goal", np.zeros(0, dtype=np.float32)), dtype=np.float32).reshape(-1)
        r_ankle_goal = np.asarray(getattr(self.policy_output, "r_ankle_goal", np.zeros(0, dtype=np.float32)), dtype=np.float32).reshape(-1)
        if l_ankle_goal.shape[0] >= 3 and r_ankle_goal.shape[0] >= 3 and self.left_ankle_body_id >= 0 and self.right_ankle_body_id >= 0:
            left_ankle_pos = np.asarray(self.d.xpos[self.left_ankle_body_id], dtype=np.float32).reshape(-1)
            right_ankle_pos = np.asarray(self.d.xpos[self.right_ankle_body_id], dtype=np.float32).reshape(-1)
            ankle_err = 0.5 * (
                float(np.linalg.norm(left_ankle_pos[:3] - l_ankle_goal[:3]))
                + float(np.linalg.norm(right_ankle_pos[:3] - r_ankle_goal[:3]))
            )
            self._ankle_tracking_error_sum += ankle_err
            self._ankle_tracking_error_samples += 1

    def _collect_episode_metrics(self):
        control_dt = self.simulation_dt * self.control_decimation
        metrics = {
            "action_rate_mean": float("nan"),
            "action_rate_rms": float("nan"),
            "action_rate_peak": float("nan"),
            "torso_tracking_error_mean": float("nan"),
            "wrist_tracking_error_mean": float("nan"),
            "ankle_tracking_error_mean": float("nan"),
            "action_rate_samples": int(self._action_delta_samples),
            "torso_tracking_error_samples": int(self._torso_tracking_error_samples),
            "wrist_tracking_error_samples": int(self._wrist_tracking_error_samples),
            "ankle_tracking_error_samples": int(self._ankle_tracking_error_samples),
        }
        if self._action_delta_samples > 0:
            mean_abs_delta = self._action_delta_abs_sum / self._action_delta_samples
            rms_delta = np.sqrt(self._action_delta_sq_sum / self._action_delta_samples)
            metrics["action_rate_mean"] = float(mean_abs_delta / control_dt)
            metrics["action_rate_rms"] = float(rms_delta / control_dt)
            metrics["action_rate_peak"] = float(self._action_delta_max / control_dt)
        if self._torso_tracking_error_samples > 0:
            metrics["torso_tracking_error_mean"] = float(
                self._torso_tracking_error_sum / self._torso_tracking_error_samples
            )
        if self._wrist_tracking_error_samples > 0:
            metrics["wrist_tracking_error_mean"] = float(
                self._wrist_tracking_error_sum / self._wrist_tracking_error_samples
            )
        if self._ankle_tracking_error_samples > 0:
            metrics["ankle_tracking_error_mean"] = float(
                self._ankle_tracking_error_sum / self._ankle_tracking_error_samples
            )
        self.last_episode_metrics = metrics
        return metrics
