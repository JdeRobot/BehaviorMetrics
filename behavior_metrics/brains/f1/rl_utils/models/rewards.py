import math

import numpy as np


class F1GazeboRewards:
    @staticmethod
    def rewards_followlane_centerline(center, rewards):
        """
        works perfectly
        rewards in function of center of Line
        """
        done = False
        if 0.65 >= center > 0.25:
            reward = rewards["from_10"]
        elif (0.9 > center > 0.65) or (0.25 >= center > 0):
            reward = rewards["from_02"]
        elif 0 >= center > -0.9:
            reward = rewards["from_01"]
        else:
            reward = rewards["penal"]
            done = True

        return reward, done

    def rewards_followlane_v_centerline_step(self, vel_cmd, center, step, rewards):
        """
        rewards in function of velocity, angular v and center
        """

        done = False
        if 0.65 >= center > 0.25:
            reward = (rewards["from_10"] + vel_cmd.linear.x) - math.log(step)
        elif (0.9 > center > 0.65) or (0.25 >= center > 0):
            reward = (rewards["from_02"] + vel_cmd.linear.x) - math.log(step)
        elif 0 >= center > -0.9:
            # reward = (self.rewards["from_01"] + vel_cmd.linear.x) - math.log(step)
            reward = -math.log(step)
        else:
            reward = rewards["penal"]
            done = True

        return reward, done

    def rewards_followlane_v_w_centerline(
        self, vel_cmd, center, rewards, beta_1, beta_0
    ):
        """
        v and w are linear dependents, plus center to the eq.
        """

        w_target = beta_0 - (beta_1 * abs(vel_cmd.linear.x))
        w_error = abs(w_target - abs(vel_cmd.angular.z))
        done = False

        if abs(center) > 0.9 or center < 0:
            done = True
            reward = rewards["penal"]
        elif center >= 0:
            reward = (
                (1 / math.exp(w_error)) + (1 / math.exp(center)) + 2
            )  # add a constant to favor right lane
            # else:
            #    reward = (1 / math.exp(w_error)) + (math.exp(center))

        return reward, done

    def calculate_reward(self, error: float) -> float:
        d = np.true_divide(error, self.center_image)
        reward = np.round(np.exp(-d), 4)
        return reward

    def rewards_followline_center(self, center, rewards):
        """
        original for Following Line
        """
        done = False
        if center > 0.9:
            done = True
            reward = rewards["penal"]
        elif 0 <= center <= 0.2:
            reward = rewards["from_10"]
        elif 0.2 < center <= 0.4:
            reward = rewards["from_02"]
        else:
            reward = rewards["from_01"]

        return reward, done

    def rewards_followline_v_w_centerline(
        self, vel_cmd, center, rewards, beta_1, beta_0
    ):
        """
        Applies a linear regression between v and w
        Supposing there is a lineal relationship V and W. So, formula w = B_0 + x*v.

        Data for Formula1:
        Max W = 5 r/s we take max abs value. Correctly it is w left or right
        Max V = 100 m/s
        Min V = 20 m/s
        B_0 = B_1 * Max V
        B_1 = (W Max / (V Max - V Min))

        w target = B_0 - B_1 * v
        error = w_actual - w_target
        reward = 1/exp(reward + center))) where Max value = 1

        Args:
                linear and angular velocity
                center

        Returns: reward
        """

        # print_messages(
        #    "in reward_v_w_center_linear()",
        #    beta1=self.beta_1,
        #    beta0=self.beta_0,
        # )

        w_target = beta_0 - (beta_1 * abs(vel_cmd.linear.x))
        w_error = abs(w_target - abs(vel_cmd.angular.z))
        done = False

        if abs(center) > 0.9:
            done = True
            reward = rewards["penal"]
        elif center > 0:
            reward = (
                (1 / math.exp(w_error)) + (1 / math.exp(center)) + 2
            )  # add a constant to favor right lane
        else:
            reward = (1 / math.exp(w_error)) + (math.exp(center))

        return reward, done
