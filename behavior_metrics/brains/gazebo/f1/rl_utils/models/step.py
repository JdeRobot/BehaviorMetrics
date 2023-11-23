from geometry_msgs.msg import Twist
import numpy as np
from brains.gazebo.f1.rl_utils.models.f1_env import F1Env


class StepFollowLine(F1Env):
    def __init__(self, **config):
        self.name = config["states"]

    def step_followline_state_image_actions_discretes(self, action, step):
        self._gazebo_unpause()
        vel_cmd = Twist()
        vel_cmd.linear.x = self.actions[action][0]
        vel_cmd.angular.z = self.actions[action][1]
        self.vel_pub.publish(vel_cmd)

        ##==== get image from sensor camera
        f1_image_camera, _ = self.f1gazeboimages.get_camera_info()
        self._gazebo_pause()

        ##==== get center
        points_in_red_line, _ = self.simplifiedperception.processed_image(
            f1_image_camera.data, self.height, self.width, self.x_row, self.center_image
        )
        if self.state_space == "spn":
            self.point = points_in_red_line[self.poi]
        else:
            self.point = points_in_red_line[0]

        center = abs(float(self.center_image - self.point) / (float(self.width) // 2))

        ##==== get State
        ##==== image as observation
        state = np.array(
            self.f1gazeboimages.image_preprocessing_black_white_32x32(
                f1_image_camera.data, self.height
            )
        )

        ##==== get Rewards
        if self.reward_function == "followline_center":
            reward, done = self.f1gazeborewards.rewards_followline_center(
                center, self.rewards
            )

        return state, reward, done, {}

    def step_followline_state_sp_actions_discretes(self, action, step):
        vel_cmd = Twist()
        vel_cmd.linear.x = self.actions[action][0]
        vel_cmd.angular.z = self.actions[action][1]
        self.vel_pub.publish(vel_cmd)

        ##==== get image from sensor camera
        f1_image_camera, _ = self.f1gazeboimages.get_camera_info()

        ##==== get center
        points_in_red_line, centrals_normalized = self.simplifiedperception.processed_image(
            f1_image_camera.data, self.height, self.width, self.x_row, self.center_image
        )
        if self.state_space == "spn":
            self.point = points_in_red_line[self.poi]
        else:
            self.point = points_in_red_line[0]

        center = abs(float(self.center_image - self.point) / (float(self.width) // 2))
        # center = float(self.center_image - self.point) / (float(self.width) // 2)

        ##==== get State
        ##==== simplified perception as observation
        # state = self.simplifiedperception.calculate_observation(
        #     points_in_red_line, self.center_image, self.pixel_region
        # )

        ##==== get Rewards
        # if self.reward_function == "followline_center":
        #     reward, done = self.f1gazeborewards.rewards_followline_center(
        #         center, self.rewards
        #     )

        return centrals_normalized, 0, False, {}

    def step_followline_state_image_actions_continuous(self, action, step):
        self._gazebo_unpause()
        vel_cmd = Twist()
        vel_cmd.linear.x = action[0][0]
        vel_cmd.angular.z = action[0][1]
        self.vel_pub.publish(vel_cmd)

        ##==== get image from sensor camera
        f1_image_camera, _ = self.f1gazeboimages.get_camera_info()
        self._gazebo_pause()

        ##==== get center
        points_in_red_line, _ = self.simplifiedperception.processed_image(
            f1_image_camera.data, self.height, self.width, self.x_row, self.center_image
        )
        if self.state_space == "spn":
            self.point = points_in_red_line[self.poi]
        else:
            self.point = points_in_red_line[0]

        center = abs(float(self.center_image - self.point) / (float(self.width) // 2))

        ##==== get State
        state = np.array(
            self.f1gazeboimages.image_preprocessing_black_white_32x32(
                f1_image_camera.data, self.height
            )
        )

        ##==== get Rewards
        if self.reward_function == "followline_center":
            reward, done = self.f1gazeborewards.rewards_followline_center(
                center, self.rewards
            )
        else:
            reward, done = self.f1gazeborewards.rewards_followline_v_w_centerline(
                vel_cmd, center, self.rewards, self.beta_1, self.beta_0
            )

        return state, reward, done, {}

    def step_followline_state_sp_actions_continuous(self, action, step):
        vel_cmd = Twist()
        vel_cmd.linear.x = action[0]
        vel_cmd.angular.z = action[1]

        self.vel_pub.publish(vel_cmd)
        ##==== get image from sensor camera
        f1_image_camera, _ = self.f1gazeboimages.get_camera_info()

        self.previous_image = f1_image_camera
        while np.array_equal(self.previous_image, f1_image_camera.data):
            f1_image_camera, _ = self.f1gazeboimages.get_camera_info()

        ##==== get center
        points_in_red_line, centrals_normalized = self.simplifiedperception.processed_image(
            f1_image_camera.data, self.height, self.width, self.x_row, self.center_image
        )
        if self.state_space == "spn":
            self.point = points_in_red_line[self.poi]
        else:
            self.point = points_in_red_line[0]

        ##==== get State
        ##==== simplified perception as observation
        # state = self.simplifiedperception.calculate_observation(
        #     points_in_red_line, self.center_image, self.pixel_region
        # )

        return centrals_normalized, 0, False, None


class StepFollowLane(F1Env):
    def __init__(self, **config):
        self.name = config["states"]

    def step_followlane_state_sp_actions_discretes(self, action, step):
        self._gazebo_unpause()
        vel_cmd = Twist()
        vel_cmd.linear.x = self.actions[action][0]
        vel_cmd.angular.z = self.actions[action][1]
        self.vel_pub.publish(vel_cmd)

        ##==== get image from sensor camera
        f1_image_camera, _ = self.f1gazeboimages.get_camera_info()
        self._gazebo_pause()

        ##==== get center
        centrals_in_lane, centrals_in_lane_normalized = self.simplifiedperception.processed_image(
            f1_image_camera.data, self.height, self.width, self.x_row, self.center_image
        )
        if self.state_space == "spn":
            self.point = centrals_in_lane[self.poi]
        else:
            self.point = centrals_in_lane[0]

        ##==== get State
        ##==== simplified perception as observation
        state = self.simplifiedperception.calculate_observation(
            centrals_in_lane, self.center_image, self.pixel_region
        )

        return state, 0, False, {}

    def step_followlane_state_image_actions_discretes(self, action, step):
        self._gazebo_unpause()
        vel_cmd = Twist()
        vel_cmd.linear.x = self.actions[action][0]
        vel_cmd.angular.z = self.actions[action][1]
        self.vel_pub.publish(vel_cmd)

        ##==== get image from sensor camera
        f1_image_camera, _ = self.f1gazeboimages.get_camera_info()
        self._gazebo_pause()

        ##==== get center
        points_in_red_line, _ = self.simplifiedperception.processed_image(
            f1_image_camera.data, self.height, self.width, self.x_row, self.center_image
        )
        if self.state_space == "spn":
            self.point = points_in_red_line[self.poi]
        else:
            self.point = points_in_red_line[0]

        # center = abs(float(self.center_image - self.point) / (float(self.width) // 2))
        center = float(self.center_image - self.point) / (float(self.width) // 2)

        ##==== get State
        ##==== image as observation
        state = np.array(
            self.f1gazeboimages.image_preprocessing_black_white_32x32(
                f1_image_camera.data, self.height
            )
        )

        ##==== get Rewards
        if self.reward_function == "follow_right_lane_center_v_step":
            reward, done = self.f1gazeborewards.rewards_followlane_v_centerline_step(
                vel_cmd, center, step, self.rewards
            )
        else:
            reward, done = self.f1gazeborewards.rewards_followlane_centerline(
                center, self.rewards
            )

        return state, reward, done, {}

    def step_followlane_state_image_actions_continuous(self, action, step):
        self._gazebo_unpause()
        vel_cmd = Twist()
        vel_cmd.linear.x = action[0][0]
        vel_cmd.angular.z = action[0][1]
        self.vel_pub.publish(vel_cmd)

        ##==== get image from sensor camera
        f1_image_camera, _ = self.f1gazeboimages.get_camera_info()
        self._gazebo_pause()

        ##==== get center
        points_in_red_line, _ = self.simplifiedperception.processed_image(
            f1_image_camera.data, self.height, self.width, self.x_row, self.center_image
        )
        if self.state_space == "spn":
            self.point = points_in_red_line[self.poi]
        else:
            self.point = points_in_red_line[0]

        # center = abs(float(self.center_image - self.point) / (float(self.width) // 2))
        center = float(self.center_image - self.point) / (float(self.width) // 2)

        ##==== get State
        state = np.array(
            self.f1gazeboimages.image_preprocessing_black_white_32x32(
                f1_image_camera.data, self.height
            )
        )

        ##==== get Rewards
        if self.reward_function == "follow_right_lane_center_v_step":
            reward, done = self.f1gazeborewards.rewards_followlane_v_centerline_step(
                vel_cmd, center, step, self.rewards
            )
        else:
            reward, done = self.f1gazeborewards.rewards_followlane_centerline(
                center, self.rewards
            )

        return state, reward, done, {}

    def step_followlane_state_sp_actions_continuous(self, action, step):
        self._gazebo_unpause()
        vel_cmd = Twist()
        vel_cmd.linear.x = action[0][0]
        vel_cmd.angular.z = action[0][1]
        self.vel_pub.publish(vel_cmd)

        ##==== get image from sensor camera
        f1_image_camera, _ = self.f1gazeboimages.get_camera_info()
        self._gazebo_pause()

        ##==== get center
        points_in_red_line, _ = self.simplifiedperception.processed_image(
            f1_image_camera.data, self.height, self.width, self.x_row, self.center_image
        )
        if self.state_space == "spn":
            self.point = points_in_red_line[self.poi]
        else:
            self.point = points_in_red_line[0]

        # center = abs(float(self.center_image - self.point) / (float(self.width) // 2))
        center = float(self.center_image - self.point) / (float(self.width) // 2)

        ##==== get State
        ##==== simplified perception as observation
        state = self.simplifiedperception.calculate_observation(
            points_in_red_line, self.center_image, self.pixel_region
        )

        ##==== get Rewards
        if self.reward_function == "follow_right_lane_center_v_step":
            reward, done = self.f1gazeborewards.rewards_followlane_v_centerline_step(
                vel_cmd, center, step, self.rewards
            )
        else:
            reward, done = self.f1gazeborewards.rewards_followlane_centerline(
                center, self.rewards
            )

        return state, reward, done, {}
