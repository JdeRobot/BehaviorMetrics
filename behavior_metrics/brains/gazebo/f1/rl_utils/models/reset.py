import numpy as np

from brains.f1.rl_utils.models.f1_env import F1Env


class Reset(F1Env):
    """
    Works for Follow Line and Follow Lane tasks
    """

    def reset_f1_state_image(self):
        """
        reset for
        - State: Image
        - tasks: FollowLane and FollowLine
        """
        self._gazebo_reset()
        # === POSE ===
        if self.alternate_pose:
            self._gazebo_set_random_pose_f1_follow_rigth_lane()
        else:
            self._gazebo_set_fix_pose_f1_follow_right_lane()

        self._gazebo_unpause()

        ##==== get image from sensor camera
        f1_image_camera, _ = self.f1gazeboimages.get_camera_info()
        self._gazebo_pause()

        ##==== calculating State
        # image as observation
        state = np.array(
            self.f1gazeboimages.image_preprocessing_black_white_32x32(
                f1_image_camera.data, self.height
            )
        )
        state_size = state.shape

        return state, state_size

    def reset_f1_state_sp(self):
        """
        reset for
        - State: Simplified perception
        - tasks: FollowLane and FollowLine
        """
        # === POSE ===
        # if self.alternate_pose:
        #     self._gazebo_set_random_pose_f1_follow_rigth_lane()
        # else:
        # self._gazebo_set_fix_pose_f1_follow_right_lane()
        self._gazebo_reset()

        self._gazebo_unpause()

        ##==== get image from sensor camera
        f1_image_camera, _ = self.f1gazeboimages.get_camera_info()
        self._gazebo_pause()

        ##==== calculating State
        # simplified perception as observation
        centrals_in_pixels, _ = self.simplifiedperception.calculate_centrals_lane(
            f1_image_camera.data,
            self.height,
            self.width,
            self.x_row,
            self.lower_limit,
            self.center_image,
        )
        states = self.simplifiedperception.calculate_observation(
            centrals_in_pixels, self.center_image, self.pixel_region
        )
        state = [states[0]]

        return state
