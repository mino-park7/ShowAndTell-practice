# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Helper functions for image preprocessing."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf


def distort_image(image, thread_id):
    """
    Perform random distortions on an image.

    :param image: A float32 Tensor of shape [height, width, 3] with values in [0, 1).
    :param thread_id: Preprocessing thread id used to select the ordering of color
        distortions. There should be a multiple of 2 preprocessing threads.

    :return:
        distorted_image: A float32 Tensor of shape [height, width, 3] with values in
        [0,1].
    """

    #Randomly flip horizontally.
    with tf.name_scope("flip_horizontal", values=[image]):
        image = tf.image.random_flip_left_right(image)

    #Randomly distort the color based on thread id.
    color_ordering = thread_id % 2
    with tf.name_scope("distort_color", values=[image]):
        if color_ordering==0 :


