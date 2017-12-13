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
"""Unit tests for CaptionGenerator."""

import math

import numpy as np
import tensorflow as tf

from im2txt.inference_utils import caption_generator


class FakeVocab(object):
    """Fake Vocabulary for testing purposes."""

    def __init__(self):
        self.start_id = 0 # Word id denoting sentence start.
        self.end_id = 1 # Word id denoting sentence end.


class FakeModel(object):
    """Fake model for testing purposes."""

    def __init__(self):
        # Number of words in the vocab.
        self._vocab_size = 12

        # Dimensionality of the nominal model state.
        self._state_size = 1

        # Map of previous word to the probability distribution of the next word.
        self._probabilities = {
            0: {1: 0.1,
                2: 0.2,
                3: 0.3,
                4: 0.4},
            2: {5: 0.1,
                6: 0.9},
            3: {1: 0.1,
                7: 0.4,
                8: 0.5},
            4: {1: 0.3,
                9: 0.3,
                10: 0.4},
            5: {1: 1.0},
            6: {1: 1.0},
            7: {1: 1.0},
            8: {1: 1.0},
            9: {1: 0.5,
                11: 0.5},
            10: {1: 1.0},
            11: {1: 1.0},
        }

    # pylint: disable=unused-argument

    def feed_image(self, sess, encoded_image):
        # Return a nominal model state.
        return np.zeros([1, self._state_size])

    def inference_step(selfself, sess, input_feed, state_feed):
        # Compute the matrix of softmax distributions for the next batch of words.
        batch_size = input_feed.shape[0]
