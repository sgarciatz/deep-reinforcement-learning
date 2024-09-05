import unittest
from deep_reinforcement_learning.PrioritizedExperienceReplay import PrioritizedExperienceReplay
from deep_reinforcement_learning.Experience import Experience
import numpy as np
import torch

class test_PrioritizedExperienceReplay(unittest.TestCase):


    def test_simple_PrioritizedExperienceReplay_initialization(self):
        """Check that the buffer is initialized with the correct
        parameters.
        """
        buffer_max_size = 20480
        device = "cpu"
        epsilon = 1e-4
        alpha = 1.0
        per = PrioritizedExperienceReplay(
            buffer_max_size=buffer_max_size,
            device=device,
            epsilon=epsilon,
            alpha=alpha)
        self.assertEqual(buffer_max_size,
                         per.experience_buffer.maxlen)
        self.assertEqual(device,
                         per.device)
        self.assertAlmostEqual(epsilon,
                               per.epsilon.item(),
                               delta=1e-4)
        self.assertAlmostEqual(alpha,
                               per.alpha.item(),
                               delta=1e-4)

    def test_add_experience(self):
        """Check that experiences are added to the buffer.
        """
        buffer_max_size = 20480
        device = "cpu"
        epsilon = 1e-4
        alpha = 1.0
        per = PrioritizedExperienceReplay(
            buffer_max_size=buffer_max_size,
            device=device,
            epsilon=epsilon,
            alpha=alpha)

        state = np.array([0, 0, 0, 0, 1])
        action = 2
        reward = 1
        next_state = np.array([0, 0, 1, 0, 1])
        done = False
        priority = 2.0
        experience = Experience(state,
                                action,
                                reward,
                                next_state,
                                done,
                                priority)

        per.add_experience(experience)
        self.assertEqual(1, len(per.experience_buffer))

    def test_sample_experience(self):
        """Check that experiences that are sampled are consistent with
        what was stored in first place.
        """
        buffer_max_size = 20480
        device = "cpu"
        epsilon = 1e-4
        alpha = 1.0
        per = PrioritizedExperienceReplay(
            buffer_max_size=buffer_max_size,
            device=device,
            epsilon=epsilon,
            alpha=alpha)

        state = np.array([0, 0, 0, 0, 1])
        action = 2
        reward = 1
        next_state = np.array([0, 0, 1, 0, 1])
        done = False
        priority = 2.0
        experience = Experience(state,
                                action,
                                reward,
                                next_state,
                                done,
                                priority)

        per.add_experience(experience)

        sampled_experience = per.sample_experience(n_samples=1)
        self.assertEqual(sampled_experience["actions"][0], action)
        self.assertEqual(sampled_experience["rewards"][0], reward)


    def test_sample_experience_prob(self):
        """Test that experiences with higher priorities are more
        frequently sampled.
        """
        buffer_max_size = 20480
        device = "cpu"
        epsilon = 1e-4
        alpha = 1.0
        per = PrioritizedExperienceReplay(
            buffer_max_size=buffer_max_size,
            device=device,
            epsilon=epsilon,
            alpha=alpha)
        priority1 = 5.0
        priority2 = 2.0
        priority3 = 1.0
        experience1 = Experience(np.array([0, 0, 0, 0, 1]),
                                 2,
                                 1,
                                 np.array([0, 0, 1, 0, 1]),
                                 False,
                                 priority1)
        experience2 = Experience(np.array([0, 0, 0, 1, 1]),
                                 2,
                                 1,
                                 np.array([0, 0, 1, 1, 1]),
                                 False,
                                 priority2)
        experience3 = Experience(np.array([0, 1, 0, 1, 1]),
                                 2,
                                 1,
                                 np.array([0, 1, 1, 1, 1]),
                                 False,
                                 priority3)
        per.add_experience(experience1)
        per.add_experience(experience2)
        per.add_experience(experience3)
        n_trials = 100
        expe1_outcomes = 0
        expe2_outcomes = 0
        expe3_outcomes = 0

        for _ in range(n_trials):
            sampled_experience_state = per.sample_experience(
                n_samples=1)["states"][0]
            if (torch.all(torch.eq(sampled_experience_state, torch.tensor(experience1.state, dtype=torch.float)))): expe1_outcomes += 1
            elif (torch.all(torch.eq(sampled_experience_state, torch.tensor(experience2.state, dtype=torch.float)))): expe2_outcomes += 1
            elif (torch.all(torch.eq(sampled_experience_state, torch.tensor(experience3.state, dtype=torch.float)))): expe3_outcomes += 1
        self.assertGreater(expe1_outcomes, expe2_outcomes)
        self.assertGreater(expe2_outcomes, expe3_outcomes)


    def test_update_batch_priorities(self):
        """Test that experiences with higher TD error end up with higher
        priorities.
        """
        buffer_max_size = 20480
        device = "cpu"
        epsilon = 1e-4
        alpha = 1.0
        per = PrioritizedExperienceReplay(
            buffer_max_size=buffer_max_size,
            device=device,
            epsilon=epsilon,
            alpha=alpha)
        priority1 = 5.0
        priority2 = 2.0
        priority3 = 1.0
        priority4 = 100.0
        experience1 = Experience("Experience 1",
                                 2,
                                 1,
                                 np.array([0, 0, 1, 0, 1]),
                                 False,
                                 priority1)
        experience2 = Experience("Experience 2",
                                 2,
                                 1,
                                 np.array([0, 0, 1, 1, 1]),
                                 False,
                                 priority2)
        experience3 = Experience("Experience 3",
                                 2,
                                 1,
                                 np.array([0, 1, 1, 1, 1]),
                                 False,
                                 priority3)
        experience4 = Experience("Experience 4",
                                 2,
                                 1,
                                 np.array([0, 1, 1, 1, 1]),
                                 False,
                                 priority4)
        per.add_experience(experience1)
        per.add_experience(experience2)
        per.add_experience(experience3)
        per.add_experience(experience4)
        td_error1 = 2.0
        td_error2 = 1.5
        td_error3 = 5.0
        td_errors = torch.tensor(np.array([td_error1, td_error2, td_error3]),
                                 dtype=torch.float32)

        batch = [experience1, experience2, experience3]
        per.update_batch_priorities(batch=batch, td_error=td_errors)
        sampled_experiences = [per.experience_buffer[0],
                               per.experience_buffer[1],
                               per.experience_buffer[2],
                               per.experience_buffer[3]]
        self.assertEqual(sampled_experiences[3].priority, priority4)
        self.assertGreater(sampled_experiences[2].priority,
                           per.experience_buffer[0].priority)
        self.assertGreater(per.experience_buffer[0].priority,
                           per.experience_buffer[1].priority)

if __name__ == '__main__':
    unittest.main()