import unittest

import torch
import torch.nn.functional as F

from models.lm_loss import (
    chunked_token_superposition_cross_entropy,
    token_superposition_cross_entropy,
    token_superposition_embeddings,
)


class TokenSuperpositionTest(unittest.TestCase):
    def test_embeddings_average_contiguous_bags(self):
        emb = torch.nn.Embedding(6, 2)
        with torch.no_grad():
            emb.weight.copy_(torch.tensor([
                [0.0, 0.0],
                [1.0, 3.0],
                [3.0, 5.0],
                [10.0, 20.0],
                [14.0, 24.0],
                [9.0, 9.0],
            ]))

        folded = token_superposition_embeddings(
            emb,
            torch.tensor([[1, 2, 3, 4]]),
            bag_size=2,
        )

        expected = torch.tensor([[[2.0, 4.0], [12.0, 22.0]]])
        torch.testing.assert_close(folded, expected)

    def test_next_bag_loss_matches_mean_ce(self):
        logits = torch.tensor([[[0.2, 1.0, -0.5], [1.5, -0.2, 0.3]]])
        labels = torch.tensor([[0, 1, 2, 1]])

        actual = token_superposition_cross_entropy(logits, labels, bag_size=2)
        expected = 0.5 * (
            F.cross_entropy(logits[:, :1, :].reshape(-1, 3), torch.tensor([2]))
            + F.cross_entropy(logits[:, :1, :].reshape(-1, 3), torch.tensor([1]))
        )

        torch.testing.assert_close(actual, expected)

    def test_chunked_next_bag_loss_matches_full_logits_and_grads(self):
        torch.manual_seed(123)
        hidden = torch.randn(2, 3, 4, requires_grad=True)
        weight = torch.randn(7, 4, requires_grad=True)
        labels = torch.tensor([
            [0, 1, 2, 3, 4, 5],
            [1, 2, -100, 0, 6, 3],
        ])

        full_hidden = hidden.detach().clone().requires_grad_(True)
        full_weight = weight.detach().clone().requires_grad_(True)
        full_logits = full_hidden @ full_weight.t()
        full_loss = token_superposition_cross_entropy(
            full_logits, labels, bag_size=2,
        )
        full_loss.backward()

        chunk_hidden = hidden.detach().clone().requires_grad_(True)
        chunk_weight = weight.detach().clone().requires_grad_(True)
        chunk_loss = chunked_token_superposition_cross_entropy(
            chunk_hidden,
            chunk_weight,
            labels,
            bag_size=2,
            chunk_size=2,
        )
        chunk_loss.backward()

        torch.testing.assert_close(chunk_loss, full_loss)
        torch.testing.assert_close(chunk_hidden.grad, full_hidden.grad)
        torch.testing.assert_close(chunk_weight.grad, full_weight.grad)


if __name__ == "__main__":
    unittest.main()
