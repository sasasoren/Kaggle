import torch


def get_batch_top3score(target, output):
    # weights = torch.as_tensor([1, 1 / 2, 1 / 3]).cuda()
    weights = torch.as_tensor([1, 1 / 2, 1 / 3])
    _, pred = torch.topk(output, k=3, dim=1)
    target = target.reshape(target.shape[0], 1)
    target = target.repeat(1, 3)
    return torch.sum(torch.einsum("ij,j->i", (pred == target).float(), weights)).item()