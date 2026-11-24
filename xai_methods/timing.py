import torch
from data import build_sequences, collate
from run import move_transactions

from xai_methods.ig import get_client, score_model


def timing(
    n_segments,
    model,
    df,
    client_id,
    key,
    batch_args,
    device,
    len_min=None,
    len_max=None,
    IG_PRECISION=50,
    seed=42,
    get_mask=False,
):
    torch.manual_seed(seed)
    df_client = get_client(df, client_id)

    alphas = torch.linspace(0, 1, IG_PRECISION, device=batch_args["device"], dtype=torch.float32)

    x_input = df_client[key].to_torch().to(device=batch_args["device"], dtype=torch.float32)

    # отсоеднияем, т.к. мы используем здесь маскированную часть по которой градиенты не текут.
    x_fixed = x_input.detach()
    x_inter = alphas[:, None] * x_input[None, :]

    T = len(df_client[key])
    N = IG_PRECISION

    if len_min is None:
        len_min = 1
    if len_max is None:
        len_max = T

    seg_lens = torch.randint(len_min, len_max + 1, (N, n_segments), device=device)

    t_starts = (torch.rand(N, n_segments, device=device) * (T - seg_lens + 1)).long()
    time_mask = torch.ones_like(x_inter)

    time_mask = torch.ones((N, T), device=device)

    for s in range(n_segments):
        max_len = int(seg_lens[:, s].max())

        indices = t_starts[:, s, None] + torch.arange(max_len, device=device)[None, :]

        valid = indices < (t_starts[:, s] + seg_lens[:, s])[:, None]

        sample_idx, offset_idx = valid.nonzero(as_tuple=True)
        time_idx = indices[sample_idx, offset_idx]

        time_mask[sample_idx, time_idx] = 0

    masked_inputs = time_mask * x_inter + (1 - time_mask) * x_fixed
    masked_inputs.requires_grad_(True)

    grads = torch.zeros((N, T))
    for rep in range(N):
        df_new = df_client
        # df_new[keys] = masked_inputs[rep].detach().to('cpu')
        cur = build_sequences(
            df_new.to_pandas(), batch_args["metadata"], batch_args["keys"], require_grad=True
        )
        batch = collate(list(cur.values()), batch_args["keys"])
        batch["transactions"][key] = masked_inputs[rep][None, :]

        embed = model.encode(move_transactions(batch, batch_args["device"]))
        logits = model.classifier(embed)

        probab = torch.softmax(logits, dim=-1)

        grad = torch.autograd.grad(
            inputs=batch["transactions"][key], outputs=probab[:, 1].sum(), allow_unused=True
        )

        # print(grad[0])
        grads[rep] = grad[0]

    avg_grad = (grads[:-1] + grads[1:]) / torch.scalar_tensor(2.0, dtype=torch.float32)
    avg_grads = torch.mean(avg_grad, dim=0) * df_client[key].to_torch()

    if get_mask:
        return avg_grads, time_mask
    return avg_grads


def CPP(attr, model, df, key, batch_args, top_k=None):
    "Предполагается что attr является одномерным тензором."
    if len(attr.shape) != 1:
        raise RuntimeError("Masking работает только с одномерными тензорами.")

    df_copy = df.clone()
    index = torch.argsort(attr.abs(), stable=True, descending=False)[:top_k]
    value = torch.zeros_like(attr)
    result_past = torch.zeros(1)
    for idx, id in enumerate(index):
        df_copy[id.item(), key] = 0
        result = score_model(model, df_copy, batch_args).detach().cpu()
        value[idx] = torch.abs(result_past - result) + (value[idx - 1] if idx > 0 else 0)
        result_past = result
    return value


def CPD(attr, model, df, key, batch_args, top_k=None):
    "Предполагается что attr является одномерным тензором."
    if len(attr.shape) != 1:
        raise RuntimeError("Masking работает только с одномерными тензорами.")

    df_copy = df.clone()
    index = torch.argsort(attr.abs(), stable=True, descending=True)[:top_k]
    value = torch.zeros_like(attr)
    result_past = torch.zeros(1)
    for idx, id in enumerate(index):
        df_copy[id.item(), key] = 0
        result = score_model(model, df_copy, batch_args).detach().cpu()
        value[idx] = torch.abs(result_past - result) + (value[idx - 1] if idx > 0 else 0)
        result_past = result
    return value


def build_cp_from_scores(scores, top_k=None):
    return scores.diff(dim=0).abs().cumsum(dim=0)
