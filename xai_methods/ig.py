from functools import partial

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
from data import FLOAT_KEYS, ClientDataset, build_sequences, collate
from run import move_transactions
from torch.utils.data import DataLoader
from tqdm import tqdm


# честно сгепетешеный код, для отрисовки matpotlib в таком формате
def plot_color_bar(
    values,
    temperatures,
    labels=None,
    title="Важность признака",
    ylabel="None",
    cmap_name="coolwarm",
    vmin=None,
    vmax=None,
):
    """
    Строит столбчатую диаграмму, где высота задаётся массивом values,
    а цвет и подписи над столбцами — массивом temperatures.
    """
    values = np.asarray(values)
    temperatures = np.asarray(temperatures)

    if len(values) != len(temperatures):
        raise ValueError(f"Длины массивов не совпадают: {len(values)} и {len(temperatures)}")

    # Если подписи оси X не переданы, нумеруем столбцы по порядку (0, 1, 2...)
    if labels is None:
        labels = np.arange(len(values))

    # Настройка нормализации и цветовой палитры
    t_min = temperatures.min() if vmin is None else vmin
    t_max = temperatures.max() if vmax is None else vmax
    norm = mcolors.Normalize(vmin=t_min, vmax=t_max)
    cmap = plt.get_cmap(cmap_name)
    colors = cmap(norm(temperatures))

    fig, ax = plt.subplots(figsize=(9, 4.8))

    # Построение столбцов
    bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.7)

    # Подписи температур над каждым столбцом
    temp_labels = [f"{t}" for t in temperatures]
    ax.bar_label(bars, labels=temp_labels, padding=4, fontsize=9, fontweight="bold")

    # Запас по оси Y, чтобы подпись над самым высоким столбцом не обрезалась
    y_max = values.max()
    ax.set_ylim(0, y_max * 1.15 if y_max > 0 else 1)

    # Цветовая шкала (colorbar)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Важность признака")

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.show()


def ig_describe_segment(df, client_id, attribution, key):
    value = get_client(df, client_id)[key]

    plot_color_bar(
        value, attribution, ylabel=f"Значение {key}", title="Совместный график значений/атрибутов"
    )


def run_ig_segment(
    model,
    df,
    client_id,
    key,
    window_size,
    batch_args,
    top_k_attr,
    IG_PRECISION=50,
    keep_per_windows_attr=False,
    output=False,
):
    df_client = get_client(df, client_id)

    if window_size is None:
        window_size = len(df_client[key])

    if top_k_attr is None:
        top_k_attr = len(df_client[key])

    reps = len(df_client[key]) - window_size + 1

    dict_cnt = torch.zeros(len(df_client[key]))

    top_k = None
    if keep_per_windows_attr:
        top_k = torch.zeros((reps, len(df_client[key])))

    pbar = tqdm(range(0, reps))
    for rep in pbar:
        pbar.set_description(f"Step of ig is: {rep}/{reps}")
        df_window, df_nwindow = get_windows(df_client, rep, window_size, key)

        attr = ig_path(model, df_nwindow, df_window, key, batch_args, client_id, IG_PRECISION)
        cur_top_k = torch.topk(attr, top_k_attr)[1]
        if output:
            print(cur_top_k)
            print(attr)
        dict_cnt[cur_top_k.type(torch.long)] += 1

        if keep_per_windows_attr:
            top_k[rep] = attr

    if keep_per_windows_attr:
        return dict_cnt, top_k
    return dict_cnt


def ig_path(
    model, df_base, df_pred, key, batch_args, client_id, IG_PRECISION=50, output=False, seed=42
):
    torch.manual_seed(seed)
    if key not in FLOAT_KEYS:
        raise RuntimeError("IG работает только на непрерывных данных.")

    grads = torch.zeros((IG_PRECISION, len(df_base[key])))
    path = torch.linspace(0, 1, IG_PRECISION)
    for idx, k in enumerate(range(0, IG_PRECISION)):
        model.zero_grad()
        if output:
            # plt.plot(df_base[key] + df_pred[key] * path[k])
            plt.plot(df_pred[key].to_torch() * path[k] + df_base[key].to_torch())
        df_new = df_base
        new_tensor = df_new[key].to_torch() + df_pred[key].to_torch() * path[k]
        df_new = df_new.with_columns(pl.Series(new_tensor).alias(key))

        # model.eval()
        cur = build_sequences(
            df_new.to_pandas(), batch_args["metadata"], batch_args["keys"], require_grad=True
        )

        batch = collate(list(cur.values()), batch_args["keys"])
        x = batch["transactions"][key]
        x.requires_grad_(True)
        embed = model.encode(move_transactions(batch, batch_args["device"]))
        logits = model.classifier(embed)
        probabilities = torch.softmax(logits, dim=-1)

        # logits.backward()
        # if (k == 0):
        #     print(df[key].to_list())
        grad = torch.autograd.grad(
            inputs=cur[client_id][key], outputs=probabilities[:, 1].sum(), allow_unused=True
        )

        # if idx == 0:
        #     print(grad)

        grads[idx] = grad[0]

    avg_grad = (grads[:-1] + grads[1:]) / torch.scalar_tensor(2.0, dtype=torch.float32)
    avg_grads = torch.mean(avg_grad, dim=0) * df_pred[key].to_torch()
    return avg_grads


def get_windows(df, step, window_size, key):
    if key not in FLOAT_KEYS:
        raise RuntimeError("значения должны быть непрерывными.")

    df_window = df.with_columns(
        pl.when((pl.int_range(pl.len()) >= step) & (pl.int_range(pl.len()) < step + window_size))
        .then(pl.col(key))
        .otherwise(0)
        .alias(key)
    )

    df_nwindow = df.with_columns(
        pl.when((pl.int_range(pl.len()) >= step) & (pl.int_range(pl.len()) < step + window_size))
        .then(0)
        .otherwise(pl.col(key))
        .alias(key)
    )
    return df_window, df_nwindow


def get_client(df, id):
    return df.filter(pl.col("client_id").eq(id))


"""
# Код который я пока оставлю тут, может ещё пригодиться.
def build_fake_ig_clients(client_id : str, df_set : pl.DataFrame, key : str, value : np.array, approx_value : np.array, IG_PRECISION : int):
    df_list = pl.DataFrame()

    for rep in range(IG_PRECISION):
        df_new = df_set.clone()
        df_new[key] = df_set[key].to_numpy() + value * approx_value[rep]
        df_list = pl.concat([df_list, df_set.with_columns(pl.col('client_id').replace(client_id, client_id + '-' + str(rep)).alias(client_id))], how='vertical')

    return df_list



path = np.linspace(0, 1, 50)
accum_grads = torch.zeros((len(client_full[key])))

key_nwind = batch_cut_window[choosed_client][key].to(device)
key_wind = batch_window[choosed_client][key].to(device)


batch_size = 16
dataloader = make_loader(build_sequences(build_fake_ig_clients(choosed_client, client_full, key, key_wind, path, 50).to_pandas(), metadata, keys, require_grad=True), keys, batch_size)

for batch in dataloader:
    batch = move_transactions(batch))
    embed = model.encode(batch)

    logits = model.classifier(embed)
    probab = logits.softmax(dim=-1)

    for rep in range(batch_size):
        accum_grads += torch.autograd.grad(probab, batch[client_id + '-' + str(rep)])


"""


@torch.inference_mode()
def score_model(model, client, batch_args, faster=False):
    model.eval()
    model.zero_grad()
    if faster:
        cur = build_sequences(client.to_pandas(), batch_args["metadata"], batch_args["keys"])
        dataloader = DataLoader(
            ClientDataset(cur, client["client_id"].unique().cast(pl.String).to_list()),
            batch_size=batch_args["batch_size"],
            shuffle=batch_args["shuffle"],
            generator=torch.Generator().manual_seed(batch_args["seed"]),
            num_workers=0,
            collate_fn=partial(collate, keys=batch_args["keys"]),
        )
        frames = []
        for batch in dataloader:
            logits = model.classifier(model.encode(move_transactions(batch, batch_args["device"])))
            probab = logits.softmax(dim=-1).cpu().numpy()
            frames.append(
                pl.DataFrame(
                    {
                        "client_id": batch["client_id"],
                        "target": batch["target"].numpy(),
                        "logit_0": logits[:, 0].cpu().numpy(),
                        "logit_1": logits[:, 1].cpu().numpy(),
                        "probab_0": probab[:, 0],
                        "probab_1": probab[:, 1],
                        "n_transactions": batch["lengths"].numpy(),
                    }
                )
            )

        frame = pl.concat(frames, how="vertical")
        return frame
    else:
        cur = build_sequences(client.to_pandas(), batch_args["metadata"], batch_args["keys"])
        batch = collate(list(cur.values()), batch_args["keys"])
        embed = model.encode(move_transactions(batch, batch_args["device"]))
        logits = model.classifier(embed)
        probab = torch.softmax(logits, dim=-1)
        return probab[:, 1]


def make_fake_clients(df, attr_order, key, start_id):
    result = []
    current = df.clone()

    result.append(current.with_columns(pl.lit(start_id).alias("client_id")))

    for step, idx in enumerate(attr_order, start=1):
        current[int(idx), key] = 0

        result.append(current.with_columns(pl.lit(start_id + step).alias("client_id")))

    return pl.concat(result, how="vertical")


def MaskingClientFaster(
    attr, model, df, key, batch_args, device, top_k=None, descending=False, random=False, seed=42
):
    "Предполагается что attr является одномерным тензором."
    if len(attr.shape) != 1:
        raise RuntimeError("Masking работает только с одномерными тензорами.")

    torch.manual_seed(seed)

    index = None
    if random:
        index = torch.randperm(len(df[key])).numpy().tolist()
    else:
        index = (
            torch.argsort(attr.abs(), stable=True, descending=descending)[:top_k]
            .detach()
            .cpu()
            .numpy()
            .tolist()
        )

    df_fake_clients = make_fake_clients(df, index, key, int(df["client_id"][0]))
    value = np.zeros(attr.shape[0] + 1)
    result = score_model(model, df_fake_clients, batch_args, faster=True)
    for x in range(attr.shape[0] + 1):
        value[x] = result.filter(pl.col("client_id").eq(str(x + int(df["client_id"][0]))))[
            "probab_1"
        ].to_list()[0]
    return value


def MaskingClient(attr, model, df, key, batch_args, device, top_k=None, descending=False):
    "Предполагается что attr является одномерным тензором."
    if len(attr.shape) != 1:
        raise RuntimeError("Masking работает только с одномерными тензорами.")

    df_copy = df.clone()
    index = (
        torch.argsort(attr.abs(), stable=True, descending=descending)[:top_k]
        .detach()
        .cpu()
        .numpy()
        .tolist()
    )
    value = torch.zeros(attr.shape[0] + 1, device=device)
    value[0] = score_model(model, df_copy, batch_args)
    for idx, id in enumerate(index):
        df_copy[id, key] = 0
        result = score_model(model, df_copy, batch_args)
        value[idx + 1] = result
    return value.detach().cpu()


def RandomMaskingClient(model, df, key, batch_args, device, top_k=None, seed=42):
    "Предполагается что attr является одномерным тензором."
    torch.manual_seed(seed)

    df_copy = df.clone()
    index = torch.randperm(len(df[key])).numpy().tolist()
    value = torch.zeros(len(df[key]) + 1, device=device)
    value[0] = score_model(model, df_copy, batch_args)
    for idx, id in enumerate(index):
        df_copy[id, key] = 0
        result = score_model(model, df_copy, batch_args)
        value[idx + 1] = result
    return value.detach().cpu()


def RandomFeaturePermutation(model, df, key, batch_args, device, seed=42):
    torch.manual_seed(seed)

    index = torch.randperm(len(df[key])).numpy().tolist()
    df_copy = df.clone()
    df_copy = df_copy.with_columns(pl.Series(key, df_copy[key].to_torch()[index]))
    return score_model(model, df_copy, batch_args)


def make_fake_clients_ig(df, series_tuple, key):
    result = []
    current = df.clone()

    for series, name in series_tuple:
        result.append(current.with_columns(pl.lit(name).alias("client_id"), series.alias(key)))

    return pl.concat(result, how="vertical")


def create_client_series_ig(df_client, window_size, key, IG_PRECISION, client):

    path = torch.linspace(0, 1, IG_PRECISION)
    result = []
    name = f"{client} {window_size}"

    reps = range(len(df_client[key]) - window_size + 1)
    for rep in reps:
        df_window, df_nwindow = get_windows(df_client, rep, window_size, key)
        values_window = df_window[key].to_torch()
        values_nwindow = df_nwindow[key].to_torch()

        for idx, x in enumerate(path):
            value = values_nwindow + values_window * x
            result += [(pl.Series(key, value), f"{name} {rep} {idx}")]
    return result


def run_ig_faster(
    model,
    dfs,
    key,
    window_size: None | float,
    batch_args,
    IG_PRECISION=50,
):

    if isinstance(window_size, int) and np.abs(window_size) > 1:
        raise RuntimeError(
            "Размер окна задается в процентном соотношении, численно размер указывать нельзя!"
        )

    torch.manual_seed(batch_args["seed"])
    cur_window_size = None
    values = {}
    fake_clients = pl.DataFrame()
    for df_client in tqdm(dfs, desc="Collecting all-set"):
        if window_size is None:
            cur_window_size = len(df_client[key])
        else:
            cur_window_size = int(len(df_client[key]) * window_size)

        fake_clients = pl.concat(
            [
                fake_clients,
                make_fake_clients_ig(
                    df_client,
                    create_client_series_ig(
                        df_client, cur_window_size, key, IG_PRECISION, df_client["client_id"][0]
                    ),
                    key,
                ),
            ],
            how="vertical",
        )
        values[df_client["client_id"][0]] = df_client[key].to_torch()

    dataset = build_sequences(
        fake_clients.to_pandas(), batch_args["metadata"], batch_args["keys"], require_grad=True
    )

    dataloader = DataLoader(
        ClientDataset(dataset, fake_clients["client_id"].unique().cast(pl.String).to_list()),
        batch_size=batch_args["batch_size"],
        shuffle=batch_args["shuffle"],
        generator=torch.Generator().manual_seed(batch_args["seed"]),
        num_workers=0,
        collate_fn=partial(collate, keys=batch_args["keys"]),
    )

    all_attributions = {}
    for batch in tqdm(dataloader, desc="Batching throw dataloader"):
        embed = model.encode(move_transactions(batch, batch_args["device"]))
        logits = model.classifier(embed)
        probab = torch.softmax(logits, dim=-1)

        # client_id, window_size_cur, id_window, ig_id = int(batch['client_id'][0].split())
        # print(batch['client_id'])
        # print(dataset)
        # print(dataset[batch['client_id']][key])
        # break

        input_list = [dataset[x][key] for x in batch["client_id"]]
        grad = torch.autograd.grad(inputs=input_list, outputs=probab[:, 1].sum(), allow_unused=True)
        for idx, client in enumerate(batch["client_id"]):
            client_cur, window_size_cur, id_window, ig_id = client.split()
            window_size_cur, ig_id, id_window = int(window_size_cur), int(ig_id), int(id_window)
            if client_cur not in all_attributions:
                all_attributions[f"{client_cur}"] = torch.zeros(
                    (
                        values[client_cur].shape[0] - window_size_cur + 1,
                        IG_PRECISION,
                        values[client_cur].shape[0],
                    )
                )

            all_attributions[f"{client_cur}"][id_window, ig_id, :] = grad[idx]

    # print(all_attributions['10345'].shape)
    for client in all_attributions:
        grads = torch.as_tensor(all_attributions[client])

        avg_grads = (grads[:, :-1, :] + grads[:, 1:, :]) / 2

        integrated = avg_grads.mean(dim=1)

        n = values[client].numel()
        w = n if window_size is None else int(window_size * n)

        delta = torch.zeros_like(integrated)

        original = values[client].to(dtype=integrated.dtype, device=integrated.device)

        for start in range(integrated.shape[0]):
            delta[start, start : start + w] = original[start : start + w]

        all_attributions[client] = integrated * delta
    return all_attributions
