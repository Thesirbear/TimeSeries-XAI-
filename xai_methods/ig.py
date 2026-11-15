import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
from data import FLOAT_KEYS, build_sequences, collate
from run import move_transactions
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


def ig_path(model, df_base, df_pred, key, batch_args, client_id, IG_PRECISION=50, output=False):

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
