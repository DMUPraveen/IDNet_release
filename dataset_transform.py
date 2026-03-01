import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np
    from scipy.io import loadmat, savemat
    from hyperVca import hyperVca
    import matplotlib.pyplot as plt


@app.cell
def _():
    EXAMPLE = "DATA/real_Samson/alldata_real_Samson.mat"
    EXAMPLE_DATA = loadmat(EXAMPLE)


    def visualize_dataset(x):
        if type(x) is np.ndarray:
            return tuple(x.shape)
        return x


    def visualize_data_dict(dd):
        return {k: visualize_dataset(x) for (k, x) in dd.items()}


    visualize_data_dict(EXAMPLE_DATA)
    return EXAMPLE_DATA, visualize_data_dict


@app.cell
def _(EXAMPLE_DATA):
    plt.imshow(EXAMPLE_DATA["A_init"][2, :].reshape(95, 95))
    return


@app.cell
def _(visualize_data_dict):
    OUR_DATA_PATH = "/storage2/HSI/hsi_datasets/samson.mat"
    OUR_DATA = loadmat(OUR_DATA_PATH)

    visualize_data_dict(OUR_DATA)
    return (OUR_DATA_PATH,)


@app.function
def transform_data(data):
    Y = data["Y"]
    H, W = [int(x) for x in data["HW"].ravel()]
    Yim = Y.reshape(-1, H, W).transpose(1, 2, 0)
    M0 = hyperVca(Y, data["A"].shape[0])[0]
    output_dic = dict(Y=Y, Yim=Yim, M0=M0)
    return output_dic


@app.cell
def _(OUR_DATA_PATH, visualize_data_dict):
    visualize_data_dict(transform_data(loadmat(OUR_DATA_PATH)))
    return


@app.cell
def _():
    datasets = [
        "/storage2/HSI/hsi_datasets/samson_K_3.mat",
        "/storage2/HSI/hsi_datasets/samson.mat",
        "/storage2/HSI/hsi_datasets/urban_4_K_4.mat",
        "/storage2/HSI/hsi_datasets/urban_4.mat",
        "/storage2/HSI/hsi_datasets/Spheric_Synthetic_0_3_perlin_corrected.mat",
        "/storage2/HSI/hsi_datasets/Spheric_Synthetic_0_3_perlin_raw.mat",
        "/storage2/HSI/hsi_datasets/Matern_Synthetic_0_3_perlin_corrected.mat",
        "/storage2/HSI/hsi_datasets/Matern_Synthetic_0_3_perlin_raw.mat",
    ]
    return (datasets,)


@app.cell
def _():
    run_button = mo.ui.run_button(label="Create Datasets")
    run_button
    return (run_button,)


@app.cell
def _(datasets, run_button):
    mo.stop(not run_button.value)
    from pathlib import Path

    for dataset in datasets:
        print(f"Processing {dataset}....")
        _data = loadmat(dataset)
        _output_data = transform_data(_data)
        savemat(Path("DATA") / list(Path(dataset).parts)[-1], _output_data)
    return


if __name__ == "__main__":
    app.run()
