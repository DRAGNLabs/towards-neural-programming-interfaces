from matplotlib import pyplot as plt


def make_training_plots(
    title: str, save_file_path: str, epoch_data, test_data, test_data_generated=None
):
    """
    Make plots for training progress of NPI network

    Arguments:
    epoch_losses -- List of tuples: (epoch, avg_epoch_loss)
    test_losses -- List of tuples: (epoch, avg_test_loss)
    """
    # Get epochs for tests into a list.
    test_epochs = list(map(lambda x: x[0], test_data))
    test_values = list(map(lambda x: x[1], test_data))

    fig1, ax1 = plt.subplots()
    ax1.plot(list(map(lambda x: x[0], epoch_data)), label="train")
    ax1.plot(test_epochs, test_values, label="test")
    if test_data_generated:
        ax1.plot(test_epochs, list(map(lambda x: x[0], test_data_generated)), label="generated test")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Average")
    ax1.set_title(title)
    plt.legend()
    plt.draw()
    fig1.savefig(f"{save_file_path}visualization_{title.replace(' ', '_')}.png")
