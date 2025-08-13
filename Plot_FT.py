import matplotlib.pyplot as plt
import numpy as np

def plot_ft_data(ft_data, title="Force/Torque Sensor Data", figsize=(10, 6)):
    """
    靜態繪製六軸力/力矩資料的時間變化圖。

    :param ft_data: shape = (N, 6)，每一列為 [Fx, Fy, Fz, Tx, Ty, Tz]
    :param title: 圖片標題
    :param figsize: 畫布大小
    """
    ft_data = np.asarray(ft_data)
    t = np.arange(ft_data.shape[0])

    fig, axs = plt.subplots(2, 1, figsize=figsize, sharex=True)
    
    # 力 - 使用實線，RGB
    axs[0].plot(t, ft_data[:, 0], label='Fx', color='r', linestyle='-')
    axs[0].plot(t, ft_data[:, 1], label='Fy', color='g', linestyle='-')
    axs[0].plot(t, ft_data[:, 2], label='Fz', color='b', linestyle='-')
    axs[0].set_ylabel("Force (N)", fontname='Times New Roman')
    axs[0].legend(loc='upper left', prop={'family': 'Times New Roman'})
    axs[0].grid(True)

    # 力矩 - 使用虛線，RGB
    axs[1].plot(t, ft_data[:, 3], label='Tx', color='r', linestyle='--')
    axs[1].plot(t, ft_data[:, 4], label='Ty', color='g', linestyle='--')
    axs[1].plot(t, ft_data[:, 5], label='Tz', color='b', linestyle='--')
    axs[1].set_xlabel("Time Step", fontname='Times New Roman')
    axs[1].set_ylabel("Torque (Nm)", fontname='Times New Roman')
    axs[1].legend(loc='upper left', prop={'family': 'Times New Roman'})
    axs[1].grid(True)

    fig.suptitle(title, fontname='Times New Roman')
    plt.tight_layout()
    plt.show()


def live_plot_ft_data(get_data_func, interval=0.05, max_len=200, title='Wrench in World Frame'):
    """
    即時顯示六軸力/力矩資料（FIFO buffer 記錄最多 max_len 筆），持續更新最新資料。

    :param get_data_func: 傳回 shape=(6,) 的函數，如 lambda: np.array(shared_data)
    :param interval: 更新時間（秒）
    :param max_len: 顯示的最大時間點數（超過會自動捨棄最舊資料）
    """
    import time
    from collections import deque

    force_data = deque(maxlen=max_len)
    torque_data = deque(maxlen=max_len)
    t_data = deque(maxlen=max_len)

    plt.ion()
    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    fig.suptitle(title, fontname='Times New Roman')
    force_lines = axs[0].plot([], [], label='Fx', color='r', linestyle='-')[0], \
               axs[0].plot([], [], label='Fy', color='g', linestyle='-')[0], \
               axs[0].plot([], [], label='Fz', color='b', linestyle='-')[0]
    torque_lines = axs[1].plot([], [], label='Tx', color='r', linestyle='--')[0], \
                axs[1].plot([], [], label='Ty', color='g', linestyle='--')[0], \
                axs[1].plot([], [], label='Tz', color='b', linestyle='--')[0]

    axs[0].set_ylabel("Force (N)", fontname='Times New Roman')
    axs[1].set_ylabel("Torque (Nm)", fontname='Times New Roman')
    axs[1].set_xlabel("Time Step", fontname='Times New Roman')
    axs[0].legend(loc='upper left', prop={'family': 'Times New Roman'})
    axs[1].legend(loc='upper left', prop={'family': 'Times New Roman'})

    count = 0
    while plt.fignum_exists(fig.number):
        ft = get_data_func()
        if ft is None or len(ft) != 6:
            time.sleep(interval)
            continue

        force_data.append(ft[:3])
        torque_data.append(ft[3:])
        t_data.append(count)
        count += 1

        F = np.array(force_data)
        T = np.array(torque_data)
        t = np.array(t_data)

        for i in range(3):
            force_lines[i].set_data(t, F[:, i])
            torque_lines[i].set_data(t, T[:, i])

        for ax in axs:
            if len(t) > 1:
                ax.set_xlim(t[0], t[-1])
            else:
                ax.set_xlim(0, 1)
            ax.set_ylim(-15, 15)
            ax.grid(True)

        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(interval)

