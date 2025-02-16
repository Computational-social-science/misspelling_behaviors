import pandas as pd
from flask import Flask, render_template, request, jsonify
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator, MultipleLocator

matplotlib.use('Agg')

app = Flask(__name__)
default_params = {
    "se": 0.3, # Probability of transition from Unknown state (U) to Copycatted Misspelling state (CM)
    "si": 0.08, # Probability of transition from Unknown state (U) to Author Misspelling state (AM)
    "ir": 0.61, # Probability of transition from Author Misspelling state (AM) to Correct Spelling state (CS)
    "sx": 0.88, # Probability of transition from Unknown state (U) to Copycatted Correct Spelling state (CC)
    "xr": 0.55, # Probability of transition from Copycatted Correct Spelling state (CC) to Correct Spelling state (CS)
    "er": 0.33, # Probability of transition from Copycatted Misspelling state (CM) to Correct Spelling state (CS)
    "rs": 0.25, # Probability of transition from Correct Spelling state (CS) to Unknown state (U)
    "theta_t": 0.5, # Word Salience at time t
    "omega_t": 0.13 # Collation by proofreader at time t
}

I = 0  # (AM states)
S = 255  # (U states)
R = 127.5  # (CS states)
E = 0  # (CM states)
X = 127.5  # (CC state)

def initializeGrid(N):
    grid = np.zeros(N * N).reshape(N, N)
    for i in range(N):
        for j in range(N):
            grid[i, j] = S
    grid[N // 2, N // 2] = E
    return grid

def calc_neighbour_count_EI(cells, x, y):
    count_0_1 = 0
    count_1_1 = 0
    N = len(cells)
    for i in range(max(0, x - 1), min(N, x + 2)):
        for j in range(max(0, y - 1), min(N, y + 2)):
            if i == x and j == y:
                continue
            if cells[i][j] in [E,I]:
                if (i == x and j == y - 1) or \
                        (i == x and j == y + 1) or \
                        (i == x - 1 and j == y) or \
                        (i == x + 1 and j == y):
                    count_0_1 += 1
                else:
                    count_1_1 += 1
    return count_0_1 + count_1_1

def calc_neighbour_count_XR(cells, x, y):
    count_0_1 = 0
    count_1_1 = 0
    N = len(cells)
    for i in range(max(0, x - 1), min(N, x + 2)):
        for j in range(max(0, y - 1), min(N, y + 2)):
            if i == x and j == y:
                continue
            if cells[i][j] in [R,X]:
                if (i == x and j == y - 1) or \
                        (i == x and j == y + 1) or \
                        (i == x - 1 and j == y) or \
                        (i == x + 1 and j == y):
                    count_0_1 += 1
                else:
                    count_1_1 += 1
    return count_0_1 + count_1_1

def update(frameNum, grid, N, params, stats):
    newGrid = grid.copy()

    se = params["se"]
    si = params["si"]
    ir = params["ir"]
    sx = params["sx"]
    xr = params["xr"]
    er = params["er"]
    rs = params["rs"]
    theta_t = params["theta_t"]
    omega_t = params["omega_t"]

    none_count = 0
    misspelling_count = 0
    spelling_count = 0

    for i in range(N):
        for j in range(N):
            cell = grid[i, j]
            if cell == S:
                none_count += 1
                if random.random() < se * (1 - theta_t) and calc_neighbour_count_EI(grid, i, j) != 0:
                    newGrid[i, j] = E
                elif random.random() < si * (1 - theta_t) and calc_neighbour_count_EI(grid, i, j) == 0:
                    newGrid[i, j] = E
                elif random.random() < sx * theta_t and calc_neighbour_count_XR(grid, i, j) != 0:
                    newGrid[i, j] = X
            elif cell == I:
                misspelling_count += 1
                if random.random() < ir and calc_neighbour_count_XR(grid, i, j) != 0:
                    newGrid[i, j] = R
                elif random.random() < omega_t:
                    newGrid[i, j] = R
            elif cell == R:
                spelling_count += 1
                if random.random() < rs:
                    newGrid[i, j] = S
            elif cell == E:
                misspelling_count += 1
                if random.random() < er and calc_neighbour_count_XR(grid, i, j) != 0:
                    newGrid[i, j] = R
                elif random.random() < omega_t:
                    newGrid[i, j] = R
            elif cell == X:
                spelling_count += 1
                if random.random() < xr:
                    newGrid[i, j] = R

    stats[0].append(none_count)
    stats[1].append(misspelling_count)
    stats[2].append(spelling_count)

    if N == 200: # save the grid state
        save_grid_to_txt(grid)

    return newGrid

def generate_animation_data(N, params, frames):
    grid = initializeGrid(N)
    stats = ([], [], [])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.5))

    ax1.set_position([0.07, 0.1, 0.4, 0.8])  # [left, bottom, width, height]
    ax2.set_position([0.57, 0.1, 0.4, 0.8])  # [left, bottom, width, height]
    img = ax1.imshow(grid, interpolation='nearest', cmap='viridis')
    ax1.set_xticks([])
    ax1.set_yticks([])

    def update_animation(frameNum):
        nonlocal grid
        grid = update(frameNum, grid, N, params, stats)
        img.set_array(grid)
        plot_stats(ax2, stats)
        return img,

    ani = FuncAnimation(fig, update_animation, frames=frames, blit=True, repeat=False)
    plt.close()  # Prevents initial plot from showing

    legend_elements = [
        Line2D([0], [0], color=plt.cm.viridis(R / 255),marker='o', markersize=9.5, linestyle='None',label='Spelling'),
        Line2D([0], [0], color=plt.cm.viridis(E / 255), marker='o', markersize=9.5, linestyle='None',label='Misspelling'),
        Line2D([0], [0], color=plt.cm.viridis(S / 255), marker='o', markersize=9.5, linestyle='None',label='Unknown')
    ]

    font = FontProperties(family='Times New Roman', style='normal', weight='normal', size=11)
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1), ncol=3,
              fancybox=False, shadow=False, frameon=False, prop=font)

    html_content = ani.to_jshtml()

    # Save HTML content to a file
    filename = "animation_output.html"
    with open("./kk.html", "w") as file:
        file.write(html_content)
    return ani.to_jshtml()

def plot_stats(ax, stats):
    ax.clear()
    ax.plot(stats[0], label='None',color=plt.cm.viridis(S / 255))
    ax.plot(stats[1], label='Misspelling',color=plt.cm.viridis(E / 255))
    ax.plot(stats[2], label='Spelling', color=plt.cm.viridis(R / 255))
    if len(stats[1]) == 100:
        save_stats_to_csv(stats)
    # ax.legend(loc='upper right')
    # ax.set_xlabel('Frame')
    # ax.set_ylabel('Count')
    font = FontProperties(family='Times New Roman', style='normal', weight='normal', size=12)
    ax.set_xlabel('Time', fontproperties=font)
    ax.set_ylabel('Count', fontproperties=font,labelpad=0.01)

    ax.tick_params(axis='y', which='both', labelsize=9, width=1,length=4)
    font_1 = FontProperties(family='Times New Roman', style='normal', weight='normal', size=9)
    for label in ax.get_yticklabels():
        label.set_fontproperties(font_1)

    ax.tick_params(axis='x', which='both', labelsize=9, width=1)
    font_2 = FontProperties(family='Times New Roman', style='normal', weight='normal', size=9)
    for label in ax.get_xticklabels():
        label.set_fontproperties(font_2)


    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(axis='y', which='minor', length=3, width=0.7)
    # ax.set_title('State Counts Over Time')

    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(axis='x', which='minor', length=3, width=0.7)

def save_grid_to_txt(grid):
    with open('grid_states.txt', 'w') as f:
        f.write("X, Y, State\n")
        N = grid.shape[0]
        for x in range(N):
            for y in range(N):
                state = grid[x, y]
                f.write(f"{x}, {y}, {state}\n")

def save_stats_to_csv(stats):
    data = {
        'None': stats[0],
        'Misspelling': stats[1],
        'Spelling': stats[2]
    }
    df = pd.DataFrame(data)

    # Save DataFrame to CSV
    df.to_csv('simulation_stats.csv', index=False)


@app.route('/')
def index():
    animation_html = generate_animation_data(N=100, params=default_params,frames=100)
    return render_template('index.html', animation_html=animation_html)

@app.route('/run_simulation', methods=['POST'])
def run_simulation():
    params = {
        "se": float(request.form.get("se", 0.3)),# CM
        "si": float(request.form.get("si", 0.08)),# AM
        "ir": float(request.form.get("ir", 0.61)),# SP(AM)
        "sx": float(request.form.get("sx", 0.88)),# CS
        "xr": float(request.form.get("xr", 0.55)),# AC
        "er": float(request.form.get("er", 0.33)),# SP(CM)
        "rs": float(request.form.get("rs", 0.25)),# FS
        "theta_t": float(request.form.get("theta_t", 0.5)),# W_S
        "omega_t": float(request.form.get("omega_t", 0.13))# C_P
    }
    n = int(request.form['n'])
    frame_count = int(request.form['frame_count'])
    animation_html = generate_animation_data(N=n, params=params,frames=frame_count)
    return jsonify({'animation_html': animation_html})

if __name__ == '__main__':
    app.run(debug=True)