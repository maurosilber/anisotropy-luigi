import pathlib
from enum import IntEnum
from functools import partial

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, widgets, gridspec
from scipy.signal import savgol_filter

from .cell import CellJumpCurves
from .files import sensor_data


class JumpEnum(IntEnum):
    undefined = 0
    jump = 1
    no_jump = 2
    noisy = 3


class FilteredData:
    def __init__(self, path, index, fluorophores, ix=None):
        self.path = path
        self.df = self.load() if path.exists() else self.create(index, fluorophores)
        self.ix = ix or 0

    def create(self, index, fluorophores):
        return pd.DataFrame(JumpEnum.undefined.value, index=index, columns=fluorophores)

    def load(self):
        return pd.read_csv(self.path, index_col=0)

    def save(self):
        self.df.to_csv(self.path)

    @property
    def i(self):
        return self.df.index[self.ix]

    def get(self, fp):
        return self.df.loc[self.i, fp]

    def set(self, fp, value):
        self.df.loc[self.i, fp] = value
        self.save()

    def stats(self):
        dg = self.df == JumpEnum.jump
        return dg.sum(1).value_counts()


def activity(ani, ani_min, ani_max, delta_brightness, derivative=None):
    if derivative is None:
        return NotImplementedError

    factor = (1 + delta_brightness) / (ani_max - ani_min)
    normalized_anisotropy = (ani - ani_min) / (ani_max - ani_min)
    return factor * derivative / (1 + delta_brightness * normalized_anisotropy) ** 2


def savgol_activity(ani, window, order, delta_brightness, ret_anisotropy=False):
    der = savgol_filter(ani, window, order, deriv=1)
    ani = savgol_filter(ani, window, order)
    ani_max, ani_min = ani.max(), ani.min()
    act = activity(
        ani, ani_min, ani_max, delta_brightness=delta_brightness, derivative=der
    )
    return act if not ret_anisotropy else (act, ani)


def max_from_quadratic(x, y):
    ix = np.argmax(y)
    if ix == 0:
        s = slice(3)
    elif ix == y.size - 1:
        s = slice(-4, -1)
    else:
        s = slice(ix - 1, ix + 2)
    p = np.polynomial.Polynomial.fit(x[s], y[s], 2)
    root = p.deriv().roots()[0]
    return root, p(root)


def call_callback(event, callback):
    callback()


class GUI:
    def __init__(self, jumps_df, filtered_df, npz, fluorophores):
        self.jumps_df = jumps_df
        self.filtered_df = filtered_df
        self.npz = npz
        self.fluorophores = fluorophores

        self.buttons = {}
        self.key_callbacks = {}

    def key_press(self, event):
        callback = self.key_callbacks.get(event.key)
        if callback:
            callback()

    def create_buttons(self, gs):
        axes = gs.subgridspec(1, len(self.buttons)).subplots()

        for ax, (name, button) in zip(axes, self.buttons.items()):
            button, label, callback, key = button
            button = button(ax, label, color="none", hovercolor="none")
            if callback is not None:
                button.on_clicked(partial(call_callback, callback=callback))
            if key is not None:
                self.key_callbacks[key] = callback
            self.buttons[name] = button  # Save reference

    @staticmethod
    def get_indexes(jump_row, cjs, t, fluorophore):
        ani = cjs.anisotropy(fluorophore)
        jump_ix = np.searchsorted(t, jump_row[f"{fluorophore}_jump_time"])
        s = slice(max(jump_ix - 20, 0), jump_ix + 20)

        act, act_loc, act_max = None, None, None
        try:
            act = savgol_activity(
                ani[s],
                11,
                2,
                delta_brightness=sensor_data.loc[fluorophore].delta_brightness,
            )
            try:
                act_loc, act_max = max_from_quadratic(t[s], act)
                if act_loc < t[s][0] or act_loc > t[s][-1]:
                    act_loc, act_max = None, None
            except IndexError:
                pass
        except ValueError:
            pass

        return {
            "anisotropy": ani,
            "jump_ix": jump_ix,
            "slice": s,
            "activity": act,
            "act_loc": act_loc,
            "act_max": act_max,
        }

    def plot(self):
        for a in self.ax:
            a.cla()
        self.ax_border.cla()

        jump_row = self.jumps_df.loc[self.filtered_df.i]
        cjs = CellJumpCurves(self.npz, jump_row.date, jump_row.position, jump_row.label)
        t = cjs.time()

        for fp in self.fluorophores:
            data = self.get_indexes(jump_row, cjs, t, fp)

            self.ax[0].plot(t, data["anisotropy"])
            self.ax[0].scatter(t[data["jump_ix"]], data["anisotropy"][data["jump_ix"]])
            if data["activity"] is not None:
                self.ax[1].plot(t[data["slice"]], data["activity"])
            if data["act_loc"] is not None:
                self.ax[1].scatter(data["act_loc"], data["act_max"])

            self.ax[2].plot(t, cjs.non_saturated_size(fp))
        self.ax[0].set(ylim=(0.2, 0.33))

        self.ax[2].plot(t, cjs.cell_size())
        self.ax[2].set(ylim=(0.0, cjs.cell_size().max() * 1.05))
        self.ax_border.plot(t, cjs.cell_on_border(), color="k")
        self.ax_border.set(ylim=(-1, 2))

    def move(self, ix, delta=False):
        if delta:
            self.filtered_df.ix += ix
        else:
            self.filtered_df.ix = ix
        self.draw()

    def next(self):
        self.move(1, delta=True)

    def prev(self):
        self.move(-1, delta=True)

    def first_undefined(self):
        i = self.filtered_df.i + 1
        ix = (self.filtered_df.df[i:] == 0).all(1).idxmax()
        self.move(ix)

    def all_no_jump(self):
        for fp in self.fluorophores:
            self.buttons[fp].ix = JumpEnum.no_jump
            self.set_value(fp, self.buttons[fp].ix)
        self.fig.canvas.draw()

    def all_noisy(self):
        for fp in self.fluorophores:
            self.buttons[fp].ix = JumpEnum.noisy
            self.set_value(fp, self.buttons[fp].ix)
        self.fig.canvas.draw()

    def get_value(self, fluorophore):
        return self.filtered_df.get(fluorophore)

    def set_value(self, fluorophore, value):
        return self.filtered_df.set(fluorophore, value)

    def set_buttons(self):
        for fp in self.fluorophores:
            self.buttons[fp].ix = self.get_value(fp)

    def fp_button(self, fp):
        self.buttons[fp].next()
        self.set_value(fp, self.buttons[fp].ix)
        self.fig.canvas.draw()

    def draw(self):
        self.plot()
        i, n = self.filtered_df.i, len(self.filtered_df.df)
        jump_row = self.jumps_df.loc[i]
        date, position, label = jump_row.date, jump_row.position, jump_row.label
        title = f"{i} / {n}  -  Date: {date}  #{position}  cell: {label}"
        title += "\n" + str(self.filtered_df.stats().to_dict())
        self.fig.suptitle(title)
        self.set_buttons()
        self.fig.canvas.draw()

    def start(self):
        self.fig = plt.figure()
        gs = gridspec.GridSpec(7, 1, figure=self.fig)
        self.ax = gs[:-2].subgridspec(3, 1).subplots(sharex=True)
        self.ax_border = self.ax[2].twinx()

        self.create_buttons(gs[-1,-1])
        self.fig.canvas.mpl_connect("key_press_event", self.key_press)
        self.draw()
        plt.show()


class CylingButton(widgets.Button):
    colors = ("gray", "g", "r", "y")
    _ix = None

    @property
    def ix(self):
        return self._ix

    @ix.setter
    def ix(self, value):
        self._ix = value
        self.draw()

    def next(self):
        self.ix = (self.ix + 1) % 4

    def draw(self):
        self.ax.set_xlabel(JumpEnum(self.ix).name)
        self.ax.set_fc(self.colors[self.ix])


def main(folder, start=False):
    path = pathlib.Path(folder)

    fluorophores = ["BFP", "Cit", "Kate"]
    jumps_df = pd.read_pickle(path / "anisotropy_jumps.pandas")
    npz = np.load(path / "jump_curves.npz")
    filtered_df = FilteredData(
        path / "filtered_data.pandas", jumps_df.index, fluorophores
    )

    gui = GUI(jumps_df, filtered_df, npz, fluorophores)
    gui.buttons["prev"] = (widgets.Button, "Prev", gui.prev, ",")
    gui.buttons["next"] = (widgets.Button, "Next", gui.next, ".")
    gui.buttons["first_undef"] = (
        widgets.Button,
        "First\nundef.",
        gui.first_undefined,
        None,
    )
    gui.buttons["all_no_jump"] = (widgets.Button, "All\nno jump.", gui.all_no_jump, "v")
    gui.buttons["all_noisy"] = (widgets.Button, "All\nnoisy.", gui.all_noisy, "b")

    for key, fp in zip(("z", "x", "c"), gui.fluorophores):
        gui.buttons[fp] = (CylingButton, fp, partial(gui.fp_button, fp=fp), key)

    if start:
        gui.start()
    return gui


if __name__ == "__main__":
    import sys

    main(sys.argv[1], start=True)
