import os
import bisect
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.patches import Rectangle, Circle


FPS = 30
INIT_FRM = int(FPS / 5)
PT_MICRON = 8
MUS_PER_FRM = 50 / FPS


class Animator:
    """Export an MP4 of atom positions and global Rydberg illumination from ``results_code``."""

    def __init__(self,
                 slm_sites: list,
                 results_code: dict,
                 architecture: dict
                 ):
        self.CANVAS_PADDING = 10
        self.RYDBERG_PADDING = 5
        self.FIG_X_PADDING = 10
        self.FIG_Y_PADDING = 20

        matplotlib.use('Agg')
        matplotlib.rcParams.update({'font.size': 10})
        self.storage_slms, self.entanglement_slms = slm_sites
        plt.rcParams['animation.ffmpeg_path'] = "ffmpeg"
        self.architecture = architecture
        self.n_q = results_code['n_q']
        self.instructions = results_code['instructions']

        self.set_architecture()

        self.keyframes = []
        self.setup_canvas()
        self.create_schedule()

        anim = FuncAnimation(
            self.fig,
            self.update,
            init_func=self.update_init,
            frames=self.keyframes[-1],
            interval=MUS_PER_FRM
        )

        anim_path = os.path.join(
            "results",
            results_code["output_dir"],
            "animations",
            f"{results_code['benchmark']}.mp4",
        )
        os.makedirs(os.path.dirname(anim_path), exist_ok=True)
        anim.save(anim_path, writer=FFMpegWriter(FPS))

    def set_architecture(self):
        self.X_LOW = min([x for x, _ in self.storage_slms+ self.entanglement_slms])
        self.X_HIGH = max([x for x, _ in self.storage_slms+ self.entanglement_slms])
        self.Y_LOW = min([y for _, y in self.storage_slms+ self.entanglement_slms])
        self.Y_HIGH = max([y for _, y in self.storage_slms+ self.entanglement_slms])

        self.GLOBAL_LIGHT_X_LOW = min([x for x, _ in self.entanglement_slms])
        self.GLOBAL_LIGHT_X_HIGH = max([x for x, _ in self.entanglement_slms])
        self.GLOBAL_LIGHT_Y_LOW = min([y for _, y in self.entanglement_slms])
        self.GLOBAL_LIGHT_Y_HIGH = max([y for _, y in self.entanglement_slms])

    def create_schedule(self):
        """Assign ``f_begin`` / ``f_end`` per instruction; stretch short gates so they are visible on video."""

        frame = 0
        for inst in self.instructions:
            if inst['type'] == "1qGate" or inst['type'] == "2qGate":
                inst['duration'] = [MUS_PER_FRM * 8 for _ in range(len(inst['qs']))]
            elif inst['type'] == "Activate" or inst['type'] == "Deactivate":
                inst['duration'] = [MUS_PER_FRM for _ in range(self.n_q)]
            elif inst['type'] == "Park" or inst['type'] == "BigMove":
                pass
            elif inst['type'] == "Init":
                inst['duration'] = [MUS_PER_FRM * INIT_FRM for _ in range(self.n_q)]
            elif inst['type'] == "Crosstalk":
                pass
            else:
                raise ValueError(f"unknown inst type {inst['type']}")

            inst['f_begin'] = frame
            new_frame = frame + int((max(inst['duration'])) / MUS_PER_FRM)
            inst['f_end'] = new_frame - 1
            self.keyframes.append(new_frame)
            frame = new_frame

        # Build a sorted list of keyframe starts for O(log n) lookup in update().
        # keyframes[i] is the first frame of instruction i+1, so keyframes[i-1] is
        # the first frame of instruction i — we binary-search on this boundary list.
        self._kf_starts = [inst['f_begin'] for inst in self.instructions]

    def setup_canvas(self):
        """Create figure, trap grid, and the semi-transparent entangling-region overlay."""
        px = 1 / plt.rcParams['figure.dpi'] * PT_MICRON
        self.fig, self.ax, = plt.subplots(
            figsize=((self.X_HIGH - self.X_LOW) * px, (self.Y_HIGH - self.Y_LOW) * px))

        self.title = self.ax.set_title("")
        self.ax.set_xlim(self.X_LOW - self.CANVAS_PADDING, self.X_HIGH + self.CANVAS_PADDING)
        self.ax.set_ylim(self.Y_LOW - self.CANVAS_PADDING, self.Y_HIGH + self.CANVAS_PADDING)
        self.ax.set_aspect('equal', adjustable='box')

        self.global_lights = Rectangle(
            (self.GLOBAL_LIGHT_X_LOW - self.RYDBERG_PADDING, self.GLOBAL_LIGHT_Y_LOW - self.RYDBERG_PADDING),
            self.GLOBAL_LIGHT_X_HIGH - self.GLOBAL_LIGHT_X_LOW + 2 * self.RYDBERG_PADDING,
            self.GLOBAL_LIGHT_Y_HIGH - self.GLOBAL_LIGHT_Y_LOW + 2 * self.RYDBERG_PADDING,
            linewidth=2,
            facecolor=(0, 0, 1, 0.2)
        )
        self.draw_slm_sites()

    def _sync_qubit_text_positions(self):
        for q in range(self.n_q):
            x, y = self.current_locs[q]
            self.qubit_text[q].set_position((x - 1, y + 1))

    def draw_slm_sites(self):
        for x, y in self.storage_slms + self.entanglement_slms:
            self.ax.add_patch(Circle((x, y), 1, fill=False, edgecolor="#515252", linewidth=1))

    def update_init(self):
        # Qubits the AOD is holding right now; drives the red ring (see
        # _refresh_rings). FuncAnimation re-runs init between passes, so this
        # must be reset here rather than in __init__.
        self._aod_held = set()
        self.current_locs = [(self.instructions[0]['locs'][q]['x'], self.instructions[0]['locs'][q]['y']) for q
                             in range(self.n_q)]
        self.col_plots = [self.ax.axvline(0, self.Y_LOW, self.Y_HIGH, c=(1, 0, 0, 0), ls="--") for _
                          in range(self.n_q)]
        self.row_plots = [self.ax.axhline(0, self.X_LOW, self.X_HIGH, c=(1, 0, 0, 0), ls="--") for _
                          in range(self.n_q)]

        self.qubit_scat = self.ax.scatter(
            [loc['x'] for loc in self.instructions[0]['locs']],
            [loc['y'] for loc in self.instructions[0]['locs']],
            s=16, edgecolors=(1, 0, 0, 0)
            )
        self.qubit_scat.set_color("black")

        locs0 = self.instructions[0]["locs"]
        self.qubit_text = [
            self.ax.text(
                locs0[q]["x"] - 1,
                locs0[q]["y"] + 1,
                f"{q}",
                fontsize=8,
            )
            for q in range(self.n_q)
        ]

        return (
            [self.qubit_scat]
            + self.col_plots
            + self.row_plots
            + self.qubit_text
        )

    def _default_artists(self):
        return [self.qubit_scat] + self.col_plots + self.row_plots + self.qubit_text

    def update(self, f: int):
        if f < self.keyframes[0]:
            return self._default_artists()

        # Binary search: find the last instruction whose f_begin <= f.
        i = bisect.bisect_right(self._kf_starts, f) - 1
        if i < 0 or i >= len(self.instructions):
            return self._default_artists()

        inst = self.instructions[i]
        t = inst['type']

        if t == "1qGate":
            return self.update_1q_gate(f, inst)
        elif t == "2qGate":
            return self.update_2q_gate(f, inst)
        elif t == "Activate":
            return self.update_activate(f, inst)
        elif t == "Deactivate":
            return self.update_deactivate(f, inst)
        elif t in ("Park", "BigMove"):
            return self.update_move(f, inst)
        elif t in ("Init", "Crosstalk"):
            return self._default_artists()
        else:
            raise ValueError(f"unknown inst type {t}")

    def update_1q_gate(self, f: int, inst: dict):
        if f == inst['f_begin']:
            self.title.set_text(f"Stage {inst['stage']} \n{inst['type']}")
            self.circles = []
            for i, q in enumerate(inst['qs']):
                circle = Circle((inst['locs'][i]['x'], inst['locs'][i]['y']), 2, color=(0, 0, 1, 0.2))
                self.ax.add_patch(circle)
                self.circles.append(circle)

            for i, q in enumerate(inst['qs']):
                self.current_locs[q] = (inst['locs'][i]['x'], inst['locs'][i]['y'])
            self.qubit_scat.set_offsets(self.current_locs)
            self._sync_qubit_text_positions()

        if f == inst['f_end']:
            self.ax.set_facecolor('w')
            for circle in self.circles:
                circle.remove()
        return self._default_artists()

    def update_2q_gate(self, f: int, inst: dict):
        if f == inst['f_begin']:
            self.title.set_text(f"Stage {inst['stage']} \n{inst['type']}")

            self.ax.add_patch(self.global_lights)

            for i, q in enumerate(inst['qs']):
                self.current_locs[q] = (inst['locs'][i]['x'], inst['locs'][i]['y'])
            self.qubit_scat.set_offsets(self.current_locs)
            self._sync_qubit_text_positions()

        if f == inst['f_end']:
            self.global_lights.remove()
        return self._default_artists()

    def interpolate(self, progress: int, duration: int, begin: int, end: int):
        """Smooth cubic position (Bluvstein-style) for ``progress`` in ``[0, duration-1]``."""

        D = end - begin
        if D == 0:
            return begin
        r = (1 + progress) / duration
        return begin + 3 * D * (r ** 2) - 2 * D * (r ** 3)

    def _refresh_rings(self):
        """Red ring exactly on the atoms the AOD is currently holding.

        Repaints from the tracked held set rather than from the qubits named
        in the current instruction: a wave can pick up or drop atoms in more
        than one step (the normal case after waypoint decomposition), so
        painting per-instruction would erase rings of atoms still held and
        leave them reading as "released but still steered".
        """
        self.qubit_scat.set_edgecolor(
            [(1, 0, 0, 1) if q in self._aod_held else (1, 0, 0, 0)
             for q in range(self.n_q)])

    def update_activate(self, f: int, inst: dict):
        if f == inst['f_begin']:
            self.title.set_text(inst['type'])
            qs_set = set(inst['qs'])
            # Build index map once per instruction instead of calling list.index() per qubit.
            q_to_i = {q: i for i, q in enumerate(inst['qs'])}
            for q in range(self.n_q):
                if q in qs_set:
                    i = q_to_i[q]
                    self.col_plots[q].set_xdata((inst['locs'][i]['x'],))
                    self.col_plots[q].set_color((1, 0, 0, 0.5))
                    self.row_plots[q].set_ydata((inst['locs'][i]['y'],))
                    self.row_plots[q].set_color((1, 0, 0, 0.5))

            self._aod_held |= qs_set
            self._refresh_rings()
        return self._default_artists()

    def update_deactivate(self, f: int, inst: dict):
        if f == inst['f_begin']:
            self.title.set_text(inst['type'])
            qs_set = set(inst['qs'])
            for q in range(self.n_q):
                if q in qs_set:
                    self.col_plots[q].set_color((1, 0, 0, 0))
                    self.row_plots[q].set_color((1, 0, 0, 0))
            self._aod_held -= qs_set
            self._refresh_rings()
        return self._default_artists()

    def update_move(self, f: int, inst: dict):
        if f == inst['f_begin']:
            self.title.set_text(inst['type'])
            # Cache index map so the per-frame loop below is O(moving qubits) not O(n_q²).
            inst['_q_to_i'] = {q: i for i, q in enumerate(inst['qs'])}

        progress = f - inst['f_begin']
        duration = inst['f_end'] - inst['f_begin'] + 1
        q_to_i = inst['_q_to_i']

        for q, i in q_to_i.items():
            curr_x = self.interpolate(
                progress, duration, inst['locs'][i]['x_begin'], inst['locs'][i]['x_end'])
            self.col_plots[q].set_xdata((curr_x,))

            curr_y = self.interpolate(
                progress, duration, inst['locs'][i]['y_begin'], inst['locs'][i]['y_end'])
            self.row_plots[q].set_ydata((curr_y,))

            self.current_locs[q] = (curr_x, curr_y)

        self.qubit_scat.set_offsets(self.current_locs)
        self._sync_qubit_text_positions()
        return self._default_artists()
