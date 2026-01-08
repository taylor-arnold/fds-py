import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io

P = 1.4          # power
A = 0.6          # weight on power term
B = 0.8          # weight on cosine term
C = 2.5          # cosine frequency
EPS = 1e-6       # smoothing; try 1e-4 if you want it "more rounded" near 0
def f(x):
    x = np.asarray(x)
    smooth_abs_p = (x*x + EPS)**(P/2)  # ~|x|^P
    tilt = 0.4 * x * (x > 0)
    return A * smooth_abs_p + B * np.cos(C * x) + tilt


def df(x):
    x = np.asarray(x)

    # Power term gradient
    grad_power = A * (P * x * (x*x + EPS)**(P/2 - 1.0))

    # Cosine term gradient
    grad_cos = -B * C * np.sin(C * x)

    # Linear tilt gradient: derivative of 0.4*x for x>0
    grad_tilt = 0.4 * (x > 0)

    return grad_power + grad_cos + grad_tilt


def ddf(x):
    x = np.asarray(x)

    u = x*x + EPS
    a = P/2 - 1.0

    # Second derivative of power term
    power_dd = A * P * (u**a + 2.0 * x*x * a * u**(a - 1.0))

    # Second derivative of cosine term
    cos_dd = -B * (C**2) * np.cos(C * x)

    # Second derivative of linear tilt is zero (almost everywhere)
    return power_dd + cos_dd


def gradient_descent(x0, lr=0.08, steps=35):
    xs = [x0]
    x = float(x0)
    for _ in range(steps):
        g = df(x)
        x = x - lr * g
        xs.append(x)
    return np.array(xs)


def make_gd_gif(
    x0=-2.5,
    lr=0.08,
    steps=35,
    xlim=(-4, 4),
    ngrid=600,
    out_path="gradient_descent_1d.gif",
    fps=10
):
    xs = gradient_descent(x0, lr=lr, steps=steps)

    # Precompute curve for background
    grid = np.linspace(xlim[0], xlim[1], ngrid)
    ygrid = f(grid)

    # For stable y-limits (avoid bouncing axes)
    ymin = min(ygrid.min(), f(xs).min()) - 0.5
    ymax = max(ygrid.max(), f(xs).max()) + 0.5

    frames = []
    duration_ms = int(1000 / fps)

    for k, xk in enumerate(xs):
        yk = f(xk)
        gk = df(xk)
        xnext = xs[k + 1] if k < len(xs) - 1 else xk
        ynext = f(xnext)

        fig, ax = plt.subplots(figsize=(6.8, 4.2), dpi=120)

        # Objective curve
        ax.plot(grid, ygrid, linewidth=2, label="f(x)")

        # Current point
        ax.scatter([xk], [yk], s=55, zorder=3, label=f"iter {k}: x={xk:.3f}")

        # Show the next step as an arrow (in x-direction)
        if k < len(xs) - 1:
            ax.annotate(
                "",
                xy=(xnext, ynext),
                xytext=(xk, yk),
                arrowprops=dict(arrowstyle="->", linewidth=2),
            )

        # Optional: tangent line at current point (local linearization)
        # y = f(xk) + f'(xk)(x-xk)
        tangent_x = np.array([xk - 1.0, xk + 1.0])
        tangent_y = yk + gk * (tangent_x - xk)
        ax.plot(tangent_x, tangent_y, linestyle="--", linewidth=1.5, label="tangent")

        # Cosmetics
        ax.set_title(f"Gradient Descent on 1D Function (lr={lr}, step {k}/{steps})")
        ax.set_xlim(*xlim)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper left", fontsize=9)

        # Render to in-memory PNG, then to Pillow image
        buf = io.BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        frames.append(Image.open(buf).convert("RGBA"))

    # Save GIF (loop forever)
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        disposal=2,
    )

    return out_path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io

def make_2nd_gif(
    x0=2.8,
    lr=0.08,
    steps=35,
    xlim=(-4, 4),
    ngrid=600,
    out_path="second_order_1d.gif",
    fps=10
):
    """
    2nd-order optimization GIF using a local quadratic approximation each step.

    Requires you to have defined in scope:
        f(x): objective
        df(x): first derivative
        ddf(x): second derivative

    Uses a damped Newton step:
        x_{k+1} = x_k - lr * df(x_k) / ddf(x_k)

    If ddf(x_k) is nonpositive or too small, falls back to a gradient step:
        x_{k+1} = x_k - lr * df(x_k)

    Visualization per frame:
      - f(x)
      - current iterate
      - local quadratic q_k(x)
      - quadratic minimizer (Newton target)
      - arrow to the actual next iterate
    """

    # ---- Precompute background curve ----
    grid = np.linspace(xlim[0], xlim[1], ngrid)
    ygrid = f(grid)

    # ---- Generate iterates using 2nd-order logic ----
    xs = [float(x0)]
    step_types = []  # "newton" or "gd_fallback"
    newton_targets = []  # store x* = x - g/h when defined, else nan

    x = float(x0)
    eps_curv = 1e-10  # curvature threshold to avoid divide-by-(near)-zero

    for _ in range(steps):
        g = float(df(x))
        h = float(ddf(x))

        if np.isfinite(h) and h > eps_curv:
            x_star = x - g / h                 # minimizer of local quadratic
            x_next = x - lr * (g / h)          # damped Newton using lr as damping
            step_types.append("newton")
            newton_targets.append(float(x_star))
        else:
            # Nonconvex / flat local quadratic; no meaningful local minimizer
            x_star = np.nan
            x_next = x - lr * g                # gradient fallback
            step_types.append("gd_fallback")
            newton_targets.append(float(x_star))

        # Keep iterates within plotting bounds (visualization safeguard)
        x_next = float(np.clip(x_next, xlim[0], xlim[1]))

        xs.append(x_next)
        x = x_next

    xs = np.array(xs, dtype=float)
    newton_targets = np.array(newton_targets, dtype=float)

    # ---- Stable y-limits across frames ----
    yxs = f(xs)
    ymin = min(np.min(ygrid), np.min(yxs)) - 0.5
    ymax = max(np.max(ygrid), np.max(yxs)) + 0.5

    frames = []
    duration_ms = int(1000 / fps)
    quad_window = 1.2  # width (around xk) to plot the quadratic approximation

    for k in range(len(xs)):
        xk = float(xs[k])
        yk = float(f(xk))
        gk = float(df(xk))
        hk = float(ddf(xk))

        # Next iterate (if exists)
        if k < len(xs) - 1:
            xnext = float(xs[k + 1])
            ynext = float(f(xnext))
        else:
            xnext, ynext = xk, yk

        fig, ax = plt.subplots(figsize=(6.8, 4.2), dpi=120)

        # Objective curve
        ax.plot(grid, ygrid, linewidth=2, label="f(x)")

        # Current point
        ax.scatter([xk], [yk], s=55, zorder=6, label=f"iter {k}: x={xk:.3f}")

        # Local quadratic approximation (2nd-order Taylor)
        qxs = np.linspace(xk - quad_window, xk + quad_window, 240)
        qys = yk + gk * (qxs - xk) + 0.5 * hk * (qxs - xk) ** 2
        ax.plot(qxs, qys, linewidth=2, linestyle=":", label="local quadratic q_k(x)")

        # Newton target (quadratic minimizer) if curvature is positive
        if np.isfinite(hk) and hk > eps_curv:
            x_star = xk - gk / hk
            x_star = float(np.clip(x_star, xlim[0], xlim[1]))
            y_star = float(f(x_star))
            ax.scatter([x_star], [y_star], s=55, marker="x", zorder=7, label="quad min x*")

        # Arrow for the actual step taken
        if k < len(xs) - 1:
            ax.annotate(
                "",
                xy=(xnext, ynext),
                xytext=(xk, yk),
                arrowprops=dict(arrowstyle="->", linewidth=2),
            )
            ax.scatter([xnext], [ynext], s=32, zorder=5, label="next iterate")

        # Title and annotations
        if k == 0:
            stype = "start"
        elif k - 1 < len(step_types):
            stype = step_types[k - 1]
        else:
            stype = "end"

        subtitle = f"g={gk:.3f}, h={hk:.3f}"
        if stype == "newton":
            subtitle += f" | damped Newton: x←x - {lr}·g/h"
        elif stype == "gd_fallback":
            subtitle += f" | fallback GD: x←x - {lr}·g"

        ax.set_title(f"2nd-Order (Quadratic-Min) Optimization, step {k}/{steps}\n{subtitle}")

        ax.set_xlim(*xlim)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper left", fontsize=9)

        buf = io.BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        frames.append(Image.open(buf).convert("RGBA"))

    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        disposal=2,
    )
    return out_path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io

def make_gd_momentum_gif(
    x0=2.8,
    lr=0.08,
    steps=35,
    xlim=(-4, 4),
    ngrid=600,
    out_path="gradient_descent_momentum_1d.gif",
    fps=10
):
    """
    Creates a GIF illustrating 1D gradient descent with momentum.

    Requires you to have defined in scope:
        f(x): objective
        df(x): derivative

    Update rule (heavy-ball momentum):
        v_{k+1} = beta * v_k - lr * df(x_k)
        x_{k+1} = x_k + v_{k+1}

    Notes:
      - beta is set internally to a sensible default (0.9). Adjust below if desired.
      - Visualization shows:
          * f(x) curve
          * current point
          * tangent line at current point
          * arrow to next iterate
          * velocity annotation (magnitude/sign)
    """

    beta = 0.5          # momentum coefficient (heavy-ball)
    v_clip = 2.0         # cap velocity for visualization stability (optional safeguard)

    # ---- Generate iterates with momentum ----
    xs = [float(x0)]
    vs = [0.0]
    x = float(x0)
    v = 0.0

    for _ in range(steps):
        g = float(df(x))
        v = beta * v - lr * g
        v = float(np.clip(v, -v_clip, v_clip))         # visualization safeguard
        x = x + v
        x = float(np.clip(x, xlim[0], xlim[1]))         # keep in view
        xs.append(x)
        vs.append(v)

    xs = np.array(xs, dtype=float)
    vs = np.array(vs, dtype=float)

    # ---- Background curve + stable y-limits ----
    grid = np.linspace(xlim[0], xlim[1], ngrid)
    ygrid = f(grid)
    yxs = f(xs)

    ymin = min(np.min(ygrid), np.min(yxs)) - 0.5
    ymax = max(np.max(ygrid), np.max(yxs)) + 0.5

    frames = []
    duration_ms = int(1000 / fps)

    for k, xk in enumerate(xs):
        yk = float(f(xk))
        gk = float(df(xk))
        vk = float(vs[k])

        # Next point
        if k < len(xs) - 1:
            xnext = float(xs[k + 1])
            ynext = float(f(xnext))
        else:
            xnext, ynext = xk, yk

        fig, ax = plt.subplots(figsize=(6.8, 4.2), dpi=120)

        # Objective curve
        ax.plot(grid, ygrid, linewidth=2, label="f(x)")

        # Current point
        ax.scatter([xk], [yk], s=55, zorder=6, label=f"iter {k}: x={xk:.3f}")

        # Tangent line (local linearization)
        tangent_halfwidth = 0.6
        tx = np.array([xk - tangent_halfwidth, xk + tangent_halfwidth])
        ty = yk + gk * (tx - xk)
        ax.plot(tx, ty, linestyle="--", linewidth=1.5, label="tangent")

        # Arrow to next iterate
        if k < len(xs) - 1:
            ax.annotate(
                "",
                xy=(xnext, ynext),
                xytext=(xk, yk),
                arrowprops=dict(arrowstyle="->", linewidth=2),
            )
            ax.scatter([xnext], [ynext], s=32, zorder=5, label="next iterate")

        # Title + dynamics annotation
        ax.set_title(
            "Gradient Descent with Momentum (Heavy-Ball)\n"
            f"lr={lr}, beta={beta}, step {k}/{steps} | v={vk:+.4f}, g={gk:+.4f}"
        )

        ax.set_xlim(*xlim)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper left", fontsize=9)

        buf = io.BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        frames.append(Image.open(buf).convert("RGBA"))

    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        disposal=2,
    )
    return out_path



if __name__ == "__main__":
    path = make_gd_gif(
        x0=-2,
        lr=0.04,
        steps=15,
        xlim=(-4, 4),
        out_path="gd_1d_slow.gif",
        fps=1
    )
    path = make_gd_gif(
        x0=-2,
        lr=0.45,
        steps=8,
        xlim=(-4, 4),
        out_path="gd_1d_fast.gif",
        fps=1
    )
    path = make_2nd_gif(
        x0=-2,
        lr=0.4,
        steps=6,
        xlim=(-4, 4),
        out_path="gd_1d_quadratic.gif",
        fps=1
    )
    path = make_gd_momentum_gif(
        x0=-2,
        lr=0.04,
        steps=8,
        xlim=(-4, 4),
        out_path="gd_1d_momentum.gif",
        fps=1        
    )
    path = make_gd_momentum_gif(
        x0=-2,
        lr=0.6,
        steps=8,
        xlim=(-4, 4),
        out_path="gd_1d_overshoot.gif",
        fps=1
    )