import plotly.graph_objects as go

def build_animated_fig(surface, xs, ys, zs, title=None):
    """
    surface: an instance of LossSurface (e.g. QuadraticBowl() or Rosenbrock())
    xs, ys, zs: 1D arrays from run_sgd(surface, ...)
    title: optional string for plot title
    """

    # 1. Build the static loss surface (the bowl / banana)
    X, Y, Z = surface.make_grid(n=100)
    surface_trace = go.Surface(
        x=X,
        y=Y,
        z=Z,
        showscale=False,
        opacity=0.8,
        name="surface",
    )

    # We'll generate frames where we:
    # - draw the trail up to step k
    # - draw the current head point at step k
    frames = []
    for k in range(len(xs)):
        trail = go.Scatter3d(
            x=xs[:k+1],
            y=ys[:k+1],
            z=zs[:k+1],
            mode="lines+markers",
            marker=dict(size=4),
            line=dict(width=6),
            name="SGD path"
        )

        head = go.Scatter3d(
            x=[xs[k]],
            y=[ys[k]],
            z=[zs[k]],
            mode="markers+text",
            marker=dict(size=6, symbol="circle"),
            text=[f"f={zs[k]:.3f}"],
            textposition="top center",
            name="current point"
        )

        # each frame contains ALL visible traces at that timestep
        frames.append(
            go.Frame(
                data=[surface_trace, trail, head],
                name=f"step{k}"
            )
        )

    # 2. Initial state (before animation runs, show step 0)
    init_trail = go.Scatter3d(
        x=[xs[0]],
        y=[ys[0]],
        z=[zs[0]],
        mode="lines+markers",
        marker=dict(size=4),
        line=dict(width=6),
        name="SGD path"
    )

    init_head = go.Scatter3d(
        x=[xs[0]],
        y=[ys[0]],
        z=[zs[0]],
        mode="markers+text",
        marker=dict(size=6, symbol="circle"),
        text=[f"f={zs[0]:.3f}"],
        textposition="top center",
        name="current point"
    )

    # 3. Build the figure with buttons & slider
    fig = go.Figure(
        data=[surface_trace, init_trail, init_head],
        layout=go.Layout(
            title = title if title is not None else f"{type(surface).__name__} descent",
            scene = dict(
                xaxis_title='x',
                yaxis_title='y',
                zaxis_title='f(x,y)',
            ),
            width=800,
            height=700,

            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    x=0.05,
                    y=0,
                    xanchor="left",
                    yanchor="bottom",
                    buttons=[
                        dict(
                            label="Play",
                            method="animate",
                            args=[
                                None,  # None => play all frames in order
                                {
                                    "frame": {"duration": 200, "redraw": True},
                                    "transition": {"duration": 0},
                                    "fromcurrent": True,
                                },
                            ],
                        ),
                        dict(
                            label="Pause",
                            method="animate",
                            args=[
                                [None],
                                {
                                    "frame": {"duration": 0, "redraw": False},
                                    "transition": {"duration": 0},
                                },
                            ],
                        ),
                    ],
                )
            ],

            sliders=[
                dict(
                    steps=[
                        dict(
                            method="animate",
                            label=str(k),
                            args=[
                                [f"step{k}"],
                                {
                                    "frame": {"duration": 0, "redraw": True},
                                    "transition": {"duration": 0},
                                },
                            ],
                        )
                        for k in range(len(frames))
                    ],
                    active=0,
                    x=0.2,
                    y=0,
                    xanchor="left",
                    yanchor="bottom",
                    len=0.7,
                )
            ],
        ),
        frames=frames
    )

    # optional: add contour projection to the surface for nice visuals
    fig.update_traces(
        selector=dict(type="surface"),
        contours=dict(
            z=dict(
                show=True,
                usecolormap=True,
                project=dict(z=True)
            )
        )
    )

    return fig
