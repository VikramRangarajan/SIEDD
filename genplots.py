import marimo

__generated_with = "0.14.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import plotly.express as px
    import plotly.graph_objects as go
    import polars as pl
    import plotly
    plotly.io.templates.default = "plotly"
    plotly.io.templates
    return pl, px


@app.cell
def _(pl):
    df = pl.read_excel("data.xlsx", sheet_name="Sheet1")
    df
    return (df,)


@app.cell
def _(df, px):
    fig6a = px.line(
        df,
        x="Bits Per Pixel (BPP)",
        y="PSNR (dB)",
        range_x=[0, 2.4],
        range_y=[30.01, 37],
        text="Label",
        color="Ablation",
        title="Ablation: PSNR (dB) vs. BPP",
        color_discrete_sequence=["blue", "red"],
        line_dash="Ablation",
        line_dash_sequence=["solid", "dash"],
    )
    fig6a.update_traces(textposition="top center")
    fig6a.update_layout(title_x=0.5, plot_bgcolor="white")
    fig6a.update_xaxes(gridcolor="lightgray")
    fig6a.update_yaxes(gridcolor="lightgray")
    fig6a.write_html("static/js/fig6a.html", include_plotlyjs="cdn")
    fig6a
    return


@app.cell
def _(df, px):
    fig6b = px.line(
        df,
        x="Average Encoding Time per Video (seconds)",
        y="PSNR (dB)",
        range_x=[850, 1800],
        range_y=[30.01, 37],
        text="Label",
        color="Ablation",
        title="Ablation: Encoding Time (s) vs. PSNR (dB)",
        color_discrete_sequence=["blue", "red"],
        line_dash="Ablation",
        line_dash_sequence=["solid", "dash"],
    )
    fig6b.update_traces(textposition="top center")
    fig6b.update_layout(title_x=0.5, plot_bgcolor="white")
    fig6b.update_yaxes(gridcolor="lightgray")
    fig6b.update_xaxes(gridcolor="lightgray")
    fig6b.write_html("static/js/fig6b.html", include_plotlyjs="cdn")
    fig6b
    return


@app.cell
def _(pl):
    fpsdf = pl.read_excel("data.xlsx", sheet_name="Sheet2").sort(
        by=["Model", "Frames per Second (FPS)"]
    )
    fpsdf
    return (fpsdf,)


@app.cell
def _(fpsdf, px):
    fig5a = px.bar(
        fpsdf,
        x="Resolution",
        y="Frames per Second (FPS)",
        color="Model",
        title="FPS vs Resolution across SIEDD Variants",
        color_discrete_sequence=["green", "blue", "orange"],
        barmode="group",
        log_y=True,
    )
    fig5a.update_layout(title_x=0.5, plot_bgcolor="white")
    fig5a.write_html("static/js/fig5a.html", include_plotlyjs="cdn")
    fig5a
    return


@app.cell
def _(pl):
    gpudf = (
        pl.read_excel(
            "data.xlsx",
            sheet_name="Sheet3",
            schema_overrides={"Number of GPUs": pl.String},
        )
        .sort(["Encoding Time (s)"], descending=True)
        .sort(["Number of GPUs"])
    )
    gpudf
    return (gpudf,)


@app.cell
def _(gpudf, px):
    fig5b = px.bar(
        gpudf,
        x="Number of GPUs",
        y="Encoding Time (s)",
        color="Method",
        title="Encoding Time vs. Number of GPUs",
        color_discrete_map={
            "NeRV": "lightblue",
            "Nirvana": "yellow",
            "HiNerv": "green",
            "HNerv": "orange",
            "Ours - L": "pink",
            "Ours - M": "chocolate",
            "Ours - S": "blue",
        },
        barmode="group",
        log_y=True,
    )
    fig5b.update_layout(title_x=0.5, plot_bgcolor="white")
    fig5b.write_html("static/js/fig5b.html", include_plotlyjs="cdn")
    fig5b
    return


@app.cell
def _(pl):
    modelsdf = pl.read_excel("data.xlsx", sheet_name="Sheet4")
    modelsdf
    return (modelsdf,)


@app.cell
def _(modelsdf, pl, px):
    fig2a = px.scatter(
        modelsdf.filter(
            ~pl.col("Model").is_in(
                [f"SIEDD-{x} (Ours)" for x in ["S", "M", "L"]] + ["Ours"]
            )
        ),
        x="Bits per Pixel (BPP)",
        y="PSNR (dB)",
        range_x=[-0.1, 1],
        range_y=[29.8, 39],
        text="Model",
        color="Model",
        title="Ablation: Bits per Pixel (BPP) vs. PSNR (dB)",
        color_discrete_sequence=[
            "blue",
            "orange",
            "brown",
            "green",
            "purple",
            "pink",
            "red",
        ],
        symbol_sequence=["star"],
    )
    fig2a.update_traces(textposition="top center", marker={"size": 20})
    fig2a.update_layout(title_x=0.5, plot_bgcolor="white")
    fig2a.update_xaxes(gridcolor="lightgray")
    fig2a.update_yaxes(gridcolor="lightgray")
    modelsdf2 = modelsdf.filter(pl.col("Model") == "Ours")
    # fig2a.add_trace(go.Scatter(x=modelsdf2["Bits per Pixel (BPP)"], y=modelsdf2["PSNR (dB)"]))
    fig2a.add_traces(
        list(
            px.line(
                modelsdf2,
                x="Bits per Pixel (BPP)",
                y="PSNR (dB)",
                color="Model",
                color_discrete_sequence=["red"],
                markers=["."],
            ).select_traces()
        )
    )
    fig2a.write_html("static/js/fig2a.html", include_plotlyjs="cdn")
    fig2a
    return


@app.cell
def _(modelsdf, pl, px):
    fig2b = px.scatter(
        modelsdf.filter(~pl.col("Model").is_in(["NVP", "Ours"])),
        x="Encoding Time (Minutes)",
        y="PSNR (dB)",
        range_x=[10, 500],
        range_y=[29.8, 39],
        text="Model",
        color="Model",
        title="PSNR (dB) vs. Encoding Time (Minutes) on UVG-HD",
        color_discrete_sequence=[
            "blue",
            "orange",
            "brown",
            "green",
            "purple",
            "red",
            "red",
            "red",
        ],
        log_x=True,
        size="Bits per Pixel (BPP)",
    )
    fig2b.update_traces(textposition="top center")
    fig2b.update_layout(title_x=0.5, plot_bgcolor="white")
    fig2b.update_xaxes(gridcolor="lightgray")
    fig2b.update_yaxes(gridcolor="lightgray")
    fig2b.write_html("static/js/fig2b.html", include_plotlyjs="cdn")
    fig2b
    return


if __name__ == "__main__":
    app.run()
