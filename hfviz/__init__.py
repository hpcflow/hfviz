import hpcflow.app as hf

import numpy as np
import plotly.express as px
from plotly import graph_objects as go


def make_workflow_figure(wk_path, submission_idx=0, max_jobscripts=None):
    """Generate a Plotly Figure object to visualise the specified workflow."""

    wk = hf.Workflow(wk_path)

    all_elems = []
    uniq_res = []
    is_array = []
    res_hashes = {}  # keys are hashes, values are tuples of IDs (first seen index), and list jobscript indices
    res_id = -1
    yax_labels = []
    task_iID = []
    task_names = []
    task_name_ids = []
    loop_iters = []
    loop_names = [i.name for i in wk.loops]
    plt_data = []
    max_elems = 0
    num_act_cumulative = 0
    hovertext = []
    js_count = 0
    for js in wk.submissions[submission_idx].jobscripts:
        js_count += 1
        num_act = 0
        res_hash = js.resources.get_jobscript_hash()

        if res_hash not in res_hashes:
            uniq_res.append(js.resources)
            is_array.append(js.is_array)
            res_id += 1
            res_hashes[res_hash] = (res_id, [js.index])
        else:
            res_id = res_hashes[res_hash][0]
            res_hashes[res_hash][1].append(js.index)

        js_max_elems = 0
        for blk_idx, blk in enumerate(js.blocks):
            if len(blk.task_elements) > js_max_elems:
                js_max_elems = len(blk.task_elements)

            for row_idx in range(len(blk.task_actions)):
                all_elems.append([res_id] * len(blk.task_elements))
                yax_labels.append(
                    f"{wk.tasks[blk.task_actions[row_idx][0]].insert_ID} ({blk.task_actions[row_idx][1]})"
                )

                task_iID.append(wk.tasks[blk.task_actions[row_idx][0]].insert_ID)

                task_name = wk.tasks[blk.task_actions[row_idx][0]].unique_name
                if task_name not in task_names:
                    task_names.append(task_name)
                    task_name_id = len(task_names) - 1
                else:
                    task_name_id = task_names.index(task_name)
                task_name_ids.append(task_name_id)

                loop_iters_i = []
                for loop_name in loop_names:
                    loop_idx = blk.task_loop_idx[blk.task_actions[row_idx][2]].get(
                        loop_name, -1
                    )
                    loop_iters_i.append(loop_idx)
                loop_iters.append(loop_iters_i)

                num_act_cumulative += 1
                num_act += 1
                hovertext.append(
                    [
                        f"js: {js.index} blk: {blk_idx}; elem: {elem_idx}"
                        for elem_idx in range(len(blk.task_elements))
                    ]
                )

        if js_max_elems > max_elems:
            max_elems = js_max_elems

        plt_data_i = {
            "num_actions": num_act,
            "num_actions_cumulative": num_act_cumulative,
            "max_num_elements": js_max_elems,
            "is_array": js.is_array,
        }
        plt_data.append(plt_data_i)

        if max_jobscripts and js.index >= max_jobscripts:
            break

    A = np.ones((len(all_elems), max_elems)) * np.nan
    for idx, i in enumerate(all_elems):
        A[idx, : len(i)] = i

    traces = []
    for idx, res_obj in enumerate(uniq_res):
        B = np.copy(A)
        B[B != idx] = np.nan
        legend_name = ""
        for k, v in res_obj._get_repr_fields().items():
            # v = textwrap.shorten(str(v), width=20, placeholder="...")
            legend_name += f"{k}: {v}<br>"

        legend_name += f"is_array: {is_array[idx]}<br>"

        colour = px.colors.qualitative.Plotly[idx % len(px.colors.qualitative.Plotly)]
        heatmap = go.Heatmap(
            z=B,
            colorscale=[[0.0, colour], [1.0, colour]],
            showscale=False,
            zauto=False,
            zmin=0,
            zmax=2,
            showlegend=True,
            name=legend_name,
            hoverinfo="text",
            text=hovertext,
            ygap=2,
            xgap=2,
        )
        # for the final (i.e. on-top) heatmap, include the jobscript and block indices as hover
        # info
        traces.append(heatmap)

        # jobscript index:
        traces.append(
            go.Heatmap(
                z=[
                    [js_idx]
                    for js_idx, i in enumerate(plt_data)
                    for j in range(i["num_actions"])
                ],
                xaxis="x2",
                # colorscale=cscale,
                showscale=False,
                showlegend=False,
            )
        )

    # task indices:
    hovertext_tasknames = [[f"{task_names[i]}"] for i in task_name_ids]
    task_name_ids_arr = np.array(task_name_ids)[:, None].astype("float")
    for task_name_id, task_name in enumerate(task_names):
        task_i_arr = np.copy(task_name_ids_arr)
        task_i_arr[task_i_arr != task_name_id] = np.nan
        legend_name = f"{task_name}"
        colour = px.colors.qualitative.Pastel[
            task_name_id % len(px.colors.qualitative.Pastel)
        ]
        heatmap = go.Heatmap(
            z=task_i_arr,
            xaxis="x3",
            colorscale=[[0.0, colour], [1.0, colour]],
            showscale=False,
            showlegend=True,
            name=legend_name,
            hoverinfo="text",
            text=hovertext_tasknames,
            legend="legend2",
        )
        # for the final (i.e. on-top) heatmap, include the jobscript and block indices as hover
        # info
        traces.append(heatmap)

    loop_cscales = [
        "Blues",
        "Greens",
        "Oranges",
        "Purples",
        "Reds",
    ]

    # loop indices
    loop_iters_arr = np.array(loop_iters, dtype=float)
    loop_iters_arr[loop_iters_arr == -1] = np.nan
    loop_task_boundaries = np.abs(
        np.diff(np.isfinite(loop_iters_arr).astype(int), axis=0)
    )
    loop_borders = np.where(loop_task_boundaries == 1)
    hovertext_loopnames = [
        [
            f"{loop_name}: {act_row[idx]}" if act_row[idx] != -1 else ""
            for idx, loop_name in enumerate(loop_names)
        ]
        for act_row in loop_iters
    ]
    for loop_name_idx, loop_name in enumerate(loop_names):
        loop_i_arr = np.copy(loop_iters_arr)
        loop_i_arr[:, [i for i in range(len(loop_names)) if i != loop_name_idx]] = (
            np.nan
        )
        legend_name = f"{loop_name}"
        colour_scale = loop_cscales[loop_name_idx % len(loop_cscales)]
        loop_heatmap_i = go.Heatmap(
            z=loop_i_arr,
            xaxis="x4",
            showscale=False,
            showlegend=True,
            name=legend_name,
            legendgroup=legend_name,
            hoverinfo="text",
            text=hovertext_loopnames,
            legend="legend3",
            colorscale=colour_scale,
        )
        traces.append(loop_heatmap_i)

        # boxes around loops:
        loop_borders_idx = np.where(loop_borders[1] == loop_name_idx)
        y_i = loop_borders[0][loop_borders_idx]
        if y_i.shape[0] % 2 == 1:
            y_i = np.concatenate((y_i, [num_act_cumulative - 1]))
        y_i = y_i.astype(float)
        y_i += 0.5
        y_i = y_i.reshape((int(y_i.shape[0] / 2), 2))
        y_i = np.repeat(y_i, 2, axis=1)
        y_i = np.hstack((y_i, y_i[:, 0:1]))
        y_i = np.hstack((y_i, np.ones((y_i.shape[0], 1)) * np.nan))
        y_i = np.reshape(y_i, -1)[:-1]

        x_i = np.array(
            [
                loop_name_idx - 0.5,
                loop_name_idx + 0.5,
                loop_name_idx + 0.5,
                loop_name_idx - 0.5,
                loop_name_idx - 0.5,
            ]
        )
        x_i = np.tile(x_i, (y_i.shape[0], 1))
        x_i = np.hstack((x_i, np.ones((x_i.shape[0], 1)) * np.nan))
        x_i = np.reshape(x_i, -1)[:-1]

        traces.append(
            go.Scatter(
                x=x_i,
                y=y_i,
                showlegend=False,
                name=legend_name,
                legendgroup=legend_name,
                xaxis="x4",
                mode="lines",
                line={
                    "color": getattr(px.colors.sequential, colour_scale)[-1],
                    "width": 0.5,
                },
                hoverinfo="none",
            )
        )

    plot_height = num_act_cumulative * 10  # height in pixels
    task_legend_height = (
        len(task_names) * 28
    )  # estimated height of task-legend in pixels
    task_legend_height_frac = (
        task_legend_height / plot_height
    )  # height of task-legend as plot height fraction

    fig = go.Figure(
        data=traces,
        layout={
            "xaxis": {
                "side": "top",
                "domain": [0.150, 1.0],
                "title": "Elements",
                "constraintoward": "left",
                # "showgrid": False,
                "zeroline": False,
            },
            "yaxis": {
                "autorange": "reversed",
                "autorangeoptions": {"minallowed": -0.5},
                "showticklabels": False,
                "scaleanchor": "x1",
                "constraintoward": "top",
                "showgrid": False,
                "zeroline": False,
            },
            "xaxis2": {
                "domain": [0.100, 0.125],
                "showticklabels": False,
                "side": "top",
                "title": "Jobscript",
                "showgrid": False,
                "zeroline": False,
            },
            "xaxis4": {
                "domain": [0.050, 0.075],
                "showticklabels": False,
                "side": "top",
                "title": "Loops",
                "showgrid": False,
                "zeroline": False,
            },
            "xaxis3": {
                "domain": [0, 0.025],
                "showticklabels": False,
                "side": "top",
                "title": "Task",
                "showgrid": False,
                "zeroline": False,
            },
            "legend": {
                "title": {
                    "text": "Unique resource requirements",
                    "side": "top center",
                    "font": {
                        "size": 15,
                    },
                },
                "font": {
                    "size": 10,
                },
                "orientation": "h",
                "x": 0.2,
                "y": 1,
                "yref": "container",
                "xanchor": "left",
                "yanchor": "top",
            },
            "legend2": {
                "title": {
                    "text": "Tasks",
                    "side": "top center",
                    "font": {
                        "size": 15,
                    },
                },
                "font": {
                    "size": 10,
                },
                "x": 0,
                "y": 1.0,
                "xref": "container",
                "bgcolor": "rgb(220,220,220)",
                "xanchor": "left",
                "yanchor": "top",
            },
            "legend3": {
                "title": {
                    "text": "Loops",
                    "side": "top center",
                    "font": {
                        "size": 15,
                    },
                },
                "font": {
                    "size": 10,
                },
                "x": 0,
                "y": 1.0 - task_legend_height_frac,
                "xref": "container",
                "bgcolor": "rgb(220,220,220)",
                "xanchor": "left",
                "yanchor": "top",
            },
        },
    )

    for js_idx, i in enumerate(plt_data):
        y0 = 0 if js_idx == 0 else plt_data[js_idx - 1]["num_actions_cumulative"]
        y1 = i["num_actions_cumulative"] - 0.5
        (
            fig.add_hrect(
                y0=y0 - 0.5,
                y1=y1,
                xref="x2",
                line={"width": 1, "color": "silver"},
                fillcolor="lightgray",
            )
        )
        fig.add_annotation(
            x=0,
            y=((y0 + y1) / 2) - 0.3,
            text=f"{js_idx}",
            xref="x2",
            showarrow=False,
            font={"color": "gray"},
        )

    fig.update_layout(height=plot_height)

    return fig
