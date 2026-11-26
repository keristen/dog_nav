#!/usr/bin/env python3
"""Show a live elevation_mapping_cupy GridMap as a draggable 3D surface."""

import argparse
import threading

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - registers the 3D projection

import rclpy
from grid_map_msgs.msg import GridMap
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node


def decode_layer(message, layer_name):
    """Decode the GridMap column-major Float32MultiArray into (rows, columns)."""
    if layer_name not in message.layers:
        raise KeyError(f"Layer '{layer_name}' is absent. Available layers: {', '.join(message.layers)}")
    layer_index = message.layers.index(layer_name)
    array_message = message.data[layer_index]
    dimensions = array_message.layout.dim
    values = np.asarray(array_message.data, dtype=np.float32)

    if len(dimensions) >= 2:
        first, second = dimensions[:2]
        if first.label == "column_index" and second.label == "row_index":
            rows, columns = second.size, first.size
            result = values.reshape((rows, columns), order="F")
        elif first.label == "row_index" and second.label == "column_index":
            rows, columns = first.size, second.size
            result = values.reshape((rows, columns), order="C")
        else:
            columns, rows = first.size, second.size
            result = values.reshape((rows, columns), order="F")
    else:
        raise ValueError(f"Layer '{layer_name}' has no 2D GridMap layout")
    return result


class GridMapViewer(Node):
    def __init__(self, topic):
        super().__init__("gridmap_3d_viewer")
        self._lock = threading.Lock()
        self._latest = None
        self.create_subscription(GridMap, topic, self._callback, 1)
        self.get_logger().info(f"Waiting for GridMap on {topic}")

    def _callback(self, message):
        with self._lock:
            self._latest = message

    def latest(self):
        with self._lock:
            return self._latest


def quaternion_yaw(orientation):
    x, y, z, w = orientation.x, orientation.y, orientation.z, orientation.w
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def make_coordinates(message, shape):
    rows, columns = shape
    resolution = message.info.resolution
    center = message.info.pose.position
    x = center.x + message.info.length_x / 2 - (np.arange(rows) + 0.5) * resolution
    y = center.y + message.info.length_y / 2 - (np.arange(columns) + 0.5) * resolution
    xx, yy = np.meshgrid(x, y, indexing="ij")
    yaw = quaternion_yaw(message.info.pose.orientation)
    if abs(yaw) > 1e-8:
        c, s = np.cos(yaw), np.sin(yaw)
        xx, yy = center.x + c * (xx - center.x) - s * (yy - center.y), center.y + s * (xx - center.x) + c * (yy - center.y)
    return xx, yy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topic", default="/elevation_mapping_node/elevation_map_raw")
    parser.add_argument("--height-layer", default="elevation", help="GridMap layer used as surface height")
    parser.add_argument("--color-layer", default="elevation", help="GridMap layer used for surface color (e.g. risk)")
    parser.add_argument("--stride", type=int, default=2, help="Plot every Nth cell to keep the interactive view responsive")
    parser.add_argument("--refresh-ms", type=int, default=700, help="GUI refresh interval in milliseconds")
    args = parser.parse_args()
    if args.stride < 1 or args.refresh_ms < 100:
        parser.error("--stride must be >= 1 and --refresh-ms must be >= 100")

    rclpy.init()
    node = GridMapViewer(args.topic)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    ros_thread = threading.Thread(target=executor.spin, daemon=True)
    ros_thread.start()

    fig = plt.figure(figsize=(11, 8))
    state = {"stamp": None, "elev": 35, "azim": -55, "zoom": 1.0}
    waiting = fig.add_subplot(111)
    waiting.text(0.5, 0.5, f"Waiting for {args.topic}", ha="center", va="center")
    waiting.axis("off")
    warned = set()

    def refresh():
        message = node.latest()
        if message is None:
            return
        stamp = (message.header.stamp.sec, message.header.stamp.nanosec)
        if stamp == state["stamp"]:
            return
        try:
            height = decode_layer(message, args.height_layer)
            color_values = decode_layer(message, args.color_layer)
        except (KeyError, ValueError) as exc:
            warning = str(exc)
            if warning not in warned:
                node.get_logger().warning(warning)
                warned.add(warning)
            return

        step = args.stride
        shape = height.shape
        height = height[::step, ::step]
        color_values = color_values[::step, ::step]
        x, y = make_coordinates(message, shape)
        x, y = x[::step, ::step], y[::step, ::step]
        valid = np.isfinite(height) & np.isfinite(color_values)
        if valid.sum() < 4:
            return

        if state["stamp"] is not None:
            previous = fig.axes[0]
            state["elev"], state["azim"] = previous.elev, previous.azim
            state["limits"] = previous.get_xlim(), previous.get_ylim(), previous.get_zlim()
        fig.clear()
        ax = fig.add_subplot(111, projection="3d")
        if args.color_layer == "risk":
            vmin, vmax, cmap = 0.0, 1.0, "RdYlGn_r"
        else:
            vmin, vmax = np.nanpercentile(color_values[valid], [2, 98])
            if not np.isfinite(vmin + vmax) or vmax <= vmin:
                vmin, vmax = float(np.nanmin(color_values[valid])), float(np.nanmax(color_values[valid]) + 1e-6)
            cmap = "terrain"
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        ax.plot_surface(
            x, y, np.where(valid, height, np.nan),
            facecolors=cm.get_cmap(cmap)(norm(color_values)),
            linewidth=0, antialiased=False, shade=False,
            rstride=1, cstride=1,
        )
        scalar = cm.ScalarMappable(norm=norm, cmap=cmap)
        scalar.set_array(color_values[valid])
        fig.colorbar(scalar, ax=ax, shrink=0.65, pad=0.1, label=args.color_layer)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_zlabel(f"{args.height_layer} [m]")
        ax.set_title(f"Fused elevation map  |  height: {args.height_layer}, color: {args.color_layer}")
        ax.view_init(elev=state["elev"], azim=state["azim"])
        aspect = (message.info.length_x, message.info.length_y, max(message.info.length_x * 0.25, 1.0))
        ax.set_box_aspect(aspect, zoom=state["zoom"])
        if "limits" in state:
            ax.set_xlim(state["limits"][0])
            ax.set_ylim(state["limits"][1])
            ax.set_zlim(state["limits"][2])
        fig.tight_layout()
        fig.canvas.draw_idle()
        state["stamp"] = stamp
        node.get_logger().info(
            f"Displayed {args.height_layer}/{args.color_layer}: {height.shape[0]}x{height.shape[1]} cells, "
            f"{valid.sum()} valid"
        )

    def on_scroll(event):
        if state["stamp"] is None or event.inaxes is not fig.axes[0]:
            return
        direction = event.step if event.step else (1 if event.button == "up" else -1)
        state["zoom"] = float(np.clip(state["zoom"] * 1.2 ** direction, 0.3, 4.0))
        fig.axes[0].set_box_aspect(aspect, zoom=state["zoom"])
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("scroll_event", on_scroll)
    timer = fig.canvas.new_timer(interval=args.refresh_ms)
    timer.add_callback(refresh)
    timer.start()
    try:
        plt.show()
    finally:
        timer.stop()
        executor.shutdown(timeout_sec=1.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        ros_thread.join(timeout=1.0)


if __name__ == "__main__":
    main()
