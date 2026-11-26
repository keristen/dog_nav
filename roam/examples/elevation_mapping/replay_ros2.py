#!/usr/bin/env python3
"""Replay the Seealpsee clip as ROS 2 sensors and TF (wall time, no /clock)."""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import cv2
from PIL import Image as PILImage
from scipy.spatial.transform import Rotation

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from std_msgs.msg import Header
from tf2_ros import TransformBroadcaster


class Replay(Node):
    def __init__(self, args):
        super().__init__('seealpsee_replay')
        self.args = args
        with np.load(args.data / 'clip.npz') as data:
            self.clip = {key: data[key] for key in data.files}
        self.events = sorted(
            [(float(t), 'cloud', i) for i, t in enumerate(self.clip['cloud_stamps'])]
            + [(float(t), 'image', i) for i, t in enumerate(self.clip['sem_stamps'])]
        )
        self.first = self.events[0][0]
        self.duration = (self.events[-1][0] - self.first) / args.rate
        self.pubs = {
            'cloud': self.create_publisher(PointCloud2, '/lidar/points', 10),
            'image': self.create_publisher(Image, '/zed/zed_node/rgb/image_rect_color', 10),
            'info': self.create_publisher(CameraInfo, '/zed/zed_node/rgb/camera_info', 10),
        }
        if args.publish_recorded_semantics:
            self.pubs['sem'] = self.create_publisher(Image, '/zed/zed_node/rgb/vlm_nav_sem', 10)
            self.pubs['sem_info'] = self.create_publisher(CameraInfo, '/zed/zed_node/rgb/camera_info_sem', 10)
            self.pubs['sem_viz'] = self.create_publisher(Image, '/zed/zed_node/rgb/vlm_nav_sem_viz', 10)
        self.tf = TransformBroadcaster(self)
        self.reset()
        self.create_timer(0.01, self.tick)
        self.get_logger().info(
            f'Replaying {len(self.events)} events at {args.rate}x; recorded semantics: '
            f'{args.publish_recorded_semantics}. Starting in {args.start_delay}s.'
        )

    def reset(self):
        self.wall_start = time.monotonic() + self.args.start_delay
        self.epoch_ns = self.get_clock().now().nanoseconds + int(self.args.start_delay * 1e9)
        self.tf_index = self.data_index = 0
        self.finished = False

    def stamp(self, t):
        return Time(nanoseconds=self.epoch_ns + round((t - self.first) / self.args.rate * 1e9)).to_msg()

    def transform(self, matrix, child, stamp):
        msg = TransformStamped()
        msg.header = Header(stamp=stamp, frame_id='odom')
        msg.child_frame_id = child
        p, q = matrix[:3, 3], Rotation.from_matrix(matrix[:3, :3]).as_quat()
        msg.transform.translation.x, msg.transform.translation.y, msg.transform.translation.z = map(float, p)
        msg.transform.rotation.x, msg.transform.rotation.y, msg.transform.rotation.z, msg.transform.rotation.w = map(float, q)
        return msg

    def camera_info(self, header):
        height, width = self.clip['sem'].shape[1:3]
        k = self.clip['K'].astype(float)
        msg = CameraInfo(header=header, height=height, width=width)
        distortion = str(self.clip['distortion_model'])
        msg.distortion_model = 'plumb_bob' if distortion == 'radtan' else distortion
        msg.d = self.clip['D'].astype(float).tolist()
        if msg.distortion_model == 'plumb_bob' and len(msg.d) == 4:
            msg.d.append(0.0)
        msg.k = k.ravel().tolist()
        msg.r = np.eye(3).ravel().tolist()
        msg.p = np.column_stack((k, np.zeros(3))).ravel().tolist()
        return msg

    @staticmethod
    def image(array, header, encoding):
        array = np.ascontiguousarray(array)
        return Image(header=header, height=array.shape[0], width=array.shape[1],
                     encoding=encoding, is_bigendian=sys.byteorder == 'big',
                     step=array.strides[0], data=array.tobytes())

    def publish_event(self, event):
        t, kind, i = event
        header = Header(stamp=self.stamp(t), frame_id='lidar' if kind == 'cloud' else 'camera_optical_frame')
        if kind == 'cloud':
            start, end = self.clip['cloud_splits'][i:i + 2]
            # The recording stores centimetres; PointCloud2/TF use metres.
            xyz = np.ascontiguousarray(self.clip['cloud_xyz'][start:end], dtype='<f4') * 0.01
            fields = [PointField(name=name, offset=j * 4, datatype=PointField.FLOAT32, count=1)
                      for j, name in enumerate(('x', 'y', 'z'))]
            msg = PointCloud2(header=header, height=1, width=len(xyz), fields=fields,
                             is_bigendian=False, point_step=12, row_step=len(xyz) * 12,
                             is_dense=bool(np.isfinite(xyz).all()), data=xyz.tobytes())
            self.pubs['cloud'].publish(msg)
        else:
            h, w = self.clip['sem'].shape[1:3]
            # K belongs to the semantic resolution, so resize RGB to the same size.
            with PILImage.open(self.args.data / 'images' / f'{i:04d}.jpg') as im:
                rgb = np.asarray(im.convert('RGB').resize((w, h), PILImage.Resampling.BILINEAR))
            info = self.camera_info(header)
            self.pubs['info'].publish(info)
            self.pubs['image'].publish(self.image(rgb, header, 'rgb8'))
            if self.args.publish_recorded_semantics:
                self.pubs['sem_info'].publish(info)
                sem = self.clip['sem'][i].astype(np.float32) / 255.0
                self.pubs['sem'].publish(self.image(sem, header, '32FC3'))
                heatmaps = [cv2.applyColorMap(self.clip['sem'][i, :, :, channel], cv2.COLORMAP_JET)
                            for channel in range(3)]
                viz = np.concatenate([cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), *heatmaps], axis=1)
                self.pubs['sem_viz'].publish(self.image(viz, header, 'bgr8'))

    def tick(self):
        if self.finished:
            return
        elapsed = time.monotonic() - self.wall_start
        # Send TF 0.2 seconds before the matching sensor data so listeners can buffer it.
        while self.tf_index < len(self.events):
            t, kind, i = self.events[self.tf_index]
            if (t - self.first) / self.args.rate > elapsed:
                break
            stamp = self.stamp(t)
            if kind == 'cloud':
                transforms = [self.transform(self.clip['cloud_T_odom_base'][i], 'base', stamp),
                              self.transform(self.clip['cloud_T_odom_lidar'][i], 'lidar', stamp)]
            else:
                transforms = [self.transform(self.clip['sem_T_odom_cam'][i], 'camera_optical_frame', stamp)]
            self.tf.sendTransform(transforms)
            self.tf_index += 1
        while self.data_index < len(self.events):
            event = self.events[self.data_index]
            if (event[0] - self.first) / self.args.rate + 0.2 > elapsed:
                break
            self.publish_event(event)
            self.data_index += 1
        if self.data_index == len(self.events):
            if self.args.loop:
                self.get_logger().info('Clip finished; looping (robot returns to its initial pose).')
                self.reset()
            else:
                self.finished = True
                self.get_logger().info('Clip finished. Publishers remain alive; Ctrl+C to exit.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path(__file__).parent / 'data/seealpsee_trail')
    parser.add_argument('--rate', type=float, default=1.0, help='Playback speed; 0.25 is four times slower')
    parser.add_argument('--start-delay', type=float, default=5.0)
    parser.add_argument('--loop', action='store_true')
    parser.add_argument('--publish-recorded-semantics', action='store_true',
                        help='Also publish saved semantic layers; do not run vlm_nav_node at the same time')
    args = parser.parse_args()
    if not np.isfinite(args.rate) or args.rate <= 0 or not np.isfinite(args.start_delay) or args.start_delay < 0:
        parser.error('--rate must be positive and --start-delay must be nonnegative (both finite)')
    rclpy.init()
    node = None
    try:
        node = Replay(args)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
